"""Radio sources for the live watcher (``sferic_watch.live``): the real SDRplay RSPdx through SoapySDR, and a mock that
replays banked .cs16 captures with fault injection.

Both deliver ``Block`` objects (interleaved int16 I,Q, the bytes the recorder writes) from a reader thread through a
bounded ring buffer, so a slow consumer never blocks the radio: when the ring is full the OLDEST block is dropped,
counted (``ring_drops``) and the next block is marked ``gap_before``.

The recipe (``SoapyRadio``) is the one every capture in the corpus was made with (dlayer_run.py ``open_sdr`` /
``capture``): driver ``sdrplay``, 62.5 kS/s, centre 15 kHz, antenna "Antenna C" (configurable), AGC off, IFGR 40,
RFGR 0, bias-T / RF notch / DAB notch / HDR off, CS16 stream, a 1 s settle read that is thrown away, then
``readStream`` of up to 65536 samples with a 1 s timeout. Return codes as in the recorder: > 0 samples, -4 overflow,
-1 timeout.

Time stamps
-----------
The recorder stamps its sidecar ``utc`` from the host clock right after the settle reads, before the first kept
read. Against GOES GLM that stamp is late: sample 0 is +25.9 ms older than it (pooled lag, bench/glm_truth.py; per
capture p25-p75 24.8-26.9 ms), because the first read returns samples already waiting in the driver's buffers. Here
the same stamp is taken (host clock just before the first read after an anchor) and corrected by
``STAMP_OFFSET_S`` = 0.026 s, so ``Block.t0`` is the best estimate of the UTC time of the block's first sample.
After the anchor the time of every later sample is anchor + samples / fs (the sample clock; its error against the
MSK transmitters is about -0.7 ppm, i.e. 2.5 ms per hour).

The host clock is still checked on every read: ``lag`` = host time when a read returns minus the sample-clock
time of its last sample. Its minimum over 16 reads is steady (buffering only adds); the first such minimum after an
open is kept as the steady lag. A sudden change of that minimum by more than ``SLIP_S`` (20 ms) between two 16-read
windows is a host clock step or a silent sample slip (dropped or duplicated samples): ``clock_steps`` counts it and
the next block starts a new segment (``gap_before``) with a new anchor. It is seen 16-32 reads (about 17-34 s at
65536 samples a read) after it happened; blocks in between keep the old time line. A slow walk past SLIP_S (sample
clock drift against NTP) re-anchors without a gap (``reanchored``). ``clock_drift_ppm`` is the slope of that
minimum against time (host clock vs sample clock, i.e. NTP vs the radio's crystal), None until 10 minutes have been
read on one anchor. Slips smaller than 20 ms (about 1250 samples) are NOT detectable this way; they shift later
event times by the slip.

Anchors after the first: after a gap the block's time is (host time when the read returned) - (its length) -
(steady lag), the same calibration as the stamp rule, because the steady lag was measured against that first anchor.
One read's return time is late by that read's jitter (scheduling, USB), so this anchor is refined once its first
16-read lag minimum is in: the anchor moves by (that minimum - steady lag) and the next block carries the move in
``anchor_shift_s`` for the samples already delivered (simulated at 10 ms mean jitter: median error 8.3 -> 0.1 ms,
max 41 -> 3 ms). A slow-walk re-anchor moves the continuous time line by the walk measured on the lag minimum (no
single read involved), and the drift reference moves with it so ``clock_drift_ppm`` keeps reading the walk.

``SFERIC_WATCH_NO_SDR`` set in the environment makes ``SoapyRadio`` refuse to open the device (tests and the stress
bench set it).

Faults, and what the stream does with them
------------------------------------------
overflow (-4)   samples lost, count unknown: ``overflows`` += 1, next block ``gap_before`` with a new anchor.
timeout (-1)    no samples for 1 s: ``timeouts`` += 1. Not a gap by itself (the driver may still deliver them): the
                next read's lag is compared with the steady lag, and more than ``TIMEOUT_LOSS_S`` (0.1 s) behind
                means samples were lost (``timeout_gaps``): that block starts a new segment.
short read      fewer samples than asked: normal for this driver, counted in ``short_reads`` only.
exception       device gone or driver error: ``read_errors`` += 1, the stream is closed and the device reopened
                with backoff 1, 2, 4 ... 60 s (``reopens``, ``open_failures``); the next block is ``gap_before``.
restart()       the consumer's watchdog asks for a close and reopen (stalled stream).
"""
from __future__ import annotations

import collections
import json
import os
import threading
import time
from dataclasses import asdict, dataclass, field

import numpy as np

BACKOFF_RESET_S = 60.0        # a stream that ran this long resets the reopen backoff
FS = 62500.0
FC = 15000.0
ANTENNA = "Antenna C"
IFGR, RFGR = 40, 0
SETTINGS = {"biasT_ctrl": "false", "rfnotch_ctrl": "false", "dabnotch_ctrl": "false", "hdr_ctrl": "false"}
CHUNK = 65536                 # samples per readStream call (the recorder's buffer)
READ_TIMEOUT_US = 1_000_000
SETTLE_S = 1.0                # settle: this much is read and thrown away after activateStream (recorder: 1 s)
SETTLE_MAX_S = 5.0
STAMP_OFFSET_S = 0.026        # sample 0 is this much older than the host stamp taken before the first read
SLIP_S = 0.020                # host-vs-sample clock change that counts as a step / slip
REFINE_MAX_S = 0.25           # a gap anchor (one read) is late by that read's jitter; refined by up to this much
TIMEOUT_LOSS_S = 0.1          # after a timeout, a read this much later than the steady lag means samples were lost
LAG_WIN = 16                  # reads per lag-minimum window
DRIFT_MIN_S = 600.0           # clock drift slope reported after this much anchored time
RING_S = 120.0                # ring buffer capacity, seconds of signal
BACKOFF_S = (1, 2, 4, 8, 16, 32, 60)
RET_TIMEOUT, RET_OVERFLOW = -1, -4
NO_SDR_ENV = "SFERIC_WATCH_NO_SDR"   # set (tests, bench, scratch runs): SoapyRadio refuses to open the device


@dataclass
class Block:
    """One read's worth of samples. ``iq`` is interleaved int16 (I0, Q0, I1, ...). ``t0`` the UTC epoch of the
    first sample (stamp offset corrected). ``gap_before``: samples were (or may have been) lost just before this
    block, so it starts a new segment. ``reanchored``: continuous with the previous block, but ``t0`` comes from a
    new host-clock anchor (slow drift correction). ``anchor_shift_s``: the time line since the last anchor moved by
    this much (the refinement after a gap anchor); the consumer shifts the samples it already holds from that
    anchor."""
    seq: int
    t0: float
    iq: np.ndarray
    fs: float = FS
    gap_before: bool = False
    reanchored: bool = False
    anchor_shift_s: float = 0.0       # the anchor in force was refined by this (t0 includes it; see the module doc)
    counters: dict | None = None      # the fault counters when this block was read (``RadioStats.counters``)

    @property
    def n(self) -> int:
        return len(self.iq) // 2


@dataclass
class RadioStats:
    blocks: int = 0
    samples: int = 0
    overflows: int = 0
    timeouts: int = 0
    timeout_gaps: int = 0
    short_reads: int = 0
    read_errors: int = 0
    reopens: int = 0
    open_failures: int = 0
    ring_drops: int = 0
    ring_dropped_samples: int = 0
    clock_steps: int = 0
    reanchors: int = 0
    anchor_refines: int = 0
    restarts: int = 0
    lag_min_ms: float | None = None
    clock_drift_ppm: float | None = None
    last_error: str | None = None
    last_block_wall: float | None = None
    state: str = "idle"           # idle / opening / settling / streaming / backoff / stopped

    def counters(self) -> dict:
        """The fault counters a minute row reports (their change over the minute)."""
        return {"overflows": self.overflows, "timeouts": self.timeouts, "ring_drops": self.ring_drops,
                "clock_steps": self.clock_steps, "read_errors": self.read_errors, "reopens": self.reopens}

    def as_dict(self) -> dict:
        d = asdict(self)
        for k in ("lag_min_ms", "clock_drift_ppm"):
            if d[k] is not None:
                d[k] = round(d[k], 3)
        return d


class RadioSource:
    """Reader thread + bounded ring. Subclasses implement ``_open``, ``_read(buf) -> int`` (samples into buf, or a
    negative return code), ``_close`` and may override ``now`` (the host clock, UTC epoch seconds)."""

    kind = "base"
    settle_s = SETTLE_S

    def __init__(self, fs: float = FS, fc: float = FC, ring_s: float = RING_S, chunk: int = CHUNK,
                 lossless: bool = False):
        self.fs, self.fc, self.chunk = float(fs), float(fc), int(chunk)
        self.cap = int(ring_s * fs)
        self.lossless = lossless          # block the reader instead of dropping (unpaced mock replay only)
        self.stats = RadioStats()
        self._ring: collections.deque[Block] = collections.deque()
        self._ring_n = 0
        self._cv = threading.Condition()
        self._stop = threading.Event()
        self._restart = threading.Event()
        self._thread: threading.Thread | None = None
        self._seq = 0
        self._need_anchor = True
        self._gap = True
        self._anchor_t = 0.0
        self._anchor_n = 0                # samples since the anchor
        self._lags: list[float] = []
        self._lag_ref: float | None = None
        self._lag_prev: float | None = None
        self._drift_t0: tuple[float, float] | None = None
        self._lag_steady: float | None = None     # first 16-read lag minimum after an open
        self._verify = False                       # a timeout happened: check the next read for lost time
        self._refine = False                       # the anchor came from one read: refine it on the lag minimum
        self._walk_shift = 0.0                     # slow walk measured on the lag minimum (re-anchor by this)
        self._shift_out = 0.0                      # anchor refinement to announce on the next block
        self.finished = False             # the source has nothing more to give (mock end of files)

    # ------------------------------------------------------------------ to override
    def now(self) -> float:
        return time.time()

    def _open(self) -> None:
        raise NotImplementedError

    def _read(self, buf: np.ndarray) -> int:
        raise NotImplementedError

    def _close(self) -> None:
        pass

    def describe(self) -> dict:
        return {"kind": self.kind, "fs_hz": self.fs, "fc_hz": self.fc}

    # ------------------------------------------------------------------ consumer side
    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(target=self._run, name=f"radio-{self.kind}", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> bool:
        """Stop the reader. Returns False (state "stop_timeout") when the thread is still alive after ``timeout``:
        a driver call that never returns. The thread is a daemon thread, so it cannot keep the process alive."""
        self._stop.set()
        with self._cv:
            self._cv.notify_all()
        if self._thread is not None:
            self._thread.join(timeout)
            if self._thread.is_alive():
                self.stats.state = "stop_timeout"
                return False
        self.stats.state = "stopped"
        return True

    def restart(self) -> None:
        """Ask the reader to close and reopen the device (watchdog)."""
        self.stats.restarts += 1
        self._restart.set()

    def get(self, timeout: float = 1.0) -> Block | None:
        """The oldest block in the ring, or None after ``timeout`` seconds."""
        with self._cv:
            if not self._ring:
                self._cv.wait(timeout)
            if not self._ring:
                return None
            b = self._ring.popleft()
            self._ring_n -= b.n
            self._cv.notify_all()
            return b

    @property
    def alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def ring_fill(self) -> float:
        return self._ring_n / self.cap if self.cap else 0.0

    # ------------------------------------------------------------------ reader thread
    def _put(self, b: Block) -> None:
        with self._cv:
            if self.lossless:
                while self._ring_n + b.n > self.cap and self._ring and not self._stop.is_set():
                    self._cv.wait(0.5)
            while self._ring and self._ring_n + b.n > self.cap:
                old = self._ring.popleft()
                self._ring_n -= old.n
                self.stats.ring_drops += 1
                self.stats.ring_dropped_samples += old.n
                if self._ring:
                    self._ring[0].gap_before = True
                else:
                    b.gap_before = True
            self._ring.append(b)
            self._ring_n += b.n
            self._cv.notify_all()

    def _mark_gap(self) -> None:
        self._need_anchor = True
        self._gap = True

    def _check_clock(self, t_ret: float) -> None:
        """Host-vs-sample clock check after a read (see the module doc)."""
        lag = t_ret - (self._anchor_t + self._anchor_n / self.fs)
        self._lags.append(lag)
        if len(self._lags) < LAG_WIN:
            return
        m = min(self._lags)
        self._lags = []
        self.stats.lag_min_ms = m * 1e3
        if self._lag_ref is None:
            if self._refine and self._lag_steady is not None:
                # the gap anchor came from ONE read's return time, late by that read's jitter (median +8 ms at
                # 10 ms mean jitter); the minimum over this first window is the steady lag plus the anchor's error
                d = m - self._lag_steady
                if -REFINE_MAX_S <= d <= SLIP_S:          # late anchor -> lags read short -> d < 0
                    self._anchor_t += d
                    self._shift_out += d
                    m -= d
                    self.stats.anchor_refines += 1
                self._refine = False
            self._lag_ref = self._lag_prev = m
            if self._lag_steady is None:
                self._lag_steady = m
            if self._drift_t0 is None:
                self._drift_t0 = (t_ret, m)
            return
        if abs(m - self._lag_prev) > SLIP_S:                 # sudden: host clock step or silent sample slip
            self.stats.clock_steps += 1
            self._mark_gap()
            return
        self._lag_prev = m
        if self._drift_t0 is not None and t_ret - self._drift_t0[0] >= DRIFT_MIN_S:
            self.stats.clock_drift_ppm = (m - self._drift_t0[1]) / (t_ret - self._drift_t0[0]) * 1e6
        if abs(m - self._lag_ref) > SLIP_S:                   # slow walk: re-anchor, samples stay continuous
            self.stats.reanchors += 1
            self._walk_shift = m - (self._lag_steady if self._lag_steady is not None else self._lag_ref)
            self._need_anchor = True

    def _stream_once(self) -> None:
        """Open, settle and read until stop / restart / error."""
        self._streaming_since = time.monotonic()
        self.stats.state = "opening"
        self._open()
        self.stats.state = "settling"
        buf = np.empty(2 * self.chunk, np.int16)
        got, t_s = 0, time.monotonic()
        while got < self.fs * self.settle_s and time.monotonic() - t_s < SETTLE_MAX_S and not self._stop.is_set():
            r = self._read(buf)
            if r > 0:
                got += r
        self._mark_gap()
        self._lag_steady = None                       # a new open may buffer differently
        self._drift_t0 = None
        self._verify = False
        self.stats.state = "streaming"
        self._streaming_since = time.monotonic()
        while not self._stop.is_set() and not self._restart.is_set():
            t_call = self.now()
            r = self._read(buf)
            t_ret = self.now()
            if r > 0:
                if self._verify and not self._need_anchor:
                    lag = t_ret - (self._anchor_t + (self._anchor_n + r) / self.fs)
                    if self._lag_steady is None or lag - self._lag_steady > TIMEOUT_LOSS_S:
                        self.stats.timeout_gaps += 1
                        self._mark_gap()
                self._verify = False
                shift = 0.0
                if self._need_anchor:
                    gap = self._gap
                    if self._lag_steady is None:      # first anchor after an open: the recorder's stamp rule
                        self._anchor_t = t_call - STAMP_OFFSET_S
                        self._refine = False
                    elif gap:                         # after lost samples: this read, refined after LAG_WIN reads
                        self._anchor_t = t_ret - r / self.fs - self._lag_steady
                        self._refine = True
                    else:                             # slow walk: move the continuous time line by the walk itself,
                        old = self._anchor_t + self._anchor_n / self.fs   # measured on the lag minimum (no jitter)
                        self._anchor_t = old + self._walk_shift
                        if self._drift_t0 is not None:                    # lags now read against the new line
                            self._drift_t0 = (self._drift_t0[0], self._drift_t0[1] - self._walk_shift)
                        self._walk_shift = 0.0
                        self._refine = False
                    self._anchor_n = 0
                    self._lags, self._lag_ref, self._lag_prev = [], None, None
                    if gap:
                        self._drift_t0 = None
                    self._need_anchor, self._gap = False, False
                    self._shift_out = 0.0
                    re = not gap
                else:
                    gap = re = False
                    shift, self._shift_out = self._shift_out, 0.0
                t0 = self._anchor_t + self._anchor_n / self.fs
                self._anchor_n += r
                if r < self.chunk:
                    self.stats.short_reads += 1
                self.stats.blocks += 1
                self.stats.samples += r
                self.stats.last_block_wall = time.time()
                self._put(Block(self._seq, t0, buf[:2 * r].copy(), self.fs, gap_before=gap, reanchored=re,
                                anchor_shift_s=shift, counters=self.stats.counters()))
                self._seq += 1
                self._check_clock(t_ret)
            elif r == RET_OVERFLOW:
                self.stats.overflows += 1
                self._mark_gap()
            elif r == RET_TIMEOUT:
                self.stats.timeouts += 1
                self._verify = True
            elif r is None or r == 0:
                if self.finished:
                    return
            else:
                raise RuntimeError(f"readStream returned {r}")

    def _run(self) -> None:
        fails = 0
        while not self._stop.is_set():
            try:
                self._stream_once()
                fails = 0
                if self.finished:
                    break
            except Exception as e:                            # device gone, driver error: reopen with backoff
                self.stats.last_error = f"{type(e).__name__}: {e}"
                if self.stats.state in ("opening",):
                    self.stats.open_failures += 1
                else:
                    self.stats.read_errors += 1
                    if time.monotonic() - getattr(self, "_streaming_since", time.monotonic()) > BACKOFF_RESET_S:
                        fails = 0                             # a stream that worked for a while: start the ladder again
                fails += 1
            finally:
                try:
                    self._close()
                except Exception:
                    pass
                self._mark_gap()
            if self._restart.is_set():
                self._restart.clear()
                self.stats.reopens += 1
                continue
            if self._stop.is_set() or self.finished:
                break
            if fails:
                self.stats.state = "backoff"
                self._stop.wait(self._backoff(fails))
                self.stats.reopens += 1
        self.stats.state = "stopped"
        with self._cv:
            self._cv.notify_all()

    def _backoff(self, fails: int) -> float:
        return BACKOFF_S[min(fails, len(BACKOFF_S)) - 1]


# ======================================================================================= the real radio

class SoapyRadio(RadioSource):
    """SDRplay RSPdx through SoapySDR, the corpus recipe (module doc). SoapySDR is imported on open only, so this
    module imports on machines without it (tests, the bench harness)."""

    kind = "soapy"

    def __init__(self, antenna: str = ANTENNA, fs: float = FS, fc: float = FC, ifgr: int = IFGR, rfgr: int = RFGR,
                 driver: str = "driver=sdrplay", **kw):
        super().__init__(fs, fc, **kw)
        self.antenna, self.ifgr, self.rfgr, self.driver = antenna, ifgr, rfgr, driver
        self._sdr = None
        self._st = None
        self.readback: dict = {}

    def describe(self) -> dict:
        return {**super().describe(), "antenna": self.antenna, "ifgr_db": self.ifgr, "rfgr_db": self.rfgr,
                "agc": False, "format": "CS16", "settings": SETTINGS, "readback": self.readback,
                "stamp_offset_s": STAMP_OFFSET_S}

    def _open(self) -> None:
        if os.environ.get(NO_SDR_ENV):
            raise RuntimeError(f"{NO_SDR_ENV} is set: the real SDR may not be opened in this environment")
        import SoapySDR
        from SoapySDR import SOAPY_SDR_CS16, SOAPY_SDR_RX
        SoapySDR.SoapySDR_setLogLevel(SoapySDR.SOAPY_SDR_FATAL)
        self._rx = SOAPY_SDR_RX
        sdr = SoapySDR.Device(self.driver)
        sdr.setAntenna(SOAPY_SDR_RX, 0, self.antenna)
        sdr.setSampleRate(SOAPY_SDR_RX, 0, self.fs)
        sdr.setFrequency(SOAPY_SDR_RX, 0, self.fc)
        sdr.setGainMode(SOAPY_SDR_RX, 0, False)
        sdr.setGain(SOAPY_SDR_RX, 0, "IFGR", self.ifgr)
        sdr.setGain(SOAPY_SDR_RX, 0, "RFGR", self.rfgr)
        for k, v in SETTINGS.items():
            sdr.writeSetting(k, v)
        self.readback = {"antenna": sdr.getAntenna(SOAPY_SDR_RX, 0), "freq_hz": sdr.getFrequency(SOAPY_SDR_RX, 0),
                         "fs_hz": sdr.getSampleRate(SOAPY_SDR_RX, 0), "agc": sdr.getGainMode(SOAPY_SDR_RX, 0),
                         "IFGR": sdr.getGain(SOAPY_SDR_RX, 0, "IFGR"), "RFGR": sdr.getGain(SOAPY_SDR_RX, 0, "RFGR"),
                         **{k: sdr.readSetting(k) for k in SETTINGS}}
        self._sdr = sdr
        self._st = sdr.setupStream(SOAPY_SDR_RX, SOAPY_SDR_CS16)
        sdr.activateStream(self._st)

    def _read(self, buf: np.ndarray) -> int:
        r = self._sdr.readStream(self._st, [buf], self.chunk, timeoutUs=READ_TIMEOUT_US)
        return int(r.ret)

    def _close(self) -> None:
        sdr, st = self._sdr, self._st
        self._sdr = self._st = None
        if sdr is not None and st is not None:
            try:
                sdr.deactivateStream(st)
            finally:
                sdr.closeStream(st)
        del sdr                                   # the device is released when the last reference goes


# ======================================================================================= the mock

@dataclass
class Fault:
    """One injected fault, at ``at_s`` seconds of stream time (samples taken from the files / fs, settle reads and
    lost samples included; the virtual clock).

    kind:  "overflow"  the next ``n`` reads return -4 and their samples are lost
           "timeout"   reads return -1 for ``dur_s`` (paced at 1 s each); the samples of that time are lost
           "drop"      ``n`` samples silently skipped (no error code)
           "dup"       ``n`` samples silently repeated
           "vanish"    the device disappears for ``dur_s``: reads and opens raise
           "stall"     one read blocks for ``dur_s`` and then returns -4 (the driver overflowed meanwhile)
           "clock"     the host clock steps by ``dur_s`` seconds (may be negative)"""
    kind: str
    at_s: float
    n: int = 1
    dur_s: float = 0.0
    done: bool = False


class MockRadio(RadioSource):
    """Replays .cs16 captures (or in-memory int16 IQ arrays) as a radio: real time (``speed`` 1), N x real time, or
    unpaced (``speed`` 0, which also makes the ring lossless so nothing is dropped). ``loop`` restarts at the first
    file after the last; otherwise the source ends (``finished``).

    Its host clock (``now``) is virtual: stream start epoch + samples delivered / fs + STAMP_OFFSET_S + injected clock
    steps, so the stamp rule of ``RadioSource`` gives exact sample-0 times and replays are deterministic. The start
    epoch is the first file's sidecar ``utc`` minus STAMP_OFFSET_S (the recorder's stamp is late by that), or
    ``t_start`` when given. The 1 s settle read consumes the first second of the first file, as on the radio, unless
    ``settle=False``."""

    kind = "mock"

    def __init__(self, files=(), arrays=(), speed: float = 1.0, loop: bool = False, faults=(), t_start=None,
                 settle: bool = True, fs: float = FS, fc: float = FC, chunk: int = CHUNK, jitter_s: float = 0.0,
                 seed: int = 0, **kw):
        kw.setdefault("lossless", speed == 0)
        super().__init__(fs, fc, chunk=chunk, **kw)
        self.files = [os.path.expanduser(f) for f in files]
        self.arrays = [np.asarray(a, np.int16) for a in arrays]
        if not self.files and not self.arrays:
            raise ValueError("MockRadio needs files or arrays")
        self.speed, self.loop, self.settle = float(speed), bool(loop), bool(settle)
        if self.loop and sum(self._src_len(i) for i in range(len(self.files) or len(self.arrays))) == 0:
            self.loop = False                     # nothing to loop over: end at once instead of spinning for ever
        self.jitter_s = float(jitter_s)           # mean extra delay of a read's return (exponential), host clock
        self._rng = np.random.default_rng(seed)
        self._jit = 0.0
        self.settle_s = SETTLE_S if settle else 0.0
        self.faults = sorted([f if isinstance(f, Fault) else Fault(**f) for f in faults], key=lambda f: f.at_s)
        if t_start is None:
            t_start = 0.0
            if self.files:
                side = os.path.splitext(self.files[0])[0] + ".json"
                if os.path.exists(side):
                    from datetime import datetime
                    with open(side) as f:
                        u = json.load(f).get("utc")
                    if u:
                        # a recorder sidecar stamps late by STAMP_OFFSET_S; the live watcher's raw sidecars hold the
                        # sample-0 time already (utc_is_sample0)
                        side_doc = {}
                        with open(side) as f:
                            side_doc = json.load(f)
                        off = 0.0 if side_doc.get("utc_is_sample0") else STAMP_OFFSET_S
                        t_start = datetime.fromisoformat(u).timestamp() - off
        self.t_start = float(t_start)
        self._src_i = 0                   # which file / array
        self._src_pos = 0                 # sample position in it
        self._emitted = 0                 # samples taken from the sources (delivered or lost), the virtual clock
        self._clock_off = 0.0
        self._vanish_until = None         # stream seconds
        self._wall0 = None
        self._open_count = 0
        self._pending_overflow = 0

    def describe(self) -> dict:
        return {**super().describe(), "speed": self.speed, "loop": self.loop,
                "files": [os.path.basename(f) for f in self.files], "n_arrays": len(self.arrays),
                "faults": [asdict(f) for f in self.faults]}

    # -- virtual time
    def stream_s(self) -> float:
        return self._emitted / self.fs

    def now(self) -> float:
        return self.t_start + self._emitted / self.fs + STAMP_OFFSET_S + self._clock_off + self._jit

    def _pace(self) -> None:
        if self.speed <= 0:
            return
        if self._wall0 is None:
            self._wall0 = (time.monotonic(), self._emitted)
        due = self._wall0[0] + (self._emitted - self._wall0[1]) / (self.fs * self.speed)
        d = due - time.monotonic()
        if d > 0:
            self._stop.wait(d)

    def _sleep_stream(self, secs: float) -> None:
        """Let ``secs`` of stream time pass with no data (lost samples), paced like real time."""
        n = int(secs * self.fs)
        self._skip(n)
        self._pace()

    # -- sources
    def _src_len(self, i: int) -> int:
        if self.files:
            return os.path.getsize(self.files[i]) // 4
        return len(self.arrays[i]) // 2

    def _src_read(self, n: int) -> np.ndarray:
        """Up to n samples (int16 interleaved) from the current source, advancing; moves to the next source at the
        end of one; empty when every source is done (and not looping)."""
        while True:
            nsrc = len(self.files) or len(self.arrays)
            if self._src_i >= nsrc:
                if not self.loop or self._stop.is_set():
                    return np.zeros(0, np.int16)
                self._src_i = 0
            L = self._src_len(self._src_i)
            if self._src_pos < L:
                k = min(n, L - self._src_pos)
                if self.files:
                    raw = np.fromfile(self.files[self._src_i], np.int16, count=2 * k, offset=4 * self._src_pos)
                else:
                    raw = self.arrays[self._src_i][2 * self._src_pos:2 * (self._src_pos + k)]
                self._src_pos += k
                return raw
            self._src_i += 1
            self._src_pos = 0

    def _skip(self, n: int) -> int:
        """Throw away n source samples (lost); returns how many there were."""
        got = 0
        while got < n:
            r = self._src_read(min(n - got, 1 << 20))
            if r.size == 0:
                break
            got += r.size // 2
        self._emitted += got
        return got

    def _catch_up(self) -> None:
        """Paced replay: the wall time that passed while the device was closed (backoff before a reopen) is stream
        time with no data, as on the radio: skip those samples (lost) and restart the pacer. Without this the pacer
        found itself minutes "behind" after a long outage and replayed the backlog at full speed, which overran the
        ring (ring drops that a real radio, resuming at 1x, would not show)."""
        if self.speed > 0 and self._wall0 is not None:
            due = self._wall0[0] + (self._emitted - self._wall0[1]) / (self.fs * self.speed)
            behind = time.monotonic() - due
            if behind > 0:
                self._skip(int(behind * self.speed * self.fs))
            self._wall0 = None

    # -- RadioSource hooks
    def _open(self) -> None:
        self._catch_up()
        if self._vanish_until is not None and self.stream_s() < self._vanish_until:
            self._sleep_stream(min(1.0, self._vanish_until - self.stream_s()))
            raise OSError("mock device not found")
        self._vanish_until = None
        self._open_count += 1

    def _backoff(self, fails: int) -> float:
        return 0.0 if self.speed <= 0 else super()._backoff(fails) / max(self.speed, 1.0)

    def _due_faults(self) -> list[Fault]:
        t = self.stream_s()
        out = []
        for f in self.faults:
            if not f.done and f.at_s <= t:
                f.done = True
                out.append(f)
        return out

    def _read(self, buf: np.ndarray) -> int:
        if self._vanish_until is not None:
            raise OSError("mock device vanished")
        if self._pending_overflow > 0:
            self._pending_overflow -= 1
            self._skip(self.chunk)
            self._pace()
            return RET_OVERFLOW
        extra = None
        for f in self._due_faults():
            if f.kind == "overflow":
                self._pending_overflow += max(0, f.n - 1)
                self._skip(self.chunk)
                self._pace()
                return RET_OVERFLOW
            if f.kind == "timeout":
                t_end = self.stream_s() + f.dur_s
                while self.stream_s() < t_end and not self._stop.is_set():
                    self._sleep_stream(min(1.0, t_end - self.stream_s()))
                return RET_TIMEOUT
            if f.kind == "drop":
                self._skip(f.n)
            elif f.kind == "dup":
                extra = f.n
            elif f.kind == "vanish":
                self._vanish_until = self.stream_s() + f.dur_s
                raise OSError("mock device vanished")
            elif f.kind == "stall":
                self._sleep_stream(f.dur_s)
                return RET_OVERFLOW
            elif f.kind == "clock":
                self._clock_off += f.dur_s
            else:
                raise ValueError(f"unknown fault kind {f.kind!r}")
        if extra:
            # repeat the last `extra` samples delivered (or the next ones) without advancing the source
            keep = (self._src_i, self._src_pos)
            raw = self._src_read(min(self.chunk, extra))
            self._src_i, self._src_pos = keep
            buf[:raw.size] = raw
            return raw.size // 2
        raw = self._src_read(self.chunk)
        if raw.size == 0:
            self.finished = True
            return 0
        n = raw.size // 2
        buf[:raw.size] = raw
        self._emitted += n
        self._pace()
        if self.jitter_s > 0:                     # the host sees this read return late by a random delay
            self._jit = float(self._rng.exponential(self.jitter_s))
        return n
