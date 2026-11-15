"""Live storm watcher: a long-running daemon that reads the radio, runs the v2 chain causally, and writes per-stroke,
per-minute and status files plus an honest storm level.

    python -m sferic_watch live --radio soapy [--antenna "Antenna C"] [--state-dir DIR]
    python -m sferic_watch live --radio mock --mock-files a.cs16 b.cs16 [--speed 10] [--loop] [--duration S]

The streaming chain (``Chain``)
-------------------------------
The batch engine (``report.analyze_iq``) needs a whole capture: the front end runs zero-phase frames, the detector
floor is a trailing 4 s median (backfilled at the start), the hum flag works on 10 s blocks with a 60 s window
test, and the mains gate folds 60 s segments and then the whole capture ("whole" trains, >= 3 blocks). The live
chain keeps the expensive, local parts per minute and reruns the cheap, global part on a rolling window:

1. Per minute m of a segment, once the stream holds minute m plus ``GUARD_S`` (2 s) after it: the front end powers
   (``detect.band_powers`` on the minute with 2 s of context each side, on the absolute frame grid: equal to the
   whole-capture powers more than one 1 s frame from the slice edges, and the context covers that), the hum-flag
   blocks of the minute with their edge times (``humflag.block_edges`` / ``block_stats``, exactly the cut
   ``flag_iq`` and ``pipeline.hum_edges`` make) and health numbers. Raw samples older than that are dropped.
2. Minute c is committed once minute c + ``lookahead_min`` is cached: ``pipeline.run`` on the cached powers, flags
   and edges of the window [c - history_min, c + lookahead_min] (default 8 + 1 + 1 = 10 minutes, the length of a
   banked capture, so the "whole" fold sees as much as it does offline), then the report's own minute row
   (``report.rate_row``: gated rate, confirmed rate, two-band coincidence, hum_frac, gated time share as the union
   of reject windows, comb share, trusted). Only the events of minute c are written.
3. At a segment end (gap in the samples, or shutdown) every cached minute is committed with whatever lookahead
   exists, like the last minute of a banked file.
4. Warm-up (default; ``--no-warmup`` turns it off): the window keeps its full 10 minutes wherever the segment allows
   (``Chain.window``). The first minutes after a start or a gap wait until minutes 0..9 are in; at a segment end the
   window reaches back. A segment of up to 10 minutes is therefore gated exactly as the batch engine gates a banked
   capture.

Latency: in steady state a stroke is written ``lookahead_min`` minutes + 2 s + processing after the END of its
minute, i.e. 61-121 s + about 1-3 s of CPU with the defaults. After a start or a gap the first minutes wait for the
warm-up: up to 10 minutes + 2 s after their end (minute 0) and the backlog then commits in one go.
``lookahead_min=0 --no-warmup`` gives 2-62 s, at the cost of the gate's look-ahead (agreement with batch in the
live README section).

Segments, minutes and time
--------------------------
A segment is a run of samples with no known loss (``radio.Block.gap_before`` starts a new one). With ``align="utc"``
(the default) a segment starts at the next whole UTC minute, so minute rows are UTC minutes; ``align="start"``
starts at the first sample (replays compared with the batch engine). Event and minute times are the sample-clock
time from the block stamps (``radio`` module doc: host clock with the measured 26 ms recorder offset removed),
re-anchored when the radio re-anchors.

Outputs (``Store``, under the state dir, default ``~/.local/state/sferic-watch``)
-----------------------------------------------------------------------------------
events.jsonl   one line per stroke that passed the mains gate: ``utc`` (ISO, microseconds), ``t`` (epoch s),
               ``peak_db`` (over the detector floor), ``conf`` / ``conf_db`` (16-30 kHz confirmation), ``coinc_hi``
               (a 16-30 kHz impulse peaks within +-0.5 ms), ``hum`` (it lies in a hum-flagged 10 s block),
               ``width_us``, ``floor_dbfs``. Impulses the gate rejected are not written.
minutes.jsonl  one line per committed minute: ``rate_per_min`` (gated sky rate), ``conf_per_min`` (gated AND
               confirmed), ``v2_per_min`` (two-band coincidence) with its chance level, ``hum_frac``,
               ``gated_frac``, ``comb_share``, ``trusted`` / ``untrusted_because`` (report rule: comb share over 0.25
               or gated share over 0.20; and ``frontend_dead`` / ``frontend_floor_high`` / ``frontend_clipping``
               from the health numbers, ``frontend_faults``), ``health`` (detector-band floor dBFS, NAA level, 0 Hz
               tone, clipped samples, radio fault counters during the minute, host-vs-sample clock drift ppm),
               ``delay_s`` (data delay: radio clock at commit minus the minute's end), ``cpu_s``.
               Gap rows: ``{"gap": true, "t", "t_end", "missing_min", "reason", "radio"}`` just before the first
               minute after a stretch no minute covers (radio loss, a chain error, the watcher not running);
               readers skip rows with ``gap``.
alerts.jsonl   one line per storm-level change (``Alert``).
state.json     status, heartbeat, uptime, radio stats, the last minute (``last_minute``), the alert level, disk;
               written atomically (temp file + rename) every few seconds.
archive/       the .jsonl files of earlier UTC days (``events-YYYYMMDD.jsonl`` ...), kept ``keep_days`` (30).
raw/           optional (``--raw``) per-minute .cs16 + .json sidecar, kept ``keep_raw_days``; raw recording stops
               (and state.json warns) when the disk has less than ``min_free_gb`` (5 GB) free; below 0.5 GB the
               .jsonl writing stops too.

The radio lock (``RadioLock``)
------------------------------
One radio, one program. The daemon holds an exclusive ``flock`` on ``~/.local/state/radio.lock`` while it owns the
SDR and refuses to start when the lock is held or a process running ``dlayer_run.py`` exists (the D-layer recorder
does not take the lock; ``runs_script`` reads interpreter options such as ``python3 -X dev -u``). It keeps looking
every FOREIGN_CHECK_S (10 s): a dlayer_run.py started later makes the watcher close the radio and exit with code 3
(the unit does not restart on 3). The mock radio takes no lock.

Restart
-------
On start the storm level is rebuilt from the minute files of the last 8 days (``Alert.restore``: baseline, hour
bands, level), and the time the watcher was not running gets a gap row (reason ``watcher_not_running``).
"""
from __future__ import annotations

import argparse
import bisect
import copy
import fcntl
import glob
import json
import math
import multiprocessing as mp
import os
import queue
import shutil
import signal
import sys
import threading
import time
import traceback
from collections import deque
from datetime import datetime, timedelta, timezone

import numpy as np

import sferic_watch as sw
from sferic_watch import detect as dt
from sferic_watch import health as hl
from sferic_watch import humflag as hf
from sferic_watch import mainsgate as mg
from sferic_watch import pipeline as pl
from sferic_watch import radio as rd
from sferic_watch import report as rp

FS = rd.FS
MINUTE_S = 60.0
GUARD_S = 2.0                 # front-end context each side of a cached minute (> one 1 s frame + the boxcar)
HISTORY_MIN = 8               # gate window: this many minutes before the committed one...
LOOKAHEAD_MIN = 1             # ...and this many after it
STATE_DIR = os.path.join("~", ".local", "state", "sferic-watch")
LOCK_PATH = os.path.join("~", ".local", "state", "radio.lock")
FOREIGN = ("dlayer_run.py",)  # programs that use the radio without the lock
HEARTBEAT_S = 5.0
STALL_S = 15.0                # no block for this long: state "stalled"
RESTART_S = 60.0              # ...this long: ask the radio to reopen
MAX_DEAD_RESTARTS = 3         # reopen requests with no block in between before exiting non-zero
WORKER_STUCK_S = 300.0        # the worker took no message for this long (queue full): it is hung, kill it
FOREIGN_CHECK_S = 10.0        # soapy: look for a dlayer_run.py started after us this often (then yield, exit 3)
KEEP_DAYS = 30
KEEP_RAW_DAYS = 2
MIN_FREE_GB = 5.0
MIN_FREE_JSONL_GB = 0.5
LIVE_VERSION = "live-1"


def iso(t: float, us: bool = False) -> str:
    """UTC ISO string of an epoch time (Z suffix)."""
    d = datetime.fromtimestamp(t, timezone.utc)
    return d.strftime("%Y-%m-%dT%H:%M:%S.%fZ" if us else "%Y-%m-%dT%H:%M:%SZ")


def _r(x, nd):
    return rp._f(x, nd)


def _med(v, nd):
    return _r(float(np.median(v)), nd) if v else None


FLOOR_DEAD_DBFS = -120.0      # detector-band floor below this: no signal at all (corpus: -76.6 ... -74.2 dBFS)
FLOOR_HIGH_DBFS = -40.0       # ...above this: the front end is swamped (35 dB over anything in the corpus)
CLIP_SHARE = 1e-4             # share of int16 values at full scale that makes a minute untrusted (corpus: 0)


def frontend_faults(h: dict, n_samples: int) -> list[str]:
    """Reasons a minute's front end cannot be read as sky (dead input, swamped floor, clipping), from its health
    numbers. A dead or clipped front end gave trusted quiet minutes (rate 0, or noise counted as strokes) that fed the
    storm level and its baseline; such minutes are now untrusted."""
    out = []
    fl = h.get("floor_dbfs")
    if fl is None or fl < FLOOR_DEAD_DBFS:
        out.append("frontend_dead")
    elif fl > FLOOR_HIGH_DBFS:
        out.append("frontend_floor_high")
    if h.get("clipped") is not None and h["clipped"] > CLIP_SHARE * 2 * n_samples:
        out.append("frontend_clipping")
    return out


def atomic_write_json(path: str, doc: dict) -> None:
    """Write JSON through a temp file and an atomic rename, so a reader never sees half a file."""
    tmp = f"{path}.tmp{os.getpid()}"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, allow_nan=False)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


# ======================================================================================= the streaming chain

class Chain:
    """The causal v2 chain for ONE segment (module doc). ``push`` int16 interleaved samples, ``end`` at the segment
    end; both return committed minute results (dicts) in order. Times inside results are seconds from the segment's
    first sample; the daemon maps them to UTC. Pure computation: no files, no clocks, deterministic."""

    def __init__(self, fs: float = FS, minute_s: float = MINUTE_S, history_min: int = HISTORY_MIN,
                 lookahead_min: int = LOOKAHEAD_MIN, guard_s: float = GUARD_S, health: bool = True,
                 warmup: bool = True):
        self.fs = float(fs)
        self.N = int(round(minute_s * fs))
        self.G = int(round(guard_s * fs))
        self.B = int(round(hf.BLOCK_S * fs))
        if self.N % self.B:
            raise ValueError("a minute must hold whole 10 s hum blocks")
        self.H, self.L = int(history_min), int(lookahead_min)
        self.W = self.H + 1 + self.L              # gate window length, minutes
        self.warmup = bool(warmup)
        self.health = health
        self.raw = np.zeros(0, np.int16)        # interleaved, samples [raw0, n)
        self.raw0 = 0
        self.n = 0
        self.cache: dict[int, dict] = {}
        self.next_cache = 0
        self.next_commit = 0
        self.ended = False

    # -- helpers
    def _iq(self, a: int, b: int) -> np.ndarray:
        """Samples [a, b) as complex64, full scale 1, their mean removed (``io.read_slice``'s DC rule; the LO leak
        at baseband 0 Hz is under 1 LSB and outside both bands, so the slice mean against the whole-file mean the
        batch run uses moves nothing that matters)."""
        r = self.raw[2 * (a - self.raw0):2 * (b - self.raw0)]
        x = np.empty(len(r) // 2, np.complex64)
        x.real = r[0::2]
        x.imag = r[1::2]
        x /= np.float32(rp.sio.FULL_SCALE)
        if x.size:
            x -= x.mean(dtype=np.complex128).astype(np.complex64)
        return x

    def _cache_minute(self, m: int, final: bool) -> None:
        N, G, fs = self.N, self.G, self.fs
        a = max(0, m * N - G)
        b = min(self.n, (m + 1) * N + G)
        x = self._iq(a, b)
        p, pc = dt.band_powers(x, fs, n0=a)
        k = m * N - a
        p, pc = p[k:k + N].copy(), pc[k:k + N].copy()
        flags, edges = [], []
        pad = int(round(hf.PAD_S * fs))
        for j in range(N // self.B):
            s0 = m * N + j * self.B
            lo, hi = max(a, s0 - pad), min(b, s0 + self.B + pad)
            e = hf.block_edges(x[lo - a:hi - a], s0 - lo, self.B, n0=lo, fs=fs)
            st = hf.block_stats(e / fs, self.B / fs)
            st.update(block=s0 // self.B, t0_s=s0 / fs, dur_s=self.B / fs)
            flags.append(st)
            edges.append((s0 + e) / fs)
        h = {}
        if self.health:
            bm = dt.block_means(p)
            h["floor_dbfs"] = _r(10 * math.log10(max(float(np.median(bm)), 1e-30)), 2) if bm.size else None
            seg = x[k:k + min(N, int(10 * fs))]
            if len(seg) >= 6250:
                lv = hl.block_levels(seg, fs)
                h["naa_db"] = _r(lv["NAA"][0], 2)
                h["naa_dbfs"] = _r(lv["NAA"][1], 2)
                h["dc_dbfs"] = _r(lv["DC"], 2)
            r = self.raw[2 * (m * N - self.raw0):2 * ((m + 1) * N - self.raw0)]
            h["clipped"] = int(np.count_nonzero(np.abs(r.astype(np.int32)) >= 32760))
        self.cache[m] = {"p": p, "pc": pc, "flags": flags, "edges": np.concatenate(edges), "health": h}

    def _commit(self, c: int, last: int) -> dict:
        """Run the gate window for minute c (cached minutes up to ``last``) and build its result."""
        t_cpu = time.process_time()
        fs, N = self.fs, self.N
        w0, w1 = self.window(c, min(self.cache), last)
        mins = range(w0, w1 + 1)
        off = w0 * N / fs
        P = np.concatenate([self.cache[m]["p"] for m in mins])
        PC = np.concatenate([self.cache[m]["pc"] for m in mins])
        flags = []
        for m in mins:
            for b in self.cache[m]["flags"]:
                b = copy.deepcopy(b)
                b["t0_s"] -= off
                flags.append(b)
        edges = np.concatenate([self.cache[m]["edges"] for m in mins]) - off
        secs = len(P) / fs
        h0 = max(w0, c - 1)                                   # 16-30 kHz peaks: c-1 .. window end is enough
        hi_t = rp.hi_band_peaks(PC[(h0 - w0) * N:], fs) + (h0 - w0) * N / fs
        live = []
        with rp.keep_trains(live):
            ev, info = pl.run(None, fs, flags=flags, powers=(P, PC), edges=edges)
        del P, PC
        coinc = rp.coincident(ev["t_peak"], hi_t)
        rej = rp.reject_intervals(live, secs)
        t0, t1 = (c - w0) * N / fs, (c - w0 + 1) * N / fs
        row = rp.rate_row(t0, t1, ev, coinc, hi_t, flags, rej, live, secs)
        k = (ev["t_start"] >= t0) & (ev["t_start"] < t1)
        events = {"t_start": ev["t_start"][k] + off, "t_peak": ev["t_peak"][k] + off, "peak_db": ev["peak_db"][k],
                  "conf": ev["conf"][k], "conf_db": ev["conf_db"][k], "coinc_hi": coinc[k], "hum": ev["hum"][k],
                  "width_s": ev["width_s"][k], "floor_dbfs": ev["floor_dbfs"][k]}
        row["t0_s"] = _r(c * N / fs, 3)
        bad = frontend_faults(self.cache[c]["health"], N) if self.health else []
        if bad:                                   # dead or saturated front end: no sky reading, never trusted
            row["trusted"] = False
            row["untrusted_because"] = list(row.get("untrusted_because") or []) + bad
        return {"minute": c, "t0_s": c * N / fs, "row": row, "events": events,
                "health": dict(self.cache[c]["health"], too_wide=int(info["too_long"])),
                "window_min": [w0, w1], "lookahead_min": w1 - c, "cpu_s": time.process_time() - t_cpu}

    def window(self, c: int, first: int, last: int) -> tuple[int, int]:
        """Gate window [w0, w1] (minutes) for committing minute c with minutes first..last cached.

        Normal: [c - H, c + L]. With ``warmup`` the window keeps its full length W = H + 1 + L wherever the segment
        allows: in the first W - 1 minutes of a segment it reaches further ahead (minute 0 waits until minutes
        0..W-1 are in), and at the segment end further back. Why: the gate's "whole" trains (a buzz source of a few
        pulses a minute) are only visible over many minutes; a cold window of 2-9 minutes after every start or gap
        missed them, kept 13-80 % more events than batch on hum captures and flipped 94 batch-untrusted minutes to
        trusted in the corpus soak. On a segment of up to W minutes the window is the whole segment, as offline."""
        if not self.warmup:
            return max(first, c - self.H), min(last, c + self.L)
        w1 = min(last, max(c + self.L, first + self.W - 1))
        w0 = max(first, min(c - self.H, w1 - self.W + 1))
        return w0, w1

    def _ready(self, c: int, last: int) -> bool:
        if c + self.L > last:
            return False
        return not self.warmup or last >= min(self.cache) + self.W - 1

    def _advance(self, final: bool) -> list[dict]:
        out = []
        while (self.next_cache + 1) * self.N + (0 if final else self.G) <= self.n:
            self._cache_minute(self.next_cache, final)
            self.next_cache += 1
        last = self.next_cache - 1
        while self.next_commit <= last and (final or self._ready(self.next_commit, last)):
            res = self._commit(self.next_commit, last)
            out.append(res)
            self.next_commit += 1
        # forget what no later commit or cache needs (a segment-end window may reach W - 1 minutes back)
        keep_from = self.next_commit - (self.H + self.L if self.warmup else self.H)
        for m in [m for m in self.cache if m < keep_from]:
            del self.cache[m]
        trim = max(0, self.next_cache * self.N - self.G)
        if trim > self.raw0:
            self.raw = self.raw[2 * (trim - self.raw0):].copy()
            self.raw0 = trim
        return out

    def push(self, iq: np.ndarray) -> list[dict]:
        """Add samples (int16 interleaved). Returns the minutes committed by this push."""
        if self.ended:
            raise RuntimeError("Chain.push after end")
        iq = np.asarray(iq, np.int16)
        self.raw = np.concatenate([self.raw, iq]) if self.raw.size else iq.copy()
        self.n += iq.size // 2
        return self._advance(False)

    def end(self) -> list[dict]:
        """Segment end: cache and commit every whole minute left (the partial one is dropped)."""
        self.ended = True
        return self._advance(True)

    @property
    def pending_samples(self) -> int:
        return self.n - self.next_commit * self.N


# ======================================================================================= storm level

LEVELS = ("quiet", "active", "storm", "severe")
STAT_MIN = 10                 # the level statistic: median over the last 10 minutes...
STAT_NEED = 5                 # ...of at least 5 trusted ones (else the level holds)
BASE_HOURS = 6.0              # rolling baseline: trusted minutes of the last 6 h...
BASE_SKIP_MIN = 30            # ...except the last 30 min (a storm must not raise its own baseline at once)
BASE_NEED = 60                # baseline needs this many trusted minutes (else the level stays "learning")
BAND_DAYS = 7                 # hour-band baseline: same UTC hour +-1 h on the previous 7 days...
BAND_NEED = 120               # ...used when it has this many trusted minutes
BASE_FLOOR = 10.0             # baseline never below this (confirmed strokes per minute)
FREEZE_H = 6.0                # minutes while the level is above quiet stay out of the baseline, for at most 6 h
# level k needs stat >= UP_RATIO[k] x baseline AND stat >= UP_MIN[k] (confirmed / min), for UP_HOLD minutes
UP_RATIO = (None, 3.0, 6.0, 12.0)
UP_MIN = (None, 40.0, 80.0, 200.0)
UP_HOLD = 3
DOWN_FRAC = 0.7               # level k drops when stat < DOWN_FRAC x its up condition...
DOWN_HOLD = 15                # ...for 15 committed minutes in a row
BLIND_HOLD = 15               # no statistic (too few trusted minutes) this long: step down one level


class Alert:
    """Storm level from the gated, 16-30 kHz confirmed rate of TRUSTED minutes against a rolling baseline, with
    hysteresis (constants above). Untrusted minutes (report rule: comb share > 0.25 or gated share > 0.20; this also
    covers hum: a hum-flagged minute only stays trusted when the gate cleaned it; and a dead or clipped front end,
    ``frontend_faults``) never enter the statistic or the baseline.

    The baseline is the larger of the rolling 6 h median and the same-UTC-hour-band median of earlier days (when it
    has enough minutes): night propagation lifts the distant sferic rate by itself, so a fixed baseline would call
    every sunset a storm. While the level is above quiet the minutes are kept out of the baseline (a storm must not
    become its own baseline: in the corpus replay the 11-06 04Z outbreak, GOES GLM 65-260 matched flashes/min for
    15 h, fell back to "active" after 3 h without this), but only for FREEZE_H (6 h): a lasting change of the
    background (propagation, a new antenna) is then learned.

    Only new evidence moves the level up: the UP_HOLD count advances on a trusted minute only (an untrusted minute
    re-checks the same stale trusted values; 4 trusted minutes followed by untrusted ones used to raise "severe").
    When the statistic has too few trusted minutes for BLIND_HOLD minutes in a row, the level steps down one level
    per BLIND_HOLD minutes (reason ``no_trusted_minutes``): an alert must not stand for ever on minutes nobody can
    read any more. ``restore`` rebuilds the state from earlier minute rows (the daemon replays the files on start, so
    a restart keeps the baseline, the hour bands and the level). ``feed`` returns an alert record when the level
    changes, else None."""

    def __init__(self, notifier=None):
        self.ht: list[float] = []                 # times of trusted minutes (sorted), BAND_DAYS + 1 days
        self.hv: list[float] = []                 # ...and their confirmed rates
        self.recent: deque = deque(maxlen=STAT_MIN)  # (t, conf rate or None when untrusted)
        self.level = 0
        self.since = None
        self.up_run = 0
        self.down_run = 0
        self.blind_run = 0
        self.last_t = None
        self.notifier = notifier
        self.last = {}

    @property
    def hist(self) -> list[tuple[float, float]]:
        return list(zip(self.ht, self.hv))

    def baseline(self, t: float) -> tuple[float | None, str]:
        ht, hv = self.ht, self.hv
        roll = hv[bisect.bisect_left(ht, t - BASE_HOURS * 3600):bisect.bisect_left(ht, t - BASE_SKIP_MIN * 60)]
        band = []
        for d in range(1, BAND_DAYS + 1):
            c = t - d * 86400
            band += hv[bisect.bisect_left(ht, c - 3600):bisect.bisect_left(ht, c + 3600)]
        cands = []
        if len(roll) >= BASE_NEED:
            cands.append((float(np.median(roll)), "rolling_6h"))
        if len(band) >= BAND_NEED:
            cands.append((float(np.median(band)), "hour_band"))
        if not cands:
            return None, "learning"
        v, src = max(cands)
        return max(v, BASE_FLOOR), src

    def _cond(self, k: int, stat: float, base: float, frac: float = 1.0) -> bool:
        return stat >= frac * UP_RATIO[k] * base and stat >= frac * UP_MIN[k]

    def feed(self, t: float, row: dict) -> dict | None:
        """One committed minute (``t`` its start, epoch s; ``row`` its minute record)."""
        if self.last_t is not None and t - self.last_t > STAT_MIN * 60:
            self.up_run = self.down_run = 0        # a long hole: earlier runs do not carry over it
        self.last_t = t
        ok = bool(row.get("trusted")) and row.get("conf_per_min") is not None
        v = float(row["conf_per_min"]) if ok else None
        self.recent.append((t, v))
        frozen = self.level > 0 and self.since is not None and t - self.since < FREEZE_H * 3600
        if ok and not frozen:
            i = bisect.bisect_right(self.ht, t)
            self.ht.insert(i, t)
            self.hv.insert(i, v)
        cut = bisect.bisect_left(self.ht, t - (BAND_DAYS + 1) * 86400)
        if cut:
            del self.ht[:cut], self.hv[:cut]
        vals = [x for (u, x) in self.recent if x is not None and u > t - STAT_MIN * 60]
        base, src = self.baseline(t)
        stat = float(np.median(vals)) if len(vals) >= STAT_NEED else None
        status = "ok"
        if stat is None:
            status = "too_few_trusted_minutes"
        elif base is None:
            status = "learning_baseline"
        old = self.level
        reason = "rate"
        if stat is None:
            self.up_run = self.down_run = 0
            self.blind_run += 1
            if self.level > 0 and self.blind_run >= BLIND_HOLD:
                self.level -= 1
                self.blind_run = 0
                reason = "no_trusted_minutes"
        else:
            self.blind_run = 0
        if stat is not None and base is not None:
            nxt = self.level + 1
            if nxt < len(LEVELS) and self._cond(nxt, stat, base):
                if ok:                                 # only a new trusted minute is new evidence
                    self.up_run += 1
                if self.up_run >= UP_HOLD:
                    k = nxt
                    while k + 1 < len(LEVELS) and self._cond(k + 1, stat, base):
                        k += 1
                    self.level, self.up_run, self.down_run = k, 0, 0
            else:
                self.up_run = 0
            if self.level > 0 and self.level == old and not self._cond(self.level, stat, base, DOWN_FRAC):
                self.down_run += 1
                if self.down_run >= DOWN_HOLD:
                    self.level -= 1
                    self.down_run = 0
            elif self.level == old:
                self.down_run = 0
        self.last = {"level": self.level, "name": LEVELS[self.level], "stat_conf_per_min": _r(stat, 2),
                     "baseline_conf_per_min": _r(base, 2), "baseline_source": src,
                     "ratio": _r(stat / base, 2) if stat is not None and base else None,
                     "trusted_recent": len(vals), "status": status,
                     "since": self.since}
        if self.level != old:
            self.since = t
            self.last["since"] = t
            rec = {"utc": iso(t), "t": t, "from": LEVELS[old], "to": LEVELS[self.level], "reason": reason,
                   **{k: self.last[k] for k in ("stat_conf_per_min", "baseline_conf_per_min", "baseline_source",
                                                "ratio", "trusted_recent")}}
            if self.notifier is not None:
                try:
                    self.notifier(rec)
                except Exception:                      # a notifier must never take the watcher down
                    pass
            return rec
        return None

    def restore(self, rows) -> int:
        """Rebuild the state from earlier minute rows (time order; gap rows skipped) without notifying. Returns how
        many rows were fed."""
        notifier, self.notifier = self.notifier, None
        n = 0
        try:
            for r in rows:
                if r.get("gap") or r.get("t") is None:
                    continue
                self.feed(float(r["t"]), r)
                n += 1
        finally:
            self.notifier = notifier
        return n


def null_notifier(rec: dict) -> None:
    """Default notifier: nothing (no network). Replace with any callable(record)."""
    return None


def command_notifier(cmd: str):
    """A notifier that runs ``cmd`` with the alert record as JSON on stdin (e.g. a local notify-send wrapper)."""
    import subprocess

    def notify(rec):
        subprocess.run(cmd, shell=True, input=json.dumps(rec), text=True, timeout=20, check=False)
    return notify


# ======================================================================================= files

class Store:
    """The state dir: jsonl outputs with UTC-daily rotation, retention, disk cap, atomic state.json, optional raw."""

    def __init__(self, state_dir: str, keep_days: int = KEEP_DAYS, raw: bool = False,
                 keep_raw_days: int = KEEP_RAW_DAYS, min_free_gb: float = MIN_FREE_GB):
        self.dir = os.path.expanduser(state_dir)
        os.makedirs(os.path.join(self.dir, "archive"), exist_ok=True)
        self.keep_days, self.keep_raw_days = keep_days, keep_raw_days
        self.raw_wanted, self.raw_on = raw, raw
        self.min_free_gb = min_free_gb
        self.jsonl_on = True
        self.warning = None
        self.repaired = self._repair_tails()
        self.day = self._day_on_disk()
        self.free_gb = None
        self._raw_f = None
        self._raw_key = None
        self.check_disk()

    def path(self, name: str) -> str:
        return os.path.join(self.dir, name)

    def _repair_tails(self) -> dict:
        """A kill or power cut in the middle of an append can leave a last line without its newline; the next append
        would glue a record onto it (one corrupt line). Cut each .jsonl back to its last complete line on start.
        Returns {file: bytes removed}. Lines that are not JSON (a power cut on ext4 can leave a NUL-filled block
        INSIDE the file, followed by later complete lines) are dropped too: ``corrupt_dropped`` {file: lines}."""
        out = {}
        self.corrupt_dropped = {}
        for name in ("events", "minutes", "alerts"):
            p = self.path(f"{name}.jsonl")
            try:
                with open(p, "rb+") as f:
                    size = f.seek(0, 2)
                    if size == 0:
                        continue
                    f.seek(size - 1)
                    if f.read(1) != b"\n":
                        pos = size
                        while pos > 0:
                            k = max(0, pos - 65536)
                            f.seek(k)
                            j = f.read(pos - k).rfind(b"\n")
                            if j >= 0:
                                pos = k + j + 1
                                break
                            pos = k
                        f.truncate(pos)
                        out[f"{name}.jsonl"] = size - pos
                bad = self._drop_corrupt_lines(p)
                if bad:
                    self.corrupt_dropped[f"{name}.jsonl"] = bad
            except FileNotFoundError:
                continue
        return out

    @staticmethod
    def _drop_corrupt_lines(p: str) -> int:
        """Rewrite p (temp file + rename) without the lines that are not a JSON object; only when it holds a NUL
        byte or its last line does not parse (the fast path reads the file once and parses nothing more)."""
        with open(p, "rb") as f:
            data = f.read()
        lines = data.split(b"\n")
        if lines and lines[-1] == b"":
            lines.pop()
        if b"\0" not in data:
            try:
                if not lines or isinstance(json.loads(lines[-1]), dict):
                    return 0
            except ValueError:
                pass
        keep, bad = [], 0
        for ln in lines:
            try:
                ok = isinstance(json.loads(ln), dict)
            except ValueError:
                ok = False
            if ok:
                keep.append(ln)
            else:
                bad += 1
        if bad:
            tmp = f"{p}.tmp{os.getpid()}"
            with open(tmp, "wb") as f:
                f.write(b"".join(x + b"\n" for x in keep))
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, p)
        return bad

    def _last_record(self, name: str, want=lambda r: True) -> dict | None:
        """The last JSON record of a .jsonl that satisfies ``want`` (reads back from the end)."""
        p = self.path(f"{name}.jsonl")
        try:
            with open(p, "rb") as f:
                size = f.seek(0, 2)
                back = 4096
                while True:
                    start = max(0, size - back)
                    f.seek(start)
                    lines = f.read().splitlines()
                    if start > 0:
                        lines = lines[1:]
                    for ln in reversed(lines):
                        try:
                            r = json.loads(ln)
                        except ValueError:
                            continue
                        if isinstance(r, dict) and want(r):
                            return r
                    if start == 0 or back > 1 << 24:
                        return None
                    back *= 8
        except OSError:
            return None

    def _day_on_disk(self) -> str | None:
        """The UTC day of the current .jsonl files (their newest record), so a restart rotates yesterday's lines."""
        days = []
        for name in ("minutes", "events", "alerts"):
            r = self._last_record(name, lambda r: isinstance(r.get("utc"), str))
            if r is not None:
                days.append(r["utc"][:10].replace("-", ""))
        if days:
            return max(days)
        mt = [os.path.getmtime(self.path(f"{n}.jsonl")) for n in ("minutes", "events", "alerts")
              if os.path.exists(self.path(f"{n}.jsonl")) and os.path.getsize(self.path(f"{n}.jsonl"))]
        return datetime.fromtimestamp(max(mt), timezone.utc).strftime("%Y%m%d") if mt else None

    def last_minute_end(self) -> float | None:
        """End (epoch s) of the newest minute row on disk (current file, else the newest archived day)."""
        r = self._last_record("minutes", lambda r: not r.get("gap") and r.get("t") is not None)
        if r is None:
            arch = sorted(f for f in os.listdir(os.path.join(self.dir, "archive")) if f.startswith("minutes-"))
            for f in reversed(arch[-2:]):
                try:
                    with open(os.path.join(self.dir, "archive", f), "rb") as fh:
                        for ln in reversed(fh.read().splitlines()):
                            try:
                                x = json.loads(ln)
                            except ValueError:
                                continue
                            if isinstance(x, dict) and not x.get("gap") and x.get("t") is not None:
                                r = x
                                break
                except OSError:
                    continue
                if r is not None:
                    break
        if r is None:
            return None
        return float(r["t"]) + float(r.get("dur_s") or MINUTE_S)

    def recent_minutes(self, days: float) -> list[dict]:
        """Minute rows (not gap rows) of the last ``days`` days before the newest one on disk, time order: the
        storm level's memory (``Alert.restore``)."""
        files = sorted(glob.glob(os.path.join(self.dir, "archive", "minutes-*.jsonl")))[-int(math.ceil(days) + 1):]
        files.append(self.path("minutes.jsonl"))
        rows = []
        for p in files:
            try:
                with open(p, "rb") as f:
                    for ln in f:
                        try:
                            r = json.loads(ln)
                        except ValueError:
                            continue
                        if isinstance(r, dict) and not r.get("gap") and isinstance(r.get("t"), (int, float)):
                            rows.append(r)
            except OSError:
                continue
        if not rows:
            return []
        rows.sort(key=lambda r: r["t"])
        t1 = rows[-1]["t"]
        return [r for r in rows if r["t"] >= t1 - days * 86400]

    def rotate_if_needed(self, t: float) -> None:
        day = datetime.fromtimestamp(t, timezone.utc).strftime("%Y%m%d")
        if self.day is not None and day != self.day:
            for name in ("events", "minutes", "alerts"):
                src = self.path(f"{name}.jsonl")
                if os.path.exists(src):
                    dst = os.path.join(self.dir, "archive", f"{name}-{self.day}.jsonl")
                    if os.path.exists(dst):               # restart on the same day after a rotation: append
                        with open(src) as a, open(dst, "a") as b:
                            shutil.copyfileobj(a, b)
                        os.remove(src)
                    else:
                        os.replace(src, dst)
            self.prune(t)
        self.day = day

    def prune(self, t: float) -> None:
        """Delete archived days older than keep_days and raw days older than keep_raw_days."""
        cut = (datetime.fromtimestamp(t, timezone.utc) - timedelta(days=self.keep_days)).strftime("%Y%m%d")
        for f in os.listdir(os.path.join(self.dir, "archive")):
            d = f.rsplit("-", 1)[-1].split(".")[0]
            if d.isdigit() and d < cut:
                os.remove(os.path.join(self.dir, "archive", f))
        rawd = self.path("raw")
        if os.path.isdir(rawd):
            cut = (datetime.fromtimestamp(t, timezone.utc) - timedelta(days=self.keep_raw_days)).strftime("%Y%m%d")
            for d in os.listdir(rawd):
                if d.isdigit() and d < cut:
                    shutil.rmtree(os.path.join(rawd, d), ignore_errors=True)

    def check_disk(self) -> None:
        try:
            self.free_gb = shutil.disk_usage(self.dir).free / 1e9
        except OSError:
            return
        self.warning = None
        if self.free_gb < self.min_free_gb:
            if self.raw_on:
                self.close_raw()
            self.raw_on = False
            self.warning = f"disk low: {self.free_gb:.1f} GB free (< {self.min_free_gb} GB): raw recording off"
        else:
            self.raw_on = self.raw_wanted
        self.jsonl_on = self.free_gb >= MIN_FREE_JSONL_GB
        if not self.jsonl_on:
            self.warning = f"disk nearly full: {self.free_gb:.2f} GB free: event and minute files paused"

    def append(self, name: str, recs: list[dict], t: float) -> None:
        if not recs or not self.jsonl_on:
            return
        self.rotate_if_needed(t)
        path = self.path(f"{name}.jsonl")
        text = "".join(json.dumps(r, allow_nan=False, separators=(",", ":")) + "\n" for r in recs)
        size0 = os.path.getsize(path) if os.path.exists(path) else 0
        try:
            with open(path, "a") as f:
                f.write(text)
        except OSError:
            try:                                              # never leave a torn line for the next append
                os.truncate(path, size0)
            except OSError:
                pass
            raise

    def write_state(self, doc: dict) -> None:
        atomic_write_json(self.path("state.json"), doc)

    # -- raw recording, per minute file
    def raw_write(self, t_minute: float, iq: np.ndarray, meta: dict) -> None:
        if not self.raw_on:
            return
        key = int(t_minute)
        if key != self._raw_key:
            self.close_raw()
            d = datetime.fromtimestamp(t_minute, timezone.utc)
            folder = os.path.join(self.dir, "raw", d.strftime("%Y%m%d"))
            os.makedirs(folder, exist_ok=True)
            stem = os.path.join(folder, d.strftime("vlf_%H%MZ"))
            self._raw_f = open(stem + ".cs16", "ab")
            self._raw_key = key
            atomic_write_json(stem + ".json", {"utc": d.isoformat(), "utc_is_sample0": True,
                                               "stamp_offset_removed_s": rd.STAMP_OFFSET_S, **meta})
        iq.tofile(self._raw_f)

    def close_raw(self) -> None:
        if self._raw_f is not None:
            self._raw_f.close()
        self._raw_f, self._raw_key = None, None


# ======================================================================================= the radio lock

_PY_OPTS_WITH_ARG = {"-X", "-W", "-Q"}


def runs_script(argv: list[str], names=FOREIGN) -> bool:
    """True when a command line runs one of ``names`` as a program: the program itself (``./dlayer_run.py``), or a
    Python interpreter's script argument after any interpreter options (``python3 -X dev -u dlayer_run.py``; the
    first check only looked at argv[:3] and missed that). An editor or ``less dlayer_run.py`` is not a run."""
    if not argv:
        return False
    if os.path.basename(argv[0]) in names:
        return True
    exe = os.path.basename(argv[0])
    if not exe.startswith("python") and exe not in ("uv", "pypy3", "pypy"):
        return False
    i = 1
    while i < len(argv):
        a = argv[i]
        if a in ("-c", "-m"):
            return False
        if a in _PY_OPTS_WITH_ARG:
            i += 2
            continue
        if a == "run" and exe == "uv":
            i += 1
            continue
        if a.startswith("-"):
            i += 1
            continue
        return os.path.basename(a) in names
    return False


def foreign_radio_users(names=FOREIGN) -> list[int]:
    """PIDs of other processes whose command line runs one of ``names`` (the programs that use the radio without
    the lock)."""
    me = os.getpid()
    out = []
    for d in os.listdir("/proc"):
        if not d.isdigit() or int(d) == me:
            continue
        try:
            with open(f"/proc/{d}/cmdline", "rb") as f:
                argv = [a.decode(errors="replace") for a in f.read().split(b"\0") if a]
        except OSError:
            continue
        if runs_script(argv, names):
            out.append(int(d))
    return out


class RadioBusy(RuntimeError):
    pass


class RadioLock:
    """Exclusive, non-blocking ``flock`` on the radio lock file, held while the daemon owns the SDR. The file holds
    the owner's pid and program name (informational; the flock is the lock and dies with the process)."""

    def __init__(self, path: str = LOCK_PATH, check_foreign: bool = True):
        self.path = os.path.expanduser(path)
        self.check_foreign = check_foreign
        self.fd = None

    def acquire(self) -> None:
        if self.check_foreign:
            pids = foreign_radio_users()
            if pids:
                raise RadioBusy(f"dlayer_run.py is running (pid {', '.join(map(str, pids))}): one radio, one program")
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        fd = os.open(self.path, os.O_RDWR | os.O_CREAT, 0o644)
        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            try:
                holder = os.pread(fd, 200, 0).decode(errors="replace").strip()
            except OSError:
                holder = "?"
            os.close(fd)
            raise RadioBusy(f"radio lock is held ({holder or 'unknown holder'})")
        os.ftruncate(fd, 0)
        os.pwrite(fd, f"{os.getpid()} sferic-watch live\n".encode(), 0)
        self.fd = fd

    def release(self) -> None:
        if self.fd is not None:
            try:
                os.ftruncate(self.fd, 0)
                fcntl.flock(self.fd, fcntl.LOCK_UN)
            finally:
                os.close(self.fd)
                self.fd = None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, *a):
        self.release()


# ======================================================================================= the worker

def _die_with_parent(parent_pid: int) -> None:
    """The chain worker must not outlive the daemon (SIGKILL, OOM): the kernel's parent-death signal (Linux prctl),
    and a watchdog thread that exits when the parent pid changes. The thread is what catches the case the old 5 s
    poll in the main loop missed: the daemon killed while a 256 KB data message is only partly in the pipe leaves
    the worker blocked in that read for ever (it never reached the poll)."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6", use_errno=True)
        libc.prctl(1, signal.SIGKILL, 0, 0, 0)           # PR_SET_PDEATHSIG
    except (OSError, AttributeError):
        pass
    if os.getppid() != parent_pid:                        # it died before the prctl
        os._exit(0)

    def watch():
        while True:
            time.sleep(1.0)
            if os.getppid() != parent_pid:
                os._exit(0)
    threading.Thread(target=watch, name="parent-watch", daemon=True).start()


def _malloc_tamer():
    """glibc's dynamic mmap threshold grows to 32 MB after the first big free, so the 15 MB per-minute arrays then
    come from the heap and the worker's resident floor ratchets up after the 1.7 GB per-commit peaks (soak: 254 ->
    463 MB in steps). A fixed 4 MB threshold keeps big arrays in mmap (given back on free), and ``malloc_trim``
    after each commit returns the freed heap. Returns the trim function (a no-op off glibc)."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 4 << 20)                         # M_MMAP_THRESHOLD (fixing it also stops the growth)
        return lambda: libc.malloc_trim(0)
    except (OSError, AttributeError):
        return lambda: None


def _close_unused_ends(q, keep: str) -> None:
    """Close this process's copy of the pipe end of ``q`` it never uses (``keep`` "reader" or "writer"). Then a
    peer that dies is seen as end-of-file, not a read that waits for ever on bytes that will never come."""
    try:
        (q._writer if keep == "reader" else q._reader).close()
    except (AttributeError, OSError):
        pass


def _worker_main(inq, outq, chain_kw, parent_pid=None):
    """Chain worker process: ("start", seg) / ("data", seg, iq) / ("end", seg) / ("stop",) in, results out. It
    runs in its own process so a long numpy call can never starve the radio reader thread of the GIL."""
    parent_pid = parent_pid or os.getppid()
    _die_with_parent(parent_pid)
    trim = _malloc_tamer()
    _close_unused_ends(inq, "reader")
    _close_unused_ends(outq, "writer")
    try:
        os.nice(5)
    except OSError:
        pass
    signal.signal(signal.SIGINT, signal.SIG_IGN)          # the parent decides when to stop
    signal.signal(signal.SIGTERM, signal.SIG_IGN)         # systemd signals the whole group; the parent drains us
    w = InlineWorker(chain_kw)
    while True:
        try:
            msg = inq.get(timeout=5.0)
        except queue.Empty:
            if os.getppid() != parent_pid:
                break
            continue
        except (EOFError, OSError):                       # the daemon is gone (its pipe end closed)
            break
        if msg[0] == "stop":
            outq.put(("stopped",))
            return
        w.put(msg)
        if w.out:
            while w.out:
                outq.put(w.out.popleft())
            trim()
    outq.cancel_join_thread()                             # nobody will read what is still queued


class InlineWorker:
    """Same protocol as the worker process, run synchronously (tests, unpaced replays)."""

    def __init__(self, chain_kw):
        self.chain_kw = chain_kw
        self.chains = {}
        self.out = deque()

    def put(self, msg):
        """A chain that raises is dropped with its segment (an "error" reply); later data for it is ignored and
        the daemon starts a new segment, so one bad minute cannot stop the watcher."""
        if msg[0] == "stop":
            self.out.append(("stopped",))
            return
        seg = msg[1]
        try:
            if msg[0] == "start":
                self.chains[seg] = Chain(**self.chain_kw)
            elif msg[0] == "data" and seg in self.chains:
                for r in self.chains[seg].push(msg[2]):
                    self.out.append(("minute", seg, r))
            elif msg[0] == "end":
                ch = self.chains.pop(seg, None)
                for r in (ch.end() if ch is not None else []):
                    self.out.append(("minute", seg, r))
                self.out.append(("ended", seg))
        except Exception:
            self.chains.pop(seg, None)
            self.out.append(("error", seg, traceback.format_exc()))

    def get(self, timeout=None):
        if not self.out:
            raise queue.Empty
        return self.out.popleft()

    def close(self):
        pass

    def terminate(self):
        pass

    def alive(self):
        return True


def _quiet_queue(ctx, maxsize: int = 0):
    """An mp.Queue whose feeder thread ends quietly on a broken pipe (the peer process died) instead of printing a
    traceback for every message still buffered. With hundreds of 256 KB messages queued, that print storm from a
    daemon thread at interpreter exit aborted the daemon (SIGABRT, core dump) after a worker kill."""
    q = ctx.Queue(maxsize)
    q._ignore_epipe = True                    # read when the feeder thread starts (first put)
    return q


class ProcessWorker:
    def __init__(self, chain_kw, maxsize: int = 400):
        ctx = mp.get_context("spawn")
        self.inq, self.outq = _quiet_queue(ctx, maxsize), _quiet_queue(ctx)
        self.proc = ctx.Process(target=_worker_main, args=(self.inq, self.outq, chain_kw, os.getpid()),
                                name="sferic-chain", daemon=True)
        self.proc.start()
        _close_unused_ends(self.inq, "writer")            # a dead worker = a broken pipe, not a blocked feeder
        _close_unused_ends(self.outq, "reader")           # ...and EOF on a half-sent result, not a hang
        self.dead = False

    def put(self, msg, timeout=None):
        self.inq.put(msg, timeout=timeout)

    def get(self, timeout=None):
        try:
            return self.outq.get(timeout=timeout) if timeout else self.outq.get_nowait()
        except (EOFError, OSError):                       # the worker died in the middle of a result
            self.dead = True
            raise queue.Empty

    def alive(self):
        return not self.dead and self.proc.is_alive()

    def terminate(self):
        self.proc.terminate()
        self.proc.join(5)
        if self.proc.is_alive():
            self.proc.kill()
            self.proc.join(5)

    def close(self):
        self.proc.join(10)
        if self.proc.is_alive():
            self.terminate()
        # whatever is still queued for a dead worker will never be read: do not let interpreter exit wait for
        # the feeder thread (it may be blocked part-way through a message)
        for q in (self.inq, self.outq):
            try:
                q.cancel_join_thread()
                q.close()
            except (OSError, ValueError, AttributeError):
                pass


# ======================================================================================= the daemon

class Segment:
    def __init__(self, sid: int, t0: float, fs: float, minute_n: int):
        self.id, self.fs, self.N = sid, fs, minute_n
        self.n = 0
        self.anchors = [(0, t0)]
        self.snap = {}                    # minute -> radio counters when its last sample arrived

    def epoch(self, n: float) -> float:
        i, t = self.anchors[0]
        for a, ta in self.anchors:
            if a <= n:
                i, t = a, ta
        return t + (n - i) / self.fs


class Daemon:
    """Radio -> segments -> chain worker -> files and alert. See the module doc."""

    def __init__(self, radio: rd.RadioSource, store: Store, align: str = "utc", minute_s: float = MINUTE_S,
                 history_min: int = HISTORY_MIN, lookahead_min: int = LOOKAHEAD_MIN, inline: bool = False,
                 notifier=None, duration_s: float | None = None, log=print, lock: RadioLock | None = None,
                 warmup: bool = True, restore: bool = True):
        self.radio, self.store, self.align = radio, store, align
        self.fs = radio.fs
        self.minute_s = minute_s
        self.N = int(round(minute_s * self.fs))
        self.chain_kw = {"fs": self.fs, "minute_s": minute_s, "history_min": history_min,
                         "lookahead_min": lookahead_min, "warmup": warmup}
        self.inline = inline
        self.alert = Alert(notifier or null_notifier)
        self.restored_minutes = 0
        if restore:                                # the level and its baselines survive a restart
            try:
                self.restored_minutes = self.alert.restore(store.recent_minutes(BAND_DAYS + 1))
            except Exception as e:                 # a bad archive must not stop the watcher
                log(f"storm level not restored: {type(e).__name__}: {e}")
        self.prev_end = store.last_minute_end() if restore else None   # for the gap row after a restart
        self.duration_s = duration_s
        self.log = log
        self.lock = lock
        self.stop_ev = threading.Event()
        self.seg: Segment | None = None
        self.segs: dict[int, Segment] = {}
        self.next_sid = 0
        self.started = time.time()
        self.last_block_wall = time.time()
        self.latest_minute = None
        self.n_minutes = self.n_events = self.n_segments = 0
        self.errors = []
        self.status = "starting"
        self.last_hb = 0.0
        self.cpu_recent = deque(maxlen=60)
        self.latency_recent = deque(maxlen=60)
        self._last_snap = {}
        self.exit_reason = None
        self.worker = None
        self.open_segments = set()
        self.write_errors = 0
        self.last_write_error = None
        self.worker_hung = False
        self._chain_error_gap = False
        self._write_gap = False
        self.n_gaps = 0
        self.yielded = None

    # -- radio counters snapshot
    def _counters(self, b: rd.Block | None = None) -> dict:
        """Fault counters as of block b's read (the reader may run ahead of the consumer), else now. Ring drops are
        counted when the consumer side drops, so they are always taken now."""
        c = dict(b.counters) if b is not None and b.counters else self.radio.stats.counters()
        c["ring_drops"] = self.radio.stats.ring_drops
        return c

    # -- segments
    def _new_segment(self, t0: float) -> Segment:
        seg = Segment(self.next_sid, t0, self.fs, self.N)
        self.next_sid += 1
        self.segs[seg.id] = seg
        self.open_segments.add(seg.id)
        self.n_segments += 1
        self._send(("start", seg.id))
        return seg

    def _end_segment(self) -> None:
        if self.seg is not None:
            self._send(("end", self.seg.id))
            self.seg = None

    def _send(self, msg) -> None:
        """Hand a message to the worker. A full queue waits while the worker drains it (results are taken and the
        heartbeat kept meanwhile), but never for a dead worker (the main loop then exits with ``worker_died``) and
        never for longer than WORKER_STUCK_S (a hung worker is killed: ``worker_hung``)."""
        if self.inline:
            self.worker.put(msg)
            self._drain()
            return
        t0 = time.time()
        while True:
            if not self.worker.alive():
                return
            try:
                self.worker.put(msg, timeout=1.0)
                return
            except queue.Full:
                self._drain()
                self._heartbeat()
                if time.time() - t0 > WORKER_STUCK_S:
                    self.worker_hung = True
                    self.log(f"chain worker took no data for {WORKER_STUCK_S:.0f} s: killing it")
                    self.worker.terminate()
                    return

    def on_block(self, b: rd.Block) -> None:
        if not self.worker.alive():
            return
        iq, t0 = b.iq, b.t0
        if b.gap_before or self.seg is None:
            self._end_segment()
            if self.align == "utc":
                boundary = math.ceil(t0 / self.minute_s - 1e-9) * self.minute_s
                k = int(round((boundary - t0) * self.fs))
                if k >= b.n:
                    return                                   # wait for a block that crosses the boundary
                iq, t0 = iq[2 * k:], t0 + k / self.fs
            self.seg = self._new_segment(t0)
        else:
            if b.anchor_shift_s:                             # the radio refined the anchor in force (radio doc)
                a, ta = self.seg.anchors[-1]
                self.seg.anchors[-1] = (a, ta + b.anchor_shift_s)
            if b.reanchored:
                self.seg.anchors.append((self.seg.n, t0))
        seg = self.seg
        n_before = seg.n
        seg.n += len(iq) // 2
        for m in range(n_before // self.N, seg.n // self.N):  # minutes completed by this block
            seg.snap[m] = self._counters(b)
        if self.store.raw_on:
            pos = 0
            while pos < len(iq) // 2:
                m = (n_before + pos) // self.N
                take = min(len(iq) // 2 - pos, (m + 1) * self.N - (n_before + pos))
                try:
                    self.store.raw_write(seg.epoch(m * self.N), iq[2 * pos:2 * (pos + take)],
                                         {"fs_hz": self.fs, "freq_hz": self.radio.fc, **self.radio.describe()})
                except OSError as e:                     # disk full / I/O error: raw off, the watch goes on
                    self.write_errors += 1
                    self.last_write_error = f"raw: {e}"
                    self.store.close_raw()
                    self.store.raw_on = self.store.raw_wanted = False
                    self.log(f"raw recording stopped: {e}")
                    break
                pos += take
        self._send(("data", seg.id, iq))

    # -- results
    def _drain(self, timeout=None) -> None:
        while True:
            try:
                msg = self.worker.get(timeout)
            except queue.Empty:
                return
            timeout = None
            kind = msg[0]
            if kind == "minute" and msg[1] in self.segs:
                self._on_minute(self.segs[msg[1]], msg[2])
            elif kind == "ended":
                self.open_segments.discard(msg[1])
                self.segs.pop(msg[1], None)
            elif kind == "error":
                self.errors.append(msg[2].strip().splitlines()[-1])
                self._chain_error_gap = True
                self.log(f"chain error (segment {msg[1]} dropped): {msg[2]}")
                self.open_segments.discard(msg[1])
                self.segs.pop(msg[1], None)
                if self.seg is not None and self.seg.id == msg[1]:
                    self.seg = None                          # the next block starts a new segment
            elif kind == "stopped":
                self._worker_stopped = True

    def _on_minute(self, seg: Segment, r: dict) -> None:
        m = r["minute"]
        t_min = seg.epoch(m * self.N)
        row = r["row"]
        snap = seg.snap.pop(m, self._counters())
        delta = {k: v - self._last_snap.get(k, 0) for k, v in snap.items()}
        self._last_snap = snap
        st = self.radio.stats
        latency = self.radio.now() - seg.epoch((m + 1) * self.N)   # mock: stream time (meaningless unpaced)
        rec = {"utc": iso(t_min), "t": round(t_min, 6), "dur_s": row["dur_s"],
               "n_events": row["n_events"], "rate_per_min": row["rate_per_min"],
               "rate_open_per_min": row["rate_open_per_min"], "n_conf": row["n_conf"],
               "conf_per_min": row["conf_per_min"], "v2_per_min": row["v2_per_min"],
               "v2_chance_per_min": row["v2_chance_per_min"], "hum_frac": row["hum_frac"],
               "gated_frac": row["gated_frac"], "comb_share": row["comb_share"], "comb_chance": row["comb_chance"],
               "trusted": row["trusted"], "untrusted_because": row["untrusted_because"],
               "health": {**r["health"], **delta, "clock_drift_ppm": _r(st.clock_drift_ppm, 3)},
               "segment": seg.id, "seg_minute": m, "lookahead_min": r["lookahead_min"],
               "cpu_s": _r(r["cpu_s"], 3), "delay_s": _r(latency, 1),
               "engine": pl.VERSION, "live": LIVE_VERSION}
        ev = r["events"]
        evs = []
        for i in range(len(ev["t_start"])):
            t = seg.epoch(ev["t_start"][i] * self.fs)
            evs.append({"utc": iso(t, us=True), "t": round(t, 6), "peak_db": _r(ev["peak_db"][i], 2),
                        "conf": bool(ev["conf"][i]), "conf_db": _r(ev["conf_db"][i], 2),
                        "coinc_hi": bool(ev["coinc_hi"][i]), "hum": bool(ev["hum"][i]),
                        "width_us": _r(ev["width_s"][i] * 1e6, 1), "floor_dbfs": _r(ev["floor_dbfs"][i], 2)})
        self._write("events", evs, t_min)
        gap = self._gap_row(t_min, delta)
        if self._write("minutes", ([gap] if gap else []) + [rec], t_min):
            self.prev_end = t_min + float(row["dur_s"] or self.minute_s)
            self._write_gap = False
        else:                                            # lost (disk full / paused): the next row's gap row says so
            self._write_gap = True
        a = self.alert.feed(t_min, rec)
        if a is not None:
            self._write("alerts", [a], t_min)
            self.log(f"alert: {a['from']} -> {a['to']} at {a['utc']} (stat {a['stat_conf_per_min']}, "
                     f"baseline {a['baseline_conf_per_min']})")
        self.latest_minute = rec
        self.n_minutes += 1
        self.n_events += len(evs)
        self.cpu_recent.append(r["cpu_s"])
        if latency is not None:
            self.latency_recent.append(latency)
        if self.n_minutes % 10 == 0:
            self.store.check_disk()

    def _gap_row(self, t_min: float, delta: dict) -> dict | None:
        """An explicit row for the minutes no segment covers (radio loss, a chain error, the watcher not running),
        written just before the first minute after them: ``gap`` true, ``t``..``t_end``, ``missing_min`` and the
        likely ``reason`` (the radio fault counters that moved, else "unknown"). Readers of minutes.jsonl skip rows
        with ``gap`` (the panel and the storm level do)."""
        prev, self_first = self.prev_end, self.n_minutes == 0
        if prev is None or t_min - prev < 0.5:
            return None                                  # contiguous, or a host clock step back (no gap)
        reasons = [k for k, v in delta.items() if v > 0]
        if self._chain_error_gap:
            reasons.append("chain_error")
            self._chain_error_gap = False
        if self._write_gap:
            reasons.append("not_written")
        if self_first:
            reasons.append("watcher_not_running")
        self.n_gaps += 1
        return {"gap": True, "utc": iso(prev), "t": round(prev, 6), "t_end": round(t_min, 6),
                "dur_s": _r(t_min - prev, 3), "missing_min": _r((t_min - prev) / 60.0, 2),
                "reason": reasons or ["unknown"], "radio": delta, "live": LIVE_VERSION}

    def _write(self, name: str, recs: list[dict], t: float) -> bool:
        """Append to an output file; a failed write (disk full, I/O error) is counted, logged and shown in
        state.json, and the watcher keeps running (those records are lost; the disk check pauses writing when the
        free space is known to be low)."""
        if not self.store.jsonl_on:
            return False
        try:
            self.store.append(name, recs, t)
            return True
        except OSError as e:
            self.write_errors += 1
            self.last_write_error = f"{name}.jsonl: {e}"
            if self.write_errors == 1 or self.write_errors % 100 == 0:
                self.log(f"write failed ({self.write_errors} so far, records lost): {self.last_write_error}")
            return False

    # -- state
    def state_doc(self) -> dict:
        now = time.time()
        st = self.radio.stats
        return {"schema": "sferic-watch/live-state/1", "status": self.status, "pid": os.getpid(),
                "exit_reason": self.exit_reason if self.status in ("stopping", "stopped") else None,
                "heartbeat_utc": iso(now), "heartbeat_t": round(now, 3),
                "started_utc": iso(self.started), "uptime_s": round(now - self.started, 1),
                "engine": pl.VERSION, "live": LIVE_VERSION, "package": sw.__version__,
                "radio": {**self.radio.describe(), "stats": st.as_dict(), "ring_fill": round(self.radio.ring_fill(), 3),
                          "seconds_since_block": round(now - self.last_block_wall, 1)},
                "chain": {**{k: v for k, v in self.chain_kw.items() if k != "fs"}, "align": self.align,
                          "segments": self.n_segments, "minutes": self.n_minutes, "events": self.n_events,
                          "gaps": self.n_gaps, "restored_minutes": self.restored_minutes,
                          "open_segments": len(self.open_segments),
                          "cpu_s_per_minute_median": _med(self.cpu_recent, 3),
                          "delay_s_median": _med(self.latency_recent, 1),
                          "errors": self.errors[-5:]},
                "last_minute": self.latest_minute,
                "alert": self.alert.last or {"level": 0, "name": LEVELS[0], "status": "no_minutes_yet"},
                "disk": {"free_gb": _r(self.store.free_gb, 2), "raw_recording": self.store.raw_on,
                         "jsonl_writing": self.store.jsonl_on, "warning": self.store.warning,
                         "write_errors": self.write_errors, "last_write_error": self.last_write_error,
                         "torn_lines_cut_on_start": getattr(self.store, "repaired", {}),
                         "corrupt_lines_dropped_on_start": getattr(self.store, "corrupt_dropped", {})}}

    def _heartbeat(self, force: bool = False) -> None:
        now = time.time()
        if force or now - self.last_hb >= HEARTBEAT_S:
            self.last_hb = now
            if now - getattr(self, "_trim_at", now) >= 60.0:
                self._trim_at = now
                self._trim()
            try:
                self.store.write_state(self.state_doc())
            except OSError as e:
                self.log(f"state.json write failed: {e}")

    # -- main loop
    def request_stop(self, *_a) -> None:
        self.stop_ev.set()

    def run(self) -> int:
        self.worker = InlineWorker(self.chain_kw) if self.inline else ProcessWorker(self.chain_kw)
        # the queue feeder and the ring hold up to ~135 MB of 256 KB blocks during a warm-up burst; give the freed
        # heap back afterwards (soak: the daemon's RSS otherwise stayed at its high-water, 112 -> 235 MB)
        self._trim = _malloc_tamer() if not self.inline else (lambda: None)
        self._trim_at = time.time()
        self._worker_stopped = False
        self.radio.start()
        self.status = "running"
        self._heartbeat(force=True)
        t_end = None if not self.duration_s else time.time() + self.duration_s
        restarted_at = 0.0
        foreign_at = time.time()
        self.exit_reason = "stop requested"
        try:
            while not self.stop_ev.is_set():
                if t_end is not None and time.time() >= t_end:
                    self.exit_reason = "duration reached"
                    break
                if self.worker_hung:
                    self.status = self.exit_reason = "worker_hung"
                    self.errors.append("chain worker hung (killed)")
                    break
                if not self.worker.alive():
                    self.status = self.exit_reason = "worker_died"
                    self.errors.append("chain worker process died")
                    break
                if self.lock is not None and self.lock.check_foreign and time.time() - foreign_at > FOREIGN_CHECK_S:
                    foreign_at = time.time()
                    pids = foreign_radio_users()
                    if pids:                                 # one radio, one program: the D-layer run wins
                        self.yielded = f"dlayer_run.py started (pid {', '.join(map(str, pids))})"
                        self.exit_reason = f"yielded the radio: {self.yielded}"
                        self.log(f"{self.exit_reason}: closing the radio and exiting (code 3)")
                        break
                b = self.radio.get(timeout=0.5)
                now = time.time()
                if b is not None:
                    self.last_block_wall = now
                    if self.status == "stalled":
                        self.status = "running"
                    self.restarts_without_block = 0
                    self.on_block(b)
                elif self.radio.finished and not self.radio._ring:
                    self.exit_reason = "source ended"
                    break
                elif not self.radio.alive and not self.radio.finished:
                    self.status = self.exit_reason = "radio_thread_died"
                    self.errors.append("radio reader thread died")
                    break
                else:
                    idle = now - self.last_block_wall
                    if idle > STALL_S and self.radio.stats.state == "streaming":
                        self.status = "stalled"
                        if idle > RESTART_S and now - restarted_at > RESTART_S:
                            self.log(f"no samples for {idle:.0f} s: reopening the radio")
                            self.radio.restart()
                            restarted_at = now
                            self.restarts_without_block = getattr(self, "restarts_without_block", 0) + 1
                            if self.restarts_without_block > MAX_DEAD_RESTARTS:
                                # a readStream stuck in the driver never sees the restart flag: give up and let
                                # systemd (Restart=on-failure) start a fresh process
                                self.status = self.exit_reason = "radio_wedged"
                                self.errors.append("radio_wedged: no samples after repeated reopen requests")
                                break
                self._drain()
                self._heartbeat()
        except Exception as e:
            self.exit_reason = f"crashed: {type(e).__name__}: {e}"
            self.errors.append(self.exit_reason)
            raise
        finally:
            self.shutdown()
        if self.yielded:
            return 3
        return 0 if not self.errors else 1

    def shutdown(self) -> None:
        self.status = "stopping"
        self._heartbeat(force=True)
        if self.radio.stop() is False:
            self.errors.append("radio reader thread did not stop")
            self.log("radio reader thread did not stop within its timeout (driver hung?)")
        while True:                                          # what the reader already holds is still processed
            b = self.radio.get(timeout=0)
            if b is None:
                break
            self.on_block(b)
        if self.worker.alive():
            self._end_segment()
            self._send(("stop",))
        t0 = time.time()
        while not self._worker_stopped and self.worker.alive() and time.time() - t0 < 300:
            self._drain(timeout=1.0)
            self._heartbeat()
        self.worker.close()
        self.store.close_raw()
        self.status = "stopped"
        self._heartbeat(force=True)
        if self.lock is not None:
            self.lock.release()


# ======================================================================================= command line

def build_parser(ap: argparse.ArgumentParser | None = None) -> argparse.ArgumentParser:
    ap = ap or argparse.ArgumentParser(prog="sferic_watch live", description=__doc__,
                                       formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--radio", choices=("soapy", "mock"), default="soapy")
    ap.add_argument("--antenna", default=rd.ANTENNA, help="SoapySDR antenna name (default 'Antenna C')")
    ap.add_argument("--state-dir", default=STATE_DIR)
    ap.add_argument("--lock-file", default=LOCK_PATH)
    ap.add_argument("--mock-files", nargs="*", default=[], help="captures the mock radio replays")
    ap.add_argument("--speed", type=float, default=1.0, help="mock replay speed (0 = as fast as possible)")
    ap.add_argument("--loop", action="store_true", help="mock: start again after the last file")
    ap.add_argument("--mock-faults", default=None,
                    help='mock: JSON list of faults, e.g. \'[{"kind": "overflow", "at_s": 300, "n": 3}]\' '
                         "(radio.Fault: overflow, timeout, drop, dup, vanish, stall, clock)")
    ap.add_argument("--duration", type=float, default=None, help="stop after this many wall-clock seconds")
    ap.add_argument("--align", choices=("utc", "start"), default=None,
                    help="segment start: next UTC minute (default for soapy) or first sample (default for mock)")
    ap.add_argument("--history-min", type=int, default=HISTORY_MIN)
    ap.add_argument("--lookahead-min", type=int, default=LOOKAHEAD_MIN)
    ap.add_argument("--no-warmup", action="store_true",
                    help="commit the first minutes of a segment with a short gate window instead of waiting for a "
                         "full one (lower delay after a start or gap; more buzz let through, see Chain.window)")
    ap.add_argument("--no-restore", action="store_true",
                    help="start the storm level from scratch instead of replaying the minute files")
    ap.add_argument("--raw", action="store_true", help="also record raw per-minute .cs16 (off by default)")
    ap.add_argument("--keep-days", type=int, default=KEEP_DAYS)
    ap.add_argument("--keep-raw-days", type=int, default=KEEP_RAW_DAYS)
    ap.add_argument("--min-free-gb", type=float, default=MIN_FREE_GB)
    ap.add_argument("--notify-cmd", default=None, help="run this shell command with each alert as JSON on stdin")
    ap.add_argument("--inline", action="store_true", help="run the chain in this process (tests, debugging)")
    return ap


def main(argv=None, args=None) -> int:
    if args is None:
        args = build_parser().parse_args(argv)

    def log(msg):
        print(f"{iso(time.time())} {msg}", flush=True)

    lock = None
    if args.radio == "soapy":
        lock = RadioLock(args.lock_file)
        try:
            lock.acquire()
        except RadioBusy as e:
            log(f"refusing to start: {e}")
            return 3
        radio = rd.SoapyRadio(antenna=args.antenna)
        align = args.align or "utc"
    else:
        if not args.mock_files:
            log("--radio mock needs --mock-files")
            return 2
        faults = json.loads(args.mock_faults) if args.mock_faults else []
        radio = rd.MockRadio(files=args.mock_files, speed=args.speed, loop=args.loop, faults=faults)
        align = args.align or "start"
    store = Store(args.state_dir, keep_days=args.keep_days, raw=args.raw, keep_raw_days=args.keep_raw_days,
                  min_free_gb=args.min_free_gb)
    d = Daemon(radio, store, align=align, history_min=args.history_min, lookahead_min=args.lookahead_min,
               inline=args.inline, notifier=command_notifier(args.notify_cmd) if args.notify_cmd else None,
               duration_s=args.duration, log=log, lock=lock, warmup=not args.no_warmup,
               restore=not args.no_restore)
    signal.signal(signal.SIGTERM, d.request_stop)
    signal.signal(signal.SIGINT, d.request_stop)
    log(f"live: radio {args.radio}, state dir {args.state_dir}, window {args.history_min}+1+{args.lookahead_min} min"
        f"{'' if args.no_warmup else ' (warm-up)'}, storm level restored from {d.restored_minutes} minutes")
    rc = d.run()
    log(f"stopped: {d.n_minutes} minutes, {d.n_events} events, {d.n_segments} segments")
    return rc


if __name__ == "__main__":
    sys.exit(main())
