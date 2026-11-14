"""Median-floor impulse detector (plan step 4), with a July-rule compatibility column.

Idea from OnAir's sferics receiver (smoothed power over a median-of-block-means floor, a start/end hysteresis, a
width cap and a short merge); the code, structure and numbers here are ours (MIT). It runs on the step-3 front end
output (``sferic_watch.frontend``): the real 2.5-9 kHz band after receiver-tone removal and line excision.

Why this detector
-----------------
* OnAir's logic behind our 2.5-9 kHz band-pass scored chance-corrected injected recall 0.71 on the h2h pieces against
  0.45 for our July rule (bench/README.md). Its floor is a median, so a busy sky does not lift it the way a mean does.
* The July rule's 8 ms dead time is blind under 120 Hz buzz: the buzz arrives every 8.33 ms, so each detection hides
  the next 96 % of the time (h2h). There is NO dead time here; the step-5 mains gate decides about buzz instead.
* Raw IQ is useless for any floor: the 0 Hz receiver tone (-22.6 dBFS) set OnAir's floor on all 12 h2h captures and it
  found nothing. Hence: always the front end first.

The rule, with the numbers it uses
----------------------------------
1. Power: det^2 smoothed by a centred 16-sample boxcar (256 us at 62.5 kS/s). The boxcar is centred, so it adds no
   delay on average, but it spreads an edge over +-128 us: a one-sample spike becomes a 16-sample flat top, and the
   +10 dB crossing (t_start) can come up to 128 us before the spike (verifier: median -80 us against July edges).
   t_peak takes the MIDDLE of a flat top, not its first sample (a first-sample argmax ran 112 us early).
2. Floor: mean power of 1 ms blocks (62 samples = 0.992 ms); every 0.1 s the floor becomes the median of the block
   means of the previous 4 s (trailing, so a live version gives the same answer). It is trusted once 0.5 s of blocks
   exist; offline, samples before that use the first trusted floor (``backfill``). Blocks a later gate marks bad can be
   left out (``exclude_blocks``). If exclusion leaves under 0.5 s of usable blocks in a window, the last trusted floor
   is HELD (not dropped: a NaN floor made the detector silently blind) and info reports the held seconds. Input
   shorter than one block has no floor and gives no events (info ``floor_none_s``).
3. An impulse starts where power exceeds the floor by 10 dB and ends at the first sample of a 100 us (6-sample) run
   below floor + 7 dB, the floor of each sample (it tracks the floor; a threshold frozen at the start's floor made a
   sustained level step one endless impulse: 0 of 19 later pulses found in the verifier's +15 dB step). An impulse
   still open 4 s after its start (the floor window) is closed there and dropped as too long.
4. An impulse starting less than 0.5 ms after the previous one ended is merged into it (one strike: return strokes
   or multipath).
5. An event wider than 5 ms is dropped and counted (``too_long``): lightning is not that long; floor steps and
   transmitter keying are. The cap applies to the MERGED event too: a train of short pulses each under 0.5 ms apart
   (keying, noise bursts) would otherwise slip through as one long "sferic" (verifier: 100 pulses at 0.5 ms pitch gave
   one 49 ms event). Once a chain is over 5 ms, every pulse that continues it (starts < 0.5 ms after its end) is
   dropped with it, so a long train is dropped whole rather than restarting an event every 5 ms.

Per event the detector records the threshold-crossing start, the peak time, the peak in dB over the floor (the
maximum of the event; t_peak is the middle of the run within 0.2 dB of it, so a flat top peaks at its centre), the width,
the rise (start to peak), a 50 %-of-peak interpolated onset (an extra timing field, NOT yet validated), the floor in
dBFS, and the 16-30 kHz confirmation: the peak of the analytic 16-30 kHz band's smoothed power within the event
(+-0.5 ms) over that band's own median floor, flagged at +6 dB. A sferic is broadband; mains buzz is 0-2 dB above
16 kHz (census), so the flag separates the two.

``july_times`` / ``july_compat`` keep the July sferic_rate rule (k = 8 MAD-sigma, 8 ms dead time) as a column only.
The k = 6 / 3 ms storm_watch rule is retired: it gave 345-518/min on clean days with only 12-23 % confirmed in
10-14 kHz.

Health: info carries ``too_long`` (dropped events), ``too_long_s`` / ``too_long_max_s`` (seconds the detector was
blind inside them), ``level_steps`` (floor rises >= 6 dB within a 4 s window) and ``floor_rise_db_max``; a level step
reads level_steps >= 1 with too_long_max_s of 1-2 s (synthetic +15/+18 dB steps: 19 of 19 later pulses found).

Cost
----
CPU per minute of signal, harness corpus pass (15 captures, 2 workers, process CPU of the detector call), measured
the same day under the same load (2026-11-14): v2_detect 2.20 median (2.28 max) for the previous version, whose
front end ran six forward-backward IIR passes, a complex exp per sample for every mix and 7 % chunk guards (the
build run on a quieter box read 1.81, front end 1.62); this version 0.59 median, 0.70 max (the full v2 chain 3.15 ->
1.35, its hum flag and gate not included in this change). The front end is now one STFT engine
(``frontend._BandEngine``, about 0.5 s/min). The detector proper went from 0.27 to about 0.10 s/min: the floor
medians are one ``np.partition`` over sliding windows instead of 600 ``np.median`` calls per minute, the per-sample
floor is an ``np.repeat`` instead of an index array, the 16-30 kHz floor is computed only at the steps where events
start, and the power smoothing is a running sum per emitted block (``_Boxcar``). Event output is unchanged on the
corpus (10-19 15Z 79.7 -> 79.8/min, 11-05 15Z 42.2, 11-03 21Z 214.2 -> 214.0) and injected recall is unchanged.

Start threshold and merge, measured (not tuned): on fixed band powers, start 9 / 10 / 10.5 / 11 / 12 dB gives a
10-19 15Z background of 236 / 80 / 55 / 42 / 32 per minute and real_snr clean-piece recall at 12 dB of 0.85 / 0.66 /
0.55 / 0.46 / 0.13; the merge window (0.5-4 ms) moves the background by under 5 % and costs native recall above
1 ms. No setting meets both the <= 60/min background and the >= 0.70 recall; 10 dB / 0.5 ms are kept.

Registered for the bench harness as ``v2_detect`` (mains gate off; step 5 adds the gate).
"""
from __future__ import annotations

import time

import numpy as np
from scipy.ndimage import uniform_filter1d
from scipy.signal import butter, sosfiltfilt

import sferic_watch as sw
from sferic_watch import frontend as fe

FS = fe.FS
SMOOTH_N = 16                 # power smoothing, samples (256 us)
BLOCK_N = 62                  # floor block, samples (0.992 ms)
FLOOR_WIN_S = 4.0             # floor = median over the previous 4 s of blocks
FLOOR_HOP_S = 0.1             # refreshed every 0.1 s
FLOOR_TRUST_S = 0.5           # trusted after 0.5 s of blocks
FLOOR_MIN = 1e-30             # power floor clamp (an all-zero input would otherwise give a -inf dB floor)
START_DB = 10.0               # start threshold over the floor
END_DB = 7.0                  # end threshold over the floor
HOLD_S = 100e-6               # end needs this long below END_DB
MAX_WIDTH_S = 5e-3            # wider single impulses are dropped
MERGE_S = 0.5e-3              # gap below which two impulses are one event
MAX_SCAN_S = FLOOR_WIN_S      # an impulse still open this long after its start is closed (and dropped)
LEVEL_STEP_DB = 6.0           # health: a floor rise of this much within one floor window counts as a level step
PEAK_TOP_DB = 0.2             # t_peak = middle of the run within 0.2 dB of the maximum (flat tops)
CONF_DB = 6.0                 # 16-30 kHz confirmation threshold over that band's floor
CONF_PAD_S = 0.5e-3           # confirmation window: event +-0.5 ms
CHUNK_S = 60.0                # unused since the frame engine (one pass, no chunks); kept for old callers
GUARD_S = 2.0                 # unused likewise (no chunk guards needed on the absolute frame grid)

FIELDS = ("t_start", "t_peak", "t_end", "peak_db", "width_s", "rise_s", "t_half", "floor_dbfs", "conf_db", "conf")


# ----------------------------------------------------------------------------- power and floor

def smooth_power(y: np.ndarray, n: int = SMOOTH_N) -> np.ndarray:
    """|y|^2 smoothed by a centred n-sample boxcar, float32. Real or complex input."""
    p = np.abs(y).astype(np.float64) ** 2 if np.iscomplexobj(y) else np.asarray(y, np.float64) ** 2
    return uniform_filter1d(p, n, mode="nearest").astype(np.float32)


def block_means(p: np.ndarray, block_n: int = BLOCK_N) -> np.ndarray:
    """Mean of each complete block of ``block_n`` samples (a last partial block is dropped)."""
    m = len(p) // block_n
    return p[: m * block_n].reshape(m, block_n).mean(axis=1, dtype=np.float64)


def _median_rows(a: np.ndarray) -> np.ndarray:
    """Median of each row (same value as ``np.median(a, axis=1)``: the mean of the two middle values for an even row
    length), by one ``np.partition`` without np.median's per-call overhead (the floor takes 600 per minute)."""
    m = a.shape[1]
    h = m // 2
    if m % 2:
        return np.partition(a, h, axis=1)[:, h]
    q = np.partition(a, h, axis=1)            # one kth; the (h-1)th value is the largest of the part below it
    return 0.5 * (q[:, :h].max(axis=1) + q[:, h])


def sliding_floor(bm: np.ndarray, block_s: float, win_s: float = FLOOR_WIN_S, hop_s: float = FLOOR_HOP_S,
                  trust_s: float = FLOOR_TRUST_S, exclude: np.ndarray | None = None,
                  backfill: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """Floor per refresh step from block means ``bm``.

    Step r covers time [r*hop, (r+1)*hop) and uses the median of the blocks that ENDED before r*hop and lie in the
    previous ``win_s`` (excluded blocks left out). A step is trusted when it had at least ``trust_s`` worth of usable
    blocks. Returns (floor per step, trusted mask). With ``backfill`` the untrusted steps before the first trusted one
    take that step's floor (offline only: it looks 0.5 s ahead); untrusted steps without backfill are NaN. An
    untrusted step AFTER a trusted one (exclusion left too few blocks) holds the last trusted floor in both modes.

    ``exclude`` is one bool per block. A shorter mask is padded with False (the missing tail counts as usable, e.g. a
    gate that has not judged the last partial block yet); a longer one is cut to the block count."""
    nb = len(bm)
    per_hop = max(1, int(round(hop_s / block_s)))
    win = max(1, int(round(win_s / block_s)))
    need = max(1, int(round(trust_s / block_s)))
    nstep = int(np.ceil(nb / per_hop)) if nb else 0
    v = bm.astype(np.float64).copy()
    if exclude is not None:
        ex = np.zeros(nb, bool)
        e = np.asarray(exclude, bool)[:nb]
        ex[:len(e)] = e
        v[ex] = np.nan
    floor = np.full(nstep, np.nan)
    ok = np.zeros(nstep, bool)
    hi_all = np.arange(nstep) * per_hop
    full = (hi_all >= win) if exclude is None else np.zeros(nstep, bool)
    if full.any():                                    # complete windows, nothing excluded: all medians at once
        rows = np.lib.stride_tricks.sliding_window_view(v, win)
        idx = np.flatnonzero(full)
        for c in range(0, len(idx), 256):             # 256 windows of 4 s = 8 MB at a time
            r = idx[c:c + 256]
            floor[r] = np.maximum(_median_rows(rows[hi_all[r] - win]), FLOOR_MIN)
        ok[idx] = True
    seen = False                                      # a trusted step has occurred
    for r in range(nstep):
        if full[r]:
            seen = True
            continue
        hi = r * per_hop
        w = v[max(0, hi - win):hi]
        w = w[np.isfinite(w)] if exclude is not None else w
        if w.size >= need:
            floor[r] = max(float(_median_rows(w[None, :])[0]), FLOOR_MIN)
            ok[r] = True
            seen = True
        elif r and np.isfinite(floor[r - 1]) and seen:
            floor[r] = floor[r - 1]                       # hold the last trusted floor
    if backfill and ok.any():
        first = int(np.argmax(ok))
        floor[:first] = floor[first]
    elif backfill and nstep:              # shorter than trust_s in all: one median over everything usable
        w = v[np.isfinite(v)]
        if w.size:
            floor[:] = max(float(np.median(w)), FLOOR_MIN)
    return floor, ok


def floor_at_steps(bm: np.ndarray, block_s: float, steps: np.ndarray, win_s: float = FLOOR_WIN_S,
                   hop_s: float = FLOOR_HOP_S, trust_s: float = FLOOR_TRUST_S) -> np.ndarray:
    """``sliding_floor(bm, block_s, backfill=True)[0][steps]`` (no exclusion) without computing the other steps: the
    16-30 kHz floor is only read at event starts, and the full version cost as much as the detection floor."""
    nb = len(bm)
    per_hop = max(1, int(round(hop_s / block_s)))
    win = max(1, int(round(win_s / block_s)))
    need = max(1, int(round(trust_s / block_s)))
    nstep = int(np.ceil(nb / per_hop)) if nb else 0
    steps = np.minimum(np.asarray(steps, np.int64), max(nstep - 1, 0))
    out = np.full(len(steps), np.nan)
    if nstep == 0 or len(steps) == 0:
        return out
    first = -(-need // per_hop)                       # first step with >= need blocks behind it
    if first >= nstep:                                # never trusted: one median over everything (backfill rule)
        out[:] = max(float(np.median(bm)), FLOOR_MIN)
        return out
    cache = {}
    for i, r in enumerate(steps):
        r = max(int(r), first)
        if r not in cache:
            hi = r * per_hop
            cache[r] = max(float(_median_rows(np.asarray(bm[max(0, hi - win):hi], np.float64)[None, :])[0]),
                           FLOOR_MIN)
        out[i] = cache[r]
    return out


def floor_per_sample(floor_steps: np.ndarray, n: int, hop_n: int) -> np.ndarray:
    """Expand per-step floors to one value per sample (step r covers samples [r*hop_n, (r+1)*hop_n)). With no steps
    at all (input shorter than one block) every sample gets NaN: no floor, no detection."""
    if n == 0:
        return np.zeros(0)
    if len(floor_steps) == 0:
        return np.full(n, np.nan)
    out = np.repeat(np.asarray(floor_steps, np.float64)[: -(-n // hop_n)], hop_n)[:n]
    if len(out) < n:                                  # more samples than steps cover: the last step holds
        out = np.concatenate([out, np.full(n - len(out), floor_steps[-1], np.float64)])
    return out


# ----------------------------------------------------------------------------- impulse state machine

def find_impulses(p: np.ndarray, floor_s: np.ndarray, fs: float = FS, start_db: float = START_DB,
                  end_db: float = END_DB, hold_s: float = HOLD_S, max_width_s: float = MAX_WIDTH_S,
                  merge_s: float = MERGE_S, max_scan_s: float = MAX_SCAN_S,
                  stats: dict | None = None) -> tuple[np.ndarray, np.ndarray, int]:
    """Start/end sample pairs of impulses in power ``p`` against the per-sample floor ``floor_s``.

    Start: p > floor * 10^(start_db/10). End: first sample of a run of round(hold_s*fs) samples all below
    floor * 10^(end_db/10), with the floor of EACH sample (it tracks the floor; the first version froze the floor of
    the start sample, and after a sustained +15 dB level step the impulse never ended: 0 of 19 later pulses found,
    verifier step.py). An impulse still open ``max_scan_s`` (4 s, the floor window, so the floor has had time to
    follow) after its start is closed there; it is a level step, not an impulse, and is dropped. An impulse whose
    start is less than ``merge_s`` after the previous impulse's end joins its event. An event (single or merged)
    wider than ``max_width_s`` is dropped and counted once; impulses that keep extending a dropped event are dropped
    with it. No dead time.

    Returns (starts, ends, n_too_long); ends are exclusive. If ``stats`` is a dict it receives ``too_long_s`` (total
    seconds inside dropped events: time the detector was blind) and ``too_long_max_s`` (the longest one)."""
    n = len(p)
    ok = np.isfinite(floor_s)
    k_start = 10 ** (start_db / 10)
    above = np.zeros(n, bool)
    if ok.all():
        np.greater(p, floor_s * k_start, out=above)
    else:
        above[ok] = p[ok] > floor_s[ok] * k_start
    st_idx = np.flatnonzero(above)
    del above
    if stats is not None:
        stats.update(too_long_s=0.0, too_long_max_s=0.0)
    if st_idx.size == 0:
        return np.zeros(0, np.int64), np.zeros(0, np.int64), 0
    hold = max(1, int(round(hold_s * fs)))
    max_w = max_width_s * fs
    merge = merge_s * fs
    max_scan = max(hold + 1, int(round(max_scan_s * fs)))
    end_k = 10 ** (end_db / 10)
    starts, ends, too_long = [], [], 0
    dropped = []                                      # [start, end] of each dropped event
    dropped_end = None                                # end of the last dropped (too long) event
    i = 0
    while i < st_idx.size:
        s = int(st_idx[i])
        e = s + 1
        stop = min(n, s + max_scan)
        while True:                                   # scan forward in pieces for the hold run
            seg = p[e:min(e + 4096, stop)] < floor_s[e:min(e + 4096, stop)] * end_k
            if seg.size < hold:
                e = stop                              # end of data or the scan cap: close here
                break
            c = np.concatenate([[0], np.cumsum(seg)])
            hit = np.flatnonzero(c[hold:] - c[:-hold] == hold)
            if hit.size:
                e = e + int(hit[0])
                break
            if e + seg.size >= stop:
                e = stop
                break
            e += seg.size - hold + 1
        if dropped_end is not None and s - dropped_end < merge:
            dropped_end = e                           # continues a dropped train: dropped with it
            dropped[-1][1] = e
        elif starts and s - ends[-1] < merge:
            if e - starts[-1] > max_w:                # the merged event outgrew the cap: drop the whole chain
                dropped.append([starts.pop(), max(e, ends.pop())])
                dropped_end = dropped[-1][1]
                too_long += 1
            else:
                ends[-1] = e
        elif e - s > max_w:
            too_long += 1
            dropped_end = e
            dropped.append([s, e])
        else:
            starts.append(s)
            ends.append(e)
        i = int(np.searchsorted(st_idx, e))
    if stats is not None and dropped:
        w = np.array([b - a for a, b in dropped], float) / fs
        stats.update(too_long_s=float(w.sum()), too_long_max_s=float(w.max()))
    return np.array(starts, np.int64), np.array(ends, np.int64), too_long


def peak_index(p: np.ndarray, a: int, b: int) -> int:
    """Index of the power maximum in [a, b). On a flat top (a short spike after the 16-sample boxcar is 16 equal
    samples) it returns the middle of the contiguous run of samples within PEAK_TOP_DB of the maximum around the
    argmax. A first-sample argmax ran SMOOTH_N/2 samples (112 us) early on a one-sample spike, and with noise on the
    plateau an exact-tie rule landed anywhere on it (+3 samples in the unit test). 0.2 dB is far below the
    10 dB start, so on a rounded sferic peak the run is a few samples and the middle stays on the maximum."""
    seg = p[a:b]
    k = int(np.argmax(seg))
    top = seg >= seg[k] * 10 ** (-PEAK_TOP_DB / 10)
    i, j = k, k
    while i > 0 and top[i - 1]:
        i -= 1
    while j + 1 < len(seg) and top[j + 1]:
        j += 1
    return a + (i + j) // 2


def half_peak_onset(p: np.ndarray, k_peak: int, lo: int) -> float:
    """Fractional sample index where power last rose through half the peak before ``k_peak`` (search not before
    ``lo``), by linear interpolation. Idea from OnAir's pulse timer (half-peak onset). Unvalidated timing field."""
    half = 0.5 * float(p[k_peak])
    seg = p[lo:k_peak + 1]
    below = np.flatnonzero(seg < half)
    if below.size == 0:
        return float(lo)
    j = lo + int(below[-1])
    a, b = float(p[j]), float(p[j + 1])
    return j + (half - a) / (b - a) if b > a else float(j)


def events_from_power(p: np.ndarray, pc: np.ndarray | None, fs: float = FS, t0: float = 0.0,
                      exclude_blocks: np.ndarray | None = None, backfill: bool = True) -> tuple[dict, dict]:
    """The whole detector on precomputed smoothed powers: ``p`` (2.5-9 kHz) and optional ``pc`` (16-30 kHz).

    Returns (events, info): events is a dict of equal-length arrays (FIELDS; times in s from x[0] plus ``t0``),
    info holds the floor range, the too-long count, the number of floor steps, how many were trusted, the seconds
    run on a held floor (``floor_held_s``: exclusion left < 0.5 s of usable blocks) and the seconds with no floor at
    all (``floor_none_s``: no detection there).

    Level-step health (the detector is blind inside a dropped event): ``too_long_s`` / ``too_long_max_s`` (seconds
    inside dropped too-long events, total and longest), ``level_steps`` (times the floor rose by >= 6 dB within one
    4 s floor window, counted once per rise) and ``floor_rise_db_max`` (the largest such rise). A level step shows as
    level_steps >= 1 with a too_long_max_s of about 1-2 s (the time the floor median takes to cross the new level);
    a keyed train shows as too_long with level_steps 0."""
    block_s = BLOCK_N / fs
    hop_n = int(round(FLOOR_HOP_S / block_s)) * BLOCK_N
    fl, okf = sliding_floor(block_means(p), block_s, exclude=exclude_blocks, backfill=backfill)
    fsamp = floor_per_sample(fl, len(p), hop_n)
    tl = {}
    s, e, too_long = find_impulses(p, fsamp, fs, stats=tl)
    ev = {k: np.zeros(len(s)) for k in FIELDS}
    ev["conf"] = np.zeros(len(s), bool)
    if pc is not None and len(s):
        cf = floor_at_steps(block_means(pc), block_s, np.asarray(s) // hop_n)
    pad = int(round(CONF_PAD_S * fs))
    for i, (a, b) in enumerate(zip(s, e)):
        k = peak_index(p, a, b)
        flo = fsamp[a]
        top = float(p[a:b].max())                     # the true peak (k is the middle of a flat top, up to 0.2 dB lower)
        ev["t_start"][i] = a / fs
        ev["t_end"][i] = b / fs
        ev["t_peak"][i] = k / fs
        ev["peak_db"][i] = 10 * np.log10(max(top, 1e-30) / flo)
        ev["width_s"][i] = (b - a) / fs
        ev["rise_s"][i] = (k - a) / fs
        ev["t_half"][i] = half_peak_onset(p, k, max(0, a - int(round(1e-3 * fs)))) / fs
        ev["floor_dbfs"][i] = 10 * np.log10(flo)
        if pc is not None:
            lo, hi = max(0, a - pad), min(len(pc), b + pad)
            c = float(pc[lo:hi].max()) / max(float(cf[i]), 1e-30)
            ev["conf_db"][i] = 10 * np.log10(max(c, 1e-30))
            ev["conf"][i] = ev["conf_db"][i] >= CONF_DB
        else:
            ev["conf_db"][i] = np.nan
    for f in ("t_start", "t_end", "t_peak", "t_half"):
        ev[f] += t0
    finite = fl[np.isfinite(fl)]
    first = int(np.argmax(okf)) if okf.any() else len(okf)
    held = (~okf) & np.isfinite(fl)
    held[:first] = False                              # backfilled start-up is not "held"
    nan_n = int(np.isnan(fsamp).sum())
    del fsamp
    win = max(1, int(round(FLOOR_WIN_S / FLOOR_HOP_S)))
    rise = np.zeros(0)
    if len(fl) > win:
        with np.errstate(invalid="ignore", divide="ignore"):
            rise = 10 * np.log10(fl[win:] / fl[:-win])
        rise = rise[np.isfinite(rise)]
    jump = rise >= LEVEL_STEP_DB
    info = {"too_long": too_long, "too_long_s": tl.get("too_long_s", 0.0),
            "too_long_max_s": tl.get("too_long_max_s", 0.0),
            "level_steps": int(np.count_nonzero(jump[1:] & ~jump[:-1]) + (jump[:1].sum() if jump.size else 0)),
            "floor_rise_db_max": float(rise.max()) if rise.size else 0.0,
            "floor_steps": int(len(fl)), "floor_trusted_steps": int(okf.sum()),
            "floor_held_s": float(held.sum() * hop_n / fs), "floor_none_s": float(nan_n / fs),
            "floor_dbfs_min": float(10 * np.log10(finite.min())) if finite.size else None,
            "floor_dbfs_max": float(10 * np.log10(finite.max())) if finite.size else None}
    return ev, info


# ----------------------------------------------------------------------------- full chain on IQ

class _Boxcar:
    """Streaming centred n-sample boxcar of raw power blocks, the same numbers as ``smooth_power`` (scipy
    ``uniform_filter1d``: window [i - n//2, i + n - n//2 - 1], edge samples repeated at the true ends), written into
    ``out``. A float64 running sum over each block plus n samples of context, applied while the block is still in
    cache: on a 600 s capture the two-pass uniform_filter1d version cost 0.09 s per minute of signal."""

    def __init__(self, out: np.ndarray, n: int = SMOOTH_N):
        self.out, self.n = out, n
        self.lm, self.rm = n // 2, n - n // 2 - 1
        self.tail = np.zeros(0)                       # raw values [self.t0, self.done + rm) not yet fully used
        self.t0 = 0
        self.done = 0                                 # out[:done] is final

    def push(self, lo: int, raw: np.ndarray, last: bool = False) -> None:
        """Raw power for samples [lo, lo + len(raw)), contiguous with the previous push; ``last`` at the end."""
        raw = np.asarray(raw, np.float64)
        if lo == 0:
            self.tail = np.full(self.lm, raw[0]) if len(raw) else np.zeros(0)   # 'nearest' at the start
            self.t0 = -self.lm
        buf = np.concatenate([self.tail, raw]) if len(self.tail) else raw
        end = lo + len(raw)                           # buf covers [t0, end)
        if last and len(buf):
            buf = np.concatenate([buf, np.full(self.rm, buf[-1])])
            stop = end
        else:
            stop = end - self.rm                      # outputs with full right context
        if stop > self.done:
            c = np.empty(len(buf) + 1)
            c[0] = 0.0
            np.cumsum(buf, out=c[1:])
            i0, i1 = self.done - self.t0, stop - self.t0  # buf indices of the outputs
            d = c[i0 + self.rm + 1:i1 + self.rm + 1] - c[i0 - self.lm:i1 - self.lm]
            d *= 1.0 / self.n
            self.out[self.done:stop] = d
            self.done = stop
        keep = self.done - self.lm                    # raw values still needed as left context
        self.tail = buf[keep - self.t0:end - self.t0].copy()
        self.t0 = keep


def band_powers(x: np.ndarray, fs: float = FS, n0: int = 0, chunk_s: float = CHUNK_S,
                guard_s: float = GUARD_S) -> tuple[np.ndarray, np.ndarray]:
    """Front end, then smoothed power of the detection band and of the 16-30 kHz confirmation band, both float32
    for the whole of x.

    The front end (``frontend._BandEngine``) works in 1 s frames on the absolute sample grid, so the whole capture
    runs as one stream: no chunks, no guards, and a slice gives the same powers as the same samples inside a longer
    run (``frontend_offline`` doc). Working set: x, the two float32 power arrays (150 MB each for 600 s) and one
    frame. ``chunk_s`` and ``guard_s`` are kept for callers of the old chunked version and are ignored; the result
    no longer depends on them (unit test).

    CPU, measured with the harness on the corpus: see the module doc (``Cost``)."""
    del chunk_s, guard_s
    x = np.asarray(x, np.complex64)
    n = len(x)
    p = np.zeros(n, np.float32)
    pc = np.zeros(n, np.float32)
    if n == 0:
        return p, pc
    eng = fe._BandEngine(fs, conf=True, raw=False)
    bp, bc = _Boxcar(p), _Boxcar(pc)

    def emit(lo, d, c, _r):
        last = lo + len(d) >= n
        bp.push(lo, np.square(d, dtype=np.float64), last)
        bc.push(lo, c.real.astype(np.float64) ** 2 + c.imag.astype(np.float64) ** 2, last)

    fe._run_frames(eng, fe._ArraySource(x, eng.num, eng.den), int(n0), int(n0) // eng.H, emit)
    return p, pc


def detect_iq(x: np.ndarray, fs: float = FS, n0: int = 0, exclude_blocks: np.ndarray | None = None,
              july: bool = False) -> tuple[dict, dict]:
    """Detect impulses in baseband IQ ``x`` (complex64, full scale 1.0, centred 15 kHz). Times are seconds from x[0].

    Returns (events, info). info carries cpu_frontend_s and cpu_detect_s (process CPU of each part) so the cost of
    the front end and of the detector proper can be read apart. ``july=True`` adds the july_compat column (costs
    one more 2.5-9 kHz filter pass)."""
    c0 = time.process_time()
    p, pc = band_powers(x, fs, n0)
    c1 = time.process_time()
    ev, info = events_from_power(p, pc, fs, exclude_blocks=exclude_blocks)
    del p, pc
    c2 = time.process_time()
    info["cpu_frontend_s"] = c1 - c0
    info["cpu_detect_s"] = c2 - c1
    info["conf_share_16_30k"] = float(ev["conf"].mean()) if len(ev["conf"]) else None
    if july:
        ev["july_compat"] = july_compat(ev["t_start"], july_times(x, fs)[0])
    return ev, info


# ----------------------------------------------------------------------------- July rule, kept as a column

_JULY_CHUNK_S = 30.0


def july_times(x: np.ndarray, fs: float = FS, fc: float = fe.FC) -> tuple[np.ndarray, np.ndarray]:
    """The July sferic_rate rule, for comparison only: per 30 s chunk, shift to real VLF, 4th-order 2.5-9 kHz
    zero-phase band-pass, |y| over median + 8 MAD-sigma, rising edges, 8 ms dead time; a last chunk under 1 s is
    skipped. Returns (times s, peak dB over the median |y| within the 8 ms after the edge).

    Kept because the archive's old numbers were made with it. Its dead time is the known flaw: under 120 Hz buzz each
    detection blinds the next 8 ms of an 8.33 ms period (h2h steady-hum corrected recall 0.08-0.30)."""
    sos = butter(4, list(fe.DET_BAND), btype="bandpass", fs=fs, output="sos")
    nch, g = int(_JULY_CHUNK_S * fs), int(0.008 * fs)
    times, peaks = [], []
    for c0 in range(0, len(x), nch):
        xc = x[c0:c0 + nch]
        if len(xc) < fs:
            break
        n = np.arange(len(xc))
        y = np.abs(sosfiltfilt(sos, (xc.astype(np.complex128) * np.exp(2j * np.pi * (fc / fs) * n)).real))
        med = np.median(y)
        sig = 1.4826 * np.median(np.abs(y - med)) + 1e-15
        above = y > med + 8 * sig
        last = -g - 1
        for e in np.flatnonzero(~above[:-1] & above[1:]):
            if e - last > g:
                last = e
                times.append((c0 + e) / fs)
                peaks.append(20 * np.log10(y[e:e + g].max() / (med + 1e-30)))
    return np.array(times), np.array(peaks)


def july_compat(t_events: np.ndarray, t_july: np.ndarray, tol: float = 0.002) -> np.ndarray:
    """Per event: did the July rule fire within +-tol (2 ms, the harness hit window) of it?"""
    t_events = np.asarray(t_events, float)
    tj = np.sort(np.asarray(t_july, float))
    if tj.size == 0:
        return np.zeros(t_events.size, bool)
    i = np.clip(np.searchsorted(tj, t_events), 1, tj.size) - 1
    d = np.minimum(np.abs(tj[i] - t_events), np.abs(tj[np.minimum(i + 1, tj.size - 1)] - t_events))
    return d <= tol


# ----------------------------------------------------------------------------- harness plug-in

@sw.register("v2_detect", note="v2 median-floor detector on the step-3 front end, mains gate off, no dead time")
def v2_detect(x, fs=FS):
    """Bench entry: event start times, peak dB over the floor, and info (CPU split, too_long, 16-30 kHz share)."""
    ev, info = detect_iq(x, fs)
    return ev["t_start"], ev["peak_db"], info
