"""Per-10 s mains-buzz flag over a whole capture.

Why this exists
---------------
The old dlayer ``hum_check`` looked at the first 30 s of a capture only. Two
captures it called clean (2026-11-05 04:30Z and 2026-11-06 02:30Z) have 120 Hz
buzz switching on part-way through (about 450 s and 90 s in), so they are
really about 25 % and 85 % buzz. The buzz is full-wave rectified mains:
pulses about 380 us wide, 15.7-17.8 dB over the floor, on a 59.98-60.004 Hz
grid with 37-44 us rms jitter. It is on in 85-90 of the 150 banked captures.

Corpus run of this flag (bench/humflag_all.py, 150 + 1 captures): weak buzz
(N 20-40 edges per block, R 0.7-1.0) is already there in 11-05 04:30Z from
180 s, before the strong train at 450 s; and the 11-05 19:30-21:00Z daytime
captures carry sparse buzz (2-8 locked edges/s) the census called clean. The
170 blocks only this flag finds have their best frequency at 119.93-120.06 Hz
(median 120.005), not spread over the 119.85-120.15 Hz scan as chance would be.

What it does
------------
For every 10 s block of a capture:

1. Turn the complex baseband (centred 15 kHz) into the real VLF signal and
   band-pass 2.5-9 kHz (4th-order Butterworth, zero phase). That band sits
   well clear of the 0 Hz receiver tone (about -22.6 dBFS, it lands at 15 kHz
   real), which otherwise dominates raw power.
2. Envelope |x|; threshold median + 8 x 1.4826 x MAD of that block (the July
   rule); rising crossings debounced at 1 ms, so one 380 us buzz pulse gives one
   edge but a 120 Hz train (8.33 ms) is never hidden, unlike the 8 ms dead time.
3. Rayleigh phase-lock R of the edge times, maximised over 119.85-120.15 Hz in
   0.0025 Hz steps (0.3 Hz x 10 s = 3 resolution cells, oversampled 40x).
   The flag level comes from the Rice formula for the maximum of a scanned
   Rayleigh statistic (``rayleigh_levels``): P(max N R^2 > u) ~=
   exp(-u) (1 + W T sqrt(pi u / 3)). The first version used exp(-u) x M with
   M = 1 + W T = 4 trials, and two verifiers measured 2-3x its stated false
   alarm rate; the Rice level measured 0.0100 / 0.0105 at N = 50 / 300 for a
   nominal 0.01 (4000 random blocks each), conservative at N = 20 (0.006).
4. Locked count: with the best frequency and mean phase from step 3, count the
   edges within +-0.4 ms of the 120 Hz grid (about 10x the 40 us jitter;
   chance share c = 0.8 / 8.333 = 0.096) and remove the chance part:
   n_locked = (in_window - c N) / (1 - c). This estimates how many edges are
   mains buzz, whatever else is in the block.
5. A block is buzz when N >= 20 (so the Rayleigh tail holds), R is over the
   1e-4 level, and n_locked >= 20 (2 buzz edges/s). At ~9000 blocks in the
   corpus, 1e-4 means under one chance flag in the whole corpus.

Why not the plan's "interval share > 0.3 and N >= 100" rule
-----------------------------------------------------------
That rule is still computed (``hum_plan``, and the share is reported), but it
is not the flag. Two measured problems, both from the step-2 verifiers:

* Interval share is a veto under dilution. Consecutive intervals break as soon
  as other edges fall between buzz pulses: full buzz plus 120 Poisson edges/s
  gives share 0.23 (not flagged, 0/50) while R is 0.47 against a level of
  about 0.07. Sparse buzz (24/s) is lost at only 30 other edges/s. A storm
  is exactly when the gate must work.
* N >= 100 leaves the named sparse-buzz capture 11-05 20Z flagged in 3 of 60
  blocks although R is 0.84-1.00 in nearly every block (N 15-98).

The interval share (``interval_share``, k = 1..8 multiples of 8.333 ms) is kept
as a reported number because the census used it and its Poisson chance level
is known (``share_chance``, at most about 0.10 near 24 events/s).

The 100/120 Hz "impulses that sit on the mains period are not lightning" idea
is from OnAir's sferics receiver; the per-block frequency search, the chance
levels, the locked count and the whole-file block flag are ours.
"""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
from scipy.signal import butter, sosfiltfilt
from scipy.stats import binom

FS = 62500.0            # sample rate of the banked VLF captures, S/s
FC = 15000.0            # tuned centre, Hz
BLOCK_S = 10.0          # flag resolution, s
PAD_S = 0.02            # filter run-in read on each side of a block, s
BAND = (2500.0, 9000.0)
K_MAD = 8.0             # July rule threshold, x 1.4826 x MAD over median
DEBOUNCE_S = 1e-3       # shorter than a half mains cycle (8.33 ms), longer than a buzz pulse (~0.38 ms)
MAINS_PERIOD = 1.0 / 120.0
SHARE_TOL = 0.02        # +- 2 % of k x 8.333 ms
SHARE_KMAX = 8          # count multiples up to 66.7 ms (sparse buzz measured 33-58 ms gaps)
SHARE_FLAG = 0.3        # plan rule only: census found an empty gap at 0.1-0.7
N_MIN = 100             # plan rule only: at least 10 edges/s in a 10 s block
N_STAT = 20             # fewest edges for which the Rayleigh tail is trusted (MC: conservative at 20)
LOCK_WIN_S = 4e-4       # +- window around the fitted 120 Hz grid for the locked count (~10x 40 us jitter)
N_LOCK_MIN = 20         # buzz edges per block to flag (2/s; a clean day has ~1 sferic/s in this band)
N_LOCK_EV_MIN = 10      # detector starts on the grid per block for the event-source flag (1/s)
EV_WIN_BLOCKS, EV_WIN_HOP = 6, 3   # event source, long window: 60 s of blocks every 30 s
P_FA_EV_WIN = 1e-5      # its level per window (about 20 windows per 600 s capture)
N_LOCK_EV_WIN = 20      # detector starts on the grid per 60 s window
LOCK_P2 = 1e-3          # a second phase cluster counts in n_locked when over chance at this level (position trials in)
F_LO, F_HI, F_STEP = 119.85, 120.15, 0.0025
P_FA = 1e-4             # false-alarm probability per block for the R test (Rice level)

_SOS = butter(4, BAND, btype="band", fs=FS, output="sos")


# ----------------------------------------------------------------- edges

def to_real(x: np.ndarray, n0: int = 0, fs: float = FS, fc: float = FC) -> np.ndarray:
    """Shift complex baseband up by fc and keep the real part (the VLF waveform).

    n0 is the absolute index of x[0]; it keeps the mixer phase continuous across
    blocks (it does not change the envelope, but makes block joins seamless).
    """
    n = np.arange(n0, n0 + len(x), dtype=np.float64)
    return (x * np.exp(2j * np.pi * (fc / fs) * n)).real.astype(np.float64)


def debounce(idx: np.ndarray, gap: int) -> np.ndarray:
    """Keep an edge only if it is more than `gap` samples after the last kept edge."""
    kept = []
    last = None
    for v in idx.tolist():
        if last is None or v - last > gap:
            kept.append(v)
            last = v
    return np.asarray(kept, dtype=np.int64)


def block_edges(x: np.ndarray, core0: int, core_len: int, n0: int = 0, fs: float = FS,
                k_mad: float = K_MAD, debounce_s: float = DEBOUNCE_S) -> np.ndarray:
    """Debounced rising threshold crossings in the 2.5-9 kHz band, as sample indices.

    x holds the block plus its filter padding; the core block is x[core0:core0+core_len]
    and the threshold (median + k_mad x 1.4826 x MAD) is set from the core only.
    Returned indices are relative to the start of the core block.
    """
    env = np.abs(sosfiltfilt(_SOS, to_real(x, n0, fs)))
    core = env[core0:core0 + core_len]
    med = float(np.median(core))
    mad = float(np.median(np.abs(core - med))) * 1.4826 + 1e-12
    above = env > med + k_mad * mad
    rise = np.flatnonzero(~above[:-1] & above[1:]) + 1
    rise = rise[(rise >= core0) & (rise < core0 + core_len)] - core0
    return debounce(rise, int(round(debounce_s * fs)))


# ----------------------------------------------------------------- statistics

def interval_share(t: np.ndarray, period: float = MAINS_PERIOD, tol: float = SHARE_TOL,
                   kmax: int = SHARE_KMAX) -> float:
    """Share of consecutive intervals within tol x k x period of k x period, k = 1..kmax."""
    if len(t) < 2:
        return 0.0
    d = np.diff(np.asarray(t, dtype=np.float64))
    k = np.rint(d / period)
    ok = (k >= 1) & (k <= kmax) & (np.abs(d - k * period) < tol * k * period)
    return float(np.mean(ok))


def share_chance(rate: float, period: float = MAINS_PERIOD, tol: float = SHARE_TOL,
                 kmax: int = SHARE_KMAX) -> float:
    """Expected interval share for Poisson (random) events at `rate` per second.

    Exponential intervals: P(within tol*k*T of k*T) ~= rate x 2 tol k T x exp(-rate k T).
    Its maximum over all rates is about 0.10 (near 24/s), well under SHARE_FLAG.
    """
    if rate <= 0:
        return 0.0
    k = np.arange(1, kmax + 1)
    lo = k * period * (1 - tol)
    hi = k * period * (1 + tol)
    return float(np.sum(np.exp(-rate * lo) - np.exp(-rate * hi)))


def rayleigh_scan(t: np.ndarray, f_lo: float = F_LO, f_hi: float = F_HI,
                  f_step: float = F_STEP) -> tuple[float, float]:
    """Max over frequency of the Rayleigh phase-lock R = |mean exp(2 pi i f t)|.

    Returns (R_max, f_at_max). R is 1 for edges exactly on a grid and about
    1/sqrt(N) for random times. Time origin does not matter (|R| is shift-invariant);
    ``locked_count`` recomputes the phase at f_at_max with the same origin (t[0]).
    """
    t = np.asarray(t, dtype=np.float64)
    if len(t) == 0:
        return 0.0, float("nan")
    t = t - t[0]
    freqs = np.arange(f_lo, f_hi + f_step / 2, f_step)
    best_r, best_f = -1.0, float("nan")
    for i in range(0, len(freqs), 32):           # chunk to bound memory (32 x N)
        f = freqs[i:i + 32]
        r = np.abs(np.exp(2j * np.pi * np.outer(f, t)).mean(axis=1))
        j = int(np.argmax(r))
        if r[j] > best_r:
            best_r, best_f = float(r[j]), float(f[j])
    return best_r, best_f


def rice_tail(u: float, n_cells: float) -> float:
    """P(max over the scan of N R^2 > u) for random times, Rice formula.

    n_cells = (f_hi - f_lo) x T is the scan width in resolution cells. For times
    spread uniformly over T the scanned complex sum has an envelope up-crossing
    rate of T sqrt(pi u / 3) exp(-u) per Hz, so the tail is the one-frequency
    Rayleigh tail plus the expected number of up-crossings across the band.
    """
    return math.exp(-u) * (1.0 + n_cells * math.sqrt(math.pi * max(u, 0.0) / 3.0))


def rayleigh_levels(n: int, block_s: float = BLOCK_S, f_lo: float = F_LO, f_hi: float = F_HI,
                    p_fa: float = P_FA) -> tuple[float, float]:
    """(chance, critical) levels of the searched R for n random edges in one block.

    chance = 1/sqrt(n), the rms R of random phases. critical solves
    rice_tail(n r^2) = p_fa by bisection. Monte Carlo (4000 uniform-random blocks
    per point, nominal 0.01): 0.0058 at n = 20, 0.0100 at n = 50, 0.0105 at
    n = 300. The earlier exp(-u) = p_fa / (1 + W T) level gave 0.021-0.023.
    """
    if n <= 0:
        return float("nan"), float("nan")
    cells = (f_hi - f_lo) * block_s
    lo, hi = 0.0, 100.0
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        if rice_tail(mid, cells) > p_fa:
            lo = mid
        else:
            hi = mid
    return 1.0 / math.sqrt(n), math.sqrt(hi / n)


def _busiest(ph: np.ndarray, P: float, win_s: float, stride: float) -> tuple[float, int]:
    """(centre, count) of the busiest +-win_s window slid round the fold circle of phases ph (s in [0, P))."""
    ph = np.sort(ph % P)
    if ph.size == 0:
        return 0.0, 0
    cent = np.arange(int(np.ceil(P / stride))) * stride
    ext = np.concatenate([ph - P, ph, ph + P])
    cnt = np.searchsorted(ext, cent + win_s, side="right") - np.searchsorted(ext, cent - win_s, side="left")
    k = int(np.argmax(cnt))
    return float(cent[k]), int(cnt[k])


def locked_count(t: np.ndarray, f: float, win_s: float = LOCK_WIN_S) -> tuple[float, int]:
    """Estimated number of edges that sit on the mains grid at frequency f, over up to two phase clusters.

    Folds the edge times on period 1/f and centres a +-win_s window on the busiest phase (the fold's mode), counts
    the edges in it and removes the share c = 2 win_s f that random edges would put there: (in - c N) / (1 - c).
    Then a second cluster: the busiest window among the edges left, counted the same way (chance share
    2 win_s / (P - 2 win_s) of the edges left) when it is over chance at 1e-3 after its position trials
    (``LOCK_P2``).

    Why the mode and a second cluster: the first version centred on the circular-mean phase (the Rayleigh
    direction). Buzz with two pulse clusters per cycle (11-06 08:30 / 09:30 / 10:00Z) has its mean phase BETWEEN
    the clusters, so n_locked came out near 0 and 101 blocks with a significant R were not flagged; centring on the
    mode recovers about 40 of them (step-2 verifier), counting the second cluster the rest that are really buzz.
    For random edges the mode window is the largest of about 80 positions, so the estimate reads high by about 2.5
    sd of the chance count (about +35 at 2000 edges); the flag still needs the Rayleigh test first.
    Unlike the interval share this does not drop when other edges fall between buzz pulses, so it still sees buzz
    under a busy storm. Returns (n_locked, in_window)."""
    t = np.asarray(t, dtype=np.float64)
    n = len(t)
    if n == 0 or not np.isfinite(f):
        return 0.0, 0
    P = 1.0 / f
    ph = (t - t[0]) % P
    stride = win_s / 4
    c1, in1 = _busiest(ph, P, win_s, stride)
    c = min(0.999, 2 * win_s * f)
    n_lock = (in1 - c * n) / (1 - c)
    d = (ph - c1 + P / 2) % P - P / 2
    rest = ph[np.abs(d) > win_s]
    inside = in1
    if rest.size >= 2:
        c2, in2 = _busiest(rest, P, win_s, stride)
        q = min(0.999, 2 * win_s / (P - 2 * win_s))
        if binom.sf(in2 - 1, rest.size, q) * np.ceil(P / stride) < LOCK_P2:
            n_lock += (in2 - q * rest.size) / (1 - q)
            inside += in2
    return float(n_lock), int(inside)


def block_stats(t: np.ndarray, block_s: float = BLOCK_S) -> dict:
    """All hum numbers for one block from its edge times (s).

    hum is the flag (significant phase lock and at least N_LOCK_MIN buzz edges);
    hum_plan is the plan's literal rule (N >= 100, share > 0.3, R over the same
    level), kept so the two can be compared on the corpus.
    """
    n = int(len(t))
    share = interval_share(t)
    r, f = rayleigh_scan(t) if n >= 2 else (0.0, float("nan"))
    r_ch, r_crit = rayleigh_levels(n, block_s)
    n_lock, _ = locked_count(t, f) if n >= 2 else (0.0, 0)
    sig = bool(n >= N_STAT and r > r_crit)
    hum = bool(sig and n_lock >= N_LOCK_MIN)
    hum_plan = bool(n >= N_MIN and share > SHARE_FLAG and r > r_crit)
    return {"n_edges": n, "rate_hz": n / block_s, "share": share,
            "share_chance": share_chance(n / block_s), "R": r, "r_chance": r_ch,
            "r_crit": r_crit, "f_mains": f, "n_locked": n_lock,
            "buzz_rate_hz": max(0.0, n_lock) / block_s if sig else 0.0,
            "hum": hum, "hum_edge": hum, "hum_plan": hum_plan}


def event_stats(t: np.ndarray, block_s: float = BLOCK_S) -> dict:
    """The same phase-lock test on the DETECTOR's impulse starts of one block (s): the second edge source.

    Why: the July-rule edges (median + 8 MAD-sigma of the un-excised envelope, about 15 dB over the median power)
    are less sensitive than the v2 detector's 10 dB start on the excised band. On 11-03 18:00-19:30Z and 11-05
    13:00-14:30Z the edge flag saw nothing (hum_frac 0) while the gated v2 output still had a comb share of
    0.40-0.58 with only 10-16 % of its events confirmed above 16 kHz: buzz the detector sees and the flag does not.
    A block is buzz by this source when it has >= N_STAT starts, R over the same Rice level (P_FA over
    119.85-120.15 Hz) and at least N_LOCK_EV_MIN starts on the grid (``locked_count``). Keys end in ``_ev``; the
    frequency is ``f_ev`` and never ``f_mains``, because it was fitted to the very starts the mains gate judges
    (``mainsgate.edge_flagged``)."""
    t = np.sort(np.asarray(t, dtype=np.float64))
    n = int(len(t))
    r, f = rayleigh_scan(t) if n >= 2 else (0.0, float("nan"))
    _, r_crit = rayleigh_levels(n, block_s)
    n_lock, _ = locked_count(t, f) if n >= 2 else (0.0, 0)
    sig = bool(n >= N_STAT and r > r_crit)
    return {"n_ev": n, "R_ev": r, "r_crit_ev": r_crit, "f_ev": f, "n_locked_ev": n_lock,
            "hum_ev": bool(sig and n_lock >= N_LOCK_EV_MIN)}


def add_event_stats(blocks: list[dict], t_ev: np.ndarray, win_blocks: int = EV_WIN_BLOCKS,
                    hop_blocks: int = EV_WIN_HOP) -> list[dict]:
    """IN PLACE: add ``event_stats`` of the detector starts ``t_ev`` (s, same clock as the blocks) to every block,
    run the 60 s window test, and set ``hum`` = edge flag OR event flag (``hum_edge`` keeps the edge flag alone).

    The 60 s window test (``hum_ev_win``): the starts of ``win_blocks`` consecutive blocks (every ``hop_blocks``)
    get the same Rayleigh scan, at level P_FA_EV_WIN against the Rice tail of a 60 s scan (18 resolution cells), and
    need N_LOCK_EV_WIN starts on the grid; every block of a passing window is flagged. Why: on 11-03 18:00-19:30Z the
    source the flag missed is mains locked (interval share 0.74 on a 16.67 ms grid, chance 0.07) but sparse, about
    0.9 detector starts/s, so a 10 s block holds about 9 starts, under the N_STAT = 20 the Rayleigh tail needs. Over
    60 s it holds about 50. The mains phase moves by under 0.5 ms in 60 s for a 2 mHz drift, so one frequency still
    folds it. Union bound: about 20 windows per 600 s capture at 1e-5 each. ``hum_ev`` is the 10 s test OR the
    window test. Returns the same list."""
    t_ev = np.sort(np.asarray(t_ev, dtype=np.float64))
    for b in blocks:
        if "hum_edge" not in b:
            b["hum_edge"] = bool(b["hum"])
        a, z = np.searchsorted(t_ev, [b["t0_s"], b["t0_s"] + b["dur_s"]])
        b.update(event_stats(t_ev[a:z] - b["t0_s"], b["dur_s"]))
        b["hum_ev10"] = b["hum_ev"]
        b["hum_ev_win"] = False
    n = len(blocks)
    starts = list(range(0, n - win_blocks + 1, hop_blocks)) if n >= win_blocks else []
    if starts and starts[-1] != n - win_blocks:
        starts.append(n - win_blocks)
    for i in starts:
        grp = blocks[i:i + win_blocks]
        t0, t1 = grp[0]["t0_s"], grp[-1]["t0_s"] + grp[-1]["dur_s"]
        a, z = np.searchsorted(t_ev, [t0, t1])
        tw = t_ev[a:z]
        if tw.size < N_STAT:
            continue
        r, f = rayleigh_scan(tw)
        crit = rayleigh_levels(tw.size, t1 - t0, p_fa=P_FA_EV_WIN)[1]
        if r > crit and locked_count(tw, f)[0] >= N_LOCK_EV_WIN:
            for b in grp:
                b["hum_ev_win"] = True
    for b in blocks:
        b["hum_ev"] = bool(b["hum_ev10"] or b["hum_ev_win"])
        b["hum"] = bool(b["hum_edge"] or b["hum_ev"])
    return blocks


# ----------------------------------------------------------------- whole captures

def _blocks(n_total: int, fs: float, block_s: float):
    b = int(round(block_s * fs))
    for s0 in range(0, n_total, b):
        L = min(b, n_total - s0)
        if L >= int(2.0 * fs):                   # skip a tail shorter than 2 s
            yield s0, L


def flag_iq(x: np.ndarray, fs: float = FS, block_s: float = BLOCK_S, events: np.ndarray | None = None) -> list[dict]:
    """Block flags for an in-memory complex IQ array (tests, live buffers). ``events``: detector impulse starts (s
    from x[0]) for the second edge source (``add_event_stats``); without them the flag uses its own edges only."""
    pad = int(round(PAD_S * fs))
    out = []
    for s0, L in _blocks(len(x), fs, block_s):
        a = max(0, s0 - pad)
        seg = x[a:min(len(x), s0 + L + pad)]
        e = block_edges(seg, s0 - a, L, n0=a, fs=fs)
        st = block_stats(e / fs, L / fs)
        st.update(block=len(out), t0_s=s0 / fs, dur_s=L / fs)
        out.append(st)
    return out if events is None else add_event_stats(out, events)


def read_iq(path: str | Path, s0: int, n: int) -> np.ndarray:
    """Read n complex samples starting at sample s0 from an interleaved int16 .cs16 file.

    Reads only that slice (np.fromfile with offset/count) as complex64, scaled to
    full scale = 1.0, so a whole 150 MB capture is never held at once.
    """
    raw = np.fromfile(path, dtype=np.int16, count=2 * n, offset=4 * s0)
    return (raw[0::2].astype(np.float32) + 1j * raw[1::2].astype(np.float32)) / np.float32(32768.0)


def flag_file(path: str | Path, fs: float = FS, block_s: float = BLOCK_S,
              events: np.ndarray | None = None) -> list[dict]:
    """Block flags for a whole .cs16 capture, read one padded block at a time (``events`` as in ``flag_iq``)."""
    n_total = Path(path).stat().st_size // 4
    pad = int(round(PAD_S * fs))
    out = []
    for s0, L in _blocks(n_total, fs, block_s):
        a = max(0, s0 - pad)
        seg = read_iq(path, a, min(n_total, s0 + L + pad) - a)
        e = block_edges(seg, s0 - a, L, n0=a, fs=fs)
        st = block_stats(e / fs, L / fs)
        st.update(block=len(out), t0_s=s0 / fs, dur_s=L / fs)
        out.append(st)
    return out if events is None else add_event_stats(out, events)


def summarize(blocks: list[dict], key: str = "hum") -> dict:
    """Per-capture summary: hum_frac, first/last hum block start, and status.

    status is 'clean' (no block flagged), 'hum' (every block flagged) or 'partial'.
    key picks the flag ('hum', or 'hum_plan' for the plan's literal rule).
    """
    flags = np.array([b[key] for b in blocks], dtype=bool)
    idx = np.flatnonzero(flags)
    if len(flags) == 0:
        status = "empty"
    elif flags.all():
        status = "hum"
    elif flags.any():
        status = "partial"
    else:
        status = "clean"
    return {"n_blocks": int(len(flags)), "hum_blocks": int(flags.sum()),
            "hum_frac": float(flags.mean()) if len(flags) else 0.0,
            "first_hum_s": float(blocks[idx[0]]["t0_s"]) if len(idx) else None,
            "last_hum_s": float(blocks[idx[-1]]["t0_s"]) if len(idx) else None,
            "status": status,
            "hum_flag": flags.tolist(),
            "R": [b["R"] for b in blocks],
            "f_mains": [b["f_mains"] for b in blocks]}
