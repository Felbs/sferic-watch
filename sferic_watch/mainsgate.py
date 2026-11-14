"""Mains gate (plan step 5): throw away detector impulses that sit on a 120 Hz mains-buzz train, and nothing else.

Idea from OnAir's sferics receiver: impulses that arrive on the mains period are not lightning. OnAir tests each new
impulse against k = 1..5 multiples of 8.33/10 ms from its last 4 starts with a 4 % tolerance. Measured on bare Poisson
times that test throws away 4.3 / 13.5 / 18.0 % of impulses at 40 / 60 / 80 per s, and it misses sparse buzz and a
second train. This module keeps the idea and replaces the test with a phase fold that has a stated, checked
false-alarm level. Code, structure and numbers are ours (MIT).

The fold test (``hot_p``)
-------------------------
Fold the impulse starts on the mains period P (8.33 ms) and slide a +-0.2 ms window round the circle at a 0.1 ms
stride (84 positions). The busiest position is a train when its binomial tail, with q = 0.4 ms / (P - arc already
peeled), times the number of positions tested (times the frequency trials when f was fitted to the same starts, or
the windows pooled into one decision) is under the level. That product is a union bound, so it can only over-state p.
Why this form: the first version multiplied by 20.8 "independent" windows while scanning 333 overlapping positions,
and two verifiers measured 3-7x its stated level (7e-4 per search for a stated 1e-4); it also kept q = 0.4 ms / P
after a train's arc was removed. Checked on uniform phases and pure Poisson starts in tests/test_mainsgate.py.

The mains clock
---------------
Excision leaves only 0.5-3 buzz impulses/s in the detector output, too thin to fit the mains phase from. The clock of
each 10 s block therefore comes from the hum flag's own edges (``humflag.block_edges``, July-rule crossings on the
UN-excised band, about 120/s under steady buzz): f refined on them, phase tracked every 1 s (``clock_from_edges``).
The clock must be independent of the starts it judges, or the binomial level of the fold is a fiction. Only edges
outside every detector impulse's extent [start - 1 ms, end + 1 ms] are used (``extent_free``). Why the whole extent:
the first guard dropped only edges within 1 ms of a start, and on the harness's mains-free Poisson scene (5/s, 24 dB)
every edge it kept still lay inside a sferic's impulse, on its ringing tail. A clock fitted to those tails lands on
the sferics' own phases: one false 'whole' train (stated p 4.9e-5) took 55 of 622 sferics (9.0 %), and a verifier
saw false trains in 3 of 12 seeds at 5-10/s. With the extent guard no edge a judged start made reaches the clock.

Which blocks get a clock (``clocked``, ``scope``)
-------------------------------------------------
"flagged" is the plan's rule (step-2 flagged blocks only). "all", the default, makes every block a candidate. Why: on
11-05 20Z blocks 51-59 and 11-05 04:30Z blocks 2-17 the flag sees nothing (6-18 locked edges per block, under its 20)
while the detector keeps 0.5-3 buzz impulses/s there (10-11 dB, none confirmed above 16 kHz); after the flagged-only
gate those blocks held 18 of the 30 comb pairs left in 11-05 20Z. A candidate whose edges are not edge-flagged gets
a clock only when its extent-free edges are phase locked at P_CLOCK (1e-2) over 119.95-120.05 Hz and the fitted f is
not on the edge of that search (``at_scan_edge``: on the false-train scene 5 of 11 block clocks sat at 119.95 or
120.05 Hz). Otherwise the block gets the self test on its starts, which fits f to them and counts every trial.
A block flagged only from the detector's starts (``humflag.event_stats``) never lends its f to a clock
(``edge_flagged``).

The searches, per run of consecutive clocked blocks (``gate``)
-------------------------------------------------------------
1. Bursts (``find_bursts``): a 1 s fold, 0.5 s hop, up to 2 trains per window; level P_HOT per block over all its
   windows and positions. Why: on 11-05 12:30Z and 11-06 03:30Z a second source comes in bursts of 5-10 pulses
   (0.15 s, 10-12 dB, unconfirmed) at 2 x 8.33 ms, each burst tight (+-25 us) but at a phase that moves over
   5.5-7.5 ms from burst to burst. A 60 s fold smears it into a 2 ms band. Run FIRST, because a few bursts that share
   a phase also pass the 60 s test (it assumes independent starts) and would then gate a whole 60 s.
2. Segment (``find_trains_clocked``): the starts of up to 6 blocks (60 s) not taken by bursts, each folded against
   its own block's clock; up to 4 trains peeled (11-04 01Z has a second train 4.4 ms from the first).
3. Whole capture (kind "whole"): one more fold over every clocked block, on the starts still free. A train of 0.05
   impulses/s is invisible in 60 s and plain in 600 s (p 1e-9 to 1e-15 on 4 hum captures).
   A 'whole' train must hold starts in at least WHOLE_MIN_BLOCKS (3) blocks.
4. Blocks left without a clock (out of scope, under 8 extent-free edges, or not phase locked): the self test
   (``find_trains`` with f searched over 119.95-120.05 Hz, no train when the best f is on that edge; level ``P_SELF``
   = 1e-5, or ``P_HOT`` when the hum flag marked the block) unless ``self_test=False``.
Family-wise per capture: at most (blocks + segments + 1) x 1e-4 for the clocked searches; a false train costs 4.8 %
of the starts in its span. Checked by simulation (bench and tests/test_mainsgate.py): mains-free Poisson starts at
2-200/s whose edges are their own tails plus noise lose <= 1 % in every run.

Every train is then applied by ``judge``: an impulse within +-0.2 ms (``WIN_S``; buzz jitter 37-44 us rms, and the
detector's 10 dB start spreads 0.4 ms buzz pulses over 0.4 ms) of a train, inside the train's own time span, is
rejected unless its peak is at least 6 dB (``ESCAPE_DB``) over that train's median peak. Buzz holds 15.7-17.8 dB
over floor before excision and 10-13 dB after, so a strong sferic on the train still counts.

What it leaves to the caller (``sferic_watch.pipeline``): blanking the rejected impulses in the power BEFORE the 0.5 ms
merge (a rejected pulse is never a merge target), re-estimating the floor without those 1 ms blocks, and an optional
2 ms refractory.

Explicitly NOT done: OnAir's loose k = 1..5 / 4 % / last-4 test (false rejection above), and hum template subtraction
(0.2-1.8 dB, measured as a failure three times in the investigation).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.stats import binom

WIN_S = 0.2e-3          # reject half-window around a train, s
ESCAPE_DB = 6.0         # keep an on-train impulse this far over the train's median peak
P_HOT = 1e-4            # family-wise false-alarm level of one fold search (flagged blocks)
P_SELF = 1e-5           # ...of one unflagged block's self test (no hum flag, no edge clock)
MAX_TRAINS = 4          # peel at most this many trains per search
N_FOLD_MIN = 8          # fewer impulses than this: no fold (cannot reach the level anyway)
SCAN_STRIDE_S = 0.1e-3  # phase stride of the sliding window, s (84 positions; union bound counts every one)
F_SPAN_HZ = 0.01        # frequency refinement half-span around the hum flag's f_mains
F_STEP_HZ = 0.0005      # frequency refinement step
SELF_F0, SELF_SPAN, SELF_STEP = 120.0, 0.05, 0.001   # unflagged blocks: 119.95-120.05 Hz (mains measured 119.96-120.008)
TRACK_S = 1.0           # phase re-measured every second
TRACK_SPAN_S = 2.0      # ...from the starts within +-2 s
TRACK_PULL_S = 0.6e-3   # ...that lie within +-0.6 ms of the train
TRACK_MIN_N = 3         # a step with fewer starts keeps the block phase
SEG_BLOCKS = 6          # edge-clocked fold pools up to 6 flagged 10 s blocks (60 s)
BURST_S, BURST_HOP_S = 1.0, 0.5   # burst fold: 1 s windows every 0.5 s
BURST_MIN_N = 4         # a burst train needs at least this many starts in its window
BURST_TRAINS = 2        # peel at most this many per burst window
N_STAT_EDGES = 20       # humflag.N_STAT: fewest edges for which the flag's Rayleigh level holds
EDGE_GUARD_S = 1e-3     # an edge within 1 ms of a detector impulse's extent [start, end] is not used for the clock
EXTENT_DEFAULT_S = 1e-3  # impulse length assumed when the caller gives no ends (``gate`` without ``t_end``)
P_CLOCK = 1e-2          # an un-flagged block's edges must be phase locked at this level to make a clock
WHOLE_MIN_BLOCKS = 3    # a "whole" train must hold starts in at least this many blocks


@dataclass
class Train:
    """One mains train. Times in s on the caller's clock.

    f: frequency (about 120 Hz). t_ref: phase reference; pulses sit at t_ref + c + k/f + offset(t). c: phase, s in
    [0, 1/f). track_t / track_off: centres of the 1 s steps and the phase offset measured there (s). t_lo / t_hi: the
    time span the train applies to (its block, or its 1 s burst window). n: starts within WIN_S of it found by the
    search. median_db: their median peak. p: family-wise p-value of the search that found it. kind: "segment",
    "block", "burst", "whole" or "self"."""
    f: float
    t_ref: float
    c: float
    track_t: np.ndarray = field(default_factory=lambda: np.zeros(0))
    track_off: np.ndarray = field(default_factory=lambda: np.zeros(0))
    n: int = 0
    median_db: float = float("nan")
    p: float = 1.0
    t_lo: float = float("nan")
    t_hi: float = float("inf")
    kind: str = "block"

    def __post_init__(self):
        if not np.isfinite(self.t_lo):
            self.t_lo = self.t_ref

    def residual(self, t: np.ndarray) -> np.ndarray:
        """Signed time from each t to the nearest pulse of this train, s, in [-P/2, P/2)."""
        t = np.asarray(t, float)
        off = np.interp(t, self.track_t, self.track_off) if self.track_t.size else 0.0
        P = 1.0 / self.f
        return (t - self.t_ref - self.c - off + P / 2) % P - P / 2


def rayleigh_r(t: np.ndarray, f: float) -> float:
    """Rayleigh phase lock R = |mean exp(2 pi i f t)| of the times t at frequency f (1 = all on one phase)."""
    t = np.asarray(t, float)
    if t.size == 0:
        return 0.0
    return float(np.abs(np.exp(2j * np.pi * f * (t - t[0])).mean()))


def f_grid(f0: float, span: float = F_SPAN_HZ, step: float = F_STEP_HZ) -> np.ndarray:
    """The trial frequencies f0 +- span in steps of ``step`` (their count enters the union bound)."""
    return np.arange(f0 - span, f0 + span + step / 2, step)


def refine_f(t: np.ndarray, f0: float, span: float = F_SPAN_HZ, step: float = F_STEP_HZ) -> tuple[float, float]:
    """Frequency in f0 +- span (step ``step``) that maximises R of the times t. Returns (f, R)."""
    t = np.asarray(t, float)
    if t.size < 2 or not np.isfinite(f0):
        return float(f0), 0.0
    fr = f_grid(f0, span, step)
    r = np.abs(np.exp(2j * np.pi * np.outer(fr, t - t[0])).mean(axis=1))
    k = int(np.argmax(r))
    return float(fr[k]), float(r[k])


def at_scan_edge(f: float, f0: float, span: float, step: float) -> bool:
    """True when a fitted frequency sits on the first or last trial of the f0 +- span grid.

    Measured (harness Poisson scene at 5/s, no mains): the false 'whole' train's per-block clocks sat at 119.95 /
    120.05 Hz in 5 of 11 blocks. Mains here is 119.96-120.008 Hz, so a best fit pinned to the edge of the search means
    the optimum lies outside it, i.e. the times are not on a mains clock in the searched range."""
    return bool(abs(f - f0) >= span - step / 2)


def n_positions(P: float, stride: float = SCAN_STRIDE_S) -> int:
    """Window positions tested on one fold circle (84 at 120 Hz and 0.1 ms)."""
    return int(np.ceil(P / stride))


def busiest_window(ph: np.ndarray, P: float, win: float = WIN_S, stride: float = SCAN_STRIDE_S) -> tuple[float, int]:
    """Slide a +-win window around the fold circle (period P) of phases ``ph`` (s in [0, P)); return (centre, count)
    of the busiest of the ``n_positions`` positions. Ties take the first."""
    ph = np.sort(np.asarray(ph, float) % P)
    if ph.size == 0:
        return 0.0, 0
    cent = np.arange(n_positions(P, stride)) * stride
    ext = np.concatenate([ph - P, ph, ph + P])
    cnt = np.searchsorted(ext, cent + win, side="right") - np.searchsorted(ext, cent - win, side="left")
    k = int(np.argmax(cnt))
    return float(cent[k]), int(cnt[k])


def hot_p(count: int, n: int, f: float, win: float = WIN_S, removed: float = 0.0, trials: float = 1.0,
          stride: float = SCAN_STRIDE_S) -> float:
    """Family-wise p-value (union bound) of ``count`` of ``n`` random phases in the busiest +-win window.

    Under no mains the n free starts are uniform on the part of the circle not yet peeled, length P - removed, so
    one position holds a binomial(n, q) count with q = 2 win / (P - removed). The bound multiplies the tail by every
    position tested and by ``trials`` (other searches pooled into the same decision, e.g. fitted frequencies or burst
    windows). A window that overlaps a peeled arc covers less free circle than q assumes, which only makes p larger."""
    if n <= 0 or count <= 0:
        return 1.0
    P = 1.0 / f
    q = min(1.0, 2 * win / max(P - removed, 2 * win))
    return float(min(1.0, binom.sf(count - 1, n, q) * n_positions(P, stride) * trials))


def _peel(ph: np.ndarray, P: float, f: float, level: float, max_trains: int, win: float, trials: float = 1.0,
          removed0: float = 0.0, free: np.ndarray | None = None, min_n: int = N_FOLD_MIN):
    """Peel trains off one fold: yield (centre, own mask, p) while the busiest window passes ``level``.

    ``free`` (bool mask over ph) is updated in place; each train removes its +-win arc (2 win of circle)."""
    if free is None:
        free = np.ones(ph.size, bool)
    removed = removed0
    for _ in range(max_trains):
        n = int(free.sum())
        if n < min_n:
            return
        c, cnt = busiest_window(ph[free], P, win)
        pv = hot_p(cnt, n, f, win, removed, trials)
        if pv >= level:
            return
        d = (ph - c + P / 2) % P - P / 2
        c = (c + float(np.median(d[free & (np.abs(d) <= win)]))) % P          # re-centre on the window's starts
        d = (ph - c + P / 2) % P - P / 2
        own = free & (np.abs(d) <= win)
        if not own.any():
            return
        free &= ~own
        removed += 2 * win
        yield c, own, pv


def track(train: Train, t: np.ndarray, t0: float, t1: float, step: float = TRACK_S, span: float = TRACK_SPAN_S,
          pull: float = TRACK_PULL_S, min_n: int = TRACK_MIN_N) -> Train:
    """Measure the train's phase offset every ``step`` s over [t0, t1): median residual of the starts within
    +-``pull`` of it inside +-``span`` s of the step centre. Steps with fewer than ``min_n`` such starts take the
    nearest measured step (or 0 when none is measured). Returns the train with track_t / track_off filled."""
    t = np.asarray(t, float)
    cen = np.arange(t0 + step / 2, t1, step)
    if cen.size == 0:
        cen = np.array([(t0 + t1) / 2])
    base = Train(train.f, train.t_ref, train.c)
    r = base.residual(t)
    near = np.abs(r) <= pull
    off = np.full(cen.size, np.nan)
    for i, c in enumerate(cen):
        m = near & (t >= c - span) & (t < c + span)
        if m.sum() >= min_n:
            off[i] = float(np.median(r[m]))
    ok = np.isfinite(off)
    if ok.any():
        off = np.interp(cen, cen[ok], off[ok])
    else:
        off[:] = 0.0
    train.track_t, train.track_off = cen, off
    return train


def find_trains(t: np.ndarray, pk: np.ndarray, t0: float, t1: float, f0: float, max_trains: int = MAX_TRAINS,
                p_hot: float = P_HOT, win: float = WIN_S, span: float = F_SPAN_HZ, step: float = F_STEP_HZ,
                kind: str = "block") -> list[Train]:
    """All mains trains in one block [t0, t1) from the detector starts alone (no edge clock).

    t, pk: impulse starts (s) and peaks (dB over floor); others outside the block are ignored. f0 +- span: the
    frequencies tried (the hum flag's f_mains +- 0.01 Hz, or 119.95-120.05 Hz for an unflagged block); their count
    is a factor in the union bound because f is fitted to the same starts that are then tested. A free search
    (span over F_SPAN_HZ) whose best f is on the edge of the grid returns no train (``at_scan_edge``); a narrow
    refinement there keeps f0 (two trains half a cycle apart flatten R, 11-04 01Z shape).
    The first train is tested on the plain fold. Measured on 11-04 01Z the phase still wanders by up to 0.3-0.5 ms
    within 10 s after the frequency fit, so it is then tracked every 1 s and later trains are searched on the
    de-wandered fold (the tracker used only starts within +-0.6 ms of train 1). Returns trains strongest first."""
    t = np.asarray(t, float)
    pk = np.asarray(pk, float)
    m = (t >= t0) & (t < t1)
    t, pk = t[m], pk[m]
    if t.size < N_FOLD_MIN or not np.isfinite(f0):
        return []
    n_f = f_grid(f0, span, step).size
    f, _ = refine_f(t, f0, span, step)
    if at_scan_edge(f, f0, span, step):
        if span > F_SPAN_HZ:          # free search (self test): the optimum is outside the mains range, no train
            return []
        f = float(f0)                 # refinement round a known f: R is flat (e.g. two trains half a cycle apart)
    P = 1.0 / f
    c, cnt = busiest_window((t - t0) % P, P, win)
    p_first = hot_p(cnt, t.size, f, win, trials=n_f)
    if p_first >= p_hot:
        return []
    clock = track(Train(f, t0, c), t, t0, t1)
    tt, off = clock.track_t, clock.track_off
    ph = (t - t0 - np.interp(t, tt, off)) % P
    free = np.ones(t.size, bool)
    trains = []
    # Train 1 already passed on the plain fold: peel it unconditionally, then the others at p_hot.
    for c, own, pv in _peel(ph, P, f, np.inf, 1, win, n_f, free=free, min_n=1):
        trains.append(Train(f, t0, c, tt.copy(), off.copy(), n=int(own.sum()), median_db=float(np.median(pk[own])),
                            p=p_first, t_hi=t1, kind=kind))
    for c, own, pv in _peel(ph, P, f, p_hot, max_trains - 1, win, n_f, removed0=2 * win, free=free):
        trains.append(Train(f, t0, c, tt.copy(), off.copy(), n=int(own.sum()), median_db=float(np.median(pk[own])),
                            p=pv, t_hi=t1, kind=kind))
    return trains


def judge(t: np.ndarray, pk: np.ndarray, trains: list[Train], win: float = WIN_S,
          escape_db: float = ESCAPE_DB) -> tuple[np.ndarray, np.ndarray]:
    """Per impulse: (reject, escaped). An impulse is on a train when it is within +-win of it and inside the train's
    span [t_lo, t_hi); it is rejected unless its peak is at least ``escape_db`` over that train's median peak (then
    ``escaped``). An impulse rejected by any train is not escaped."""
    t = np.asarray(t, float)
    pk = np.asarray(pk, float)
    rej = np.zeros(t.size, bool)
    esc = np.zeros(t.size, bool)
    for tr in trains:
        inb = (t >= tr.t_lo) & (t < tr.t_hi)
        on = inb & (np.abs(tr.residual(t)) <= win)
        strong = pk >= tr.median_db + escape_db
        rej |= on & ~strong
        esc |= on & strong
    esc &= ~rej
    return rej, esc


def clock_from_edges(edges: np.ndarray, t0: float, t1: float, f0: float, span: float = F_SPAN_HZ,
                     p_lock: float | None = None) -> Train | None:
    """The mains clock of one block from the hum flag's own edges (July-rule crossings on the UN-excised
    2.5-9 kHz band, about 120/s under steady buzz; ``humflag.block_edges``).

    The dense edges fix the phase every second to a few us, where the detector's 0.5-3 buzz impulses/s cannot.
    f is refined on the edges (f0 +- span), the clock phase is their busiest +-0.2 ms window, then tracked every 1 s.
    Returns None when there are fewer than N_FOLD_MIN edges, when a free search (span wider than F_SPAN_HZ) puts f on
    its edge (``at_scan_edge``; a narrow refinement round a known f falls back to that f instead), or, with ``p_lock``, when the edges are not phase locked at that level (Rayleigh R against
    the Rice level of ``humflag.rayleigh_levels`` over the searched band). The lock test is what an un-flagged block
    needs before its edges count as a mains clock: under no mains its edges are noise, and a noise clock with a random
    f would only make the block skip the self test on its starts."""
    e = np.asarray(edges, float)
    e = e[(e >= t0) & (e < t1)]
    if e.size < N_FOLD_MIN or not np.isfinite(f0):
        return None
    f, r = refine_f(e, f0, span)
    if at_scan_edge(f, f0, span, F_STEP_HZ):
        if span > F_SPAN_HZ:          # free search: not a mains clock in the searched range
            return None
        f, r = float(f0), rayleigh_r(e, f0)
    if p_lock is not None:
        from sferic_watch.humflag import rayleigh_levels
        if r <= rayleigh_levels(e.size, t1 - t0, f0 - span, f0 + span, p_lock)[1]:
            return None
    c, _ = busiest_window((e - t0) % (1.0 / f), 1.0 / f)
    tr = track(Train(f, t0, c), e, t0, t1)
    tr.t_hi = t1
    return tr


def _clock_phases(t: np.ndarray, clocks: list[Train]) -> tuple[np.ndarray, np.ndarray, float]:
    """(indices into t, phase of each against its block's clock in [0, P), mean P) for starts inside the clocks."""
    idx, ph = [], []
    for ck in clocks:
        m = np.flatnonzero((t >= ck.t_lo) & (t < ck.t_hi))
        idx.append(m)
        ph.append(ck.residual(t[m]) % (1.0 / ck.f))
    P = float(np.mean([1.0 / ck.f for ck in clocks]))
    return np.concatenate(idx), np.concatenate(ph), P


def find_trains_clocked(t: np.ndarray, pk: np.ndarray, clocks: list[Train], max_trains: int = MAX_TRAINS,
                        p_hot: float = P_HOT, win: float = WIN_S, exclude: np.ndarray | None = None,
                        removed: float = 0.0, kind: str = "segment", min_blocks: int = 1) -> list[Train]:
    """Trains of detector impulses in one SEGMENT of consecutive flagged blocks, each block with its edge clock.

    Every start is mapped to its phase against its own block's clock, so the whole segment (up to SEG_BLOCKS x 10 s)
    folds coherently while the mains phase wanders. The clock was fitted to the edges, not to these starts, so no
    frequency trials enter the bound. ``exclude`` (bool per start) leaves out starts already taken by other trains;
    ``removed`` is the largest arc (s) those trains took from any one block's circle, so q is never under-stated.
    ``min_blocks``: a train whose own starts fall in fewer distinct blocks is dropped and the peel stops (the
    whole-capture fold uses WHOLE_MIN_BLOCKS: real sparse buzz is spread over the capture, while a chance cluster that
    survives the bound is most likely a few blocks' worth of starts).
    Returns one Train per (train, block) so ``judge`` applies each only inside its
    own block; the copies of one train share its segment-wide n, median peak and p."""
    if not clocks:
        return []
    t = np.asarray(t, float)
    pk = np.asarray(pk, float)
    idx, ph, P = _clock_phases(t, clocks)
    blk = np.concatenate([np.full(int(((t >= ck.t_lo) & (t < ck.t_hi)).sum()), i) for i, ck in enumerate(clocks)])
    if exclude is not None:
        keep = ~np.asarray(exclude, bool)[idx]
        idx, ph, blk = idx[keep], ph[keep], blk[keep]
    out = []
    for c, own, pv in _peel(ph, P, 1.0 / P, p_hot, max_trains, win, removed0=removed):
        if np.unique(blk[own]).size < min_blocks:
            break
        med = float(np.median(pk[idx[own]]))
        for ck in clocks:
            out.append(Train(ck.f, ck.t_ref, (ck.c + c) % (1.0 / ck.f), ck.track_t, ck.track_off,
                             n=int(own.sum()), median_db=med, p=pv, t_lo=ck.t_lo, t_hi=ck.t_hi, kind=kind))
    return out


def find_bursts(t: np.ndarray, pk: np.ndarray, clock: Train, done: list[Train], p_hot: float = P_HOT,
                win: float = WIN_S, span: float = BURST_S, hop: float = BURST_HOP_S,
                max_trains: int = BURST_TRAINS) -> list[Train]:
    """Short trains (bursts) inside one edge-clocked block: fold the starts of each 1 s window (0.5 s hop) that no
    train in ``done`` already rejects, against the block's clock, and peel up to ``max_trains`` per window.

    Union bound: the block's level ``p_hot`` covers every window and position (trials = number of windows); the
    arcs of the trains already found in the block are counted as removed. A burst also needs BURST_MIN_N starts.
    Each burst train applies to its own window only."""
    t = np.asarray(t, float)
    pk = np.asarray(pk, float)
    P = 1.0 / clock.f
    w0s = np.arange(clock.t_lo, clock.t_hi - span / 2 + 1e-9, hop)
    if w0s.size == 0:
        return []
    mine = [tr for tr in done if tr.t_lo < clock.t_hi and tr.t_hi > clock.t_lo]
    taken = np.zeros(t.size, bool)
    for tr in mine:
        taken |= (t >= tr.t_lo) & (t < tr.t_hi) & (np.abs(tr.residual(t)) <= win)
    removed = 2 * win * len({round(tr.c * 1e6) for tr in mine if tr.kind != "burst"})
    out = []
    for w0 in w0s:
        w1 = min(w0 + span, clock.t_hi)
        m = np.flatnonzero((t >= w0) & (t < w1) & ~taken)
        if m.size < BURST_MIN_N:
            continue
        ph = clock.residual(t[m]) % P
        for c, own, pv in _peel(ph, P, clock.f, p_hot, max_trains, win, trials=w0s.size, removed0=removed,
                                min_n=BURST_MIN_N):
            if own.sum() < BURST_MIN_N:
                break
            out.append(Train(clock.f, clock.t_ref, (clock.c + c) % P, clock.track_t, clock.track_off,
                             n=int(own.sum()), median_db=float(np.median(pk[m[own]])), p=pv, t_lo=w0, t_hi=w1,
                             kind="burst"))
    return out


def phase_locked(b: dict) -> bool:
    """The hum flag's own phase-lock test on a block's edges: N >= humflag.N_STAT and R over its 1e-4 level."""
    return bool(b.get("n_edges", 0) >= N_STAT_EDGES and b.get("R", 0.0) > b.get("r_crit", np.inf))


def edge_flagged(b: dict) -> bool:
    """Did the hum flag find buzz in this block from its OWN edges (``hum_edge``; blocks from before the event source
    existed carry only ``hum``)? Only then is its ``f_mains`` an edge frequency the clock may be refined around: a
    block flagged from the detector's own starts (``humflag`` event source) has an f fitted to the very starts the
    gate judges, so it is searched with the self test's frequency trials instead."""
    return bool(b.get("hum_edge", b.get("hum")))


def clocked(b: dict, scope: str = "all") -> bool:
    """Is this hum-flag block a candidate for an edge clock and the clocked folds?

    "flagged": the plan's rule, flagged blocks only. "locked": also blocks that pass the flag's 1e-4 phase-lock test
    but miss its 20 locked-edge count. "all" (default): every block. Why "all": on 11-05 20Z blocks 51-59 the flag
    sees neither (n_locked 6-18, R under its level) while the detector keeps 0.5-1 buzz impulse/s there (10-11 dB,
    none confirmed above 16 kHz); after the flagged-only gate those blocks held 18 of the 30 comb pairs left.
    A candidate that is not edge-flagged still needs its guarded edges phase locked at P_CLOCK to get a clock
    (``clock_from_edges``); otherwise it falls back to the self test. What makes "all" safe is ``extent_free``: the
    clock never sees an edge inside (or within 1 ms of) a detector impulse, so under no mains it is independent of
    the starts it judges and the fold keeps its level."""
    if scope == "all" or b.get("hum"):
        return True
    return scope == "locked" and phase_locked(b)


def start_free(edges: np.ndarray, t: np.ndarray, guard: float = EDGE_GUARD_S) -> np.ndarray:
    """The edges with no detector start within +-guard (the first, insufficient, independence guard; kept for
    comparison). Measured on the harness Poisson scene (5/s, no mains, 24 dB): 155 of 731 edges survive it and
    every one of them still lies inside a detector impulse (median 0.95 ms before its end), i.e. on a sferic's
    tail. A clock fitted to them is pulled onto the sferics' own phases: 55 of 622 sferics lost to a false 'whole'
    train. Use ``extent_free``."""
    e = np.sort(np.asarray(edges, float))
    t = np.sort(np.asarray(t, float))
    if t.size == 0 or e.size == 0:
        return e
    j = np.searchsorted(t, e)
    d = np.minimum(np.abs(e - t[np.clip(j - 1, 0, t.size - 1)]), np.abs(e - t[np.clip(j, 0, t.size - 1)]))
    return e[d > guard]


def extent_free(edges: np.ndarray, t: np.ndarray, t_end: np.ndarray | None = None,
                guard: float = EDGE_GUARD_S) -> np.ndarray:
    """The edges outside every detector impulse's extent [start - guard, end + guard].

    Why the whole extent and not just the start: a sferic in the 2.5-9 kHz band rings for 1-3 ms and the July rule
    (1 ms debounce) re-crosses on that ring; on the harness Poisson scenes 100 % of the edges more than 1 ms from a
    start were still inside an impulse. The July threshold (median + 8 MAD-sigma of the envelope) sits above the
    detector's 10 dB start, so an edge a sferic makes is inside that sferic's impulse; dropping the extent plus 1 ms
    leaves edges that no judged start made. Without ``t_end`` each impulse is taken as EXTENT_DEFAULT_S long.
    Under steady buzz little is lost: about 120 buzz edges/s against the detector's 0.5-3 kept buzz impulses/s
    (each takes about 2.4 ms of edges); at 300 impulses/s about 40 % of the edges remain."""
    e = np.sort(np.asarray(edges, float))
    t = np.asarray(t, float)
    if t.size == 0 or e.size == 0:
        return e
    hi_t = t + EXTENT_DEFAULT_S if t_end is None else np.maximum(np.asarray(t_end, float), t)
    o = np.argsort(t)
    lo, hi = t[o] - guard, np.maximum.accumulate(hi_t[o] + guard)
    j = np.searchsorted(lo, e, side="right") - 1
    inside = (j >= 0) & (hi[np.clip(j, 0, None)] >= e)
    return e[~inside]


def segments(blocks: list[dict], max_blocks: int = SEG_BLOCKS, key: str = "hum") -> list[list[dict]]:
    """Runs of consecutive blocks with ``key`` true (touching in time), cut into pieces of at most ``max_blocks``."""
    segs, cur = [], []
    for b in blocks:
        if not b.get(key):
            if cur:
                segs.append(cur)
            cur = []
            continue
        if cur and (abs(cur[-1]["t0_s"] + cur[-1]["dur_s"] - b["t0_s"]) > 1e-6 or len(cur) >= max_blocks):
            segs.append(cur)
            cur = []
        cur.append(b)
    if cur:
        segs.append(cur)
    return segs


def _self(t, pk, b, p_hot=P_SELF):
    """The self test of one block (no clock): frequency searched over 119.95-120.05 Hz, level ``p_hot`` (P_SELF for an
    un-flagged block, P_HOT for one the hum flag marked: the flag's own 1e-4 test and this one must both pass, so
    the joint false-alarm level is at most the smaller of the two)."""
    return find_trains(t, pk, b["t0_s"], b["t0_s"] + b["dur_s"], SELF_F0, p_hot=p_hot, span=SELF_SPAN,
                       step=SELF_STEP, kind="self")


def gate(t: np.ndarray, pk: np.ndarray, blocks: list[dict], edges: np.ndarray | None = None, scope: str = "all",
         self_test: bool = True, bursts: bool = True, whole: bool = True, t_end: np.ndarray | None = None,
         **kw) -> tuple[np.ndarray, np.ndarray, list[Train]]:
    """Gate a whole impulse list. ``blocks`` are hum-flag block dicts (``t0_s``, ``dur_s``, ``hum``, ``f_mains``,
    ``n_edges``, ``R``, ``r_crit``; optional ``hum_edge``, see ``edge_flagged``). ``t_end``: impulse ends (s), used
    to keep every edge inside an impulse out of the clock (``extent_free``).

    With ``edges`` (the hum flag's edge times, s, same clock, covering at least the blocks in ``scope``, see
    ``clocked``): the extent-free edges give each candidate block a clock (edge-flagged blocks around their flag
    frequency; others only when their edges are phase locked at P_CLOCK over 119.95-120.05 Hz, see
    ``clock_from_edges``); per run of up to SEG_BLOCKS such blocks, first 1 s bursts (``find_bursts``, if
    ``bursts``), then the 60 s segment fold (``find_trains_clocked``); then, if ``whole`` and more than SEG_BLOCKS
    blocks have clocks, one fold over all of them on the starts still free (kind "whole", at least
    WHOLE_MIN_BLOCKS blocks). An edge-flagged block whose clock cannot be built is searched alone around its flag
    frequency (``find_trains``). Without edges every edge-flagged block is searched alone. Blocks left without a
    clock get the self test when ``self_test`` (level P_HOT when flagged by any source, else P_SELF).
    Returns (reject, escaped, trains)."""
    t = np.asarray(t, float)
    trains, bare = [], []
    if edges is None:
        for b in blocks:
            if edge_flagged(b) and np.isfinite(b.get("f_mains", np.nan)):
                trains += find_trains(t, pk, b["t0_s"], b["t0_s"] + b["dur_s"], b["f_mains"], **kw)
            else:
                bare.append(b)
    else:
        e = extent_free(edges, t, t_end)
        sel = [dict(b, _clk=clocked(b, scope)) for b in blocks]
        bare = [b for b in sel if not b["_clk"]]
        all_clocks, n_seg_max = [], 0
        for seg in segments(sel, key="_clk"):
            clocks = []
            for b in seg:
                known = (edge_flagged(b) or phase_locked(b)) and np.isfinite(b.get("f_mains", np.nan))
                f0, span = (b["f_mains"], F_SPAN_HZ) if known else (SELF_F0, SELF_SPAN)
                ck = clock_from_edges(e, b["t0_s"], b["t0_s"] + b["dur_s"], f0, span=span,
                                      p_lock=None if known else P_CLOCK)
                if ck is not None:
                    clocks.append(ck)
                elif edge_flagged(b) and np.isfinite(b.get("f_mains", np.nan)):
                    trains += find_trains(t, pk, b["t0_s"], b["t0_s"] + b["dur_s"], b["f_mains"], **kw)
                else:
                    bare.append(b)
            # Bursts first: a few bursts that happen to share a phase also pass the 60 s binomial test (it assumes
            # independent starts) and would then gate the whole 60 s at that phase. Measured on the synthetic burst
            # scene of tests/test_mainsgate.py (8-pulse bursts every 3 s at random phases, 5/s Poisson): segment
            # first gave a 60 s train on 2 of 4 seeds and 8-13 % Poisson loss.
            bt = [b for ck in clocks for b in find_bursts(t, pk, ck, [])] if bursts else []
            taken = judge(t, np.full(t.size, -np.inf), bt)[0] if bt else None
            st = find_trains_clocked(t, pk, clocks, exclude=taken, **kw)
            n_seg_max = max(n_seg_max, len(st) // max(1, len(clocks)))
            trains += bt + st
            all_clocks += clocks
        if whole and len(all_clocks) > SEG_BLOCKS:
            # Whole-capture fold: every clocked block pooled, on the starts no train has taken yet. Why: a very sparse
            # train (0.05 buzz impulses/s, 10.5 dB, unconfirmed) stays below any 60 s test but is plain over 600 s:
            # pooled kept events of 11-05 20Z / 11-03 08Z / 11-05 12:30Z / 11-06 02:30Z put 30 / 32 / 66 / 30 in one
            # +-0.2 ms window (p 1e-13 / 3e-9 / 1e-15 / 1e-11) and carried comb shares of 0.35 / 0.26 / 0.38 / 0.29.
            taken = judge(t, np.full(t.size, -np.inf), trains)[0] if trains else None
            trains += find_trains_clocked(t, pk, all_clocks, exclude=taken, removed=2 * WIN_S * n_seg_max,
                                          kind="whole", min_blocks=WHOLE_MIN_BLOCKS, **kw)
    if self_test:
        for b in bare:
            trains += _self(t, pk, b, P_HOT if b.get("hum") else P_SELF)
    rej, esc = judge(t, pk, trains)
    return rej, esc, trains


def gated_seconds(trains: list[Train], win: float = WIN_S) -> float:
    """Seconds of signal inside a reject window (where a sferic weaker than the escape would be lost): per train,
    its span x 2 win f. Overlapping spans of different trains are counted once each (an upper bound)."""
    s = 0.0
    for tr in trains:
        dur = tr.t_hi - tr.t_lo if np.isfinite(tr.t_hi) else 0.0
        s += max(0.0, dur) * min(1.0, 2 * win * tr.f)
    return s
