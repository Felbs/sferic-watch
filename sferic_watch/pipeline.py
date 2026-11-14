"""The v2 engine end to end (plan step 5): front end -> detector -> hum flag -> mains gate -> events.

Order and why:
1. ``detect.band_powers``: the zero-phase front end (0 Hz receiver tone out, 2.5-9 kHz band, line excision) and the
   16-sample smoothed power of the detection and 16-30 kHz confirmation bands. Line excision alone already strips most
   buzz. On 60 s of 11-04 01Z / 10-20 08Z / 11-06 22Z the detector keeps only 2.5-3.9 impulses/s, from about 120 buzz
   edges/s before excision. What is left still has a comb share of 0.60 / 0.16 / 0.27 (chance 0.144).
2. ``humflag.flag_iq``: the step-2 per-10 s buzz flag (its own July-rule edges on the un-excised band, Rayleigh R at
   p 1e-4), plus a second source: the same test on the detector's raw impulse starts (``humflag.add_event_stats``),
   because the July edges are less sensitive than the 10 dB detector (11-03 18:00-19:30Z: flag 0, comb 0.40-0.58).
   The flag's edges (``hum_edges``) outside every raw impulse's extent become the gate's 1 s mains clock
   (``mainsgate.extent_free``: under no mains an edge inside an impulse is that sferic's own tail, and a clock fitted
   to tails gated 9 % of the harness's Poisson sferics); the 60 s fold and the 1 s burst pass run there. Blocks with
   no locked clock get the gate's self test (level 1e-5, 1e-4 when flagged; ``self_test=False`` turns it off).
3. Raw impulses (the detector's 10/7 dB state machine with NO merge) and their peaks -> ``mainsgate.gate``.
4. Rejected impulses are blanked: their power is set to 0 and every 1 ms floor block they touch is left out of the
   median floor. Then the floor is re-estimated and the raw impulses are found again. Impulses that the new, lower
   floor lets through are judged against the same trains, with peaks measured on the first-pass floor (the scale of
   the trains' median peaks, so the 6 dB escape does not loosen by the floor drop). Finally ``detect.events_from_power`` runs on the blanked
   power WITH the 0.5 ms merge, so a rejected buzz pulse can never be a merge target or stretch a sferic.
5. Optional refractory (default off): drop an event starting less than ``refractory_s`` after the previous kept one.

Per run the info dict reports hum_frac (share of flagged time), gated_frac (share of time inside a reject window, where
a sferic under the 6 dB escape is lost), the trains per flagged block, rejected and escaped counts, and the CPU split.

Registered for the bench harness as ``v2`` (gated) with the ungated twin ``v2_nogate`` (identical chain, gate off).
"""
from __future__ import annotations

import time

import numpy as np

import sferic_watch as sw
from sferic_watch import detect as dt
from sferic_watch import humflag as hf
from sferic_watch import mainsgate as mg

FS = dt.FS
VERSION = "v2-gate-3"


def raw_impulses(p: np.ndarray, fs: float = FS, exclude_blocks: np.ndarray | None = None,
                 peak_floor: np.ndarray | None = None, return_floor: bool = False):
    """Unmerged impulses of the detector on smoothed power ``p``: (starts, ends, peak dB over floor[, floor]).

    Same floor and start/end rule as ``detect.events_from_power`` (median of 1 ms block means over a trailing 4 s,
    10 dB start, 7 dB / 100 us end, 5 ms width cap) but merge_s = 0, so the gate sees every pulse on its own.
    ``peak_floor`` (per-sample power) measures the peaks against another floor than the one used to find the
    impulses: the pipeline's second pass finds impulses on the re-estimated (lower) floor but must measure peaks on
    the first-pass floor, the scale of each train's median peak, or the 6 dB escape would loosen by the floor drop."""
    block_s = dt.BLOCK_N / fs
    hop_n = int(round(dt.FLOOR_HOP_S / block_s)) * dt.BLOCK_N
    fl, _ = dt.sliding_floor(dt.block_means(p), block_s, exclude=exclude_blocks)
    fsamp = dt.floor_per_sample(fl, len(p), hop_n)
    s, e, _ = dt.find_impulses(p, fsamp, fs, merge_s=0.0)
    ref = fsamp if peak_floor is None else peak_floor
    pk = np.array([10 * np.log10(max(float(p[a:b].max()), 1e-30) / ref[a]) for a, b in zip(s, e)])
    return (s, e, pk, fsamp) if return_floor else (s, e, pk)


def hum_edges(x: np.ndarray, flags: list[dict], fs: float = FS, scope: str = "all") -> np.ndarray:
    """The hum flag's own edge times (s) in the blocks the gate clocks (``mainsgate.clocked`` with ``scope``):
    ``humflag.block_edges`` on each block plus the flag's filter padding, exactly as ``humflag.flag_iq`` cuts them.
    These edges (minus those on a detector start) are the gate's mains clock (``mainsgate.clock_from_edges``)."""
    pad = int(round(hf.PAD_S * fs))
    out = []
    for b in flags:
        if not mg.clocked(b, scope):
            continue
        s0, L = int(round(b["t0_s"] * fs)), int(round(b["dur_s"] * fs))
        a = max(0, s0 - pad)
        e = hf.block_edges(x[a:min(len(x), s0 + L + pad)], s0 - a, L, n0=a, fs=fs)
        out.append((s0 + e) / fs)
    return np.concatenate(out) if out else np.zeros(0)


def blank(p: np.ndarray, s: np.ndarray, e: np.ndarray, excl: np.ndarray) -> None:
    """In place: zero p over each [s, e) and mark every 1 ms floor block it touches in ``excl``."""
    for a, b in zip(s, e):
        p[a:b] = 0.0
        excl[a // dt.BLOCK_N:min(len(excl), (b - 1) // dt.BLOCK_N + 1)] = True


def refractory(t: np.ndarray, dead_s: float) -> np.ndarray:
    """Bool keep-mask: drop an event that starts less than dead_s after the previous KEPT event."""
    keep = np.zeros(len(t), bool)
    last = -np.inf
    for i, v in enumerate(np.asarray(t, float)):
        if v - last >= dead_s:
            keep[i] = True
            last = v
    return keep


def run(x: np.ndarray, fs: float = FS, gate: bool = True, refractory_s: float | None = None,
        flags: list[dict] | None = None, powers: tuple[np.ndarray, np.ndarray] | None = None,
        scope: str = "all", self_test: bool = True, bursts: bool = True, whole: bool = True) -> tuple[dict, dict]:
    """Full v2 chain on baseband IQ ``x`` (complex64, full scale 1, centred 15 kHz). Times are s from x[0].

    ``flags`` lets a caller pass precomputed ``humflag.flag_iq`` blocks (same clock); blocks without the event
    source get it here IN PLACE (``humflag.add_event_stats`` on the raw impulse starts, so ``hum`` may turn on), so a
    caller that reads the flags afterwards sees the flag the gate used. ``powers`` a precomputed
    ``detect.band_powers(x)`` pair; the detection power is blanked IN PLACE, so pass a copy if it is reused.
    ``scope`` picks the blocks the gate clocks ("flagged" is the plan's rule, "all" the default; see
    ``mainsgate.clocked``); ``self_test`` / ``bursts`` / ``whole`` switch the self test of unclocked blocks, the 1 s
    burst pass and the whole-capture fold (``mainsgate.gate``). scope="flagged", self_test=False is the plan's gate.
    Returns (events, info): events
    as ``detect.FIELDS`` arrays plus ``hum`` (event lies in a flagged block); info as described in the module doc."""
    c0 = time.process_time()
    p, pc = powers if powers is not None else dt.band_powers(x, fs)
    c1 = time.process_time()
    s, e, pk, fl1 = raw_impulses(p, fs, return_floor=True)
    if flags is None:
        flags = hf.flag_iq(x, fs)
    if flags and "hum_ev" not in flags[0]:
        hf.add_event_stats(flags, s / fs)                  # second edge source: the detector's own starts
    c2 = time.process_time()
    secs = len(x) / fs
    nb = len(p) // dt.BLOCK_N
    excl = np.zeros(nb, bool)
    trains, n_rej, n_esc = [], 0, 0
    if gate and (self_test or any(mg.clocked(b, scope) for b in flags)):
        edges = hum_edges(x, flags, fs, scope) if any(mg.clocked(b, scope) for b in flags) else None
        rej, esc, trains = mg.gate(s / fs, pk, flags, edges=edges, scope=scope, self_test=self_test, bursts=bursts,
                                   whole=whole, t_end=e / fs)
        n_rej, n_esc = int(rej.sum()), int(esc.sum())
        blank(p, s[rej], e[rej], excl)
        if n_rej:                                          # floor re-estimated after blanking: judge again
            s, e, pk = raw_impulses(p, fs, excl, peak_floor=fl1)
            rej2, esc2 = mg.judge(s / fs, pk, trains)
            blank(p, s[rej2], e[rej2], excl)
            n_rej += int(rej2.sum())
            n_esc = int(esc2.sum())                        # every first-pass escape is still an impulse here
    ev, info = dt.events_from_power(p, pc, fs, exclude_blocks=excl if excl.any() else None)
    del p, pc
    if refractory_s:
        k = refractory(ev["t_start"], refractory_s)
        ev = {f: v[k] for f, v in ev.items()}
    hum_t = [(b["t0_s"], b["t0_s"] + b["dur_s"]) for b in flags if b["hum"]]
    ev["hum"] = np.zeros(len(ev["t_start"]), bool)
    for a, b in hum_t:
        ev["hum"] |= (ev["t_start"] >= a) & (ev["t_start"] < b)
    c3 = time.process_time()
    per_block, kinds = {}, {}
    for tr in trains:
        kinds[tr.kind] = kinds.get(tr.kind, 0) + 1
        per_block.setdefault(round(tr.t_ref, 3), []).append(
            {"kind": tr.kind, "f": round(tr.f, 5), "phase_ms": round(tr.c * 1e3, 3), "n": tr.n,
             "median_db": round(tr.median_db, 2), "p": float(f"{tr.p:.3g}"),
             "span_s": [round(tr.t_lo, 3), round(tr.t_hi, 3)]})
    info.update({"version": VERSION, "gate": gate, "secs": secs,
                 "hum_frac": float(sum(b - a for a, b in hum_t) / secs) if secs else 0.0,
                 "scope": scope, "self_test": self_test,
                 "gated_frac": float(mg.gated_seconds(trains) / secs) if secs else 0.0,
                 "n_trains": len(trains), "train_kinds": kinds, "trains": per_block, "rejected": n_rej, "escaped": n_esc,
                 "floor_blocks_excluded": int(excl.sum()),
                 "cpu_frontend_s": c1 - c0, "cpu_humflag_s": c2 - c1, "cpu_gate_detect_s": c3 - c2})
    return ev, info


@sw.register("v2", ungated="v2_nogate", note="v2 engine: front end, median-floor detector, hum flag, mains phase gate")
def v2(x, fs=FS):
    """Bench entry: event start times, peak dB over the floor, and info (without the per-block train list)."""
    ev, info = run(x, fs)
    info.pop("trains", None)
    return ev["t_start"], ev["peak_db"], info


@sw.register("v2_nogate", note="v2 chain with the mains gate off (the ungated twin of v2)")
def v2_nogate(x, fs=FS):
    """Bench entry: the same chain with gate=False (equals v2_detect; kept so the twin always matches v2's chain)."""
    ev, info = run(x, fs, gate=False)
    info.pop("trains", None)
    return ev["t_start"], ev["peak_db"], info
