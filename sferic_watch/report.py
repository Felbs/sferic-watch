"""Honest rate report and event catalogue for one capture (plan step 6): the panel's real producer.

What it writes, per capture (see ``schema.json`` for every field):

* ``<id>.events.csv``: one row per detected event (gated v2 chain, ``pipeline.run``): start, peak, end, 50 %-of-peak
  onset, peak dB over the floor, width, rise, floor, the 16-30 kHz confirmation, the 10 s hum block it lies in and
  that block's phase lock R (block -1 and R empty when it lies outside every flag block), whether a 16-30 kHz impulse peaks within +-0.5 ms (two-band coincidence), and a tail
  score.
* ``<id>.session.json``: per-minute and whole-capture rates. Every rate carries the numbers needed to judge it:
  - ``gated_frac``: share of the time inside a mains reject window (+-0.2 ms around each train pulse), where a
    sferic weaker than the 6 dB escape is lost: the UNION of all trains' windows, tracked phase included (report-2;
    report-1 summed overlapping trains and over-stated it). ``gated_frac_sum`` keeps the old sum for comparison.
    ``rate_open_per_min`` divides the count by the open time.
  - ``hum_frac``: share of the time in 10 s blocks the hum flag (``humflag``) marks as buzz.
  - ``comb_share``: share of consecutive event intervals within 0.6 ms of a multiple of 8.333 ms, against its
    chance level 2 x 0.6 / 8.333 = 0.144 for independent events. Buzz-dominated output reads 0.9-1.0 (bench).
  - two-band coincidence (the census "v2" idea): events whose peak has a 16-30 kHz impulse peak within +-0.5 ms,
    with the rate two independent trains of n_lo and n_hi impulses would give by chance, n_lo n_hi 2w / T^2
    (w = 0.5 ms, so 2w = 1 ms as in the census). A sferic is broadband; buzz is 0-2 dB above 16 kHz (census), so
    the coincidence rate should not move when buzz switches on.
  - the detector and report versions.
  A number is marked untrusted (the panel greys it) when its comb share is above 0.25 or its gated fraction is
  above 0.20 (plan step 6).

What is deliberately NOT here: tweek ranges (step 7 found the July fit pinned to its own bound; nothing per tweek
is published until that audit passes) and anything about where the station is. The sidecar is copied through a
whitelist of recorder settings only.

The tail score is new and NOT validated: mean detection-band power (before the gate blanks it) from 1 to 5 ms
after the peak over the event's floor, in dB. A dispersed sky-wave sferic rings on for milliseconds; a ~0.38 ms buzz pulse does not. It is recorded
so it can be checked against the confirmation flag later, not used for any decision.

Field names avoid the letters the privacy grep looks for (lat|lon|grid, any case), even inside words: the
detector's ``too_long`` count is written as ``too_wide`` (events dropped for being over 5 ms wide).

Output is deterministic: no wall-clock or CPU time is written into the session or catalogue, numbers are rounded
to fixed digits, so two runs on the same capture are byte-identical (unit test).
"""
from __future__ import annotations

import contextlib
import json
import math
import os
import re

import numpy as np

import sferic_watch as sw
from sferic_watch import detect as dt
from sferic_watch import humflag as hf
from sferic_watch import io as sio
from sferic_watch import mainsgate as mg
from sferic_watch import pipeline as pl

SCHEMA_ID = "sferic-watch/session/2"
REPORT_VERSION = "report-2"
SCHEMA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "schema.json")

MAINS_HALF_S = 1.0 / 120.0      # full-wave buzz period, s
COMB_WIN_S = 0.6e-3             # storm_watch's powerline window, kept so old and new comb shares compare
COMB_CHANCE = 2 * COMB_WIN_S / MAINS_HALF_S   # 0.144
COMB_MIN_N = 10                 # fewer events: no comb share (None)
COMB_TRUST_MAX = 0.25           # plan step 6: grey a number whose comb share is above this
GATED_TRUST_MAX = 0.20          # ...or whose gated (blanked) time share is above this
COINC_WIN_S = 0.5e-3            # two-band coincidence half window (census v2)
MINUTE_S = 60.0
MIN_PART_S = 10.0               # a trailing partial minute shorter than this gets no row
TAIL_A_S, TAIL_B_S = 1e-3, 5e-3  # tail score window after the peak

SIDECAR_KEYS = {               # whitelist: recorder settings only, never anything positional
    "utc": "utc_start", "secs": "sidecar_secs", "antenna": "antenna", "agc": "agc", "IFGR": "ifgr_db",
    "RFGR": "rfgr_db", "overflows": "overflows", "timeouts": "timeouts", "ntp_synced": "ntp_synced",
    "edges_per_s": "sidecar_edges_per_s", "hum_share_120hz": "sidecar_hum_share_120hz"}

EVENT_COLUMNS = ("i", "t_start_s", "t_peak_s", "t_end_s", "t_half_s", "peak_db", "width_us", "rise_us",
                 "floor_dbfs", "conf_db", "conf", "hum", "hum_block", "block_R", "coinc_hi", "tail_db")


# ----------------------------------------------------------------------------- small measures

def comb_share(t, min_n: int = COMB_MIN_N):
    """Share of consecutive intervals of the sorted times ``t`` (s) within 0.6 ms of a multiple of 8.333 ms.

    Independent (Poisson) events give 0.144; residual buzz pushes it towards 1. None when fewer than ``min_n``
    events, because a share of a handful of intervals says nothing."""
    t = np.sort(np.asarray(t, float))
    if t.size < min_n:
        return None
    ph = np.diff(t) / MAINS_HALF_S
    return float(np.mean(np.abs(ph - np.round(ph)) * MAINS_HALF_S < COMB_WIN_S))


def coincident(t_lo, t_hi, win: float = COINC_WIN_S) -> np.ndarray:
    """Bool per ``t_lo``: is any ``t_hi`` within +-win? Both in s; t_hi need not be sorted."""
    t_lo = np.asarray(t_lo, float)
    t_hi = np.sort(np.asarray(t_hi, float))
    if t_lo.size == 0 or t_hi.size == 0:
        return np.zeros(t_lo.size, bool)
    j = np.searchsorted(t_hi, t_lo)
    jl = np.clip(j - 1, 0, t_hi.size - 1)
    jr = np.clip(j, 0, t_hi.size - 1)
    return np.minimum(np.abs(t_hi[jl] - t_lo), np.abs(t_hi[jr] - t_lo)) <= win


def chance_rate(n_lo: int, n_hi: int, secs: float, win: float = COINC_WIN_S) -> float:
    """Coincidences per second two independent trains of n_lo and n_hi events in ``secs`` give by chance:
    n_lo x (n_hi / secs) x 2 win / secs (each low event has a 2 win window that a high event hits at rate n_hi/secs)."""
    return n_lo * n_hi * 2 * win / secs ** 2 if secs > 0 else 0.0


def hi_band_peaks(pc: np.ndarray, fs: float = dt.FS) -> np.ndarray:
    """Peak times (s) of impulses in the 16-30 kHz smoothed power ``pc``, with the detector's own rule (median
    floor over 4 s, +10 dB start, +7 dB / 100 us end, 5 ms cap, 0.5 ms merge). No mains gate: buzz barely reaches
    this band, which is the point of using it as the independent half of the coincidence."""
    block_s = dt.BLOCK_N / fs
    hop_n = int(round(dt.FLOOR_HOP_S / block_s)) * dt.BLOCK_N
    fl, _ = dt.sliding_floor(dt.block_means(pc), block_s)
    fsamp = dt.floor_per_sample(fl, len(pc), hop_n)
    s, e, _ = dt.find_impulses(pc, fsamp, fs)
    return np.array([dt.peak_index(pc, a, b) for a, b in zip(s, e)], float) / fs


def tail_db(p: np.ndarray, t_peak: np.ndarray, floor_dbfs: np.ndarray, fs: float = dt.FS) -> np.ndarray:
    """Tail score per event (UNVALIDATED): mean of ``p`` from 1 to 5 ms after the peak over the event's floor, dB.
    NaN when the window runs off the end of ``p``."""
    out = np.full(len(t_peak), np.nan)
    a_n, b_n = int(round(TAIL_A_S * fs)), int(round(TAIL_B_S * fs))
    for i, (tp, fl) in enumerate(zip(t_peak, floor_dbfs)):
        k = int(round(tp * fs))
        if k + b_n <= len(p):
            m = float(np.mean(p[k + a_n:k + b_n], dtype=np.float64))
            out[i] = 10 * math.log10(max(m, 1e-30)) - fl
    return out


def overlap(a0: float, a1: float, b0: float, b1: float) -> float:
    """Length of the overlap of [a0, a1) and [b0, b1)."""
    return max(0.0, min(a1, b1) - max(a0, b0))


def _union(lo: np.ndarray, hi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Merge intervals [lo, hi) into disjoint sorted intervals (touching or overlapping ones join)."""
    keep = hi > lo
    lo, hi = lo[keep], hi[keep]
    if lo.size == 0:
        return lo, hi
    o = np.argsort(lo, kind="stable")
    lo, hi = lo[o], hi[o]
    reach = np.maximum.accumulate(hi)
    new = np.ones(lo.size, bool)
    new[1:] = lo[1:] > reach[:-1]
    return lo[new], np.maximum.reduceat(hi, np.flatnonzero(new))


def reject_intervals(trains: list, secs: float, win: float = mg.WIN_S) -> tuple[np.ndarray, np.ndarray]:
    """Disjoint [lo, hi) intervals (s) of the capture inside ANY train's reject window: the time ``mainsgate.judge``
    would reject a weak impulse in. ``trains`` are ``mainsgate.Train`` objects (or anything with f, t_ref, c,
    track_t, track_off, t_lo, t_hi). Per train, each pulse sits at t_ref + c + k/f + offset (the tracked phase,
    interpolated at the pulse), its window is +-win, clipped to the train's span [t_lo, t_hi) and to [0, secs).

    Why a union and not a sum: in one 10 s block the 'whole', 'segment' and 'burst' passes often find trains
    0.1-0.25 ms apart (verifier, vlf_20261103_0430Z minute 2: five trains, summed 0.24 of the minute, union about
    0.19), so their +-0.2 ms windows overlap and a sum counts the same blanked time two to five times."""
    los, his = [], []
    for tr in trains:
        f = float(tr.f)
        t_lo = max(0.0, float(tr.t_lo))
        t_hi = secs if not math.isfinite(float(tr.t_hi)) else min(secs, float(tr.t_hi))
        if not (f > 0 and t_hi > t_lo):
            continue
        base = float(tr.t_ref) + float(tr.c)
        k0 = math.floor((t_lo - win - base) * f) - 1
        k1 = math.ceil((t_hi + win - base) * f) + 1
        tn = base + np.arange(k0, k1 + 1) / f
        tt = np.asarray(getattr(tr, "track_t", np.zeros(0)), float)
        if tt.size:
            tn = tn + np.interp(tn, tt, np.asarray(tr.track_off, float))
        los.append(np.clip(tn - win, t_lo, t_hi))
        his.append(np.clip(tn + win, t_lo, t_hi))
    if not los:
        return np.zeros(0), np.zeros(0)
    return _union(np.concatenate(los), np.concatenate(his))


def covered_seconds(lo: np.ndarray, hi: np.ndarray, t0: float, t1: float) -> float:
    """Seconds of [t0, t1) covered by the disjoint intervals [lo, hi)."""
    if lo.size == 0:
        return 0.0
    return float(np.clip(np.minimum(hi, t1) - np.maximum(lo, t0), 0.0, None).sum())


def trains_from_info(dicts: list[dict]) -> list:
    """``mainsgate.Train`` objects rebuilt from pipeline info train dicts (f, phase_ms, span_s, t_ref_s). The 1 s
    phase track is not in the info, so these are the untracked trains; used only when the live objects are not
    available."""
    out = []
    for d in dicts:
        lo, hi = d["span_s"]
        out.append(mg.Train(f=float(d["f"]), t_ref=float(d["t_ref_s"]), c=float(d["phase_ms"]) * 1e-3,
                            t_lo=float(lo), t_hi=float("inf") if hi is None else float(hi), kind=d.get("kind", "block")))
    return out


def gated_seconds_sum(trains: list, t0: float, t1: float, secs: float, win: float = mg.WIN_S) -> float:
    """The old SUMMED measure, kept for comparison only (``gated_frac_sum``): per train, its span clipped to the
    capture and to [t0, t1), times 2 win f, added over trains. Overlapping trains are counted once each, so it can
    exceed the blanked time (see ``reject_intervals``)."""
    s = 0.0
    for tr in trains:
        lo = max(0.0, float(tr.t_lo))
        hi = secs if not math.isfinite(float(tr.t_hi)) else min(secs, float(tr.t_hi))
        s += overlap(lo, hi, t0, t1) * min(1.0, 2 * win * float(tr.f))
    return min(s, t1 - t0)


@contextlib.contextmanager
def keep_trains(store: list):
    """While active, every ``mainsgate.gate`` call (as made by ``pipeline.run``) also appends its Train objects to
    ``store``. pipeline.run only reports trains as rounded dicts without their 1 s phase track; the union of
    reject windows needs the tracked objects. The original function is always restored."""
    orig = mg.gate

    def wrapped(*a, **k):
        rej, esc, trains = orig(*a, **k)
        store.extend(trains)
        return rej, esc, trains
    mg.gate = wrapped
    try:
        yield store
    finally:
        mg.gate = orig


def trust(comb, gated_frac: float) -> list[str]:
    """Reasons a rate is not to be trusted (empty list = trusted): residual comb share above 0.25, or more than
    20 % of the time blanked by the mains gate."""
    why = []
    if comb is not None and comb > COMB_TRUST_MAX:
        why.append("comb_share")
    if gated_frac > GATED_TRUST_MAX:
        why.append("gated_frac")
    return why


def _f(x, nd: int):
    """Round to ``nd`` digits for output; None for NaN/inf/None (JSON has no NaN)."""
    if x is None:
        return None
    x = float(x)
    return round(x, nd) if math.isfinite(x) else None


# ----------------------------------------------------------------------------- rows

def rate_row(t0: float, t1: float, ev: dict, coinc: np.ndarray, hi_t: np.ndarray, blocks: list[dict],
             rej: tuple[np.ndarray, np.ndarray], trains: list, secs: float) -> dict:
    """All rate numbers for the span [t0, t1) of a capture (one minute or the whole file). ``rej`` is the union of
    reject windows (``reject_intervals``); ``trains`` only feed the old summed measure kept as gated_frac_sum."""
    dur = t1 - t0
    m = (ev["t_start"] >= t0) & (ev["t_start"] < t1)
    n = int(m.sum())
    n_hi = int(((hi_t >= t0) & (hi_t < t1)).sum())
    n_co = int(coinc[m].sum())
    hum_s = sum(overlap(b["t0_s"], b["t0_s"] + b["dur_s"], t0, t1) for b in blocks if b["hum"])
    gfrac = covered_seconds(rej[0], rej[1], t0, t1) / dur
    gsum = gated_seconds_sum(trains, t0, t1, secs) / dur
    comb = comb_share(ev["t_start"][m])
    per_min = 60.0 / dur
    open_s = dur * (1.0 - gfrac)
    why = trust(comb, gfrac)
    return {"t0_s": _f(t0, 3), "dur_s": _f(dur, 3), "n_events": n, "rate_per_min": _f(n * per_min, 2),
            "rate_open_per_min": _f(n * 60.0 / open_s, 2) if open_s > 0 else None,
            "n_conf": int(ev["conf"][m].sum()), "conf_per_min": _f(ev["conf"][m].sum() * per_min, 2),
            "gated_frac": _f(gfrac, 4), "gated_frac_sum": _f(gsum, 4), "hum_frac": _f(hum_s / dur, 4),
            "comb_share": _f(comb, 4), "comb_chance": _f(COMB_CHANCE, 4),
            "v2_n_lo": n, "v2_n_hi": n_hi, "v2_n": n_co, "v2_per_min": _f(n_co * per_min, 2),
            "v2_chance_per_min": _f(chance_rate(n, n_hi, dur) * 60.0, 4),
            "trusted": not why, "untrusted_because": why}


def minute_spans(secs: float, minute_s: float = MINUTE_S, min_part_s: float = MIN_PART_S) -> list[tuple[float, float]]:
    """[t0, t1) of each whole minute, plus a trailing partial minute if it is at least ``min_part_s`` long."""
    out, t = [], 0.0
    while t + minute_s <= secs + 1e-9:
        out.append((t, t + minute_s))
        t += minute_s
    if secs - t >= min_part_s:
        out.append((t, secs))
    return out


def capture_id(path: str) -> str:
    """Stable output name: ``vlf_YYYYMMDD_HHMMZ`` for banked captures (date folder + vlf_HHMMZ), else the stem."""
    path = os.path.expanduser(path)
    stem = os.path.splitext(os.path.basename(path))[0]
    parent = os.path.basename(os.path.dirname(path))
    if re.fullmatch(r"\d{8}", parent) and re.fullmatch(r"vlf_\d{4}Z", stem):
        return f"vlf_{parent}_{stem[4:]}"
    return stem


def capture_block(path: str, info: dict) -> dict:
    """The capture's identity and recorder settings. Only the file's last folder and name are kept (no home path),
    and the sidecar goes through ``SIDECAR_KEYS`` so nothing positional can leak into a session file."""
    path = os.path.expanduser(path)
    side = info["sidecar"]
    out = {"id": capture_id(path), "file": f"{os.path.basename(os.path.dirname(path))}/{os.path.basename(path)}",
           "secs": _f(info["secs"], 3), "fs_hz": _f(info["fs"], 3), "fc_hz": _f(info["fc"], 3)}
    for k, name in SIDECAR_KEYS.items():
        if k in side:
            v = side[k]
            out[name] = v if isinstance(v, (str, bool)) or v is None else _f(v, 4)
    out.setdefault("utc_start", None)
    exp = side.get("secs")
    out["complete"] = bool(exp is None or abs(info["secs"] - float(exp)) < 1.0)
    return out


# ----------------------------------------------------------------------------- the whole capture

def event_blocks(t_start: np.ndarray, flags: list[dict]) -> tuple[np.ndarray, np.ndarray]:
    """(hum block index, that block's R) per event start. An event outside every flag block (the flag skips a
    trailing piece shorter than its minimum) gets block -1 and R NaN instead of the last block's numbers."""
    t_start = np.asarray(t_start, float)
    blk = np.full(t_start.size, -1, int)
    R = np.full(t_start.size, np.nan)
    if not flags or t_start.size == 0:
        return blk, R
    t0 = np.array([b["t0_s"] for b in flags], float)
    t1 = t0 + np.array([b["dur_s"] for b in flags], float)
    j = np.searchsorted(t0, t_start, side="right") - 1
    ok = (j >= 0) & (t_start < t1[np.clip(j, 0, None)])
    blk[ok] = j[ok]
    R[ok] = np.array([flags[k]["R"] for k in j[ok]], float)
    return blk, R


def analyze_iq(x: np.ndarray, fs: float = dt.FS) -> tuple[dict, dict]:
    """Run the gated v2 chain on IQ ``x`` and build (report, events). ``report`` holds summary, minutes, hum blocks,
    gate and detector parts of the session (no capture block); ``events`` the catalogue columns as arrays."""
    secs = len(x) / fs
    flags = hf.flag_iq(x, fs)
    p, pc = dt.band_powers(x, fs)
    hi_t = hi_band_peaks(pc, fs)
    p_raw = p.copy()                     # pipeline.run blanks p in place; the tail score reads the unblanked power
    live = []
    with keep_trains(live):
        ev, info = pl.run(x, fs, flags=flags, powers=(p, pc))
    coinc = coincident(ev["t_peak"], hi_t)
    tails = tail_db(p_raw, ev["t_peak"], ev["floor_dbfs"], fs)
    del p, pc, p_raw
    blk, block_R = event_blocks(ev["t_start"], flags)
    if not live and info["n_trains"]:    # gate reached by another route: fall back to untracked trains
        live = trains_from_info([dict(tr, t_ref_s=float(k)) for k, lst in info["trains"].items() for tr in lst])
    rej = reject_intervals(live, secs)
    trains = sorted((dict(tr, t_ref_s=float(k)) for k, lst in info["trains"].items() for tr in lst),
                    key=lambda d: (d["t_ref_s"], d["span_s"][0], d["phase_ms"]))
    for tr in trains:
        tr["span_s"] = [_f(v, 3) for v in tr["span_s"]]
    rows = [rate_row(a, b, ev, coinc, hi_t, flags, rej, live, secs) for a, b in minute_spans(secs)]
    summ = rate_row(0.0, secs, ev, coinc, hi_t, flags, rej, live, secs)
    summ["n_minutes"] = len(rows)
    summ["n_minutes_untrusted"] = sum(not r["trusted"] for r in rows)
    hum_blocks = [{"t0_s": _f(b["t0_s"], 3), "dur_s": _f(b["dur_s"], 3), "hum": bool(b["hum"]),
                   "R": _f(b["R"], 4), "r_crit": _f(b["r_crit"], 4), "f_mains_hz": _f(b["f_mains"], 4),
                   "n_edges": int(b["n_edges"]), "n_locked": _f(b["n_locked"], 1)} for b in flags]
    gate = {"rejected": int(info["rejected"]), "escaped": int(info["escaped"]), "n_trains": int(info["n_trains"]),
            "train_kinds": {k: int(v) for k, v in sorted(info["train_kinds"].items())},
            "gated_frac_engine": _f(info["gated_frac"], 4), "floor_blocks_excluded": int(info["floor_blocks_excluded"]),
            "trains": [{"t_ref_s": _f(t["t_ref_s"], 3), "kind": t["kind"], "f_hz": _f(t["f"], 5),
                        "phase_ms": _f(t["phase_ms"], 3), "n": int(t["n"]), "median_db": _f(t["median_db"], 2),
                        "p": float(f"{t['p']:.3g}"), "span_s": t["span_s"]} for t in trains]}
    det = {"too_wide": int(info["too_long"]), "floor_dbfs_min": _f(info["floor_dbfs_min"], 2),
           "floor_dbfs_max": _f(info["floor_dbfs_max"], 2), "floor_held_s": _f(info["floor_held_s"], 2),
           "floor_none_s": _f(info["floor_none_s"], 2), "n_hi_impulses": int(len(hi_t))}
    events = {"t_start_s": ev["t_start"], "t_peak_s": ev["t_peak"], "t_end_s": ev["t_end"], "t_half_s": ev["t_half"],
              "peak_db": ev["peak_db"], "width_us": ev["width_s"] * 1e6, "rise_us": ev["rise_s"] * 1e6,
              "floor_dbfs": ev["floor_dbfs"], "conf_db": ev["conf_db"], "conf": ev["conf"], "hum": ev["hum"],
              "hum_block": blk, "block_R": block_R, "coinc_hi": coinc, "tail_db": tails}
    return {"summary": summ, "minutes": rows, "hum_blocks": hum_blocks, "gate": gate, "detector": det}, events


def session_doc(capture: dict, rep: dict, n_events: int) -> dict:
    """Assemble the session JSON document (key order fixed, so its text is deterministic)."""
    return {"schema": SCHEMA_ID, "engine_version": sw.__version__, "detector_version": pl.VERSION,
            "report_version": REPORT_VERSION, "capture": capture,
            "rules": {"minute_s": MINUTE_S, "comb_window_ms": COMB_WIN_S * 1e3, "comb_chance": _f(COMB_CHANCE, 4),
                      "comb_trust_max": COMB_TRUST_MAX, "gated_trust_max": GATED_TRUST_MAX,
                      "coinc_window_ms": COINC_WIN_S * 1e3, "conf_db": dt.CONF_DB, "reject_window_ms": mg.WIN_S * 1e3,
                      "escape_db": mg.ESCAPE_DB},
            "summary": rep["summary"], "minutes": rep["minutes"], "hum_blocks": rep["hum_blocks"],
            "gate": rep["gate"], "detector": rep["detector"],
            "events": {"file": f"{capture['id']}.events.csv", "n": n_events, "columns": list(EVENT_COLUMNS)}}


def _cell(name: str, v) -> str:
    """One CSV cell with fixed formatting per column (deterministic text)."""
    if name in ("conf", "hum", "coinc_hi"):
        return "1" if v else "0"
    if name in ("i", "hum_block"):
        return str(int(v))
    v = float(v)
    if not math.isfinite(v):
        return ""
    nd = 6 if name.endswith("_s") else 1 if name.endswith("_us") else 2 if name == "tail_db" else 4 if name == "block_R" else 2
    return f"{v:.{nd}f}"


def events_csv(events: dict) -> str:
    """The catalogue as CSV text, one row per event, columns ``EVENT_COLUMNS``."""
    n = len(events["t_start_s"])
    lines = [",".join(EVENT_COLUMNS)]
    for i in range(n):
        lines.append(",".join(_cell(c, i if c == "i" else events[c][i]) for c in EVENT_COLUMNS))
    return "\n".join(lines) + "\n"


def dumps(doc: dict) -> str:
    """Session JSON text: compact per row, indented by section, newline at the end; NaN is never written."""
    return json.dumps(doc, indent=1, allow_nan=False) + "\n"


def analyze_file(path: str) -> tuple[dict, str]:
    """Whole capture -> (session document, events CSV text). Reads the file once as complex64 (300 MB for 600 s;
    never complex128) through ``io.read_slice``."""
    info = sio.capture_info(path)
    secs = info["n_samples"] / info["fs"]
    x = sio.read_slice(path, 0.0, secs, fs=info["fs"])
    rep, events = analyze_iq(x, info["fs"])
    del x
    doc = session_doc(capture_block(path, info), rep, len(events["t_start_s"]))
    return doc, events_csv(events)


def write_outputs(doc: dict, csv_text: str, out_dir: str) -> tuple[str, str]:
    """Write ``<id>.session.json`` and ``<id>.events.csv`` into out_dir; returns both paths."""
    os.makedirs(out_dir, exist_ok=True)
    cid = doc["capture"]["id"]
    js = os.path.join(out_dir, f"{cid}.session.json")
    cs = os.path.join(out_dir, f"{cid}.events.csv")
    with open(js, "w") as f:
        f.write(dumps(doc))
    with open(cs, "w") as f:
        f.write(csv_text)
    return js, cs


CENSUS_COLUMNS = ("id", "file", "utc_start", "secs", "status", "n_events", "rate_per_min", "rate_open_per_min",
                  "conf_per_min", "gated_frac", "hum_frac", "comb_share", "v2_per_min", "v2_chance_per_min",
                  "trusted", "n_minutes_untrusted", "rejected", "escaped", "n_trains", "too_wide", "floor_dbfs_min",
                  "floor_dbfs_max", "detector_version")


def census_row(doc: dict) -> dict:
    """One census CSV row from a session document."""
    s, c = doc["summary"], doc["capture"]
    return {"id": c["id"], "file": c["file"], "utc_start": c.get("utc_start"), "secs": c["secs"],
            "status": "ok" if c["complete"] else "short", "n_events": s["n_events"],
            "rate_per_min": s["rate_per_min"], "rate_open_per_min": s["rate_open_per_min"],
            "conf_per_min": s["conf_per_min"], "gated_frac": s["gated_frac"], "hum_frac": s["hum_frac"],
            "comb_share": s["comb_share"], "v2_per_min": s["v2_per_min"], "v2_chance_per_min": s["v2_chance_per_min"],
            "trusted": int(s["trusted"]), "n_minutes_untrusted": s["n_minutes_untrusted"],
            "rejected": doc["gate"]["rejected"], "escaped": doc["gate"]["escaped"], "n_trains": doc["gate"]["n_trains"],
            "too_wide": doc["detector"]["too_wide"], "floor_dbfs_min": doc["detector"]["floor_dbfs_min"],
            "floor_dbfs_max": doc["detector"]["floor_dbfs_max"], "detector_version": doc["detector_version"]}


def census_csv(rows: list[dict]) -> str:
    """Census rows as CSV text (empty cell for None)."""
    lines = [",".join(CENSUS_COLUMNS)]
    for r in rows:
        lines.append(",".join("" if r.get(k) is None else str(r.get(k)) for k in CENSUS_COLUMNS))
    return "\n".join(lines) + "\n"


def find_captures(*dirs: str) -> list[str]:
    """Every ``vlf_*.cs16`` under the given folders (recursive), sorted by path."""
    out = []
    for d in dirs:
        d = os.path.expanduser(d)
        if os.path.isfile(d):
            out.append(d)
            continue
        for root, _, files in os.walk(d):
            out += [os.path.join(root, f) for f in files if f.startswith("vlf_") and f.endswith(".cs16")]
    return sorted(set(out))


# ----------------------------------------------------------------------------- schema check (no jsonschema dep)

def load_schema(path: str | None = None) -> dict:
    """The session JSON schema (sferic_watch/schema.json): from ``path`` if given, else as package data through
    importlib.resources (works from a zip or wheel as long as the build ships schema.json; the in-repo file
    otherwise)."""
    if path is not None:
        with open(path) as f:
            return json.load(f)
    try:
        from importlib import resources
        return json.loads(resources.files("sferic_watch").joinpath("schema.json").read_text())
    except (OSError, ModuleNotFoundError, TypeError):
        with open(SCHEMA_PATH) as f:
            return json.load(f)


_TYPES = {"object": dict, "array": list, "string": str, "boolean": bool, "null": type(None)}


def _is(v, t: str) -> bool:
    if t == "integer":
        return isinstance(v, int) and not isinstance(v, bool)
    if t == "number":
        return isinstance(v, (int, float)) and not isinstance(v, bool)
    return isinstance(v, _TYPES[t])


def validate(doc, schema: dict, path: str = "$", defs: dict | None = None) -> list[str]:
    """Check ``doc`` against the subset of JSON Schema that schema.json uses (type, const, enum, required,
    properties, additionalProperties, items, minimum, maximum, minItems, $ref to $defs). Returns a list of errors
    (empty = valid). Written here so the package needs no jsonschema dependency."""
    defs = schema.get("$defs", {}) if defs is None else defs
    if "$ref" in schema:
        return validate(doc, defs[schema["$ref"].split("/")[-1]], path, defs)
    err = []
    t = schema.get("type")
    if t is not None:
        ts = t if isinstance(t, list) else [t]
        if not any(_is(doc, x) for x in ts):
            return [f"{path}: expected {t}, got {type(doc).__name__}"]
    if "const" in schema and doc != schema["const"]:
        err.append(f"{path}: expected {schema['const']!r}")
    if "enum" in schema and doc not in schema["enum"]:
        err.append(f"{path}: {doc!r} not in {schema['enum']}")
    if isinstance(doc, (int, float)) and not isinstance(doc, bool):
        if "minimum" in schema and doc < schema["minimum"]:
            err.append(f"{path}: {doc} < {schema['minimum']}")
        if "maximum" in schema and doc > schema["maximum"]:
            err.append(f"{path}: {doc} > {schema['maximum']}")
    if isinstance(doc, dict):
        props = schema.get("properties", {})
        for k in schema.get("required", []):
            if k not in doc:
                err.append(f"{path}: missing {k}")
        for k, v in doc.items():
            if k in props:
                err += validate(v, props[k], f"{path}.{k}", defs)
            elif schema.get("additionalProperties") is False:
                err.append(f"{path}: unexpected key {k}")
            elif isinstance(schema.get("additionalProperties"), dict):
                err += validate(v, schema["additionalProperties"], f"{path}.{k}", defs)
    if isinstance(doc, list):
        if len(doc) < schema.get("minItems", 0):
            err.append(f"{path}: fewer than {schema['minItems']} items")
        if "items" in schema:
            for i, v in enumerate(doc):
                err += validate(v, schema["items"], f"{path}[{i}]", defs)
    return err


PRIVACY_RE = re.compile(r"lat|lon|latitude|longitude|grid", re.IGNORECASE)


def privacy_hits(text: str) -> list[str]:
    """Substrings of ``text`` that the plan's privacy grep (lat|lon|latitude|longitude|grid, any case) would hit.
    Deliberately a plain substring match, as strict as the grep: field names avoid words that merely contain
    those letters."""
    return sorted({m.group(0) for m in PRIVACY_RE.finditer(text)})
