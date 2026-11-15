"""Score sferic detectors against GOES GLM lightning (the per-capture npz files written by bench/glm_fetch.py).

Usage (heavy runs: wrap in systemd-run --user --scope -q -p MemoryMax=4G nice -n 10):
  python bench/glm_truth.py events [--workers 2] [--capture ID ...]   # run july/onair_bp/v2_detect/v2 on captures
  python bench/glm_truth.py match  [--out results.json]               # clock fit, coincidence, chance, bootstrap

Inputs:
  * GLM: $SFERIC_GLM_DIR (default ~/.local/share/sferic-watch/glm)/glm_<capture-id>.npz. These private files hold the
    flash/group times and each one's great-circle distance from the station (flash_dist_km, group_dist_km). This
    module never reads the station file. It reads times, distances, energies and areas; group lat/lon are read only
    inside ``load_capture`` and reduced at once to the flash-to-satellite light time (for the light-time test), then
    dropped. Nothing positional is written or printed: every output is an aggregate (counts, rates, lags, histograms
    with distance bins of 250 km or wider, no bearings).
  * Detector events: $SFERIC_GLM_EVENTS (default ./glm_events), one npz per capture written by ``events``; times are
    seconds from sample 0, plus the sidecar's UTC start ``t0``.

Method (what each number means):
  CLOCK   Detector event times (t0 + t) minus GLM times minus the waveguide delay d / v, histogrammed over +-2 s in
          1 ms bins (a cross-correlation of two point processes). The v2 events with 16-30 kHz confirmation and GLM
          flashes/groups within ``CLOCK_KM`` give the clean peak. The lag is fitted pooled and per capture; the frozen
          lag model is then used for every detector.
  MATCH   A GLM flash is "detected" when a detector event lies within +-w of (flash first-event time + lag + d/v)
          ("first"), or anywhere in [first - w, last + w] shifted the same way ("span"), or within +-w_g of one of its
          groups ("group"). A detector event is "matched" when a GLM flash (first-event rule) or group is within the
          window. w comes from the measured width of the lag peak.
  CHANCE  The same counts with the GLM times shifted by many offsets of +-5..60 s (never near the true lag). Chance
          is the mean over the shifts; the excess = observed - chance is the coincidence that is not luck.
  LAG     Default ``--lag-mode crossfit``: each capture's lag is fitted twice, on the v2_conf events of its even and
          of its odd 20 s blocks, and events of one parity are scored with the lag of the other, so no event is scored
          with a lag tuned on itself (``insample`` did that and flattered v2_conf by ~0.01; ``pooled`` = one lag).
  CI      Captures are not independent (one storm fills many consecutive captures; one day holds almost every flash
          within 500 km), so the headline interval is a moving-block bootstrap of 8 consecutive captures (ci95_block8).
          A day-level bootstrap (few clusters, crude) and the naive capture bootstrap (too narrow) are written too.
          The per-day table (by_day) shows how much a number rests on one storm.
  PRECISION proxy q = (matched - chance) / (1 - chance) of the events, matched = a GLM group closer than dmax km within
          +-w; a lower bound on the lightning share (GLM detection efficiency, GLM field of view, IC/CG).

Measured on the 151 banked captures (2026-11-14; scratch results.json of the GLM truth step):
  * v2_conf minus GLM group lag, raw, by distance: 26.1 ms at 0-500 km rising 3.75 +- 0.13 ms per 1000 km
    -> effective waveguide speed 0.89 c (0.86-0.92) for the 10 dB threshold crossing in 2.5-9 kHz; intercept 25.2 ms.
  * With d / (0.89 c) removed: pooled lag +25.9 ms (z ~1200), FWHM 7 ms after per-capture alignment; window +-8.5 ms
    (90 % of the core). Per-capture lag p25-p75 24.8-26.9 ms, p5-p95 17-33 ms (robust spread 1.5 ms).
  * Light time: the residual peak does not move with the flash-to-GEO light time. Peak-per-cell slope -0.03
    (block-bootstrap 95 % CI -0.19..0.05), within-capture slope +0.01 (-0.05..0.08); the satellite hypothesis is -1.
    The same group seen by G18 and G19 (light times 4-9 ms apart) carries the same time (bench/glm_fetch.py
    lighttime: slope -0.02). So GLM L2 times are referenced to the flash, not the satellite. The +26 ms is the
    recorder: the sidecar ``utc`` is stamped after the settle reads, so sample 0 (already buffered) is ~26 ms older
    than the stamp, varying by a few ms per capture.
  * 99 % of the GLM flashes within 500 km fall on one day (2026-11-06): the 0-500 km recall is that storm's number.
"""
from __future__ import annotations

import argparse
import datetime as dt
import glob
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

HOME = Path.home()
GLM_DIR = Path(os.environ.get("SFERIC_GLM_DIR", HOME / ".local/share/sferic-watch/glm"))
EVENTS_DIR = Path(os.environ.get("SFERIC_GLM_EVENTS", "glm_events"))
CAPTURE_GLOBS = [str(HOME / "dlayer_run/data/vlf/*/vlf_*.cs16"), str(HOME / "storm_live/vlf_*.cs16")]
DETECTORS = ("july", "onair_bp", "v2_detect", "v2")
FS = 62500.0

C_KM_S = 299792.458
V_FRAC = 0.99                 # default waveguide group speed / c for the d / v delay (refit by ``fit_speed``)
SCAN_S = 2.0                  # lag scan half-range
SCAN_STEP = 0.001             # lag scan bin
CLOCK_KM = 2000.0             # GLM flashes/groups used for the clock fit
LOCAL_S = 0.020               # per-capture lag searched within +-20 ms of the pooled lag
DIST_BINS = (0, 500, 1000, 2000, 3000, 4000, np.inf)
NIGHT_UTC = (22, 11)          # station-local night, UTC hours [22, 24) U [0, 11)
XFIT_BLOCK_S = 20.0           # cross-fitted lag: events in even 20 s blocks use the lag fitted on odd blocks, and back
BLOCK_CAPS = 8                # moving-block bootstrap: 8 consecutive captures (~4 h of one storm) per block
LT_EDGES = np.arange(0.117, 0.1301, 0.001)          # light-time test: GLM light-time bins (s)
LT_DEV_EDGES = np.array([-np.inf, -0.002, -0.001, 0.0, 0.001, 0.002, np.inf])   # light time minus capture median
LT_RES_HALF, LT_RES_STEP = 0.060, 0.0005            # residual histograms (lag removed), +-60 ms in 0.5 ms bins
LT_DBINS = (0, 500, 1000, 2000, 3000, 4000)


# ================================================================ captures and events

def capture_id(p: Path) -> str:
    """vlf_HHMMZ.cs16 in a YYYYMMDD dir -> YYYYMMDD_HHMMZ; vlf_YYYYMMDD_HHMMZ.cs16 -> YYYYMMDD_HHMMZ (as glm_fetch)."""
    stem = p.stem[4:] if p.stem.startswith("vlf_") else p.stem
    return stem if "_" in stem else f"{p.parent.name}_{stem}"


def list_captures():
    paths = sorted(set(sum((glob.glob(g) for g in CAPTURE_GLOBS), [])))
    return {capture_id(Path(p)): p for p in paths}


def _events_one(args):
    """Worker: one capture through the four detectors. Saves EVENTS_DIR/<id>.npz; returns a short status dict."""
    cid, path = args
    out_f = EVENTS_DIR / f"{cid}.npz"
    if out_f.exists():
        return {"id": cid, "skipped": True}
    import harness as hn
    from sferic_watch import io as swio, detect as dtc, humflag as hf, pipeline as pl
    hn.load_plugins()
    info = swio.capture_info(path)
    t0 = dt.datetime.fromisoformat(info["sidecar"]["utc"]).timestamp()
    x = swio.read_slice(path, 0, info["secs"], dc="none")
    secs = len(x) / FS
    out = {"t0": t0, "secs": secs}
    cpu = {}
    c = time.process_time()
    t, pk = hn.july(x)
    cpu["july"] = time.process_time() - c
    out.update(july_t=t, july_pk=pk)
    if hn.onair_available():
        t, pk, oi = hn.onair_bp(x)
        cpu["onair_bp"] = oi["cpu_self_s"]
        out.update(onair_bp_t=t, onair_bp_pk=pk, onair_bp_ok=True)
    else:
        out.update(onair_bp_t=np.zeros(0), onair_bp_pk=np.zeros(0), onair_bp_ok=False)
    c = time.process_time()
    ev, _ = dtc.detect_iq(x)
    cpu["v2_detect"] = time.process_time() - c
    out.update(v2_detect_t=ev["t_start"], v2_detect_pk=ev["peak_db"], v2_detect_conf=ev["conf"],
               v2_detect_tpeak=ev["t_peak"])
    c = time.process_time()
    flags = hf.flag_iq(x)
    ev, vi = pl.run(x, flags=flags)
    cpu["v2"] = time.process_time() - c
    out.update(v2_t=ev["t_start"], v2_pk=ev["peak_db"], v2_conf=ev["conf"], v2_tpeak=ev["t_peak"], v2_hum=ev["hum"],
               v2_width=ev["width_s"],
               hum_iv=np.array([(b["t0_s"], b["t0_s"] + b["dur_s"]) for b in flags if b["hum"]], float).reshape(-1, 2),
               hum_frac=vi["hum_frac"], gated_frac=vi["gated_frac"], cpu=json.dumps(cpu))
    del x
    EVENTS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = out_f.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, out_f)
    return {"id": cid, "n": {d: int(len(out[f"{d}_t"])) for d in DETECTORS}, "cpu": cpu}


def cmd_events(a):
    caps = list_captures()
    if a.capture:
        caps = {k: v for k, v in caps.items() if k in set(a.capture)}
    jobs = sorted(caps.items())
    t = time.time()
    if a.workers <= 1:
        it = map(_events_one, jobs)
    else:
        import multiprocessing as mp
        pool = mp.get_context("spawn").Pool(min(a.workers, 2))
        it = pool.imap_unordered(_events_one, jobs, chunksize=1)
    for k, r in enumerate(it, 1):
        print(f"[{k}/{len(jobs)} {time.time() - t:.0f}s] {json.dumps(r)}", flush=True)


# ================================================================ pure helpers (tested on synthetic data)

SAT_LON = {19: -75.2, 18: -137.0}     # GOES-East / GOES-West sub-satellite longitudes, deg
R_EARTH_EQ_KM, R_GEO_KM = 6378.137, 42164.0


def nearest(sorted_a: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(index of the nearest element of sorted ``sorted_a``, |q - it|) per q; (-1, inf) when the array is empty."""
    q = np.asarray(q, float)
    if sorted_a.size == 0:
        return np.full(q.size, -1), np.full(q.size, np.inf)
    i = np.searchsorted(sorted_a, q)
    lo = np.clip(i - 1, 0, sorted_a.size - 1)
    hi = np.clip(i, 0, sorted_a.size - 1)
    dlo, dhi = np.abs(q - sorted_a[lo]), np.abs(q - sorted_a[hi])
    pick = np.where(dhi < dlo, hi, lo)
    return pick, np.minimum(dlo, dhi)


def min_in_window(sorted_t: np.ndarray, vals: np.ndarray, q: np.ndarray, w: float, chunk: int = 20000):
    """For each q: (min of vals over sorted_t within +-w, index of that element); (inf, -1) when none is in range."""
    q = np.asarray(q, float)
    best = np.full(q.size, np.inf)
    arg = np.full(q.size, -1, np.int64)
    if not sorted_t.size or not q.size:
        return best, arg
    for c in range(0, q.size, chunk):
        qq = q[c:c + chunk]
        lo = np.searchsorted(sorted_t, qq - w, "left")
        hi = np.searchsorted(sorted_t, qq + w, "right")
        cnt = hi - lo
        n = int(cnt.sum())
        if not n:
            continue
        rep = np.repeat(np.arange(qq.size), cnt)
        idx = np.arange(n) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        o = np.lexsort((vals[idx], rep))
        rep, idx = rep[o], idx[o]
        first = np.r_[True, rep[1:] != rep[:-1]]
        best[c + rep[first]] = vals[idx[first]]
        arg[c + rep[first]] = idx[first]
    return best, arg


def lag_hist(det: np.ndarray, ref: np.ndarray, half: float = SCAN_S, step: float = SCAN_STEP,
             chunk: int = 4000) -> np.ndarray:
    """Cross-correlation of two point processes: histogram of det - ref over every pair within +-half.

    Bins are ``step`` wide and centred on k*step, k = -K..K (K = half/step), so index K is lag 0."""
    K = int(round(half / step))
    edges = (np.arange(-K, K + 2) - 0.5) * step
    h = np.zeros(2 * K + 1, np.int64)
    ref = np.sort(np.asarray(ref, float))
    det = np.asarray(det, float)
    if not det.size or not ref.size:
        return h
    for c in range(0, det.size, chunk):
        d = det[c:c + chunk]
        lo = np.searchsorted(ref, d - half - step)
        hi = np.searchsorted(ref, d + half + step)
        cnt = hi - lo
        n = int(cnt.sum())
        if not n:
            continue
        rep = np.repeat(np.arange(d.size), cnt)
        idx = np.arange(n) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        h += np.histogram(d[rep] - ref[idx], edges)[0]
    return h


def peak_fit(h: np.ndarray, step: float = SCAN_STEP, far_s: float = 0.3, core_s: float = 0.010,
             ped_s: tuple[float, float] = (0.015, 0.060), contain: float = 0.90,
             search: tuple[float, float] | None = None) -> dict:
    """Locate and size the coincidence peak of a lag histogram (bins centred on k*step, centre index = lag 0).

    bg: median of bins farther than ``far_s`` from the peak (the chance level per bin). The peak is the maximum of the
    3-bin smoothed histogram; its lag is the excess-weighted centroid over +-3 bins. Above the local pedestal (median
    excess at ``ped_s`` from the peak: correlated pairs from other groups/strokes of the same flash) the core is
    sized by its FWHM and by ``w``, the smallest half-width around the centroid holding ``contain`` of the core excess
    within +-``core_s``. ``z`` = core excess in +-w over its Poisson sigma. ``search`` (lo, hi) in s limits where the
    peak may be (a per-capture fit near the pooled lag; a +-2 s search on a sparse capture finds noise peaks)."""
    h = np.asarray(h, float)
    K = (len(h) - 1) // 2
    sm = np.convolve(h, np.ones(3) / 3, mode="same")
    lags = (np.arange(len(h)) - K) * step
    if search is None:
        k = int(np.argmax(sm))
    else:
        cand = np.nonzero((lags >= search[0]) & (lags <= search[1]))[0]
        k = int(cand[np.argmax(sm[cand])])
    far = np.abs(lags - lags[k]) > far_s
    bg = float(np.median(h[far])) if far.any() else 0.0
    ex = h - bg
    j = np.arange(max(0, k - 3), min(len(h), k + 4))
    wts = np.clip(ex[j], 0, None)
    lag = float((lags[j] * wts).sum() / wts.sum()) if wts.sum() > 0 else float(lags[k])
    dist = np.abs(lags - lag)
    ped_m = (dist >= ped_s[0]) & (dist <= ped_s[1])
    ped = float(np.median(ex[ped_m])) if ped_m.any() else 0.0
    core = ex - ped
    cm = dist <= core_s
    tot = float(np.clip(core[cm], 0, None).sum())
    w = core_s
    if tot > 0:
        for wi in np.arange(0, core_s + step / 2, step):
            if np.clip(core[dist <= wi + step / 2], 0, None).sum() >= contain * tot:
                w = float(wi + step / 2)
                break
    above = np.nonzero(sm - bg - ped >= 0.5 * (sm[k] - bg - ped))[0]
    run = above[(above >= k - 50) & (above <= k + 50)]
    lo_i, hi_i = k, k
    while lo_i - 1 in run:
        lo_i -= 1
    while hi_i + 1 in run:
        hi_i += 1
    win = dist <= w
    exc = float((h[win] - bg).sum())
    z = exc / np.sqrt(max(bg * win.sum(), 1.0))
    return {"lag_s": lag, "peak_bin_s": float(lags[k]), "bg_per_bin": bg, "pedestal_per_bin": ped,
            "fwhm_s": float((hi_i - lo_i + 1) * step), "w_s": w, "excess_in_w": exc,
            "core_excess_10ms": tot, "z": float(z)}


def in_intervals(t: np.ndarray, iv: np.ndarray) -> np.ndarray:
    """Bool per t: inside one of the sorted, non-overlapping [a, b) intervals ``iv`` (shape (n, 2))."""
    t = np.asarray(t, float)
    iv = np.asarray(iv, float).reshape(-1, 2)
    if not len(iv):
        return np.zeros(t.size, bool)
    i = np.searchsorted(iv[:, 0], t, "right") - 1
    ok = i >= 0
    out = np.zeros(t.size, bool)
    out[ok] = t[ok] < iv[i[ok], 1]
    return out


def merge_intervals(iv: np.ndarray) -> np.ndarray:
    """Sort and merge overlapping/touching [a, b) intervals."""
    iv = np.asarray(iv, float).reshape(-1, 2)
    if not len(iv):
        return iv
    iv = iv[np.argsort(iv[:, 0])]
    out = [list(iv[0])]
    for a, b in iv[1:]:
        if a <= out[-1][1]:
            out[-1][1] = max(out[-1][1], b)
        else:
            out.append([a, b])
    return np.array(out)


def light_time_s(lat: np.ndarray, lon: np.ndarray, sat_lon: np.ndarray) -> np.ndarray:
    """Flash-to-GEO light time (s) on a spherical Earth: slant range from (lat, lon) to a GEO satellite at sat_lon."""
    la, dl = np.radians(lat), np.radians(np.asarray(lon) - np.asarray(sat_lon))
    cg = np.cos(la) * np.cos(dl)
    r = np.sqrt(R_GEO_KM ** 2 + R_EARTH_EQ_KM ** 2 - 2 * R_GEO_KM * R_EARTH_EQ_KM * cg)
    return r / C_KM_S


def corrected(obs: float, chance: float, n: float) -> float:
    """Chance-corrected rate (obs/n - chance/n) / (1 - chance/n); nan when n == 0."""
    if n <= 0:
        return float("nan")
    r, c = obs / n, chance / n
    return (r - c) / (1 - c) if c < 1 else 0.0


def crossing(x: np.ndarray, y: np.ndarray, level: float) -> float | None:
    """First x where y (falling) crosses ``level``, linear interpolation; None if it never does."""
    for i in range(1, len(y)):
        if np.isfinite(y[i - 1]) and np.isfinite(y[i]) and y[i - 1] >= level > y[i]:
            return float(x[i - 1] + (level - y[i - 1]) * (x[i] - x[i - 1]) / (y[i] - y[i - 1]))
    return None


def resample_idx(rng, scheme: str, ncap: int, days: np.ndarray | None = None, L: int = BLOCK_CAPS) -> np.ndarray:
    """One bootstrap resample of capture positions 0..ncap-1 (positions in time order).

    "capture": captures with replacement (treats them as independent: too narrow when storms span many captures).
    "block": moving blocks of L consecutive captures (keeps the storm-to-storm correlation within ~L captures).
    "day": whole days with replacement (``days`` = day label per position; only a handful of clusters, so crude)."""
    if scheme == "capture":
        return rng.integers(0, ncap, ncap)
    if scheme == "block":
        L = max(1, min(L, ncap))
        st = rng.integers(0, ncap - L + 1, int(np.ceil(ncap / L)))
        return np.concatenate([np.arange(s, s + L) for s in st])[:ncap]
    if scheme == "day":
        u = np.unique(days)
        pick = rng.choice(u, u.size, replace=True)
        return np.concatenate([np.nonzero(days == d)[0] for d in pick])
    raise ValueError(scheme)


def centroid_peak(h: np.ndarray, mid: np.ndarray, bg_far: float = 0.020, search: float = 0.030, half_bins: int = 8):
    """Lag peak of a residual histogram: background = median of bins farther than ``bg_far`` from 0, peak = max of the
    5-bin smoothed excess within +-``search``, location = excess-weighted centroid over +-half_bins around it (the
    window follows the peak, so a shifted peak is not truncated). Returns (location s or nan, excess weight)."""
    h = np.asarray(h, float)
    far = np.abs(mid) > bg_far
    if h.sum() <= 0 or not far.any():
        return float("nan"), 0.0
    ex = h - float(np.median(h[far]))
    sm = np.convolve(ex, np.ones(5) / 5, "same")
    k = int(np.argmax(np.where(np.abs(mid) <= search, sm, -np.inf)))
    j = slice(max(k - half_bins, 0), k + half_bins + 1)
    w = np.clip(ex[j], 0, None)
    return (float((mid[j] * w).sum() / w.sum()) if w.sum() > 0 else float("nan")), float(w.sum())


def make_shifts(n: int, lo: float = 5.0, hi: float = 58.0, seed: int = 7) -> np.ndarray:
    """``n`` control shifts, half negative, |shift| in [lo, hi] s (well away from the true lag)."""
    rng = np.random.default_rng(seed)
    s = rng.uniform(lo, hi, n)
    s[1::2] *= -1
    return s


# ================================================================ per-capture data

SCORED = DETECTORS + ("v2_conf",)      # v2_conf: v2 events with the 16-30 kHz confirmation (the clock-fit set)
FINE_BINS = (0, 250, 500, 750, 1000, 1250, 1500, 1750, 2000, 2500, 3000, 3500, 4000, 5000, 6000, 8000, 10000, np.inf)
DBD_BINS = (0, 500, 1000, 1500, 2000, 3000, 4000, 6000, 8000, np.inf)
SPEED_BINS = (0, 500, 1000, 1500, 2000, 2500, 3000, 4000, 5000, 6000)
DB_BINS = (-np.inf, 12, 15, 20, 30, np.inf)
NCAT = (len(DIST_BINS) - 1) * 3 * 2
DBH_EDGES = np.arange(0.0, 61.0, 1.0)  # peak dB histogram bins (detector's own dB scale)


def load_capture(cid: str, latlon: bool = False) -> dict:
    """GLM flashes/groups (times, distances, energies; light time only when ``latlon``) and detector events.

    Group parent uids are mapped to flash indices. Lat/lon, when read, are reduced at once to the GLM light time and
    dropped: nothing positional leaves this function."""
    g = np.load(GLM_DIR / f"glm_{cid}.npz")
    e = np.load(EVENTS_DIR / f"{cid}.npz")
    t0, secs = float(e["t0"]), float(e["secs"])
    if abs(float(g["t0"]) - t0) > 1e-3:
        raise ValueError(f"{cid}: GLM npz t0 and events t0 disagree")
    uid = g["flash_uid"]
    o = np.argsort(uid)
    F = {"t": g["flash_t"], "t_last": g["flash_t_last"], "d": g["flash_dist_km"], "E": g["flash_energy_J"],
         "A": g["flash_area_m2"]}
    G = {"t": g["group_t"], "d": g["group_dist_km"], "f": o[np.searchsorted(uid[o], g["group_parent_uid"])],
         "sat": g["group_sat"]}
    if latlon:
        G["lt"] = light_time_s(g["group_lat"], g["group_lon"],
                               np.where(g["group_sat"] == 19, SAT_LON[19], SAT_LON[18]))
    gaps = sorted(set(np.concatenate([g["missing_g19"], g["missing_g18"]]).tolist()))
    det, pk = {}, {}
    for d in DETECTORS:
        det[d], pk[d] = np.asarray(e[f"{d}_t"], float), np.asarray(e[f"{d}_pk"], float)
    c = e["v2_conf"].astype(bool)
    det["v2_conf"], pk["v2_conf"] = det["v2"][c], pk["v2"][c]
    for d in det:
        oo = np.argsort(det[d], kind="stable")
        det[d], pk[d] = det[d][oo], pk[d][oo]
    hour = dt.datetime.fromtimestamp(t0, dt.timezone.utc).hour
    return {"id": cid, "t0": t0, "secs": secs, "F": F, "G": G, "det": det, "pk": pk,
            "gaps": merge_intervals(np.array([(s, s + 20.0) for s in gaps])),
            "hum_iv": merge_intervals(e["hum_iv"]), "hum_frac": float(e["hum_frac"]),
            "gated_frac": float(e["gated_frac"]), "onair_ok": bool(e["onair_bp_ok"]),
            "hour": hour, "night": bool(hour >= NIGHT_UTC[0] or hour < NIGHT_UTC[1])}


def capture_ids() -> list[str]:
    ev = {p.stem for p in EVENTS_DIR.glob("*.npz") if not p.stem.endswith(".tmp")}
    gl = {p.stem[4:] for p in GLM_DIR.glob("glm_*.npz")}
    return sorted(ev & gl)


# ================================================================ pass 1: clock

def _clock_one(args):
    cid, v = args
    C = load_capture(cid)
    G, F, t0 = C["G"], C["F"], C["t0"]
    out = {"id": cid}
    mg = G["d"] < CLOCK_KM
    ga = G["t"][mg] + G["d"][mg] / v - t0
    mf = F["d"] < CLOCK_KM
    fa = F["t"][mf] + F["d"][mf] / v - t0
    for d in SCORED:
        out[f"hg_{d}"] = lag_hist(C["det"][d], ga)
    tc = C["det"]["v2_conf"]
    par = (np.floor(tc / XFIT_BLOCK_S).astype(int) % 2)
    out["hg_par"] = np.array([lag_hist(tc[par == p], ga) for p in (0, 1)])   # v2_conf in even / odd 20 s blocks
    out["hf_v2_conf"] = lag_hist(C["det"]["v2_conf"], fa)
    out["hf_v2"] = lag_hist(C["det"]["v2"], fa)
    # raw lag (no waveguide delay) per distance bin, +-0.25 s, for the speed fit
    hs, dm = [], []
    for a, b in zip(SPEED_BINS[:-1], SPEED_BINS[1:]):
        m = (G["d"] >= a) & (G["d"] < b)
        hs.append(lag_hist(C["det"]["v2_conf"], G["t"][m] - t0, half=0.25))
        dm.append((float(G["d"][m].sum()), int(m.sum())))
    out["h_speed"] = np.array(hs)
    out["speed_dsum"] = np.array(dm)
    core = (F["t"] >= t0) & (F["t"] < t0 + C["secs"]) & (F["d"] < 4000)
    out["E4000"] = F["E"][core]
    return out


def _pool_map(fn, jobs, workers):
    if workers <= 1:
        return list(map(fn, jobs))
    import multiprocessing as mp
    with mp.get_context("spawn").Pool(min(workers, 2)) as pool:
        return pool.map(fn, jobs, chunksize=1)


def fit_speed(h_speed: np.ndarray, dsum: np.ndarray) -> dict:
    """Raw lag peak per distance bin, then lag = delta + d / v by weighted least squares (weights = core excess)."""
    rows = []
    for k in range(len(h_speed)):
        n = dsum[k, 1]
        if n == 0:
            continue
        pf = peak_fit(h_speed[k], far_s=0.1)
        rows.append({"bin_km": [SPEED_BINS[k], SPEED_BINS[k + 1]], "_mean_km": dsum[k, 0] / n,
                     "lag_ms": round(pf["lag_s"] * 1e3, 2), "core_excess": round(pf["core_excess_10ms"], 0),
                     "z": round(pf["z"], 1)})
    ok = [r for r in rows if r["z"] >= 5 and r["core_excess"] >= 20]
    # the mean group distance per bin is used for the fit but not written out (with public GLM data and the capture
    # times, per-bin mean distances would constrain the station position)
    res = {"bins": [{k: v for k, v in r.items() if not k.startswith("_")} for r in rows]}
    if len(ok) >= 3:
        x = np.array([r["_mean_km"] for r in ok])
        y = np.array([r["lag_ms"] for r in ok]) * 1e-3
        wt = np.sqrt(np.array([r["core_excess"] for r in ok]))
        A = np.vstack([np.ones_like(x), x]).T * wt[:, None]
        coef, *_ = np.linalg.lstsq(A, y * wt, rcond=None)
        resid = y - (coef[0] + coef[1] * x)
        cov = np.linalg.inv(A.T @ A) * (np.sum((resid * wt) ** 2) / max(len(x) - 2, 1))
        s, se = coef[1], np.sqrt(cov[1, 1])
        res.update({"delta_ms": round(coef[0] * 1e3, 2), "slope_ms_per_1000km": round(s * 1e6, 3),
                    "slope_se_ms_per_1000km": round(se * 1e6, 3),
                    "v_over_c": round(1 / (s * C_KM_S), 4) if s > 0 else None,
                    "v_over_c_range_1se": [round(1 / ((s + se) * C_KM_S), 4), round(1 / ((s - se) * C_KM_S), 4)
                                           if s - se > 0 else None],
                    "resid_ms": [round(r * 1e3, 2) for r in resid]})
    return res


def clock_stage(ids, v, workers):
    res = _pool_map(_clock_one, [(c, v) for c in ids], workers)
    by = {r["id"]: r for r in res}
    pooled = {d: np.sum([r[f"hg_{d}"] for r in res], axis=0) for d in SCORED}
    pg = peak_fit(pooled["v2_conf"])
    pf = peak_fit(np.sum([r["hf_v2_conf"] for r in res], axis=0), core_s=0.050, ped_s=(0.1, 0.2))
    pf_all = peak_fit(np.sum([r["hf_v2"] for r in res], axis=0), core_s=0.050, ped_s=(0.1, 0.2))
    per_det = {d: peak_fit(pooled[d]) for d in SCORED}
    per_cap = {}
    L = pg["lag_s"]
    per_par = {}
    for cid, r in by.items():
        per_cap[cid] = peak_fit(r["hg_v2_conf"], search=(L - LOCAL_S, L + LOCAL_S))
        per_par[cid] = [peak_fit(r["hg_par"][p], search=(L - LOCAL_S, L + LOCAL_S)) for p in (0, 1)]
    E = np.concatenate([r["E4000"] for r in res])
    speed = fit_speed(np.sum([r["h_speed"] for r in res], axis=0), np.sum([r["speed_dsum"] for r in res], axis=0))
    # pooled histogram around the peak, for the record (aggregate)
    K = (len(pooled["v2_conf"]) - 1) // 2
    kk = K + int(round(pg["lag_s"] / SCAN_STEP))
    shape = {int(j - kk): int(pooled["v2_conf"][j]) for j in range(kk - 30, kk + 31)}
    return {"hists": {cid: r["hg_par"] for cid, r in by.items()},
            "pooled_group": pg, "pooled_flash_first_v2conf": pf, "pooled_flash_first_v2": pf_all,
            "per_detector_group": per_det, "per_capture": per_cap, "per_capture_parity": per_par, "speed": speed,
            "energy_terciles_J": [float(np.percentile(E, 100 / 3)), float(np.percentile(E, 200 / 3))],
            "hist_v2conf_group_around_peak": shape}


def lag_model(clock: dict, min_z: float = 8.0, min_excess: float = 15.0, mode: str = "crossfit") -> dict:
    """Freeze the lag. ``pooled``: one constant for every capture. ``insample``: each capture's own v2_conf peak (the
    lag is then tuned on events that are scored: it flatters v2_conf a little). ``crossfit`` (default): per capture
    and per 20 s block parity, the lag fitted on the OTHER parity's v2_conf events, so no event is scored with a lag
    fitted on itself. A weak peak (z < min_z or < min_excess pairs; half thresholds for one parity) falls back to the
    pooled lag. per_capture_s maps capture -> (lag for events in even blocks, lag for events in odd blocks)."""
    L = clock["pooled_group"]["lag_s"]

    def ok(p, f=1.0):
        return p["z"] >= min_z * f and p["excess_in_w"] >= min_excess * f
    sig = {c: p["lag_s"] for c, p in clock["per_capture"].items() if ok(p)}
    lags = np.array(list(sig.values()))
    spread = float(1.4826 * np.median(np.abs(lags - np.median(lags)))) if lags.size else float("nan")
    per = {}
    if mode == "insample":
        per = {c: (x, x) for c, x in sig.items()}
    elif mode == "crossfit":
        for c, (pe, po) in clock["per_capture_parity"].items():
            le = po["lag_s"] if ok(po, 0.75) else L      # even-block events: lag from odd blocks
            lo = pe["lag_s"] if ok(pe, 0.75) else L
            if (le, lo) != (L, L):
                per[c] = (le, lo)
    elif mode != "pooled":
        raise ValueError(mode)
    return {"pooled_s": L, "mode": mode, "robust_spread_s": spread, "n_capture_fits": int(lags.size),
            "n_parity_fits": int(sum((a != L) + (b != L) for a, b in per.values())) if mode == "crossfit" else None,
            "per_capture_s": per}


def lag_for(model: dict, cid: str) -> tuple[float, float]:
    """(lag for events in even 20 s blocks, lag for events in odd blocks) of one capture."""
    L = model["pooled_s"]
    return tuple(model["per_capture_s"].get(cid, (L, L)))


def adjust_events(td: np.ndarray, lags: tuple[float, float], L: float) -> np.ndarray:
    """Event times moved onto the pooled-lag clock: t - (lag of the event's block parity - L)."""
    par = np.floor(np.asarray(td) / XFIT_BLOCK_S).astype(int) % 2
    return td - (np.where(par == 1, lags[1], lags[0]) - L)


# ================================================================ pass 2: coincidence and chance

def _flash_cats(F, ecut, hum):
    db = np.clip(np.searchsorted(DIST_BINS, F["d"], "right") - 1, 0, len(DIST_BINS) - 2)
    et = np.searchsorted(ecut, F["E"])
    return (db * 3 + et) * 2 + hum.astype(int)


def lt_histograms(td, ga, gd, glt, gdev, gsat) -> dict:
    """Light-time test input (aggregate histograms only): residual td - ga of every event-group pair within
    +-LT_RES_HALF (ga already holds the waveguide delay and the pooled lag, so the peak sits near 0), binned
    (a) by group distance bin x GLM light-time bin x satellite, and (b) by the group's light time minus the capture's
    median light time. Taking all pairs, not just the matched ones, keeps the peak from being truncated."""
    edges = np.arange(-LT_RES_HALF, LT_RES_HALF + LT_RES_STEP / 2, LT_RES_STEP)
    nb = len(edges) - 1
    H = np.zeros((len(LT_DBINS) - 1, len(LT_EDGES) - 1, 2, nb))
    Hd = np.zeros((len(LT_DEV_EDGES) - 1, nb))
    o = np.argsort(ga)
    ga, gd, glt, gdev, gsat = ga[o], gd[o], glt[o], gdev[o], gsat[o]
    lo = np.searchsorted(ga, td - LT_RES_HALF)
    hi = np.searchsorted(ga, td + LT_RES_HALF)
    cnt = hi - lo
    if cnt.sum():
        rep = np.repeat(np.arange(td.size), cnt)
        idx = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        res = td[rep] - ga[idx]
        rb = np.clip(((res + LT_RES_HALF) / LT_RES_STEP).astype(int), 0, nb - 1)
        db = np.clip(np.searchsorted(LT_DBINS, gd[idx], "right") - 1, 0, len(LT_DBINS) - 2)
        lb = np.clip(np.searchsorted(LT_EDGES, glt[idx], "right") - 1, 0, len(LT_EDGES) - 2)
        sb = (gsat[idx] == 18).astype(int)
        np.add.at(H, (db, lb, sb, rb), 1)
        vb = np.clip(np.searchsorted(LT_DEV_EDGES, gdev[idx], "right") - 1, 0, len(LT_DEV_EDGES) - 2)
        np.add.at(Hd, (vb, rb), 1)
    return {"lt_H": H, "lt_Hdev": Hd}


def _match_one(args):
    cid, lags, lag, v, wg, wf, shifts, ecut = args      # lag = the pooled lag; lags = this capture's (even, odd)
    C = load_capture(cid, latlon=True)
    F, G, t0, secs = C["F"], C["G"], C["t0"], C["secs"]
    ga = G["t"] + G["d"] / v + lag - t0
    og = np.argsort(ga, kind="stable")
    ga_s, gflash, gd, glt, gsat = ga[og], G["f"][og], G["d"][og], G["lt"][og], G["sat"][og]
    fa = F["t"] + F["d"] / v + lag - t0
    la = F["t_last"] + F["d"] / v + lag - t0
    of = np.argsort(fa, kind="stable")
    fa_s, fd_s = fa[of], F["d"][of]
    f_gap = in_intervals(F["t"], C["gaps"])
    fine = np.clip(np.searchsorted(FINE_BINS, F["d"], "right") - 1, 0, len(FINE_BINS) - 2)
    S = np.concatenate([[0.0], shifts])
    nf = len(FINE_BINS) - 1
    nd = len(DBD_BINS) - 1
    out = {"id": cid, "t0": t0, "secs": secs, "hour": C["hour"], "night": C["night"], "hum_frac": C["hum_frac"],
           "gated_frac": C["gated_frac"], "onair_ok": C["onair_ok"],
           "hum_secs": float(np.sum(C["hum_iv"][:, 1] - C["hum_iv"][:, 0])) if len(C["hum_iv"]) else 0.0, "det": {}}
    lt_m = gd < LT_DBINS[-1]
    lt_dev = glt - (np.median(glt[lt_m]) if lt_m.any() else 0.0)
    for d in SCORED:
        td0, pk = C["det"][d], C["pk"][d]
        ev_hum = in_intervals(td0, C["hum_iv"]).astype(int)          # hum is judged on the capture's own time axis
        td = adjust_events(td0, lags, lag)                           # onto the pooled-lag clock (cross-fitted lags)
        o_ = np.argsort(td, kind="stable")
        td, pk, ev_hum = td[o_], pk[o_], ev_hum[o_]
        if d == "v2_conf":       # light-time test: residual histograms of every pair within +-60 ms (lag removed)
            r_lt = lt_histograms(td, ga_s[lt_m], gd[lt_m], glt[lt_m], lt_dev[lt_m], gsat[lt_m])
        ev_db = np.clip(np.searchsorted(DB_BINS, pk, "right") - 1, 0, len(DB_BINS) - 2)
        r = {k: np.zeros((2,) + shp) for k, shp in
             {"p_n": (2, 5), "p_g": (2, 5, nd + 1), "p_f": (2, 5, nd + 1),    # events: eligible; matched, by the
              "r_n": (NCAT,), "r_g": (NCAT,), "r_f": (NCAT,), "r_s": (NCAT,),  # nearest-source distance bin (last =
              "r_mult": (NCAT,), "r_multi2": (NCAT,),                          # no GLM source in the window)
              "fine_n": (nf, 2), "fine_g": (nf, 2),                           # fine distance x hum (group rule)
              "gpf": (len(DIST_BINS) - 1, 5),                                 # events per matched flash
              "dbh": (nd, len(DBH_EDGES) - 1)}.items()}                       # detected flash: max dB x distance
        cb = np.clip(np.searchsorted(DIST_BINS, F["d"], "right") - 1, 0, len(DIST_BINS) - 2)
        for j, s in enumerate(S):
            a = 0 if j == 0 else 1
            # --- events (precision proxy): eligible when the GLM time they would match is not in a missing slot.
            # Each event's source = the GLM group (flash) in the window that is closest to the station.
            el = ~in_intervals(td + t0 - lag - s, C["gaps"])
            gmin, garg = min_in_window(ga_s + s, gd, td, wg)
            fmin, _ = min_in_window(fa_s + s, fd_s, td, wf)
            gbin = np.where(np.isfinite(gmin), np.clip(np.searchsorted(DBD_BINS, gmin, "right") - 1, 0, nd - 1), nd)
            fbin = np.where(np.isfinite(fmin), np.clip(np.searchsorted(DBD_BINS, fmin, "right") - 1, 0, nd - 1), nd)
            np.add.at(r["p_n"][a], (ev_hum[el], ev_db[el]), 1)
            np.add.at(r["p_g"][a], (ev_hum[el], ev_db[el], gbin[el]), 1)
            np.add.at(r["p_f"][a], (ev_hum[el], ev_db[el], fbin[el]), 1)
            mg_ = el & (garg >= 0)
            if mg_.any():
                cnt = np.bincount(gflash[garg[mg_]], minlength=len(F["t"]))
                k = cnt > 0
                np.add.at(r["gpf"][a], (cb[k], np.clip(cnt[k], 1, 5) - 1), 1)
                if j == 0:
                    r["pairs_d"] = gmin[mg_]
                    r["pairs_pk"] = pk[mg_]
            # --- flashes (recall): eligible when the shifted arrival lies inside the capture and GLM had the file
            fel = (fa + s >= 0.5) & (fa + s < secs - 0.5) & ~f_gap
            hum = in_intervals(fa + s, C["hum_iv"])
            cat = _flash_cats(F, ecut, hum)
            di, dg = nearest(td, ga_s + s)
            gh = dg <= wg
            hit = np.zeros(len(F["t"]), bool)
            hit[gflash[gh]] = True
            fmax = np.full(len(F["t"]), -np.inf)
            np.maximum.at(fmax, gflash[gh], pk[di[gh]])
            _, df = nearest(td, fa + s)
            hf_ = df <= wf
            lo = np.searchsorted(td, fa + s - wg, "left")
            hi = np.searchsorted(td, la + s + wg, "right")
            ns = hi - lo
            r["r_n"][a] += np.bincount(cat[fel], minlength=NCAT)
            r["r_g"][a] += np.bincount(cat[fel & hit], minlength=NCAT)
            r["r_f"][a] += np.bincount(cat[fel & hf_], minlength=NCAT)
            r["r_s"][a] += np.bincount(cat[fel & (ns > 0)], minlength=NCAT)
            r["r_mult"][a] += np.bincount(cat[fel], weights=ns[fel], minlength=NCAT)
            r["r_multi2"][a] += np.bincount(cat[fel & (ns >= 2)], minlength=NCAT)
            np.add.at(r["fine_n"][a], (fine[fel], hum[fel].astype(int)), 1)
            np.add.at(r["fine_g"][a], (fine[fel & hit], hum[fel & hit].astype(int)), 1)
            k = fel & hit
            r["dbh"][a] += np.histogram2d(np.clip(F["d"][k], 0, 1e5), np.clip(fmax[k], DBH_EDGES[0], DBH_EDGES[-1] - 1e-6),
                                          [np.r_[DBD_BINS[:-1], 1e5 + 1], DBH_EDGES])[0]
        r["n_events"] = int(td.size)
        if d == "v2_conf":
            r.update(r_lt)
        out["det"][d] = r
    return out


# ================================================================ aggregation (pooled sums over a set of captures)

DBIN_NAMES = [f"{a}-{b}" if np.isfinite(b) else f"{a}+" for a, b in zip(DIST_BINS[:-1], DIST_BINS[1:])]
DB_NAMES = ["<12", "12-15", "15-20", "20-30", "30+"]
PREC_DMAX = (1000, 2000, 3000, 4000, 6000, 8000, np.inf)
PREC_MAIN = 4000                 # source-distance cap of the headline precision proxy (see the dmax table)


def _dm_name(dm):
    return "any" if not np.isfinite(dm) else f"<={dm}"


def _cat_mask(dbins=None, terc=None, hum=None) -> np.ndarray:
    m = np.zeros(NCAT, bool)
    for db in range(len(DIST_BINS) - 1):
        for et in range(3):
            for h in range(2):
                if (dbins is None or db in dbins) and (terc is None or et == terc) and (hum is None or h == hum):
                    m[(db * 3 + et) * 2 + h] = True
    return m


class Pool:
    """Per-capture count arrays stacked as (ncap, 2, ...): [:, 0] observed, [:, 1] summed over the control shifts."""

    def __init__(self, rows, nshift):
        self.rows = rows
        self.nshift = nshift
        self.secs = np.array([r["secs"] for r in rows])
        self.days = np.array([r["id"][:8] for r in rows])     # rows are in time order (capture ids sort by time)
        self.A = {}
        for d in SCORED:
            self.A[d] = {k: np.stack([r["det"][d][k] for r in rows]) for k in rows[0]["det"][d]
                         if k.startswith(("p_", "r_", "fine_", "gpf", "dbh"))}
            self.A[d]["n_events"] = np.array([r["det"][d]["n_events"] for r in rows], float)

    def s(self, d, k, idx):
        return self.A[d][k][idx].sum(axis=0)

    def precision(self, d, idx, rule="g", hum=None, dbb=None, dmax=np.inf):
        """(observed matched share, chance share, excess, corrected share, confirmed events per minute) of eligible
        events. An event is matched when a GLM group (rule g) or flash first event (rule f) closer than ``dmax`` km is
        in its window. corrected q = (obs - chance) / (1 - chance): if a share q of events truly has a GLM-seen source,
        obs = q + (1 - q) * chance. confirmed per minute = q * eligible events / minutes."""
        n, m = self.s(d, "p_n", idx), self.s(d, f"p_{rule}", idx)
        kb = [k for k in range(len(DBD_BINS) - 1) if DBD_BINS[k + 1] <= dmax]
        sl = (slice(None) if hum is None else hum, slice(None) if dbb is None else dbb)
        n0, n1 = n[0][sl].sum(), n[1][sl].sum()
        m0, m1 = m[0][sl][..., kb].sum(), m[1][sl][..., kb].sum()
        if n0 == 0 or n1 == 0:
            return (float("nan"),) * 5
        obs, ch = m0 / n0, m1 / n1
        q = (obs - ch) / (1 - ch) if ch < 1 else float("nan")
        mins = self.secs[idx].sum() / 60
        return obs, ch, obs - ch, q, q * n0 / mins

    def recall(self, d, idx, rule="g", mask=None):
        """(n flashes, observed detected share, chance share, chance-corrected recall)."""
        mask = np.ones(NCAT, bool) if mask is None else mask
        n, x = self.s(d, "r_n", idx), self.s(d, f"r_{rule}", idx)
        n0, x0, n1, x1 = n[0][mask].sum(), x[0][mask].sum(), n[1][mask].sum(), x[1][mask].sum()
        if n0 == 0 or n1 == 0:
            return 0, float("nan"), float("nan"), float("nan")
        obs, ch = x0 / n0, x1 / n1
        return int(n0), obs, ch, ((obs - ch) / (1 - ch) if ch < 1 else 0.0)

    def fine_recall(self, d, idx, hum=None):
        n, x = self.s(d, "fine_n", idx), self.s(d, "fine_g", idx)
        if hum is not None:
            n, x = n[..., hum], x[..., hum]
        else:
            n, x = n.sum(-1), x.sum(-1)
        with np.errstate(invalid="ignore", divide="ignore"):
            obs, ch = x[0] / n[0], x[1] / n[1]
            return n[0], (obs - ch) / (1 - ch)


def _r(x, k=4):
    return None if x is None or not np.isfinite(x) else round(float(x), k)


def boot_ci(fn, ncap, B, seed=11, scheme="capture", days=None, base=None):
    """95 % percentile interval of fn(idx) over B bootstrap resamples (``resample_idx`` scheme) of the captures
    ``base`` (default all ncap, in time order); fn gets capture indices."""
    rng = np.random.default_rng(seed)
    base = np.arange(ncap) if base is None else np.asarray(base)
    dd = None if days is None else np.asarray(days)[base]
    vals = []
    for _ in range(B):
        v = fn(base[resample_idx(rng, scheme, base.size, dd)])
        vals.append(np.nan if v is None else v)
    vals = np.array(vals, float)
    vals = vals[np.isfinite(vals)]
    if vals.size < B * 0.5:
        return None
    return [_r(np.percentile(vals, 2.5)), _r(np.percentile(vals, 97.5))]


CI_SCHEMES = ("block", "day", "capture")


def cis(P, fn, B, base=None, schemes=CI_SCHEMES) -> dict:
    """95 % intervals of fn(capture idx) under each resampling scheme. "block" (8 consecutive captures) is the
    headline: captures of one storm are not independent; "day" has only a handful of clusters; "capture" is the
    naive (too narrow) interval, kept for comparison."""
    return {f"ci95_{sc if sc != 'block' else f'block{BLOCK_CAPS}'}": boot_ci(fn, len(P.rows), B, scheme=sc,
                                                                            days=P.days, base=base)
            for sc in schemes}


def n10(n) -> int:
    """Counts written out are rounded to the nearest 10 (enough to judge a rate; less to triangulate with)."""
    return int(round(float(n) / 10.0) * 10)


def fine_crossings(P, d, idx, near_ref=True):
    """Distances where the chance-corrected recall (group rule) falls to 50 % and 10 % (absolute), and to 50 % / 10 %
    of the 0-500 km level (relative). Uses the FINE_BINS midpoints (8000-10000 km bin at 9000, last bin dropped)."""
    n, rc = P.fine_recall(d, idx)
    mids = np.array([(a + b) / 2 for a, b in zip(FINE_BINS[:-2], FINE_BINS[1:-1])])
    rc = rc[:-1]
    ok = n[:-1] >= 30
    x, y = mids[ok], rc[ok]
    ref = float(np.nanmean(y[x < 500])) if (x < 500).any() else float("nan")
    out = {"abs50_km": crossing(x, y, 0.5), "abs10_km": crossing(x, y, 0.1)}
    if np.isfinite(ref) and ref > 0:
        out.update(ref_0_500=ref, rel50_km=crossing(x, y, 0.5 * ref), rel10_km=crossing(x, y, 0.1 * ref))
    return out


def summarize(P: Pool, clock, model, wg, wf, B):
    ncap = len(P.rows)
    ALL = np.arange(ncap)
    sets = {"all": ALL,
            "hum_captures": np.array([i for i, r in enumerate(P.rows) if r["hum_frac"] >= 0.5]),
            "clean_captures": np.array([i for i, r in enumerate(P.rows) if r["hum_frac"] < 0.05]),
            "night": np.array([i for i, r in enumerate(P.rows) if r["night"]]),
            "day": np.array([i for i, r in enumerate(P.rows) if not r["night"]])}
    res = {"n_captures": ncap, "capture_sets": {k: int(len(v)) for k, v in sets.items()}, "detectors": {}}
    for d in SCORED:
        D = {}
        mins = P.secs.sum() / 60
        D["events_per_min"] = _r(P.A[d]["n_events"].sum() / mins, 2)
        keys = ("matched", "chance", "excess", "corrected", "confirmed_per_min")
        for rule in ("g", "f"):
            tab = {}
            for dm in PREC_DMAX:
                obs, ch, ex, q, cpm = P.precision(d, ALL, rule, dmax=dm)
                tab[_dm_name(dm)] = {"matched_share": _r(obs), "chance_share": _r(ch), "excess": _r(ex),
                                     "corrected": _r(q), "confirmed_per_min": _r(cpm, 2)}
                if rule == "g" and dm in (2000, PREC_MAIN, np.inf):
                    tab[_dm_name(dm)].update(cis(P, lambda i: P.precision(d, i, "g", dmax=dm)[3], B))
            D[f"precision_{'group' if rule == 'g' else 'first'}_by_dmax"] = tab
        D["precision_by_peak_db"] = {_dm_name(dm): {nm: dict(zip(keys, map(_r, P.precision(d, ALL, "g", dbb=k, dmax=dm))))
                                                    for k, nm in enumerate(DB_NAMES)} for dm in (2000, PREC_MAIN)}
        D["precision_by_block_hum"] = {nm: dict(zip(keys, map(_r, P.precision(d, ALL, "g", hum=h, dmax=PREC_MAIN))))
                                       for h, nm in ((0, "clean_blocks"), (1, "hum_blocks"))}
        D["precision_by_set"] = {k: dict(zip(keys, map(_r, P.precision(d, v, "g", dmax=PREC_MAIN))))
                                 for k, v in sets.items() if len(v)}
        rec = {}
        for rule, nm in (("g", "group"), ("f", "first"), ("s", "span")):
            rr = {}
            for b, bn in enumerate(DBIN_NAMES):
                n, obs, ch, rc = P.recall(d, ALL, rule, _cat_mask(dbins={b}))
                rr[bn] = {"n_flash": n10(n), "observed": _r(obs), "chance": _r(ch), "corrected": _r(rc)}
                if rule == "g":
                    rr[bn].update(cis(P, lambda i: P.recall(d, i, "g", _cat_mask(dbins={b}))[3], B))
            rec[nm] = rr
        D["recall_by_distance"] = rec
        D["recall_by_energy_tercile"] = {
            rng_nm: {f"T{t + 1}": _r(P.recall(d, ALL, "g", _cat_mask(dbins=bins, terc=t))[3]) for t in range(3)}
            for rng_nm, bins in (("0-1000", {0, 1}), ("1000-2000", {2}), ("2000-4000", {3, 4}), ("4000+", {5}))}
        D["recall_by_block_hum"] = {
            hn: {bn: _r(P.recall(d, ALL, "g", _cat_mask(dbins={b}, hum=h))[3]) for b, bn in enumerate(DBIN_NAMES)}
            for h, hn in ((0, "clean_blocks"), (1, "hum_blocks"))}
        D["recall_by_set"] = {k: {bn: _r(P.recall(d, v, "g", _cat_mask(dbins={b}))[3])
                                  for b, bn in enumerate(DBIN_NAMES)} for k, v in sets.items() if len(v)}
        n, rc = P.fine_recall(d, ALL)
        D["recall_fine_group"] = {(f"{a}-{b}" if np.isfinite(b) else f"{a}+"): {"n": n10(nn), "corrected": _r(x)}
                                  for a, b, nn, x in zip(FINE_BINS[:-1], FINE_BINS[1:], n, rc)}
        D["recall_distance_crossings"] = {k: _r(v, 1) for k, v in fine_crossings(P, d, ALL).items()}
        if d in ("v2", "v2_conf", "onair_bp"):
            for key in ("abs50_km", "abs10_km"):
                D["recall_distance_crossings"][f"{key}_cis"] = cis(
                    P, lambda i, key=key: fine_crossings(P, d, i).get(key), min(B, 400))
        D["by_day"] = by_day(P, d)
        # multiplicity: sferics per flash inside the flash's span (+-w), observed minus chance, per distance bin
        mult = {}
        for b, bn in enumerate(DBIN_NAMES):
            m = _cat_mask(dbins={b})
            n, x, x2, xs = (P.s(d, k, ALL) for k in ("r_n", "r_mult", "r_multi2", "r_s"))
            n0, n1 = n[0][m].sum(), n[1][m].sum()
            if n0 and n1:
                ex_mean = x[0][m].sum() / n0 - x[1][m].sum() / n1
                ex_det = xs[0][m].sum() / n0 - xs[1][m].sum() / n1
                ex_2 = x2[0][m].sum() / n0 - x2[1][m].sum() / n1
                mult[bn] = {"excess_sferics_per_flash": _r(ex_mean, 3),
                            "excess_sferics_per_detected_flash": _r(ex_mean / ex_det, 2) if ex_det > 0.005 else None,
                            "excess_share_flashes_with_2plus": _r(ex_2, 4)}
        D["multiplicity_span"] = mult
        g = P.s(d, "gpf", ALL)
        D["events_per_matched_flash_group_rule"] = {
            bn: {"observed": [int(v) for v in g[0][b]], "chance_mean": [_r(v / P.nshift, 1) for v in g[1][b]]}
            for b, bn in enumerate(DBIN_NAMES)}
        dbd = P.s(d, "p_g", ALL).sum(axis=(1, 2))[:, :len(DBD_BINS) - 1]
        pairs_d = np.concatenate([r["det"][d].get("pairs_d", np.zeros(0)) for r in P.rows])
        pairs_pk = np.concatenate([r["det"][d].get("pairs_pk", np.zeros(0)) for r in P.rows])
        D["peak_db_vs_distance_events"] = db_vs_distance(pairs_d, pairs_pk, dbd, P.nshift)
        D["peak_db_vs_distance_flashes"] = flash_db_vs_distance(P.s(d, "dbh", ALL), P.nshift)
        res["detectors"][d] = D
    res["comparisons"] = compare(P, sets, B)
    res["light_time"] = light_time_test(P, B)
    res["fov_by_utc_hour"] = {d: by_hour(P, d) for d in ("v2", "v2_conf", "onair_bp")}
    return res


def db_vs_distance(dd, pk, dbd, nshift):
    """Event level: median matched peak dB per distance of the nearest GLM source in the window, with the matches the
    shifted controls put in that bin over the observed ones (not a clean chance share: under a shift an event that
    really came from a near flash is re-assigned to a farther one, so far bins can exceed 1), and an OLS fit
    dB = a + b log10(d/1000) on bins where that ratio is under 0.3. The flash-level table is the cleaner one."""
    rows, keep = {}, np.zeros(dd.size, bool)
    for k, (a, b) in enumerate(zip(DBD_BINS[:-1], DBD_BINS[1:])):
        m = (dd >= a) & (dd < b)
        ch = (dbd[1][k] / nshift) / dbd[0][k] if dbd[0][k] else float("nan")
        nm = f"{a}-{b}" if np.isfinite(b) else f"{a}+"
        rows[nm] = {"n": int(m.sum()), "control_over_observed": _r(ch, 3),
                    "median_db": _r(np.median(pk[m]), 2) if m.any() else None,
                    "p90_db": _r(np.percentile(pk[m], 90), 2) if m.any() else None}
        if np.isfinite(ch) and ch < 0.3:
            keep |= m
    fit = None
    if keep.sum() >= 30:
        x, y = np.log10(dd[keep] / 1000.0), pk[keep]
        b, a = np.polyfit(x, y, 1)
        rng = np.random.default_rng(3)
        bs = []
        for _ in range(500):
            i = rng.integers(0, x.size, x.size)
            bs.append(np.polyfit(x[i], y[i], 1)[0])
        with np.errstate(invalid="ignore", divide="ignore"):
            rho = float(np.corrcoef(x, y)[0, 1])
        fit = {"n": int(keep.sum()), "db_at_1000km": _r(a, 2), "db_per_decade": _r(b, 2),
               "db_per_decade_ci95": [_r(np.percentile(bs, 2.5), 2), _r(np.percentile(bs, 97.5), 2)],
               "pearson_r": _r(rho, 3), "range_km": [float(dd[keep].min() // 250 * 250), float(dd[keep].max() // 250 * 250 + 250)]}
    return {"bins": rows, "fit_log10_distance": fit}


def _hist_q(h, q):
    c = np.cumsum(h)
    if c[-1] <= 0:
        return None
    return float(DBH_EDGES[np.searchsorted(c, q * c[-1])] + 0.5)


def flash_db_vs_distance(dbh, nshift):
    """Strongest matched sferic of each detected GLM flash, by flash distance, with the chance part removed:
    excess histogram = observed - (sum over shifts) / nshift, negatives clipped. Median / p90 of the excess per bin,
    then a weighted fit median = a + b log10(d / 1000 km) over bins where the excess is at least half of the observed
    (bin midpoints; 8000+ at 9000 km)."""
    obs, ch = dbh[0], dbh[1] / nshift
    ex = np.clip(obs - ch, 0, None)
    rows, xs, ys, ws = {}, [], [], []
    for k, (a, b) in enumerate(zip(DBD_BINS[:-1], DBD_BINS[1:])):
        nm = f"{a}-{b}" if np.isfinite(b) else f"{a}+"
        no, ne = float(obs[k].sum()), float(ex[k].sum())
        med, p90 = _hist_q(ex[k], 0.5), _hist_q(ex[k], 0.9)
        rows[nm] = {"detected_flashes": n10(no), "excess_flashes": n10(ne),
                    "real_share": _r(ne / no, 3) if no else None, "median_db": med, "p90_db": p90}
        if no and ne / no >= 0.5 and ne >= 20 and med is not None:
            xs.append(np.log10(((a + b) / 2 if np.isfinite(b) else 9000.0) / 1000.0))
            ys.append(med)
            ws.append(ne)
    fit = None
    if len(xs) >= 3:
        w = np.sqrt(ws)
        b, a = np.polyfit(xs, ys, 1, w=w)
        fit = {"n_bins": len(xs), "median_db_at_1000km": _r(a, 2), "db_per_decade": _r(b, 2)}
    return {"bins": rows, "fit_median_vs_log10_distance": fit}


def compare(P, sets, B):
    ncap = len(P.rows)
    ALL = np.arange(ncap)
    out = {}
    near = _cat_mask(dbins={0, 1})
    mid = _cat_mask(dbins={2})
    far = _cat_mask(dbins={3, 4})
    metrics = {
        "precision_corrected": lambda d, i: P.precision(d, i, "g", dmax=PREC_MAIN)[3],
        "confirmed_per_min": lambda d, i: P.precision(d, i, "g", dmax=PREC_MAIN)[4],
        "recall_0_1000": lambda d, i: P.recall(d, i, "g", near)[3],
        "recall_1000_2000": lambda d, i: P.recall(d, i, "g", mid)[3],
        "recall_2000_4000": lambda d, i: P.recall(d, i, "g", far)[3],
    }
    for base in ("july", "onair_bp", "v2_detect"):
        for sn in ("all", "hum_captures", "clean_captures"):
            idx = sets[sn]
            if not len(idx):
                continue
            row = {}
            for mn, f in metrics.items():
                v2 = f("v2", idx)
                bv = f(base, idx)
                row[mn] = {"v2": _r(v2), base: _r(bv), "diff": _r(v2 - bv)}
                row[mn].update({f"diff_{k}": v for k, v in
                                cis(P, lambda i, f=f: f("v2", i) - f(base, i), B, base=idx).items()})
            out[f"v2_vs_{base}|{sn}"] = row
    return out


def light_time_test(P, B):
    """Does the VLF-minus-GLM residual depend on the GLM flash-to-GEO light time? Slope ~ 0: GLM times are referenced
    to the flash (light time removed); ~ -1: referenced to the satellite.

    Peak-based (not a regression over matched pairs: about half of those are chance and the +-w window truncates any
    shift, both pulling a pair regression toward 0). Two estimators on the v2_conf pair histograms (every pair within
    +-60 ms, groups < 4000 km):
      cells: peak location per (distance bin, light-time bin, satellite) cell, then a weighted fit
             peak = offset[distance bin] + slope * light time. G18 and G19 see the same area with light times several
             ms apart, which gives the fit its lever arm.
      within_capture: peak location per bin of (light time - the capture's median light time), pooled; slope fit.
    CIs: block bootstrap (8 consecutive captures) and capture bootstrap."""
    rows = [r["det"]["v2_conf"] for r in P.rows]
    if "lt_H" not in rows[0]:
        return None
    H = np.stack([r["lt_H"] for r in rows])
    Hd = np.stack([r["lt_Hdev"] for r in rows])
    mid = (np.arange(H.shape[-1]) + 0.5) * LT_RES_STEP - LT_RES_HALF
    ltm = (LT_EDGES[:-1] + LT_EDGES[1:]) / 2
    nd = len(LT_DBINS) - 1

    def cells(idx):
        h = H[idx].sum(0)
        X, y, w = [], [], []
        for a in range(nd):
            for b in range(len(ltm)):
                for sat in range(2):
                    if h[a, b, sat].sum() < 2000:
                        continue
                    pk, wt = centroid_peak(h[a, b, sat], mid)
                    if wt > 300 and np.isfinite(pk):
                        X.append(np.r_[np.eye(nd)[a], ltm[b]])
                        y.append(pk)
                        w.append(np.sqrt(wt))
        if len(y) < nd + 3:
            return None
        X, y, w = np.array(X), np.array(y), np.array(w)
        return float(np.linalg.lstsq(X * w[:, None], y * w, rcond=None)[0][-1])

    dev_mid = np.array([-0.0025, -0.0015, -0.0005, 0.0005, 0.0015, 0.0025])

    def within(idx):
        h = Hd[idx].sum(0)
        pts = [centroid_peak(h[k], mid) for k in range(len(dev_mid))]
        y = np.array([p[0] for p in pts])
        w = np.array([p[1] for p in pts])
        ok = np.isfinite(y) & (w > 100)
        if ok.sum() < 3:
            return None
        return float(np.polyfit(dev_mid[ok], y[ok], 1, w=np.sqrt(w[ok]))[0])
    ALL = np.arange(len(rows))
    Bl = min(B, 300)
    lt_all = H.sum(axis=(0, 1, 3, 4))              # pairs per light-time bin
    c = np.cumsum(lt_all) / max(lt_all.sum(), 1)
    return {"method": "peak location per cell, every pair within +-60 ms (no truncation, chance removed as background)",
            "slope_cells_s_per_s": _r(cells(ALL), 3),
            **{f"cells_{k}": v for k, v in cis(P, cells, Bl, schemes=("block", "capture")).items()},
            "slope_within_capture_s_per_s": _r(within(ALL), 3),
            **{f"within_{k}": v for k, v in cis(P, within, Bl, schemes=("block", "capture")).items()},
            "pairs_in_histograms": int(H.sum()),
            "light_time_ms_p5_p50_p95_of_pairs": [_r(ltm[min(np.searchsorted(c, q), len(ltm) - 1)] * 1e3, 1)
                                                  for q in (0.05, 0.5, 0.95)],
            "reading": "slope 0 => GLM times at the flash; slope -1 => GLM times at the satellite"}


def by_day(P, d):
    """Per day: precision proxy and recall by distance. Shows how much each number rests on one storm."""
    out = {}
    near = {"0-500": _cat_mask(dbins={0}), "500-1000": _cat_mask(dbins={1}), "1000-2000": _cat_mask(dbins={2})}
    for day in sorted(set(P.days)):
        idx = np.nonzero(P.days == day)[0]
        row = {"n_captures": int(idx.size), "precision_corrected": _r(P.precision(d, idx, "g", dmax=PREC_MAIN)[3], 3)}
        for nm, m in near.items():
            n, _, _, rc = P.recall(d, idx, "g", m)
            row[f"recall_{nm}"] = {"n_flash": n10(n), "corrected": _r(rc, 3) if n >= 30 else None}
        out[day] = row
    return out


def by_hour(P, d):
    """Precision excess by capture UTC hour (3 h bins): VLF hears Africa/Europe storms that GLM cannot see."""
    out = {}
    for h0 in range(0, 24, 3):
        idx = np.array([i for i, r in enumerate(P.rows) if h0 <= r["hour"] < h0 + 3])
        if len(idx):
            obs, ch, ex, q, cpm = P.precision(d, idx, "g", dmax=PREC_MAIN)
            out[f"{h0:02d}-{h0 + 3:02d}"] = {"n_captures": int(len(idx)), "corrected": _r(q), "chance": _r(ch),
                                            "events_per_min": _r(P.A[d]["n_events"][idx].sum() / (P.secs[idx].sum() / 60), 1)}
    return out


def cmd_match(a):
    ids = capture_ids()
    if a.capture:
        ids = [c for c in ids if c in set(a.capture)]
    t = time.time()
    v_frac, speed0 = a.v_frac, None
    if a.v_frac is None:                     # auto: a first clock pass at 0.99 c, then the measured speed
        speed0 = clock_stage(ids, V_FRAC * C_KM_S, a.workers)["speed"]
        vf = speed0.get("v_over_c")
        v_frac = vf if vf and 0.8 < vf < 1.05 else V_FRAC
    v = v_frac * C_KM_S
    clock = clock_stage(ids, v, a.workers)
    model = lag_model(clock, mode=a.lag_mode)
    clock.pop("per_capture_parity")
    hists = clock.pop("hists")
    K = (hists[next(iter(hists))].shape[-1] - 1) // 2
    aligned = np.zeros(2 * K + 1)
    for cid, h in hists.items():          # each block parity moved by the lag its events are scored with
        for p_, lg in enumerate(lag_for(model, cid)):
            aligned += np.roll(h[p_], -int(round((lg - model["pooled_s"]) / SCAN_STEP)))
    clock["aligned_group"] = peak_fit(aligned)
    kk = K + int(round(clock["aligned_group"]["lag_s"] / SCAN_STEP))
    clock["hist_aligned_around_peak"] = {int(j - kk): int(aligned[j]) for j in range(kk - 30, kk + 31)}
    wg = max(clock["aligned_group"]["w_s"], 2 * SCAN_STEP) if a.w_group is None else a.w_group
    wf = max(clock["pooled_flash_first_v2conf"]["w_s"], wg) if a.w_first is None else a.w_first
    print(f"clock done {time.time() - t:.0f}s: v/c {v_frac:.4f}, pooled lag {model['pooled_s'] * 1e3:.2f} ms "
          f"({model['mode']}, spread {model['robust_spread_s'] * 1e3:.2f} ms), "
          f"w_group {wg * 1e3:.1f} ms, w_first {wf * 1e3:.1f} ms", flush=True)
    shifts = make_shifts(a.shifts)
    ecut = np.array(clock["energy_terciles_J"])
    rows = _pool_map(_match_one, [(c, lag_for(model, c), model["pooled_s"], v, wg, wf, shifts, ecut) for c in ids],
                     a.workers)
    print(f"match done {time.time() - t:.0f}s", flush=True)
    P = Pool(rows, len(shifts))
    res = summarize(P, clock, model, wg, wf, a.boot)
    pc = clock.pop("per_capture")
    lags = np.array([p["lag_s"] for p in pc.values()])
    zs = np.array([p["z"] for p in pc.values()])
    good = zs >= 8
    clock["per_capture_summary"] = {
        "n": int(len(lags)), "n_significant_z8": int(good.sum()),
        "lag_ms_p5_p25_p50_p75_p95_significant": [_r(x, 2) for x in np.percentile(lags[good] * 1e3, [5, 25, 50, 75, 95])]
        if good.any() else None,
        "share_within_2ms_of_pooled": _r(np.mean(np.abs(lags[good] - model["pooled_s"]) <= 0.002)) if good.any() else None}
    by_day = {}
    for c, p_ in pc.items():              # descriptive: each capture's own (in-sample) v2_conf peak
        if p_["z"] >= 8 and p_["excess_in_w"] >= 15:
            by_day.setdefault(c[:8], []).append(p_["lag_s"] * 1e3)
    clock["per_capture_lag_ms_by_day"] = {k: {"n": len(v), "p25_p50_p75": [_r(q, 2) for q in np.percentile(v, [25, 50, 75])]}
                                          for k, v in sorted(by_day.items())}
    model_out = {k: v for k, v in model.items() if k != "per_capture_s"}
    model_out["per_capture_lag_ms_used_even_odd"] = {c: [round(x * 1e3, 2) for x in xs]
                                                     for c, xs in model["per_capture_s"].items()}
    res.update({"clock": jsonable(clock), "lag_model": model_out, "v_over_c_used": v_frac,
                "speed_fit_first_pass_at_0.99c": jsonable(speed0),
                "window_group_s": wg, "window_first_s": wf, "shifts_s": [round(s, 3) for s in shifts],
                "per_capture": [{"id": r["id"], "hour_utc": r["hour"], "night": r["night"], "hum_frac": _r(r["hum_frac"], 3),
                                 "gated_frac": _r(r["gated_frac"], 3), "lag_ms": _r(pc[r["id"]]["lag_s"] * 1e3, 2),
                                 "lag_z": _r(pc[r["id"]]["z"], 1),
                                 "v2_events": r["det"]["v2"]["n_events"],
                                 "v2_precision_corrected": _r(P.precision("v2", np.array([k]), "g", dmax=PREC_MAIN)[3], 3)}
                                for k, r in enumerate(rows)],
                "wall_s": round(time.time() - t, 1)})
    js = json.dumps(jsonable(res), indent=1)
    if a.out:
        Path(a.out).write_text(js)
    print(js[:4000])


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, np.ndarray):
        return jsonable(o.tolist())
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ================================================================ CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("events", help="run the detectors on every capture and cache their events")
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--capture", nargs="*")
    p.set_defaults(fn=cmd_events)
    p = sub.add_parser("match", help="clock fit, coincidence, chance controls, bootstrap (aggregate output only)")
    p.add_argument("--out", default=None)
    p.add_argument("--boot", type=int, default=1000)
    p.add_argument("--shifts", type=int, default=40)
    p.add_argument("--workers", type=int, default=2)
    p.add_argument("--capture", nargs="*")
    p.add_argument("--v-frac", type=float, default=None,
                   help="waveguide group speed / c for the d/v delay (default: measured by the speed fit)")
    p.add_argument("--lag-mode", choices=("crossfit", "pooled", "insample"), default="crossfit",
                   help="per-capture lag: cross-fitted on the other 20 s blocks (default), one pooled lag, or in-sample")
    p.add_argument("--w-group", type=float, default=None, help="override the group window (s)")
    p.add_argument("--w-first", type=float, default=None, help="override the flash first-event window (s)")
    p.set_defaults(fn=cmd_match)
    a = ap.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
