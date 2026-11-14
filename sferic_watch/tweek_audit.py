"""Tweek honesty audit (plan step 7): does the July tweek method find more "textbook" tweeks than chance?

Why this module exists
----------------------
The July tweek result (dlayer-diary ``tweeks_fits.json``: 77 tweeks, median fc 1.831 kHz, h = c/2fc = 81.9 km) came
from a dispersion fit t(f) = t0 + D f / sqrt(f^2 - fc^2) whose fc upper bound is the lowest ridge frequency
(``hi = [max(rt)+8, 1e3, rf.min()]``). Measured on that JSON (``july_reaudit``):

* 29 / 44 / 56 of 77 fits have fc within 0.5 / 5 / 20 Hz of that bound, so fc is mostly the ridge's lowest bin;
* 21 / 77 have D < 0.01 ms (d = c D < 3 km: unphysical, 90 % of tweeks are 1000-5000 km away, Kumar 2008);
* the 19 fits clear of every bound have median fc 1.595 kHz (h about 94 km), not 1.831 kHz;
* the reported fc_err (median a few Hz) is curve_fit covariance AT a bound, i.e. meaningless.

And the July "textbook" gates pass 57 of 300 RANDOM 45 ms windows on the buzz capture 11-06 04:00Z (median fc
1.83 kHz, the published value), against 1/300 on clean air (ourchain/nulltest.py). So a yield means nothing until it
beats the same gates run on random windows.

What the audit does
-------------------
1. Candidates: v2 pipeline events (``sferic_watch.pipeline.run``: front end, median-floor detector, hum flag, mains
   phase gate) that are 16-30 kHz confirmed (``conf``) and not in a hum-flagged block (``hum`` False).
2. Each candidate gets the July window (6 ms before the onset, 45 ms long, real VLF waveform), the July ridge
   (``ridge``) and the July fit (``fit``), and the July textbook gates (``textbook``). The ridge and fit are a clean
   re-implementation of dlayer-diary analyze_tweeks.py. The window's real waveform is made with July's mixer by
   default (``phase="july"``: the 15 kHz mixer restarts at every window), so the audit judges the published method.
   ``phase="absolute"`` is the other fixed mixer (the first corpus run, 2026-11-14 v1, used it); neither is right:
   see ``window_real`` for why they differ (a per-window cos(phase) gain, random under both).
3. Null: 300 random window times per capture (uniform over the analysed span) go through the SAME time gates: not in
   a hum-flagged block and not inside a mains train's +-0.2 ms reject window (the pipeline's own trains). The 16-30 kHz
   confirmation is an event property; a random time has no event, so requiring it would make the null vanish by
   construction and the test would be empty. Admissible windows are then fitted exactly like candidates.
4. Flags per fit: fc within 10 Hz of either bound (lowest ridge frequency, or the 1.4 kHz floor) or D < 0.01 ms
   (D at or near its 1e-3 floor; d = c D < 3 km). A fit with no flag is "bound-free".
5. Uncertainty: fc 95 % CI from a pairs bootstrap of the kept ridge points (each resample refitted by a 1 Hz profile
   of the same model, weights and t0 / D / fc bounds, ``profile_fc``; the CI is centred on the profile's own estimate
   of the full ridge, and a fit where curve_fit stopped elsewhere is flagged ``fit_not_global``), widened
   in quadrature by the ridge grid quantisation (frequency step 91.55 Hz = 3 x 62.5 kHz / 2048, uniform, so 95 %
   half-width 0.95 x step / 2 = 43.5 Hz). Not curve_fit covariance. Every printed h carries its 95 % CI.
6. Verdict (``verdict``): "insufficient" unless the textbook yield per candidate is at least 3x the null rate per
   admissible window AND at least 10 bound-free textbook fits exist.

OnAir has no tweek code; nothing here comes from it.
"""
from __future__ import annotations

import json
import math
import os

import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import stft
from scipy.stats import poisson

FS = 62500.0                 # capture sample rate, S/s
FC = 15000.0                 # tuned centre, Hz
C_KM_PER_MS = 299.792458     # d = c D with D in ms
H_KM_KHZ = 149.896229        # h = c / (2 fc) with fc in kHz -> km

# July window and ridge (dlayer-diary analyze_tweeks.py, unchanged so the audit judges the published method)
WIN_S = 0.045                # analysis window length
PRE_S = 0.006                # window starts this long before the event onset
R_FMIN, R_FMAX = 1.55, 6.5   # ridge band, kHz
NPERSEG, NFFT, HOP = 256, 2048, 6
R_ONSET_BAND = (2.5, 7.0)    # onset = frame of max energy in this band, kHz
R_TWIN = (-1.5, 16.0)        # ridge time window around the onset, ms
R_FSTEP_KHZ = 0.08           # ridge row thinning target, kHz (gives 3 bins = 91.55 Hz)
R_REL = 0.08                 # ridge row kept if its peak >= 8 % of the band peak
R_MONO_MS = 0.6              # monotone descent tolerance, ms

# July fit
FC_LO = 1.4                  # fc lower bound, kHz
D_LO, D_HI = 1e-3, 1e3       # D bounds, ms
T0_PAD = 8.0                 # t0 bounds: ridge time span +- 8 ms
MIN_PTS = 6

# July textbook gates
TB_NPTS, TB_FC, TB_RMS, TB_FCERR = 12, (1.35, 2.6), 0.45, 0.20
TB_SPREAD, TB_RMAX, TB_RMIN = (2.0, 13.0), 5.0, 2.6

# audit
BOUND_TOL_KHZ = 0.010        # plan: flag fc within 10 Hz of a bound
D_FLAG_MS = 0.01             # D below this is at/near the 1e-3 floor (d < 3 km)
N_NULL = 300
N_BOOT = 200
PROFILE_STEP_KHZ = 0.001     # fc grid of the bootstrap refits (1 Hz)
YIELD_X = 3.0                # yield must be >= 3x the null rate...
MIN_BOUND_FREE = 10          # ...and at least this many bound-free textbook fits
SMALL_N = 5                  # below this many fits a median CI is the envelope of the per-fit CIs


def ridge_step_khz(fs: float = FS, nfft: int = NFFT) -> float:
    """Frequency step of the thinned ridge, kHz: round(0.08 kHz / bin) bins. 91.55 Hz at 62.5 kS/s, nfft 2048.
    The July ridge minima sit on this grid (1.556, 1.648, 1.739, 1.831 kHz ...)."""
    df = fs / nfft / 1000.0
    return max(1, int(round(R_FSTEP_KHZ / df))) * df


def h_km(fc_khz):
    """Reflection height from the TM1 cutoff: h = c / (2 fc). fc in kHz -> km (1.831 kHz -> 81.9 km)."""
    return H_KM_KHZ / np.asarray(fc_khz, float)


def group_delay(f, t0, D, fc):
    """Waveguide group delay in ms: t = t0 + D f / sqrt(f^2 - fc^2) (f, fc in kHz). NaN at or below the cutoff."""
    f = np.asarray(f, float)
    out = np.full(f.shape, np.nan)
    m = f > fc
    out[m] = t0 + D * f[m] / np.sqrt(f[m] ** 2 - fc ** 2)
    return out


PHASES = ("july", "absolute")


def window_real(x: np.ndarray, n0: int, fs: float = FS, fc: float = FC, phase: str = "july") -> np.ndarray:
    """Complex baseband slice (absolute first sample n0) -> real VLF waveform, float64: Re(x e^{+j 2 pi fc n / fs}).

    phase="july": n counts from 0 in every window, as dlayer-diary analyze_capture did (``nn = np.arange(len(x))``).
    phase="absolute": n is the absolute sample index n0 + k, the same mixer as humflag.

    Why it matters (measured 2026-11-14 by the step-7 verifier, explained here): the RF is real, so the IQ holds
    every sferic twice, at baseband f - 15 kHz and as a mirror image at -f - 15 kHz. After the mixer the two copies land
    on +f and -f, and taking .real adds them coherently: the output is R(f) cos(phi), where phi is the mixer phase at
    the window start relative to the receiver's LO. The mixer turns 0.24 cycles (86.4 deg) per sample, so July's
    window-local restart gives each window a different phi = 86.4 deg x (window start mod 25) + const: the tweek is
    scaled by |cos(phi)| while receiver noise and the 0 Hz tone are not. On 40 textbook candidates of 11-06 15:00Z the
    absolute mixer kept 40 textbook, July's 12; constant offsets of 0/1/2/3 samples gave 40/5/32/7 (those 40 were
    SELECTED as textbook under the absolute mixer, so part of that drop is selection; on 300 unselected random
    windows of 11-06 04:00Z the absolute mixer gave 49 textbook and July's 57, 68 windows changing status).
    ``mirror_phase`` shows the absolute mixer is not "the right phase" either: the receiver's phase drifts by tens of
    degrees per second, so under either mixer each window's gain |cos(phi)| is effectively random. Neither choice is
    phase-invariant; plan step 10 should use the two images' powers (|Y(+f)|^2 + |Y(-f)|^2) instead of .real."""
    if phase not in PHASES:
        raise ValueError(f"phase must be one of {PHASES}")
    k0 = n0 if phase == "absolute" else 0
    n = np.arange(k0, k0 + len(x), dtype=np.float64)
    return (x * np.exp(2j * np.pi * (fc / fs) * n)).real.astype(np.float64)


def mirror_phase(x: np.ndarray, fs: float = FS, fc: float = FC, band_khz=(2.5, 9.0), nfft: int = 1024,
                 top_frac: float = 0.1) -> dict:
    """Mixer phase phi under the ABSOLUTE mixer, measured from the mirror images (see ``window_real``).

    For real RF, Y(+f) Y(-f) = |R(f)|^2 e^{-2 j phi} after the absolute mixer, so phi = -angle(sum Y+ Y-) / 2 (mod
    180 deg) over the sferic band. Measured per nfft block on the ``top_frac`` strongest blocks (sferics):
    ``coherence`` = median per-block |sum Y+ Y-| / norm (1 = the two images are mirror copies, 0 = independent);
    ``phi_deg`` = the phase of sum Y+ Y- over them; ``stability`` = |sum c_b| / sum |c_b| (c_b = per-block Y+ Y-, so
    strong sferics weigh most; 1 = one fixed phase for the whole
    slice, so the absolute mixer gives every sferic the same gain |cos phi|; near 0 = the phase drifts, so the gain is
    effectively random per window under ANY fixed mixer). Measured 2026-11-14 on 30 s of four captures: per-block
    coherence 0.31-0.87 (the images are real copies) but phi of the 40 strongest blocks spread over 0-180 deg and
    walked by tens of degrees per second: the receiver LO is not locked to the 0.24 cycle/sample mixer. Over all 151
    captures (first 60 s): coherence median 0.68 (5-95 % 0.25-0.85), stability median 0.07 (5-95 % 0.02-0.60)."""
    nb = len(x) // nfft
    nan = dict(phi_deg=math.nan, coherence=math.nan, stability=math.nan, gain=math.nan, n_blocks=0)
    if nb < 1:
        return nan
    n = np.arange(nb * nfft, dtype=np.float64)
    y = x[: nb * nfft] * np.exp(2j * np.pi * (fc / fs) * n)          # absolute mixer: true frequency
    Y = np.fft.fft(y.reshape(nb, nfft) * np.hanning(nfft), axis=1)
    k = np.arange(int(np.ceil(band_khz[0] * 1e3 * nfft / fs)), int(band_khz[1] * 1e3 * nfft / fs) + 1)
    yp, ym = Y[:, k], Y[:, (-k) % nfft]                              # +f and its mirror -f, exact bins
    cr = (yp * ym).sum(1)
    pp, pm = (np.abs(yp) ** 2).sum(1), (np.abs(ym) ** 2).sum(1)
    top = np.argsort(-(pp + pm))[: max(1, int(round(top_frac * nb)))]
    coh = np.abs(cr[top]) / (np.sqrt(pp[top] * pm[top]) + 1e-30)
    m = cr[top].sum()                                               # strong blocks dominate the phase
    phi = -np.angle(m) / 2.0
    stab = abs(m) / (np.abs(cr[top]).sum() + 1e-30)
    return dict(phi_deg=float(np.degrees(phi)) % 180.0, coherence=float(np.median(coh)), stability=float(stab),
                gain=float(abs(math.cos(phi))), n_blocks=int(len(top)))


def ridge(xr: np.ndarray, fs: float = FS) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """July ridge: per thinned frequency row (1.55-6.5 kHz), the time of peak power, after subtracting each row's
    median over the window (kills steady lines and the 0 Hz tone). Rows weaker than 8 % of the band peak are dropped,
    then a monotone-descent rule (time may not go back more than 0.6 ms when moving down in frequency) keeps the
    dispersion branch. Returns (f kHz descending, t ms from window start, row peak power)."""
    f, t, Z = stft(xr, fs=fs, nperseg=NPERSEG, nfft=NFFT, noverlap=NPERSEG - HOP, window="hann",
                   boundary=None, padded=False)
    S = np.abs(Z) ** 2
    S = np.maximum(S - np.median(S, axis=1, keepdims=True), 0.0)
    fk, tms = f / 1000.0, t * 1000.0
    det = (fk >= R_ONSET_BAND[0]) & (fk <= R_ONSET_BAND[1])
    onset = tms[int(np.argmax(S[det].sum(axis=0)))]
    tw = (tms >= onset + R_TWIN[0]) & (tms <= onset + R_TWIN[1])
    band = (fk >= R_FMIN) & (fk <= R_FMAX)
    fb, tb = fk[band], tms[tw]
    Sb = S[band][:, tw]
    smax = Sb.max() + 1e-30
    step = max(1, int(round(R_FSTEP_KHZ / (fb[1] - fb[0]))))
    rows = np.arange(0, len(fb), step)
    j = np.argmax(Sb[rows], axis=1)
    pk = Sb[rows, j]
    ok = pk >= R_REL * smax
    rf, rt, rw = fb[rows][ok], tb[j][ok], pk[ok]
    if len(rf) > 3:
        o = np.argsort(-rf, kind="stable")
        rf, rt, rw = rf[o], rt[o], rw[o]
        keep = np.zeros(len(rf), bool)
        tprev = -1e9
        for k in range(len(rf)):
            if rt[k] >= tprev - R_MONO_MS:
                keep[k] = True
                tprev = max(tprev, rt[k])
        rf, rt, rw = rf[keep], rt[keep], rw[keep]
    return rf, rt, rw


def _bounds(rf, rt):
    """July bounds: t0 in [min t - 8, max t + 8] ms, D in [1e-3, 1e3] ms, fc in [1.4 kHz, lowest ridge frequency]."""
    return [np.min(rt) - T0_PAD, D_LO, FC_LO], [np.max(rt) + T0_PAD, D_HI, float(rf.min())]


def _core(rf, rt, rw, maxfev: int = 30000):
    """One weighted bounded fit (weights 1/sqrt(row power)). Returns (popt, perr) or None. maxfev 30000 is July's;
    a fit pinned at a bound can take 0.3-0.9 s, which is why the bootstrap uses ``profile_fc`` instead."""
    lo, hi = _bounds(rf, rt)
    if hi[2] <= lo[2]:
        return None
    p0 = [np.min(rt), 1.0, max(1.5, rf.min() - 0.15)]
    sigma = 1.0 / np.sqrt(np.maximum(rw, rw.max() * 1e-3))
    try:
        popt, pcov = curve_fit(group_delay, rf, rt, p0=p0, sigma=sigma, bounds=(lo, hi), maxfev=maxfev)
    except (RuntimeError, ValueError):
        return None
    return popt, np.sqrt(np.abs(np.diag(pcov)))


def fit(rf: np.ndarray, rt: np.ndarray, rw: np.ndarray) -> dict | None:
    """July dispersion fit with iterative trimming (up to 4 rounds; keep points within max(2.2 x rms, 0.4 ms)).
    Returns t0, D (ms), fc (kHz), fc_err (curve_fit, kept only for comparison), rms (ms), dspread (ms), the kept
    ridge points and the bounds in force, or None when fewer than 6 points or the fit fails."""
    if len(rf) < MIN_PTS:
        return None
    o = np.argsort(rf, kind="stable")
    rf, rt, rw = rf[o], rt[o], rw[o]
    keep = np.ones(len(rf), bool)
    for _ in range(4):
        r = _core(rf[keep], rt[keep], rw[keep])
        if r is None:
            return None
        res = rt - group_delay(rf, *r[0])
        rms = np.sqrt(np.nanmean(res[keep] ** 2))
        new = (np.abs(res) < max(2.2 * rms, 0.4)) & np.isfinite(res)
        if new.sum() < MIN_PTS or (new == keep).all():
            keep = new if new.sum() >= MIN_PTS else keep
            break
        keep = new
    r = _core(rf[keep], rt[keep], rw[keep])
    if r is None:
        return None
    popt, perr = r
    res = (rt - group_delay(rf, *popt))[keep]
    lo, hi = _bounds(rf[keep], rt[keep])
    return dict(t0=float(popt[0]), D=float(popt[1]), fc=float(popt[2]), fc_err=float(perr[2]),
                rms=float(np.sqrt(np.nanmean(res ** 2))), dspread=float(np.nanmax(rt[keep]) - np.nanmin(rt[keep])),
                npts=int(keep.sum()), rmin=float(rf[keep].min()), rmax=float(rf[keep].max()),
                fc_lo=float(lo[2]), fc_hi=float(hi[2]), rf=rf[keep], rt=rt[keep], rw=rw[keep])


def textbook(ft: dict | None) -> bool:
    """July textbook gates: >= 12 kept points, fc 1.35-2.6 kHz, rms <= 0.45 ms, fc_err <= 0.2 kHz, delay spread
    2-13 ms, ridge reaching >= 5 kHz at the top and <= 2.6 kHz at the bottom."""
    if ft is None:
        return False
    return bool(ft["npts"] >= TB_NPTS and TB_FC[0] <= ft["fc"] <= TB_FC[1] and ft["rms"] <= TB_RMS
                and ft["fc_err"] <= TB_FCERR and TB_SPREAD[0] <= ft["dspread"] <= TB_SPREAD[1]
                and ft["rmax"] >= TB_RMAX and ft["rmin"] <= TB_RMIN)


def bound_flags(fc, fc_hi, D, fc_lo: float = FC_LO, tol: float = BOUND_TOL_KHZ, d_flag: float = D_FLAG_MS) -> dict:
    """Bound flags of one fit: fc within ``tol`` kHz of its upper bound (lowest ridge frequency) or lower bound
    (1.4 kHz), D below ``d_flag`` ms. ``bound_free`` is True when none is set. Works on arrays too."""
    fc, fc_hi, D = np.asarray(fc, float), np.asarray(fc_hi, float), np.asarray(D, float)
    hi = np.abs(fc - fc_hi) < tol
    lo = np.abs(fc - fc_lo) < tol
    dfl = D < d_flag
    return dict(at_fc_hi=hi, at_fc_lo=lo, at_d_floor=dfl, bound_free=~(hi | lo | dfl))


def _box_ls(g: np.ndarray, rt: np.ndarray, w: np.ndarray, tlo: float, thi: float):
    """Weighted least squares of rt = t0 + D g for every row of g at once, with t0 in [tlo, thi] and D in [D_LO, D_HI].
    The SSE is a convex quadratic in (t0, D), so its box minimum is the free minimum when that is inside the box,
    else the best of the four edge minima (each a 1-D minimum clipped to its edge, which also covers the corners).
    Returns (t0, D, sse) arrays."""
    W = w.sum()
    Sg, St = (g * w).sum(1), (rt * w).sum()
    Sgg, Sgt = (g * g * w).sum(1), (g * rt * w).sum(1)
    Stt = (rt * rt * w).sum()

    def sse(t0, D):
        return Stt + W * t0 ** 2 + D ** 2 * Sgg - 2 * t0 * St - 2 * D * Sgt + 2 * t0 * D * Sg

    det = W * Sgg - Sg ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        Du = np.where(det > 0, (W * Sgt - Sg * St) / np.where(det > 0, det, 1.0), np.nan)
        tu = (St - Du * Sg) / W
    inside = np.isfinite(Du) & (Du >= D_LO) & (Du <= D_HI) & (tu >= tlo) & (tu <= thi)
    cands = [(np.where(inside, tu, 0.0), np.where(inside, Du, D_LO), np.where(inside, sse(tu, Du), np.inf))]
    for Dv in (D_LO, D_HI):                                   # D on an edge, best t0 clipped
        t = np.clip((St - Dv * Sg) / W, tlo, thi)
        D = np.full_like(t, Dv)
        cands.append((t, D, sse(t, D)))
    for tv in (tlo, thi):                                     # t0 on an edge, best D clipped
        D = np.clip((Sgt - tv * Sg) / np.maximum(Sgg, 1e-300), D_LO, D_HI)
        t = np.full_like(D, tv)
        cands.append((t, D, sse(t, D)))
    S = np.stack([c[2] for c in cands])
    k = np.argmin(S, axis=0)
    j = np.arange(S.shape[1])
    return (np.stack([c[0] for c in cands])[k, j], np.stack([c[1] for c in cands])[k, j], S[k, j])


def profile_fit(rf: np.ndarray, rt: np.ndarray, rw: np.ndarray, step_khz: float = PROFILE_STEP_KHZ) -> dict | None:
    """Global best (t0, D, fc) of the July model under ALL the July bounds, by profiling fc on a 1 Hz grid.

    Same problem ``_core`` hands curve_fit: weights rw (curve_fit's sigma = 1/sqrt(rw) minimises sum rw r^2, with rw
    floored at 1e-3 of its max), t0 in [min t - 8, max t + 8] ms, D in [1e-3, 1e3] ms, fc in [1.4 kHz, lowest ridge
    frequency]. For a fixed fc the model t = t0 + D g(f), g = f / sqrt(f^2 - fc^2), is linear in (t0, D), so each
    grid fc gets an exact box-constrained least squares (``_box_ls``) and the fc with the lowest weighted SSE wins.
    No local optimiser: curve_fit at a bound took 0.3-0.9 s per refit and failed to converge on up to 60 % of
    bootstrap resamples of the July ridges. (An earlier version weighted by 1/rw and ignored the t0 bounds; the step-7
    verifier measured it 42 Hz median and up to 470 Hz away from curve_fit. Fixed 2026-11-14.) None when
    rmin <= 1.4 kHz. Returns dict(t0, D, fc, sse)."""
    hi = float(rf.min())
    if hi <= FC_LO:
        return None
    fcs = np.arange(FC_LO, hi, step_khz)
    if fcs[-1] < hi:
        fcs = np.r_[fcs, hi]
    w = np.maximum(rw, rw.max() * 1e-3)
    with np.errstate(divide="ignore", invalid="ignore"):
        g = rf[None, :] / np.sqrt(rf[None, :] ** 2 - fcs[:, None] ** 2)
    ok = np.isfinite(g).all(axis=1)
    if not ok.any():
        return None
    fcs, g = fcs[ok], g[ok]
    t0, D, sse = _box_ls(g, rt, w, float(np.min(rt)) - T0_PAD, float(np.max(rt)) + T0_PAD)
    i = int(np.argmin(sse))
    return dict(t0=float(t0[i]), D=float(D[i]), fc=float(fcs[i]), sse=float(sse[i]))


def profile_fc(rf: np.ndarray, rt: np.ndarray, rw: np.ndarray, step_khz: float = PROFILE_STEP_KHZ) -> float:
    """fc (kHz) of ``profile_fit``; NaN when there is no admissible fc."""
    r = profile_fit(rf, rt, rw, step_khz)
    return math.nan if r is None else r["fc"]


def weighted_sse(rf, rt, rw, t0, D, fc) -> float:
    """The objective ``_core`` minimises: sum of rw (floored at 1e-3 max) x residual^2."""
    w = np.maximum(rw, rw.max() * 1e-3)
    return float((((rt - group_delay(rf, t0, D, fc)) ** 2) * w).sum())


GLOBAL_TOL_KHZ = 0.002      # curve_fit fc this far from the 1 Hz profile optimum = curve_fit stopped elsewhere


def bootstrap_fc(ft: dict, n_boot: int = N_BOOT, rng: np.random.Generator | None = None) -> dict:
    """fc 95 % CI (kHz) from a pairs bootstrap of the kept ridge points, every refit by ``profile_fit`` (same model,
    weights and bounds as ``fit``; the fc bound moves with the resample's lowest frequency).

    The CI is a CI of ONE estimator, the profile optimum of the full kept ridge (``fc_profile``): the 2.5-97.5
    percentiles of the resample optima, each side widened in quadrature by the ridge grid quantisation (uniform, step
    91.55 Hz: 95 % half-width 0.95 x step / 2) about fc_profile. The upper end is clipped to the full ridge's fc bound
    (its lowest frequency) because the method cannot return more; ``ci_hi_at_bound`` says so (the verifier saw a
    resample-only upper end of 4.1 kHz on a July fit). When curve_fit's fc differs from fc_profile by more than 2 Hz
    and has a larger weighted SSE, ``fit_not_global`` is set: curve_fit stopped at a local optimum or short of the
    bound, and the CI does not describe its fc. A fit pinned at a bound gets a CI hugging that bound; that is what the
    bound flags are for. The CI is statistical only: it does NOT include the method bias of ``synthetic_bias``.
    Measured over the 768 textbook fits of the July-mixer corpus run (2026-11-14): |fc - fc_profile| median 4.4 Hz,
    90th percentile 69 Hz, 99th 176 Hz; 463 flagged fit_not_global (curve_fit, maxfev 30000, stopped short of the
    global optimum); the CI covered curve_fit's fc in 763; the upper end reached the rmin bound in 732 (any fc within
    43.5 Hz of rmin reaches it through the grid term alone).
    Returns dict(fc_ci=[lo, hi], fc_profile, n_boot_ok, ci_hi_at_bound, fit_not_global, sse_fit, sse_profile)."""
    rng = rng or np.random.default_rng(0)
    rf, rt, rw = ft["rf"], ft["rt"], ft["rw"]
    q = 0.95 * ridge_step_khz() / 2.0
    cap = float(rf.min())
    pf = profile_fit(rf, rt, rw)
    c = pf["fc"] if pf else ft["fc"]
    s_fit = weighted_sse(rf, rt, rw, ft.get("t0", math.nan), ft.get("D", math.nan), ft["fc"]) if "t0" in ft else math.nan
    s_pro = pf["sse"] if pf else math.nan
    not_global = bool(pf is not None and abs(ft["fc"] - c) > GLOBAL_TOL_KHZ and not (s_fit <= s_pro * (1 + 1e-9)))
    n = len(rf)
    out = []
    for _ in range(n_boot):
        i = rng.integers(0, n, n)
        if len(np.unique(rf[i])) < MIN_PTS:
            continue
        v = profile_fc(rf[i], rt[i], rw[i])
        if np.isfinite(v):
            out.append(v)
    if len(out) < 20:
        lo, hi = c - q, c + q
    else:
        plo, phi = np.percentile(out, [2.5, 97.5])
        lo = c - math.hypot(max(c - plo, 0.0), q)
        hi = c + math.hypot(max(phi - c, 0.0), q)
    at_bound = hi >= cap
    hi = min(hi, cap)
    return dict(fc_ci=[float(lo), float(hi)], fc_profile=float(c), n_boot_ok=len(out), ci_hi_at_bound=bool(at_bound),
                fit_not_global=not_global, sse_fit=s_fit, sse_profile=s_pro)


def h_text(fc: float, lo: float, hi: float) -> str:
    """'h 94.0 km (95 % CI 90.1-98.2)' from fc and its CI in kHz. The only way this module prints a height."""
    return f"h {h_km(fc):.1f} km (95 % CI {h_km(hi):.1f}-{h_km(lo):.1f})"


def median_ci(v, n_boot: int = 2000, rng: np.random.Generator | None = None) -> tuple[float, float, float]:
    """Median of v with a 95 % percentile-bootstrap CI (resampling the values). NaNs when v is empty."""
    v = np.asarray(v, float)
    if v.size == 0:
        return (math.nan, math.nan, math.nan)
    rng = rng or np.random.default_rng(1)
    b = np.median(v[rng.integers(0, v.size, (n_boot, v.size))], axis=1)
    lo, hi = np.percentile(b, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)


def analyse_window(xr: np.ndarray, fs: float = FS, n_boot: int = 0, rng=None) -> dict:
    """Ridge + fit + gates + flags of one 45 ms real window. With n_boot > 0 a textbook fit also gets its
    bootstrap fc CI (``fc_ci``) and h with CI. Returns a flat JSON-ready dict (``fit`` False when no fit)."""
    rf, rt, rw = ridge(xr, fs)
    ft = fit(rf, rt, rw) if len(rf) >= MIN_PTS else None
    if ft is None:
        return {"fit": False, "textbook": False}
    fl = bound_flags(ft["fc"], ft["fc_hi"], ft["D"])
    out = {"fit": True, "textbook": textbook(ft), **{k: ft[k] for k in
           ("t0", "D", "fc", "fc_err", "rms", "dspread", "npts", "rmin", "rmax", "fc_hi")},
           **{k: bool(v) for k, v in fl.items()}, "d_km": ft["D"] * C_KM_PER_MS, "h_km": float(h_km(ft["fc"]))}
    if n_boot and out["textbook"]:
        b = bootstrap_fc(ft, n_boot, rng)
        lo, hi = b["fc_ci"]
        out.update(fc_ci=[lo, hi], h_ci=[float(h_km(hi)), float(h_km(lo))], n_boot_ok=b["n_boot_ok"],
                   fc_profile=b["fc_profile"], ci_hi_at_bound=b["ci_hi_at_bound"], fit_not_global=b["fit_not_global"])
    return out


# ------------------------------------------------------------------ gates and the null

def capture_trains(x: np.ndarray, fs: float = FS, flags: list[dict] | None = None):
    """Run the v2 pipeline (``pipeline.run``, defaults) and keep the mains trains it judged with.

    The pipeline reports trains only as rounded summaries, so ``mainsgate.gate`` is wrapped for the duration of the
    call (restored in ``finally``) to keep the Train objects; the null windows are then judged against exactly the
    trains the events were. Returns (events, info, flags, trains)."""
    from sferic_watch import humflag as hf
    from sferic_watch import mainsgate as mg
    from sferic_watch import pipeline as pl
    if flags is None:
        flags = hf.flag_iq(x, fs)
    kept, orig = [], mg.gate

    def spy(*a, **k):
        r = orig(*a, **k)
        kept.extend(r[2])
        return r
    mg.gate = spy
    try:
        ev, info = pl.run(x, fs, flags=flags)
    finally:
        mg.gate = orig
    return ev, info, flags, kept


def hum_mask(t: np.ndarray, flags: list[dict]) -> np.ndarray:
    """True where a time (s) lies in a hum-flagged block of ``flags`` (humflag.flag_iq blocks)."""
    t = np.asarray(t, float)
    m = np.zeros(t.size, bool)
    for b in flags:
        if b.get("hum"):
            m |= (t >= b["t0_s"]) & (t < b["t0_s"] + b["dur_s"])
    return m


def admissible(t: np.ndarray, flags: list[dict], trains: list) -> np.ndarray:
    """The time gates a null window must pass, the same ones a candidate event passed: not in a hum-flagged block,
    and not inside any mains train's reject window (+-0.2 ms, within its span). Peaks are -inf so the 6 dB escape
    never lets a random time through."""
    from sferic_watch import mainsgate as mg
    t = np.asarray(t, float)
    ok = ~hum_mask(t, flags)
    if trains:
        rej, _ = mg.judge(t, np.full(t.size, -np.inf), trains)
        ok &= ~rej
    return ok


def candidates(ev: dict) -> np.ndarray:
    """Indices of events usable as tweek candidates: 16-30 kHz confirmed and not in a hum-flagged block. (Buzz
    impulses rejected by the mains gate are no longer events at all.)"""
    return np.flatnonzero(np.asarray(ev["conf"], bool) & ~np.asarray(ev["hum"], bool))


def fit_at(x: np.ndarray, t_on: np.ndarray, fs: float = FS, n_boot: int = 0, seed: int = 0, n0: int = 0,
           phase: str = "july") -> list[dict]:
    """Analyse the July window (onset - 6 ms, 45 ms) at each onset time (s from x[0]; x[0] is absolute sample n0).
    ``phase`` picks the mixer (``window_real``). Windows that would run off x are skipped (returned as
    {"fit": False, "skipped": True}) and are not counted as analysed."""
    win, pre = int(round(WIN_S * fs)), int(round(PRE_S * fs))
    rng = np.random.default_rng(seed)
    out = []
    for t in np.asarray(t_on, float):
        s = int(round(t * fs)) - pre
        if s < 0 or s + win > len(x):
            out.append({"t": float(t), "fit": False, "textbook": False, "skipped": True})
            continue
        r = analyse_window(window_real(x[s:s + win], n0 + s, fs, phase=phase), fs, n_boot, rng)
        r["t"] = float(t)
        out.append(r)
    return out


def verdict(n_cand: int, n_tb: int, n_null: int, n_null_tb: int, n_bound_free: int,
            x: float = YIELD_X, min_bf: int = MIN_BOUND_FREE) -> dict:
    """The step-7 rule. yield = textbook / candidates, null = textbook / admissible random windows. "insufficient"
    unless yield >= x * null AND bound-free textbook fits >= min_bf. When the null has zero hits its rate is taken
    as < 1/n_null (the rule then compares against 1/n_null, so a handful of events cannot beat an empty null by
    luck). Also reports a one-sided Poisson p of seeing >= n_tb textbook events at the null rate."""
    yl = n_tb / n_cand if n_cand else math.nan
    nr = n_null_tb / n_null if n_null else math.nan
    nr_rule = max(n_null_tb, 1) / n_null if n_null else math.nan
    reasons = []
    if not n_cand:
        reasons.append("no candidates")
    if not n_null:
        reasons.append("no admissible null windows")
    if n_cand and n_null and not yl >= x * nr_rule:
        reasons.append(f"yield {yl:.4f} < {x:g} x null {nr_rule:.4f}")
    if n_bound_free < min_bf:
        reasons.append(f"{n_bound_free} bound-free textbook fits < {min_bf}")
    p = float(poisson.sf(n_tb - 1, n_cand * nr_rule)) if (n_cand and n_null and n_tb) else math.nan
    return dict(status="insufficient" if reasons else "above null", reasons=reasons, yield_rate=yl, null_rate=nr,
                null_rate_rule=nr_rule, ratio=(yl / nr_rule) if (n_cand and n_null) else math.nan, p_poisson=p,
                n_cand=n_cand, n_textbook=n_tb, n_null=n_null, n_null_textbook=n_null_tb, n_bound_free=n_bound_free)


def summarise(fits: list[dict], rng=None) -> dict:
    """Counts and, over bound-free textbook fits, median fc with a bootstrap 95 % CI and h with its CI."""
    tb = [f for f in fits if f.get("textbook")]
    bf = [f for f in tb if f.get("bound_free")]
    out = dict(n=len(fits), n_fit=sum(bool(f.get("fit")) for f in fits), n_textbook=len(tb), n_bound_free=len(bf),
               n_at_fc_hi=sum(bool(f.get("at_fc_hi")) for f in tb), n_at_fc_lo=sum(bool(f.get("at_fc_lo")) for f in tb),
               n_at_d_floor=sum(bool(f.get("at_d_floor")) for f in tb),
               n_skipped=sum(bool(f.get("skipped")) for f in fits),
               n_fit_not_global=sum(bool(f.get("fit_not_global")) for f in tb))
    if tb:
        out["fc_median_textbook_all"] = float(np.median([f["fc"] for f in tb]))
    if bf:
        m, lo, hi = median_ci([f["fc"] for f in bf], rng=rng)
        how = "bootstrap of the median"
        if len(bf) < SMALL_N:
            # a bootstrap median of 1-4 values has a near-zero width; use the envelope of the per-fit CIs instead
            cis = [f["fc_ci"] for f in bf if "fc_ci" in f]
            q = 0.95 * ridge_step_khz() / 2.0
            lo = min([c[0] for c in cis] + [m - q])
            hi = max([c[1] for c in cis] + [m + q])
            how = "envelope of per-fit CIs (n < 5)"
        out.update(fc_median_bound_free=m, fc_median_ci=[lo, hi], ci_method=how, h_bound_free=h_text(m, lo, hi))
    return out


def audit_capture(x: np.ndarray, fs: float = FS, n_null: int = N_NULL, seed: int = 7, n_boot: int = N_BOOT,
                  max_cand: int | None = None, gate: bool = True, phase: str = "july") -> dict:
    """Full step-7 audit of one in-memory capture (complex64 baseband, full scale 1).

    gate=True: candidates and null windows through the v2 gates. gate=False: the confirmation and hum gates are off
    (every v2 event is a candidate, every random window is admissible) - the July-style comparison. The pipeline's
    own mains gate still removes buzz impulses from the events in both modes; only the null is fully ungated.
    ``max_cand`` keeps the first N candidates in time (reported as truncated). Random null times are uniform on
    [1 s, T - 1 s] with ``seed``. ``phase``: the mixer of ``window_real`` (default July's). Candidates and null
    windows that run off the capture are not counted in the verdict. ``mirror_phase`` of the first 60 s is reported
    so the absolute-mixer gain |cos(phi)| of each capture is on record."""
    secs = len(x) / fs
    ev, info, flags, trains = capture_trains(x, fs)
    idx = candidates(ev) if gate else np.arange(len(ev["t_start"]))
    n_all = len(idx)
    if max_cand is not None:
        idx = idx[:max_cand]
    cf = fit_at(x, ev["t_start"][idx], fs, n_boot=n_boot, seed=seed, phase=phase)
    tnull = np.random.default_rng(seed).uniform(1.0, secs - 1.0, n_null)
    adm = admissible(tnull, flags, trains) if gate else np.ones(n_null, bool)
    nf = fit_at(x, tnull[adm], fs, n_boot=n_boot, seed=seed + 1, phase=phase)
    cs, ns = summarise(cf), summarise(nf)
    v = verdict(len(cf) - cs["n_skipped"], cs["n_textbook"], int(adm.sum()) - ns["n_skipped"], ns["n_textbook"],
                cs["n_bound_free"])
    mp = mirror_phase(x[: int(60 * fs)], fs)
    return dict(secs=secs, gate=gate, phase=phase, mirror=mp, n_events=int(len(ev["t_start"])), n_cand_total=n_all,
                n_cand_fitted=len(cf) - cs["n_skipped"],
                truncated=bool(max_cand is not None and n_all > max_cand), hum_frac=info.get("hum_frac"),
                gated_frac=info.get("gated_frac"), pipeline_version=info.get("version"),
                null=dict(n_drawn=n_null, n_admissible=int(adm.sum()), **ns), cand=cs, verdict=v,
                textbook_events=[_slim(f) for f in cf if f.get("textbook")],
                textbook_null=[_slim(f) for f in nf if f.get("textbook")])


def _slim(f: dict) -> dict:
    """JSON-ready rounding of one fit record."""
    return {k: (round(v, 5) if isinstance(v, float) else ([round(a, 5) for a in v] if isinstance(v, list) else v))
            for k, v in f.items()}


# ------------------------------------------------------------------ synthetic tweeks

def synth_tweek_iq(n: int, t_on: float, fc_khz: float = 1.8, d_ms: float = 6.67, amp: float = 1.0,
                   f_top_khz: float = 30.0, fs: float = FS, fc_tune: float = FC, real_rf: bool = False,
                   lo_phase_deg: float = 0.0) -> np.ndarray:
    """Complex baseband (centred ``fc_tune``) of an ideal TM1 tweek, peak |x| = amp: flat spectrum from the cutoff fc
    to f_top (true frequency) with phase -2 pi (t_on f + D sqrt(f^2 - fc^2)). Its group delay is then exactly
    t_on + D f / sqrt(f^2 - fc^2), the model the fit assumes, so any fc error on it is the method's own.
    d = c D: D = 6.67 ms is 2000 km. Circular (FFT) synthesis: keep n well above the tail you care about.

    real_rf=False: positive true frequencies only (an analytic signal: no mirror image, phase-insensitive).
    real_rf=True: what the receiver records from REAL RF: the tweek plus its mirror at -f - fc_tune (conjugate
    spectrum), both rotated by the receiver LO phase ``lo_phase_deg`` relative to the absolute mixer of
    ``window_real``. This is the case where .real of the re-centred IQ scales the tweek by cos(phase)."""
    fb = np.fft.fftfreq(n, 1 / fs)
    f = fb + fc_tune
    fc, D = fc_khz * 1e3, d_ms * 1e-3
    m = (f > fc) & (f < f_top_khz * 1e3)
    X = np.zeros(n, complex)
    X[m] = np.exp(-2j * np.pi * (t_on * f[m] + D * np.sqrt(f[m] ** 2 - fc ** 2)))
    if real_rf:
        fm = -f                                       # bin at baseband b holds true frequency -(b + fc_tune) mirrored
        mm = (fm > fc) & (fm < f_top_khz * 1e3) & ~m
        X[mm] = np.exp(+2j * np.pi * (t_on * fm[mm] + D * np.sqrt(fm[mm] ** 2 - fc ** 2)))
        X *= np.exp(-1j * math.radians(lo_phase_deg))
    x = np.fft.ifft(X)
    return (amp * x / np.abs(x).max()).astype(np.complex64)


def synthetic_bias(fcs=(1.6, 1.8, 2.0), d_kms=(1000.0, 2000.0, 4000.0), snr_rms: float = 0.003,
                   n_boot: int = N_BOOT, seed: int = 0) -> list[dict]:
    """The July ridge + fit on ideal synthetic tweeks (``synth_tweek_iq``, peak 1, complex noise rms ``snr_rms``),
    window placed at the high-frequency arrival t_on + D as the detector would. Per (fc, d): fitted fc, error, bound
    flags, textbook, and whether the bootstrap 95 % CI covers the truth. Measured 2026-11-14 (9 cases, noise rms
    0.003, July mixer, corrected bootstrap): fc errors from -0.26 to +0.26 kHz, about -0.2 to -0.26 kHz at
    d = 1000 km, -0.08 at 2000 km and +0.23 at 4000 km (the +16 ms ridge window truncates the near-cutoff tail of far
    tweeks; for near ones the 256-point STFT leaks energy below fc into the ridge). 8 of 9 are textbook and 8 of 9
    bound-free; the CI covered the truth in only 3 of 8 textbook cases (the bias is larger than the CI). The fitted
    d was 1600-2350 km whatever the true d (1000-4000 km). So even a bound-free July fc carries a method bias the
    bootstrap CI does not see; plan step 10 must fix the method. (The analytic synthetic has no mirror image, so it
    does not show the mixer-phase gain of ``window_real``; ``real_rf=True`` does.)"""
    rng = np.random.default_rng(seed)
    n = int(0.2 * FS)
    out = []
    for fc in fcs:
        for dk in d_kms:
            d_ms = dk / C_KM_PER_MS
            x = synth_tweek_iq(n, 0.05, fc, d_ms)
            x = x + ((rng.standard_normal(n) + 1j * rng.standard_normal(n)) * snr_rms / math.sqrt(2)).astype(np.complex64)
            r = fit_at(x, [0.05 + d_ms * 1e-3], n_boot=n_boot, seed=seed)[0]
            row = dict(fc_true=fc, d_true_km=dk, fit=r["fit"], textbook=r["textbook"])
            if r["fit"]:
                row.update(fc_fit=r["fc"], fc_err_khz=r["fc"] - fc, d_fit_km=r["d_km"], bound_free=r["bound_free"],
                           rmin=r["rmin"])
                if "fc_ci" in r:
                    row.update(fc_ci=r["fc_ci"], covers=bool(r["fc_ci"][0] <= fc <= r["fc_ci"][1]),
                               fc_profile=r["fc_profile"], fit_not_global=r["fit_not_global"])
            out.append(row)
    return out


# ------------------------------------------------------------------ the July JSON

def july_reaudit(path: str = "~/Projects/dlayer-diary/tweeks/data/tweeks_fits.json") -> dict:
    """Re-audit the published July fits (read only). Pin counts at 0.5 / 5 / 10 / 20 Hz from the fc upper bound
    (the fit's rmin), D < 0.01 ms, and the bound-free set two ways: the plan's audit tolerance (>= 20 Hz from rmin,
    > 10 Hz above 1.4 kHz, D >= 0.01 ms; expected 19, median 1.595 kHz) and this module's 10 Hz rule. Every median
    gets a bootstrap 95 % CI. The 6 fits that ship ridge points also get their per-fit bootstrap CI (equal weights: the
    July JSON has no row powers; the CI and its h are about the equal-weight profile fc, printed beside it)."""
    d = json.load(open(os.path.expanduser(path)))
    T = d["tweeks"]
    fc = np.array([t["fc"] for t in T])
    rmin = np.array([t["rmin"] for t in T])
    D = np.array([t["D"] for t in T])
    gap_hz = np.abs(fc - rmin) * 1000
    out = dict(n=len(T), published_fc_median=d.get("fc_median_khz"), published_h_median=d.get("h_median_km"),
               pinned={f"{tol:g}Hz": int((gap_hz < tol).sum()) for tol in (0.5, 5, 10, 20)},
               d_below_0p01=int((D < D_FLAG_MS).sum()),
               d_km_pct_10_25_50_75_90=[round(float(v), 1) for v in np.percentile(D * C_KM_PER_MS, [10, 25, 50, 75, 90])],
               d_in_1000_5000=int(((D * C_KM_PER_MS >= 1000) & (D * C_KM_PER_MS <= 5000)).sum()),
               fc_err_median_hz=float(np.median([t["fc_err"] for t in T]) * 1000))
    m, lo, hi = median_ci(fc)
    out["all"] = dict(fc_median=m, fc_ci=[lo, hi], h=h_text(m, lo, hi))
    plan_bf = (gap_hz >= 20) & (fc - FC_LO > 0.010) & (D >= D_FLAG_MS)
    rule_bf = bound_flags(fc, rmin, D)["bound_free"]
    for name, msk in (("bound_free_plan20Hz", plan_bf), ("bound_free_rule10Hz", rule_bf)):
        m, lo, hi = median_ci(fc[msk])
        out[name] = dict(n=int(msk.sum()), fc_median=m, fc_ci=[lo, hi], h=h_text(m, lo, hi) if msk.any() else None)
    per = []
    rng = np.random.default_rng(3)
    for t in T:
        if "ridge_f" not in t:
            continue
        rf, rt = np.array(t["ridge_f"]), np.array(t["ridge_t"])
        k = np.array(t.get("ridge_keep", [True] * len(rf)), bool)
        ft = dict(fc=t["fc"], t0=t["t0"], D=t["D"], rf=rf[k], rt=rt[k], rw=np.ones(int(k.sum())))
        # row powers were not shipped: equal weights, so fc_profile can differ from the published (weighted) fc
        bs = bootstrap_fc(ft, N_BOOT, rng)
        lo, hi = bs["fc_ci"]
        fl = bound_flags(t["fc"], t["rmin"], t["D"])
        per.append(dict(tag=t["tag"], t=t["t"], fc=t["fc"], fc_err_published_hz=t["fc_err"] * 1000, fc_ci=[lo, hi],
                        fc_profile_equal_w=bs["fc_profile"], ci_hi_at_bound=bs["ci_hi_at_bound"],
                        h=h_text(bs["fc_profile"], lo, hi), bound_free=bool(fl["bound_free"])))
    out["with_ridge"] = per
    return out
