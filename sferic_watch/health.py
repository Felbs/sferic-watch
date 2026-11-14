"""Instrument health for one VLF capture (plan step 8).

What this module measures, per 600 s capture (62.5 kS/s IQ, centred 15 kHz):

* Sample-clock error (ppm) and LO offset (Hz) from the MSK transmitters.
  Idea from OnAir's MskRefTimer (squaring MSK gives two pure lines at
  2*carrier_offset +- baud/2); re-implemented here in our own structure.
  The baud rate (200 Bd) is set by the transmitter's atomic reference, so the
  baud we *measure* only differs from 200 because of our sample clock.  The
  mid-point of the two lines is the carrier offset.  On 3 captures the
  prototype found baud 200.00013-200.0002 (clock -0.46..-0.98 ppm) and a
  carrier offset of +0.5053..+0.5069 Hz shared by NAA and NML on every date,
  so the offset belongs to the tuner LO, not to the clock.  The phase-slope
  fit is cut at phase/timing jumps (see _refine_pair); one jump in the
  first minute of 20261020 0800Z had tilted a plain fit to +0.91 ppm.
* Transmitter levels with the "shoulder" metric: median PSD at 20-120 Hz
  offsets from the carrier against the median at 300-700 Hz.  The centre bin
  is skipped on purpose: a narrow local line sits exactly on 24.000 kHz and
  would otherwise be read as NAA.
* The 0 Hz receiver tone (about -22.6 dBFS, it dominates raw power) reported
  on its own.
* The 38.5 kHz hump, which is an NAA-derived spur (census over 148 captures:
  correlation 0.999 with NAA, 6.29 +- 0.44 dB below it), and NAU 40.75 kHz,
  usable only at night.
* Per-capture quarantine flags (clock, LO, short file, clipping, sidecar
  settings, 0 Hz tone off its usual level).

Corpus-level logic (two-state NAA level, deep fades) lives in
:func:`corpus_states`, because a single file cannot know the usual level.

All numbers here are relative (dB, ppm, Hz); no station position is used.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.signal import firwin, upfirdn, welch

FS = 62500.0            # sample rate of the banked captures
FC = 15000.0            # tuner centre
BAUD = 200.0            # NAA / NML / NAU MSK symbol rate
EXPECTED_SECS = 600.0   # banked capture length

# Shoulder-metric stations (kHz).  The 38.5 kHz entry is the NAA spur.
LEVEL_STATIONS = {"NAA": 24000.0, "NLK": 24800.0, "NML": 25200.0,
                  "SPUR38": 38500.0, "NAU": 40750.0}
# MSK clock stations.  NAU is too weak for the squared lines in the census.
CLOCK_STATIONS = {"NAA": 24000.0, "NML": 25200.0}

LINE_MIN_DB = 20.0      # squared line must be this far over its noise
BLOCK_SECS = 10.0       # level time-series resolution
DEC = 125               # 62.5 kS/s -> 500 S/s for the MSK lines

# References measured before this module existed (cited in docstrings).
REF_CLOCK_PPM = -0.7        # msk_clock.py, 3 captures: -0.46..-0.98 ppm
REF_LO_HZ = 0.506           # msk_clock.py: +0.5053..+0.5069 Hz
REF_DC_DBFS = -22.6         # 0 Hz tone, census
REF_SPUR_DB = -6.29         # SPUR38 minus NAA, census (148 captures)
CLOCK_TOL_PPM = 0.5         # plan step 8 quarantine limits
LO_TOL_HZ = 0.02
DC_TOL_DB = 3.0
SPUR_TOL_DB = 2.0           # about 4.5 x the 0.44 dB census spread


# ----------------------------------------------------------------- reading
def read_slices(path, secs=60.0):
    """Yield (first_sample_index, complex64 IQ scaled to full scale 1.0).

    Reads interleaved int16 in slices with np.fromfile(count=, offset=) so a
    150 MB capture never sits in memory as complex128.  Also yields the count
    of clipped samples (|int16| >= 32760) in the slice.
    """
    n = int(secs * FS)
    off = 0
    while True:
        raw = np.fromfile(path, np.int16, count=2 * n, offset=4 * off)
        if raw.size < 2:
            return
        raw = raw[: raw.size - raw.size % 2]
        clip = int(np.count_nonzero(np.abs(raw.astype(np.int32)) >= 32760))
        x = np.empty(raw.size // 2, np.complex64)
        x.real = raw[0::2]
        x.imag = raw[1::2]
        x *= np.float32(1.0 / 32768.0)
        yield off, x, clip
        off += x.size


# ----------------------------------------------------------- level metric
def shoulder_level(f, P, f0):
    """Return (dB over local floor, signal dBFS) for the carrier at f0.

    band = median PSD at 20-120 Hz offsets (both sides, MSK shoulders),
    floor = median PSD at 300-700 Hz offsets.  Medians make the metric
    robust to sferics and to the narrow local line on 24.000 kHz.  dBFS is
    (band - floor) times 240 Hz, the same scale the census used, so levels
    such as -76 (night) / -91 (sunrise dip) / -63 (high state) compare
    directly.  Returns floor-limited -inf-like values as -300 dBFS.
    """
    df = np.abs(f - f0)
    band = np.median(P[(df >= 20) & (df <= 120)])
    fl = np.median(P[(df >= 300) & (df <= 700)])
    over = 10 * np.log10(band / fl) if fl > 0 and band > 0 else 0.0
    sig = band - fl
    dbfs = 10 * np.log10(sig * 240.0) if sig > 0 else -300.0
    return float(over), float(dbfs)


def block_levels(x, fs=FS, fc=FC):
    """Levels of every LEVEL_STATIONS entry plus the 0 Hz tone for one block.

    Welch PSD with 6250-point segments (10 Hz bins).  The "0 Hz tone" is at
    0 Hz *RF*, i.e. at -15 kHz in the IQ data (measured on 20261106 2230Z:
    a line at -14999.5 Hz baseband holding -22.8 of the -22.6 dBFS total;
    the 0.5 Hz is the LO offset).  Its power is the PSD integrated over
    |f_rf| <= 25 Hz (density scaling, so sum*df is power even for a line).
    """
    f, P = welch(x, fs=fs, nperseg=6250, return_onesided=False, detrend=False)
    f = f + fc
    out = {}
    for name, frf in LEVEL_STATIONS.items():
        out[name] = shoulder_level(f, P, frf)
    df = fs / 6250
    out["DC"] = float(10 * np.log10(np.sum(P[np.abs(f) <= 25.0]) * df + 1e-30))
    return out


# ------------------------------------------------------------ MSK lines
_DEC_TAPS = firwin(2001, 180.0, fs=FS, window=("kaiser", 10.0)).astype(np.float32)
_TAIL = 2125            # multiple of DEC and >= len(_DEC_TAPS) - 1


@dataclass
class _Mixer:
    """Phase-continuous shift of one station to 0 Hz, then 125x decimation.

    The FIR state is carried across slices (a tail of old samples is
    prepended) so the 500 S/s output has no seams at slice boundaries; seams
    would put spurious sidebands next to the MSK lines.
    """
    f_base: float
    tail: np.ndarray = field(default_factory=lambda: np.zeros(_TAIL, np.complex64))
    out: list = field(default_factory=list)

    def push(self, n0, x):
        n = np.arange(n0, n0 + x.size, dtype=np.float64)
        ph = 2 * np.pi * np.mod(self.f_base * n, FS) / FS   # exact for integer Hz
        y = x * np.exp(-1j * ph).astype(np.complex64)
        buf = np.concatenate([self.tail, y])
        z = upfirdn(_DEC_TAPS, buf, down=DEC)
        k0 = _TAIL // DEC
        self.out.append(z[k0:k0 + x.size // DEC].astype(np.complex64))
        self.tail = buf[-_TAIL:]

    def series(self):
        return np.concatenate(self.out) if self.out else np.zeros(0, np.complex64)


def _coarse_peak(f, P, f0, w=3.0):
    """Parabolic (log-power) peak near f0 and its height over the 5-20 Hz noise."""
    m = np.abs(f - f0) < w
    idx = np.flatnonzero(m)
    k = idx[np.argmax(P[m])]
    a, b, c = np.log(P[k - 1:k + 2] + 1e-300)
    den = a - 2 * b + c
    d = 0.5 * (a - c) / den if den != 0 else 0.0
    nb = np.median(P[(np.abs(f - f0) > 5) & (np.abs(f - f0) < 20)])
    return f[k] + d * (f[1] - f[0]), float(10 * np.log10(P[k] / nb))


def _seg_phasors(z, fs2, f_est, seg_secs):
    """Complex DFT of each seg_secs piece of z taken exactly at f_est."""
    L = int(seg_secs * fs2)
    nseg = z.size // L
    t = np.arange(L) / fs2
    w = np.exp(-2j * np.pi * f_est * t)
    return np.array([np.sum(z[i * L:(i + 1) * L] * w) * np.exp(-2j * np.pi * f_est * i * L / fs2)
                     for i in range(nseg)])


def _wrap(a):
    return (np.asarray(a) + np.pi) % (2 * np.pi) - np.pi


def _breaks(c, seg_secs, floor_rad=0.3, cap_rad=2.0, amp_frac=0.5):
    """Mark where one line's segment phase cannot be trusted.

    Returns (bad_seg, bad_gap): bad_seg[i] = segment i is weak (amplitude
    under amp_frac of BOTH neighbours: a dropout or a glitch inside it; a
    real level step only puts the straddling segment between its
    neighbours, and the 13 dB NAA steps must not drop half the file as a
    median test did); bad_gap[i] = the phase step from
    segment i to i+1 is off the robust median step by more than
    max(5 sigma_MAD, 0.3 rad), capped at 2 rad because a wrapped step can
    never exceed pi.  The 0.3 rad floor catches the -0.55 rad jump the
    verifiers measured on the second baud line at 51 s of that capture; on
    a 40-50 dB line the step noise is ~0.01-0.05 rad, so it never trips.
    """
    amp = np.abs(c)
    left = np.r_[np.inf, amp[:-1]]
    right = np.r_[amp[1:], np.inf]
    bad_seg = amp < amp_frac * np.minimum(left, right)
    d = _wrap(np.angle(c[1:] * np.conj(c[:-1])))
    r = _wrap(d - np.median(d))
    sig = 1.4826 * np.median(np.abs(r))
    thr = min(max(5 * sig, floor_rad), cap_rad)
    return bad_seg, np.abs(r) > thr


def _piecewise_slope(c, keep, cut, seg_secs):
    """Common phase slope (rad/s) with a free phase offset per piece.

    `keep` marks usable segments, `cut[i]` breaks the chain between i and
    i+1.  Inside a piece the phase is unwrapped by summing wrapped steps
    about the robust median step, so one noisy step cannot slip a whole
    cycle into the rest of the file (the cause of -10.5 ppm on 11-03 1100Z).
    Returns (slope, standard error, number of points used, number of pieces).
    """
    n = c.size
    tc = (np.arange(n) + 0.5) * seg_secs
    d_all = _wrap(np.angle(c[1:] * np.conj(c[:-1])))
    med = np.median(d_all) if d_all.size else 0.0
    pieces, cur = [], []
    for i in range(n):
        if not keep[i]:
            if cur:
                pieces.append(cur)
            cur = []
            continue
        if cur and cut[i - 1]:
            pieces.append(cur)
            cur = []
        cur.append(i)
    if cur:
        pieces.append(cur)
    num = den = 0.0
    resid_parts = []
    used = 0
    for pc in pieces:
        if len(pc) < 2:
            continue
        idx = np.array(pc)
        steps = med + _wrap(d_all[idx[:-1]] - med)
        ph = np.concatenate([[0.0], np.cumsum(steps)])
        tt = tc[idx]
        tm, pm = tt.mean(), ph.mean()
        num += np.sum((tt - tm) * (ph - pm))
        den += np.sum((tt - tm) ** 2)
        resid_parts.append((tt - tm, ph - pm))
        used += len(pc)
    npieces = len(resid_parts)
    if den <= 0 or used - npieces < 2:
        return float("nan"), float("nan"), used, npieces
    slope = num / den
    rss = sum(np.sum((p - slope * t) ** 2) for t, p in resid_parts)
    se = np.sqrt(rss / max(used - npieces - 1, 1) / den)
    return float(slope), float(se), used, npieces


def _robust_rate(c, keep, cut, seg_secs, nsub=4):
    """Phase slope of c (rad/s) and an honest 1-sigma.

    sigma = max(residual-based standard error, spread of the slopes of
    nsub sub-windows / nsub**1.5).  For white phase noise a slope over 1/nsub
    of the points has nsub**1.5 times the error, so both terms estimate the
    same number; the spread term also catches what the residuals cannot
    (slow wander, an undetected slip).  The verifiers asked for exactly this
    after the first version's error sat 4-8 sigma from the truth.
    """
    s, se, used, _ = _piecewise_slope(c, keep, cut, seg_secs)
    n = c.size
    sub = []
    for j in range(nsub):
        k = np.zeros(n, bool)
        k[j * n // nsub:(j + 1) * n // nsub] = True
        sj = _piecewise_slope(c, keep & k, cut, seg_secs)[0]
        if np.isfinite(sj):
            sub.append(sj)
    if len(sub) >= 3 and np.isfinite(se):
        se = max(se, float(np.std(sub, ddof=1)) / nsub ** 1.5)
    return s, se


def _refine_pair(z, fs2, f_plus, f_minus, seg_secs=20.0):
    """Refine the two baud lines by a glitch-proof phase-slope fit.

    The DFT of each seg_secs piece is taken exactly at the coarse line
    frequency; a residual error df makes the phase advance 2*pi*df*seg_secs
    per segment.  This beats FFT-bin interpolation by about 10x on a 600 s
    capture (bin 1.7 mHz = 8 ppm of 200 Hz; we need << 0.5 ppm = 0.1 mHz).

    The baud (f+ - f-) is fitted on the phase of c+ * conj(c-), and the
    carrier (f+ + f-) on c+ * c-.  In the difference the carrier phase and
    the ionospheric path phase cancel, so the clock sees only symbol timing.

    Why it is robust (verifier findings on the first version, which fitted
    one straight line per line through all 30 points):

    * 20261020 0800Z has a symbol-timing / phase discontinuity at about
      51 s (+line -2.0 rad, -line +0.55 rad, raw power flat).  One straight
      line through it read +0.91 ppm; the file without its first 60 s reads
      -0.74 ppm, the corpus median.  Breaks found on EITHER line cut the
      chain for both, and every piece between breaks gets its own phase
      offset, so a step in phase cannot tilt the slope.
    * Weak segments (amplitude < half the median) are dropped.

    Returns (baud_hz, carrier2_hz, baud_err_hz, carrier2_err_hz, n_breaks),
    where carrier2 = f+ + f- (= 4 x carrier offset).
    """
    cp = _seg_phasors(z, fs2, f_plus, seg_secs)
    cm = _seg_phasors(z, fs2, f_minus, seg_secs)
    nan = float("nan")
    if cp.size < 3:
        return f_plus - f_minus, f_plus + f_minus, nan, nan, 0
    bs_p, bg_p = _breaks(cp, seg_secs)
    bs_m, bg_m = _breaks(cm, seg_secs)
    keep = ~(bs_p | bs_m)
    cut = bg_p | bg_m
    n_breaks = int(np.sum(cut) + np.sum(~keep))
    sd, ed = _robust_rate(cp * np.conj(cm), keep, cut, seg_secs)
    ss, es = _robust_rate(cp * cm, keep, cut, seg_secs)
    two_pi = 2 * np.pi
    return (f_plus - f_minus + sd / two_pi, f_plus + f_minus + ss / two_pi,
            ed / two_pi, es / two_pi, n_breaks)


def msk_clock(y, fs2=FS / DEC, baud=BAUD):
    """Sample-clock ppm and carrier offset from a station shifted to ~0 Hz.

    Squaring MSK removes the data and leaves lines at 2*off +- baud/2.  Then
    baud_meas = f+ - f-  and  off = (f+ + f-)/4.  If our clock runs fast by
    eps, a true 200 Bd appears as 200*(1 - eps), so clock_ppm = (1 -
    baud_meas/200)*1e6 (positive = our clock fast; sign confirmed on the
    corpus by NML-minus-NAA carrier offset +0.71 mHz vs +0.88 predicted).
    Returns a dict with the weaker line's SNR (line_db), clock_ppm, its
    1-sigma, lo_hz and glitches (segments dropped + chain breaks, see
    _refine_pair).
    """
    z = (y.astype(np.complex128)) ** 2
    if z.size < 8 * int(fs2):
        return {"line_db": float("nan"), "clock_ppm": float("nan"),
                "clock_err_ppm": float("nan"), "lo_hz": float("nan"), "glitches": 0}
    nfft = 1 << int(np.ceil(np.log2(z.size)))
    Z = np.abs(np.fft.fft(z * np.hanning(z.size), nfft)) ** 2
    f = np.fft.fftfreq(nfft, 1 / fs2)
    o = np.argsort(f)
    f, Z = f[o], Z[o]
    fp, sp = _coarse_peak(f, Z, +baud / 2)
    fm, sm = _coarse_peak(f, Z, -baud / 2)
    b, c2, eb, _, nb = _refine_pair(z, fs2, fp, fm)
    return {"line_db": float(min(sp, sm)),
            "clock_ppm": float((1 - b / baud) * 1e6),
            "clock_err_ppm": float(eb / baud * 1e6),
            "lo_hz": float(c2 / 4.0),
            "glitches": int(nb)}


# ------------------------------------------------------------- one capture
def read_sidecar(path):
    """Load the JSON sidecar next to a .cs16 capture, or {} if absent."""
    p = Path(path).with_suffix(".json")
    try:
        return json.loads(p.read_text())
    except (OSError, ValueError):
        return {}


def sidecar_problems(meta):
    """List recipe deviations from the banked recipe (AGC off, IFGR 40, ...)."""
    bad = []
    if not meta:
        return ["no_sidecar"]
    if float(meta.get("fs_hz", FS)) != FS:
        bad.append("fs")
    if float(meta.get("freq_hz", FC)) != FC:
        bad.append("freq")
    if meta.get("agc") not in (False, "false", 0, None):
        bad.append("agc")
    if float(meta.get("IFGR", 40.0)) != 40.0:
        bad.append("ifgr")
    if str(meta.get("biasT_ctrl", "false")).lower() not in ("false", "0"):
        bad.append("biasT")
    if int(meta.get("overflows", 0) or 0) > 0:
        bad.append("overflow")
    return bad


def scan_capture(path, block_secs=BLOCK_SECS):
    """One pass over a capture: per-block levels and MSK clock per station.

    Returns a dict with 'secs', 'clipped', per-block arrays 'blocks'
    (name -> (over_db array, dbfs array)), 'dc' (dBFS array) and 'msk'
    (station -> msk_clock dict).
    """
    mixers = {s: _Mixer(f - FC) for s, f in CLOCK_STATIONS.items()}
    nb = int(block_secs * FS)
    lv = {s: ([], []) for s in LEVEL_STATIONS}
    dc = []
    clipped = 0
    total = 0
    # 60 s slices are a whole number of 10 s blocks and of DEC samples.
    for n0, x, clip in read_slices(path, secs=60.0):
        clipped += clip
        total += x.size
        usable = x.size - x.size % DEC
        for m in mixers.values():
            m.push(n0, x[:usable])
        for i in range(x.size // nb):
            r = block_levels(x[i * nb:(i + 1) * nb])
            for s in LEVEL_STATIONS:
                lv[s][0].append(r[s][0])
                lv[s][1].append(r[s][1])
            dc.append(r["DC"])
    msk = {s: msk_clock(m.series()) for s, m in mixers.items()}
    return {"secs": total / FS, "clipped": clipped,
            "blocks": {s: (np.array(a), np.array(b)) for s, (a, b) in lv.items()},
            "dc": np.array(dc), "msk": msk, "block_secs": block_secs}


def _holds(w, tol):
    """(holds, level): a window holds if it has >= 3 finite blocks and every
    finite block is within tol of the window median.  The shoulder metric
    is a median over 10 s, so sferics do not need an outlier allowance
    (night plateau 10 s jitter is ~0.3-1 dB against tol = 4 dB)."""
    w = w[np.isfinite(w)]
    if w.size < 3:
        return False, float("nan")
    m = float(np.median(w))
    return bool(np.all(np.abs(w - m) <= tol)), m


def find_step(dbfs, block_secs=BLOCK_SECS, min_db=8.0, hold=6, min_side=3):
    """Largest abrupt level step inside one capture.

    For each block k (the block the switch lands in, left out), the level
    before is the window of up to `hold` blocks (60 s) ending at k and the
    level after is the window of up to `hold` blocks starting at k+1.  The
    windows are cut only by the file edges, never shortened otherwise, so
    a step is accepted when the new level holds for 60 s, or for every
    remaining block (at least `min_side` = 30 s) when the file ends first;
    likewise the old level.  "Holds" means every block is within min_db/2
    of the window median (see _holds), so a 20-50 s burst in mid-file is
    not a step, and a 20 s burst at either edge is not either.
    NaN blocks (station absent, see present_dbfs) are left out of a window
    instead of vetoing it; each side still needs 3 finite blocks.

    Measured scale: a real switch (the 13 dB two-state NAA level) moves
    >= 8 dB inside one 10 s block; ionospheric dips move about 1 dB per
    20 s (sunrise 11:00Z: -96 -> -87 over 480 s).  Within one file a 30 s
    excursion that runs into the file edge cannot be told from a real step
    (11-06 0630Z/0700Z switch at 565/555 s); such steps are reported.
    Returns (step_db, time_s at the centre of block k) of the largest
    |step| >= min_db, else (0.0, nan).
    """
    s = np.asarray(dbfs, float)
    tol = min_db / 2
    best, at = 0.0, float("nan")
    for k in range(min_side, s.size - min_side):
        ok_b, before = _holds(s[max(k - hold, 0):k], tol)
        ok_a, after = _holds(s[k + 1:k + 1 + hold], tol)
        if not (ok_b and ok_a):
            continue
        d = after - before
        if abs(d) >= min_db and abs(d) > abs(best):
            best, at = d, (k + 0.5) * block_secs
    return float(best), at


def level_change_at(dbfs, t_step, block_secs=BLOCK_SECS, side=3):
    """Median level change of another station across an NAA step time.

    Uses up to `side` blocks each side of the step block, NaN blocks left
    out; needs 2 finite blocks per side (one absent NAU block used to veto
    the whole comparison, leaving 11-03 2130Z/2230Z 'unclear').
    """
    s = np.asarray(dbfs, float)
    if not np.isfinite(t_step):
        return float("nan")
    k = int(round(t_step / block_secs - 0.5))
    a = s[k + 1:k + 1 + side]
    b = s[max(k - side, 0):k]
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if a.size < 2 or b.size < 2:
        return float("nan")
    return float(np.median(a) - np.median(b))


def present_dbfs(over, dbfs, min_over_db=1.0):
    """dBFS series with NaN where the station is absent (< 1 dB over floor).

    Measured need: NAA was off the air on 10-20 15:00-15:30Z; its shoulder
    then sits at the floor and dBFS reads -300, which turned into fake
    200 dB "steps" and "fades" in the first corpus run.
    """
    d = np.asarray(dbfs, float).copy()
    d[np.asarray(over, float) < min_over_db] = np.nan
    return d


def step_kind(naa_step, nau_step, nml_step):
    """Classify an NAA step by what NAU (40.75 kHz) and NML (25.2 kHz) did.

    A transmitter power/mode change moves NAA (and its 38.5 kHz spur) only.
    A gain change in our antenna or receiver moves every station by the same
    dB.  'common' = both others moved >= half the NAA step with the same
    sign (or the one that is present did: a receiver gain change cannot
    skip a station, so one witness is enough); 'naa_only' = both present
    and both moved < a quarter of it (needs both, because a transmitter
    claim is the stronger one); else 'unclear'.
    """
    if naa_step == 0.0:
        return ""
    o = [x for x in (nau_step, nml_step) if np.isfinite(x)]
    if o and all(np.sign(x) == np.sign(naa_step) and abs(x) >= abs(naa_step) / 2 for x in o):
        return "common"
    if len(o) == 2 and all(abs(x) < abs(naa_step) / 4 for x in o):
        return "naa_only"
    return "unclear"


def capture_health(path, ref_clock_ppm=REF_CLOCK_PPM, ref_lo_hz=REF_LO_HZ):
    """Scan one capture and return a flat dict row for health.csv.

    The clock / LO come from whichever of NAA / NML has a squared line
    >= 20 dB and the smaller clock standard error (11-03 1100Z, sunrise
    fade: NAA at a 21 dB line reads -2.19 +- 0.90 ppm, NML at 36 dB
    -0.84 +- 0.13; the first, non-robust fit had NAA at -10.5 there).  If neither qualifies they are NaN and
    the capture is flagged 'no_clock' (informational, not quarantine).  When
    both stations qualify, clock_agree_ppm is their difference, a check the
    prototype passed within 0.3 ppm.
    """
    return row_from_scan(path, read_sidecar(path), scan_capture(path),
                         ref_clock_ppm, ref_lo_hz)


def row_from_scan(path, meta, sc, ref_clock_ppm=REF_CLOCK_PPM, ref_lo_hz=REF_LO_HZ):
    """Turn a scan_capture() result into the flat health.csv row."""
    row = {"file": "/".join(Path(path).parts[-2:]), "utc": meta.get("utc", ""),
           "secs": round(sc["secs"], 2)}
    for s in LEVEL_STATIONS:
        over, dbfs = sc["blocks"][s]
        row[f"{s}_db"] = round(float(np.median(over)), 2) if over.size else np.nan
        row[f"{s}_dbfs"] = round(float(np.median(dbfs)), 2) if dbfs.size else np.nan
    bs = float(sc.get("block_secs", BLOCK_SECS))
    pres = {s: present_dbfs(*sc["blocks"][s]) for s in ("NAA", "NAU", "NML")}
    naa = pres["NAA"]
    row["NAA_present_frac"] = round(float(np.mean(np.isfinite(naa))), 3) if naa.size else 0.0
    sm = _smooth3(naa)
    ok = np.isfinite(sm)
    row["NAA_min_dbfs"] = round(float(np.min(sm[ok])), 2) if ok.any() else np.nan
    # block centre, the same convention as NAA_step_t_s
    row["NAA_min_t_s"] = float((np.flatnonzero(ok)[np.argmin(sm[ok])] + 0.5) * bs) if ok.any() else np.nan
    dn = np.diff(naa)
    dn = dn[np.isfinite(dn)]
    row["NAA_jitter_db"] = round(float(np.median(np.abs(dn))), 3) if dn.size else np.nan
    step, t_step = find_step(naa, block_secs=bs)
    row["NAA_step_db"], row["NAA_step_t_s"] = round(step, 2), t_step
    # does the same step show on NAU and NML?  A transmitter change would
    # move NAA (and its spur) only; a receiver/antenna change moves all.
    # Witness series: any measurable shoulder (over > 0 dB, i.e. not the
    # -300 sentinel), not the 1 dB presence floor.  On 11-03 2230Z NAU went
    # -97 -> -82 and NML -100 -> -88 with NAA, but sat under 1 dB before
    # the step, so the strict series left the step 'unclear'.
    wit = {s: present_dbfs(*sc["blocks"][s], min_over_db=1e-9) for s in ("NAU", "NML")}
    for s in ("NAU", "NML"):
        v = level_change_at(wit[s], t_step, block_secs=bs) if step != 0.0 else np.nan
        row[f"{s}_at_step_db"] = round(float(v), 2) if np.isfinite(v) else np.nan
    row["step_kind"] = step_kind(step, row["NAU_at_step_db"], row["NML_at_step_db"])
    row["spur_minus_naa_db"] = round(row["SPUR38_dbfs"] - row["NAA_dbfs"], 2)
    row["dc_tone_dbfs"] = round(float(np.median(sc["dc"])), 2) if sc["dc"].size else np.nan
    row["clipped"] = sc["clipped"]
    for s, r in sc["msk"].items():
        row[f"{s}_line_db"] = round(r["line_db"], 1)
        row[f"{s}_clock_ppm"] = round(r["clock_ppm"], 3)
        row[f"{s}_clock_err_ppm"] = round(r["clock_err_ppm"], 3)
        row[f"{s}_lo_hz"] = round(r["lo_hz"], 5)
        row[f"{s}_glitches"] = int(r.get("glitches", 0))
    cand = [s for s in ("NAA", "NML") if sc["msk"][s]["line_db"] >= LINE_MIN_DB]
    src = min(cand, key=lambda s: sc["msk"][s]["clock_err_ppm"]) if cand else None
    row["clock_src"] = src or ""
    row["clock_ppm"] = row[f"{src}_clock_ppm"] if src else np.nan
    row["clock_err_ppm"] = row[f"{src}_clock_err_ppm"] if src else np.nan
    row["lo_hz"] = row[f"{src}_lo_hz"] if src else np.nan
    row["clock_glitches"] = row[f"{src}_glitches"] if src else 0
    both = all(sc["msk"][s]["line_db"] >= LINE_MIN_DB for s in ("NAA", "NML"))
    row["clock_agree_ppm"] = round(row["NAA_clock_ppm"] - row["NML_clock_ppm"], 3) if both else np.nan
    # agreement in units of the combined 1-sigma: the honesty check of
    # clock_err_ppm (should scatter about +-1 if the errors are right)
    row["clock_pull"] = (round(row["clock_agree_ppm"] / float(np.hypot(row["NAA_clock_err_ppm"],
                                                                       row["NML_clock_err_ppm"])), 2)
                         if both else np.nan)
    q, info = quarantine_flags(row, meta, ref_clock_ppm, ref_lo_hz)
    row["quarantine"] = ";".join(q)
    row["info"] = ";".join(info)
    return row


def _smooth3(a):
    """3-block running median, so one sferic burst cannot set the minimum."""
    a = np.asarray(a, float)
    if a.size < 3:
        return a
    p = np.concatenate([a[:1], a, a[-1:]])
    return np.median(np.stack([p[:-2], p[1:-1], p[2:]]), axis=0)


def quarantine_flags(row, meta, ref_clock_ppm=REF_CLOCK_PPM, ref_lo_hz=REF_LO_HZ):
    """Split problems into quarantine (science must not use) and info flags.

    Quarantine: clock more than 0.5 ppm (and more than 3 standard errors,
    so a noisy estimate from a weak line cannot quarantine a good file; it
    gets the info flag clock_noisy when its error exceeds 0.25 ppm) or LO
    more than 0.02 Hz from the reference (plan step 8), short file, clipping, sidecar recipe deviation,
    0 Hz tone more than 3 dB off -22.6 dBFS (a gain or front-end change).
    Info: no_clock (no line >= 20 dB), clock_glitch (the chosen station's
    phase chain had breaks or dropped segments: a symbol-timing or phase
    jump such as 20261020 0800Z at 51 s, fitted around, not a clock
    fault), clock_disagree (NAA and NML differ by > 0.5 ppm and > 4 sigma),
    spur off its -6.29 dB tie to NAA by
    > 2 dB while NAA is >= 10 dB over floor, nau_night (NAU >= 3 dB, the
    night-only second path), naa_step (abrupt >= 8 dB step inside the file),
    naa_off (NAA absent in more than half of the 10 s blocks).
    """
    q, info = [], []
    if np.isfinite(row.get("clock_ppm", np.nan)):
        dev = abs(row["clock_ppm"] - ref_clock_ppm)
        err = row.get("clock_err_ppm", 0.0)
        err = 0.0 if not np.isfinite(err) else err
        if dev > CLOCK_TOL_PPM and dev > 3 * err:
            q.append("clock")
        if err > 0.25:
            info.append("clock_noisy")
        if row.get("clock_glitches", 0) > 0:
            info.append("clock_glitch")
        if abs(row["lo_hz"] - ref_lo_hz) > LO_TOL_HZ:
            q.append("lo")
    else:
        info.append("no_clock")
    if row["secs"] < EXPECTED_SECS - 0.5:
        q.append("short")
    if row["clipped"] > 0:
        q.append("clip")
    q += [f"sidecar_{b}" for b in sidecar_problems(meta)]
    if np.isfinite(row["dc_tone_dbfs"]) and abs(row["dc_tone_dbfs"] - REF_DC_DBFS) > DC_TOL_DB:
        q.append("dc_tone")
    if row["NAA_db"] >= 10 and abs(row["spur_minus_naa_db"] - REF_SPUR_DB) > SPUR_TOL_DB:
        info.append("spur_off")
    pull = row.get("clock_pull", np.nan)
    if np.isfinite(pull) and abs(pull) > 4 and abs(row.get("clock_agree_ppm", 0.0)) > CLOCK_TOL_PPM:
        info.append("clock_disagree")
    if row["NAU_db"] >= 3:
        info.append("nau_night")
    if row["NAA_step_db"] != 0.0:
        info.append("naa_step")
    if row.get("NAA_present_frac", 1.0) < 0.5:
        info.append("naa_off")
    return q, info


# ------------------------------------------------------------ corpus level
def corpus_states(rows, low_ref=None, high_db=6.5, fade_db=10.0):
    """Add NAA state / fade columns that need the whole corpus.

    low_ref is the usual NAA level: the median of all captures' NAA_dbfs
    (census: night plateau about -76 dBFS on 11-03..05).  A capture is
    'high' when its NAA level is more than half the measured 13 dB step
    (6.5 dB) above low_ref: the night level is either about -63 or about
    -76 dBFS.  naa_flag = high state or an abrupt in-file step.  A 'fade'
    is a 3-block-smoothed minimum more than fade_db below low_ref (sunrise
    dips reach about -91 dBFS at 11:00Z) while NAA is still present (an
    absent NAA is 'naa_off', not a fade).  Returns low_ref used.
    """
    lv = np.array([r["NAA_dbfs"] for r in rows if r.get("NAA_present_frac", 1.0) >= 0.5], float)
    if low_ref is None:
        low_ref = float(np.nanmedian(lv))
    for r in rows:
        if r.get("NAA_present_frac", 1.0) < 0.5:
            r["NAA_state"] = "off"
        else:
            r["NAA_state"] = "high" if r["NAA_dbfs"] > low_ref + high_db else "low"
        r["naa_flag"] = int(r["NAA_state"] == "high" or r["NAA_step_db"] != 0.0)
        r["naa_fade"] = int(r["NAA_min_dbfs"] < low_ref - fade_db)
    return low_ref
