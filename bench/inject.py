"""Injection pieces for the scoring harness: known sferic waveforms added into real air (or synthetic noise) at known times.

Three waveform sets:
  real     - 386 sferics cut from 10-19 15Z (the h2h bank). Their start times are frozen in corpus.json. They were picked
             because BOTH the July rule (A) and OnAir+band-pass found them within 2 ms, so this set flatters those two
             detectors; it is kept so the h2h numbers can be reproduced bit for bit.
  real_snr - the same cuts, cleaned (RF < 1.5 kHz incl. the 0 Hz receiver tone and RF > 30 kHz removed) and rescaled to
             a stated peak SNR. Only cuts whose 2.5-9 kHz peak sits at the labelled onset (-0.5..+1 ms) are used, so the
             stated SNR belongs to the sferic at the truth time and not to a later, stronger one in the same 16 ms
             window; of those, the strongest (own in-band SNR) are kept so rescaling lifts their copy of the source
             floor as little as possible (see real_snr_bank).
  synth    - damped oscillations (near, local-looking) and waveguide-dispersed pulses (distant, tweek-like), generated from
             a seed. No detector chose them, which removes the selection bias of the real set.

Peak SNR, as everywhere in this harness: 16-sample (256 us) smoothed power of the analytic 2.5-9 kHz band, peak of the
pulse over the host's median of 1 ms block means. That is the same yardstick the onair-review injection used, so
"12 dB" here means what it meant there.

The injection schedule copies h2h: 80 pulses per 120 s piece at 1 + 1.43 k + U(0, 0.3) s; truth = cut start + 4 ms
(the real cuts start 4 ms before the A-rule onset; synthetic pulses are built with their onset 4 ms in).
"""
import numpy as np
from scipy.signal.windows import tukey

FS = 62500.0
FC = 15000.0
C_KM_S = 299792.458
SNIP_N = 1000                          # 16 ms cut, as h2h
ONSET_S = 0.004                        # truth = cut start + 4 ms
H2H_TAPER = tukey(SNIP_N, 2 * 62 / SNIP_N)   # 1 ms raised-cosine ends, exactly as h2h/inject.py
H2H_ORDER = ["20261105_vlf_1500Z", "20261104_vlf_0100Z", "20261106_vlf_2200Z",
             "20261105_vlf_2000Z", "20261020_vlf_0800Z", "20261106_vlf_1000Z"]   # h2h's argv order (recovered: it alone
#                                                                                   reproduces all six .truth files)


# ---------------------------------------------------------------- band power yardstick
def band_analytic(x, fs=FS, fc=FC, lo=2500.0, hi=9000.0, nfft=None):
    """Analytic (one-sided) band lo..hi Hz RF of complex baseband x, by an FFT mask. Baseband f = RF - fc.

    A brick-wall mask is fine for a yardstick (peak and floor go through the same mask)."""
    n = len(x) if nfft is None else nfft
    X = np.fft.fft(x, n)
    rf = np.fft.fftfreq(n, 1 / fs) + fc
    X[(rf < lo) | (rf > hi)] = 0
    return np.fft.ifft(X)[: len(x)]


def smoothed_power(xb, n=16):
    """|xb|^2 smoothed by an n-sample boxcar (16 samples = 256 us at 62.5 kS/s, the OnAir-style integration)."""
    return np.convolve(np.abs(xb) ** 2, np.ones(n) / n, mode="same")


def median_floor(p, fs=FS, block_s=0.001):
    """Median of block means of a power series (1 ms blocks): the floor sferic peaks are measured against."""
    b = max(1, int(round(block_s * fs)))
    m = len(p) // b * b
    return float(np.median(p[:m].reshape(-1, b).mean(axis=1)))


def host_floor(x, fs=FS, fc=FC):
    """In-band (2.5-9 kHz) floor of a host piece in full-scale power units (10*log10 -> dBFS)."""
    return median_floor(smoothed_power(band_analytic(x, fs, fc)), fs)


def snip_peak_power(snip, fs=FS, fc=FC):
    """Peak 2.5-9 kHz smoothed power of one pulse (zero padded so the mask does not wrap its tail)."""
    return float(smoothed_power(band_analytic(snip, fs, fc, nfft=4 * len(snip))).max())


# ---------------------------------------------------------------- waveform banks
def real_bank(src_path, cut_start_samples, read_slice):
    """The h2h bank: 16 ms raw cuts starting at each frozen cut start (sample index), Tukey-tapered exactly as h2h did.

    Returned as complex128 in full-scale units. Raw cuts still carry the 0 Hz receiver tone and the source floor;
    at native amplitude that is what h2h injected, so this is the reproduction set."""
    out = np.empty((len(cut_start_samples), SNIP_N), dtype=np.complex128)
    for i, s0 in enumerate(cut_start_samples):      # integers: s0/FS*FS can land a hair under s0 and int() drops a sample
        out[i] = read_slice(src_path, int(s0) / FS, SNIP_N / FS, dc="none").astype(np.complex128) * H2H_TAPER
    return out


def clean_snip(snip, fs=FS, fc=FC, lo=1500.0, hi=30000.0):
    """Remove the receiver tone (RF ~0 Hz) and out-of-band junk from a raw cut, re-taper. Used before rescaling."""
    return band_analytic(snip, fs, fc, lo, hi, nfft=4 * len(snip)) * H2H_TAPER


def snip_peak_lag_s(snip, fs=FS, fc=FC):
    """Time (s) of the 2.5-9 kHz smoothed-power peak relative to the labelled onset (ONSET_S into the cut)."""
    p = smoothed_power(band_analytic(snip, fs, fc, nfft=4 * len(snip)))
    return float(np.argmax(p) / fs - ONSET_S)


def real_snr_bank(raw_bank, src_floor, min_native_db=30.0, min_count=24, lag_lo_s=-0.0005, lag_hi_s=0.001,
                  fs=FS, fc=FC):
    """Cleaned real cuts usable for SNR-scaled injection, plus their native peak SNR (dB over the source floor).

    Peak position. A 16 ms cut can hold a second, stronger sferic a few ms after the one at the labelled onset (truth =
    cut start + 4 ms). The gain is set from the cut's peak, so in such a cut the stated SNR would describe the later
    sferic and the labelled one would go in 15-27 dB weaker (verifier probe: 7 of the first bank's 24 cuts had their
    peak 2.8-7.8 ms late; July found 4 of them 0-6 % of the time at 30 dB). Only cuts whose peak lies within
    ``lag_lo_s``..``lag_hi_s`` of the onset are eligible: 349 of the 386 (most peak 0-0.5 ms after the A-rule edge).

    Native SNR. Scaling a cut up by g also scales its copy of the source floor. A cut with native SNR S injected at T
    dB carries that floor at T - S dB relative to the host floor (above it when T > S) for the cut's 16 ms; S >= 30 dB
    keeps it at or under the host floor at the top 30 dB level. If fewer than ``min_count`` eligible cuts reach
    ``min_native_db``, the strongest ``min_count`` eligible cuts
    are used and the returned ``native_db`` says how far short they are (measured: 8 reach 30 dB, the 24th is 24.3 dB)."""
    cl = np.array([clean_snip(s, fs, fc) for s in raw_bank])
    nat = np.array([10 * np.log10(snip_peak_power(s, fs, fc) / src_floor) for s in cl])
    lag = np.array([snip_peak_lag_s(s, fs, fc) for s in cl])
    ok = (lag >= lag_lo_s) & (lag <= lag_hi_s)
    keep = np.where(ok & (nat >= min_native_db))[0]
    if len(keep) < min_count:
        idx = np.where(ok)[0]
        keep = idx[np.argsort(-nat[idx])[:min_count]]
    return cl[keep], nat[keep]


def synth_sferic(rng, fs=FS, fc=FC, kind=None):
    """One synthetic sferic, complex baseband, onset at ONSET_S. Returns (waveform, kind).

    kind "damped": exp(-t/tau) oscillation at f0 (RF 4-14 kHz, tau 0.08-0.5 ms) with a 3-sample raised-cosine rise -
        the short, near-source shape.
    kind "dispersed": one earth-ionosphere waveguide mode, phase 2*pi*(d/c)*(sqrt(f^2 - fcut^2) - f) so the group delay
        is (d/c)*(f/sqrt(f^2 - fcut^2) - 1): high frequencies arrive first (t = onset), frequencies near the cutoff
        (fcut 1.6-1.9 kHz, ~80-94 km reflection) trail - the tweek shape. d 1000-4000 km, spectrum (f/fp)*exp(1 - f/fp)
        with fp 4-10 kHz, faded in as ((f - fcut)/300 Hz)^2 above fcut to keep the near-cutoff tail small. The window is
        2048 samples (28.8 ms after the onset); the tail is NOT strictly inside it: at d = 4000 km, fcut = 1.9 kHz the
        group delay at fcut + 100 Hz (fade factor 0.11) is 29.4 ms, so the faintest tail can wrap round the circular
        IFFT. Measured (verifier probe): at most 1e-4 of the pulse energy lands in the last 300 samples, and anything
        wrapped to the start is zeroed with the pre-onset samples below - negligible for a peak-SNR yardstick.
    Both get a random overall phase (the LO phase, hence polarity, is unknown on this receiver)."""
    kind = kind or ("damped" if rng.random() < 0.5 else "dispersed")
    s_on = int(round(ONSET_S * fs))
    if kind == "damped":
        n = SNIP_N
        t = (np.arange(n) - s_on) / fs
        f0 = rng.uniform(4000, 14000)
        tau = rng.uniform(0.08e-3, 0.5e-3)
        env = np.where(t >= 0, np.exp(-np.clip(t, 0, None) / tau), 0.0)
        rise = np.clip(t * fs / 3.0, 0, 1)
        env *= 0.5 - 0.5 * np.cos(np.pi * rise)
        z = env * np.exp(1j * 2 * np.pi * (f0 - fc) * t)
    else:
        n = 2048
        d = rng.uniform(1000, 4000)
        fcut = rng.uniform(1600, 1900)
        fp = rng.uniform(4000, 10000)
        k = np.fft.fftfreq(n, 1 / fs)
        f = k + fc
        ok = (f > fcut + 1) & (f <= 30000)
        amp = np.zeros(n)
        amp[ok] = (f[ok] / fp) * np.exp(1 - f[ok] / fp)
        amp *= np.clip((f - fcut) / 300.0, 0, 1) ** 2
        ph = np.zeros(n)
        ph[ok] = 2 * np.pi * (d / C_KM_S) * (np.sqrt(f[ok] ** 2 - fcut ** 2) - f[ok])
        X = amp * np.exp(-1j * ph) * np.exp(-1j * 2 * np.pi * k * s_on / fs)   # front at sample s_on
        z = np.fft.ifft(X)
        z[: max(0, s_on - 30)] = 0          # nothing before the onset (the mask's sidelobes only)
        z *= tukey(n, 0.05)
    z = z * np.exp(1j * rng.uniform(0, 2 * np.pi))
    return z / np.sqrt(np.mean(np.abs(z) ** 2)), kind


def synth_bank(n, seed=11, fs=FS, fc=FC):
    """n synthetic sferics (list of arrays; dispersed ones are 2048 samples long) and their kinds."""
    rng = np.random.default_rng(seed)
    out = [synth_sferic(rng, fs, fc) for _ in range(n)]
    return [w for w, _ in out], [k for _, k in out]


# ---------------------------------------------------------------- schedules and injection
def schedule(rng, secs=120.0, n=80):
    """h2h's schedule: 1 + 1.43 k + U(0, 0.3) s, keeping cuts that end 1 s before the piece ends. Returns cut starts."""
    t = 1.0 + np.arange(n) * 1.43 + rng.uniform(0, 0.3, n)
    return t[t < secs - 1.0]


def h2h_schedule(n_bank=386):
    """The exact h2h cut starts and bank picks for its six pieces (one shared rng(7) stream in h2h's argv order)."""
    rng = np.random.default_rng(7)
    out = {}
    for tag in H2H_ORDER:
        t = schedule(rng)
        out[tag] = (t, rng.integers(0, n_bank, len(t)))
    return out


def inject(x, snips, cut_starts, gains=None, fs=FS, quantize=True):
    """Add pulses into a copy of x (complex64, full scale 1.0) with each pulse's first sample at a cut start.

    gains: per-pulse amplitude factors (None = native amplitude, the h2h mode).
    Pulses that run off either end of x are clipped; a pulse wholly outside x is skipped.
    quantize: emulate h2h's int16 file - float32 sum, clip, truncate toward zero (numpy astype(int16)) - so detectors see
    the same numbers as the reproduction run. Returns complex64."""
    y = x.astype(np.complex128)
    for i, (t, s) in enumerate(zip(cut_starts, snips)):
        s0 = int(t * fs)
        g = 1.0 if gains is None else gains[i]
        a = max(0, -s0)                                  # part of the pulse before the host starts is dropped
        m = min(len(s), len(y) - s0)                     # ... and so is the part after it ends
        if m <= a:
            continue                                     # pulse wholly outside the host
        y[s0 + a:s0 + m] += g * np.asarray(s)[a:m]
    y = y.astype(np.complex64)
    if quantize:
        sc = np.float32(32768.0)
        re = np.trunc(np.clip(y.real * sc, -32768, 32767)) / sc
        im = np.trunc(np.clip(y.imag * sc, -32768, 32767)) / sc
        y = (re + 1j * im).astype(np.complex64)
    return y


def snr_gains(snips, floor, snr_db, fs=FS, fc=FC):
    """Amplitude factor per pulse that puts its 2.5-9 kHz smoothed peak at snr_db over ``floor``."""
    return np.array([np.sqrt(floor * 10 ** (snr_db / 10) / snip_peak_power(s, fs, fc)) for s in snips])


def synthetic_host(secs, seed=0, band_floor_dbfs=-72.6, tone_dbfs=-22.6, fs=FS, fc=FC):
    """White complex noise whose 2.5-9 kHz band sits at band_floor_dbfs (the clean-day h2h Welch number), plus the
    RF 0 Hz receiver tone at tone_dbfs (baseband -fc). No mains, no sferics. Complex64, not quantized."""
    rng = np.random.default_rng(seed)
    n = int(secs * fs)
    p_tot = 10 ** (band_floor_dbfs / 10) * fs / (9000.0 - 2500.0)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)) * np.sqrt(p_tot / 2)
    x += 10 ** (tone_dbfs / 20) * np.exp(-1j * 2 * np.pi * fc / fs * np.arange(n))
    return x.astype(np.complex64)


def poisson_starts(rate, secs, seed, guard=0.05):
    """Poisson pulse start times at ``rate`` per second in (guard, secs - guard)."""
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.exponential(1 / rate, int(rate * secs * 1.5) + 20))
    return t[(t > guard) & (t < secs - guard)]
