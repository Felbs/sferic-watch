"""Band-limited front end: receiver-tone removal, shift to real VLF, line
excision, and the analysis bands.

Why this exists
---------------
The banked captures are complex IQ centred on 15 kHz (62.5 kS/s, int16). A
receiver tone sits at about 0.5 Hz *true* frequency (baseband -15 kHz, IQ mean
only -93.6 dBFS, so it is not a DC offset), with a phase-noise skirt out to
about 100 Hz. On 2026-10-19 15Z (10 s at 100 s) it measured:

* total power -22.6 dBFS, all of it the tone (0.3-1 Hz holds -22.6 dBFS);
* skirt 1-20 Hz about -48 dBFS, 20-100 Hz -62 dBFS;
* everything above 300 Hz together -60.6 dBFS (transmitters, 16-46 kHz noise).
  So removing the tone can lower the total rms by about 38 dB and no more:
  moving the high-pass corner to 1 or 2 kHz only gives 38.07 / 38.18 dB;
* the 2.5-9 kHz sferic band -72.6 dBFS (+side, lines included), about 50 dB
  under the tone; -73.9 to -74.1 dBFS with the steady lines taken out.

OnAir's sferics detector, fed this raw IQ, put its floor at -22.6 dBFS on all
12 test captures and found nothing (injected recall 0.00). With a 2.5-9 kHz
band-pass in front of the same logic the floor fell to -74 dBFS and the
chance-corrected injected recall rose to 0.71. This module is that band-pass,
cleaned up, plus the line excision census v1 used.

Signal chain (both paths, one short-time FFT: ``_BandEngine``)
--------------------------------------------------------------
Frames of 1 s (62500 samples, 1 Hz bins), half overlap, sine window for
analysis and synthesis (sin^2 sums to 1: perfect reconstruction when nothing
is cut), on an ABSOLUTE sample grid: frame k covers absolute samples
[(k-1)H, (k+1)H), H = 31250. In each frame:

1. Shift to true VLF: multiply by exp(+j 2 pi 15 kHz n / fs) using the
   absolute sample index (exact fraction 6/25, so no drift over 600 s and the
   same phase however a stream is chunked). Folded into the analysis window.
2. Tone removal: the 4th-order Butterworth high-pass at 300 Hz, applied as
   its zero-phase response |H|^2 on the bins (|f| < 300 Hz on both sides).
   At 0.5 Hz that is > 400 dB; at 20-100 Hz 76-190 dB. The line rule must
   see the bins after it: the tone's leakage, not the sky, would otherwise
   set the floor near 0 Hz.
3. Line excision (``_cut_lines`` / ``_cut_mask`` for the rule and its
   measured cost): the 4 kHz-multiple local lines, the +-31.5 Hz pairs and
   the transmitters (census v1 rule, frame by frame so it can stream).
   It ALSO removes most of the 120 Hz mains buzz: a pulse train repeating
   every 8.33 ms is a 120 Hz comb of lines, and the comb lines cross 8x the
   floor. On 2026-11-14 02:12Z the 120 Hz phase-fold contrast of det power
   fell from 13.2-13.5 dB to 1.6 dB (verifier, slices at 60 s and 300 s), and
   the loudest 1 % of 1 ms blocks on hum slices lose 10-13 dB. That is why the
   output also carries ``det_raw`` (the det band without excision): mains
   flagging and the phase-fold gate need the buzz, the sferic detector wants
   it gone.
4. Detection band: real part of the excised signal through the 4th-order
   Butterworth band-pass 2.5-9 kHz, zero phase (|H|^2 on the bins; the real
   part via an inverse real FFT of (Z(f) + Z*(-f))/2). Real VLF, as every
   prototype that measured the 0.71 used. Both mirror sides carry the
   sferics: on 10-19 15Z the 1 ms power in +2.5..9 kHz and -2.5..-9 kHz
   correlates at 0.95, floors within 0.7 dB.
5. Confirmation band 16-30 kHz, kept *analytic* (complex, positive
   frequencies only): the 4th-order Butterworth 7 kHz low-pass response
   centred on 23 kHz (aliased exactly as the old shift-filter-shift did).
   Taking the real part instead would fold the 32.5-46 kHz part of the IQ
   (-66.7 dBFS on 10-19 15Z, as strong as the -64.1 dBFS in 16-30 kHz) onto
   the band. Taken from the excised bins.

Then inverse FFTs, synthesis window, overlap-add. Filtering inside a
windowed STFT equals the time-domain filter up to the window's change over
the filter's impulse response (a few ms against a 1 s frame; with sin^2
windows summing to 1 the first-order term cancels). Measured against the
old time-domain chain (high-pass, excise, sosfiltfilt) on grid-aligned 10 s
slices of 10-19 15Z and 11-06 05Z: relative rms difference 3.8-4.6e-5 in
det, 0.5-27e-5 in conf, 0.5-0.9e-5 in det_raw; band levels equal to 0.01 dB.

Why the change: the old chain cost 1.62 s of CPU per minute of signal (six
forward-backward IIR passes, a complex exp per sample for every mix, a
second det band for det_raw, 7 % chunk-guard overlap), and its excision
frames started at each slice or stream start, so the same samples from a
different start differed by about 8 % rms in det. This one does one forward
FFT and two inverse FFTs (complex64 / real) per half second, about 0.5 s of
CPU per minute (see ``detect`` for the harness numbers), and the grid is
absolute: ``frontend_offline(x[s:], n0 + s)`` equals
``frontend_offline(x, n0)[s:]`` once both are a frame from their edges.

Offline and live are the same computation. ``frontend_offline`` runs the
frames over a whole slice; ``FrontendStream`` runs them as input arrives and
its output equals the offline output (unit test), zero phase included: the
filters need no causal approximation inside a frame. The cost is latency:
a sample is out once the frame after it is complete, 0.5-1 s after it
arrived. Beyond a slice's ends the input is extended by a mirror reflection
made at TRUE frequency (``_Source``): a reflection of the baseband samples
puts the tone, which sits at -15 kHz there, into a +30 dB splash at the
edge. Measured over 24 real slice starts/ends (4 captures), the first 10 ms
of det sit a median +1.0 dB over the slice's level (old chain +0.6 dB);
in-band lines splash a little at any edge, old chain included.

dBFS convention: a complex full-scale tone (int16 / 32768) is 0 dBFS; power is
mean(|x|^2). For the real detection band the same mean(x^2) is used (half of
the +side plus half of the -side power).
"""
from __future__ import annotations

from dataclasses import dataclass
from fractions import Fraction

import numpy as np
import scipy.fft as sfft
from scipy.signal import butter, sosfilt, sosfilt_zi, sosfiltfilt, sosfreqz, welch

FS = 62500.0                  # capture sample rate, S/s
FC = 15000.0                  # tuned centre, Hz (baseband 0 Hz = 15 kHz true)
TONE_HP_HZ = 300.0            # tone removal high-pass corner, Hz
DET_BAND = (2500.0, 9000.0)   # sferic detection band, Hz (true frequency)
CONF_BAND = (16000.0, 30000.0)  # confirmation band, Hz (true frequency)
FILTER_ORDER = 4
LINE_FRAME_S = 1.0            # excision frame, s (1 Hz bins)
LINE_BLOCK_HZ = 20.0          # floor block width, Hz
LINE_SMOOTH_BLOCKS = 15       # running median over 15 blocks = 300 Hz
LINE_K = 8.0                  # cut bins above 8x the local floor
LINE_NEIGHBOUR_K = 4.0        # ...and their +-1 neighbours if those are above 4x
EXT_MODE = "even"             # slice-edge extension at true frequency: mirror ('even') or 'odd'


# ----------------------------------------------------------------------------- I/O

def read_iq(path, start_s: float, dur_s: float, fs: float = FS) -> np.ndarray:
    """Read a slice of an interleaved int16 IQ capture as complex64 scaled to
    full scale 1.0. Reads only the slice (np.fromfile with count/offset), so a
    600 s, 150 MB capture is never loaded whole. Returns fewer samples at the
    end of the file."""
    n0 = int(round(start_s * fs))
    n = int(round(dur_s * fs))
    r = np.fromfile(path, dtype=np.int16, count=2 * n, offset=4 * n0)
    r = r[: len(r) // 2 * 2].astype(np.float32) * np.float32(1.0 / 32768.0)
    return (r[0::2] + 1j * r[1::2]).astype(np.complex64)


# ----------------------------------------------------------------------------- helpers

def _rotator(n0: int, n: int, f: float, fs: float) -> np.ndarray:
    """exp(+j 2 pi f k / fs) for absolute sample indices k = n0..n0+n-1.

    f/fs is taken as an exact fraction (15000/62500 = 6/25), so the phase is
    computed from (k mod denominator) with integer arithmetic: no drift over a
    600 s capture and identical values however a stream is chunked."""
    q = (Fraction(f) / Fraction(fs)).limit_denominator(10**6)
    den, num = q.denominator, q.numerator % q.denominator
    k = (np.arange(n, dtype=np.int64) + (int(n0) % den)) % den
    return np.exp(2j * np.pi * ((k * num) % den) / den)


def shift_to_vlf(x: np.ndarray, n0: int = 0, fs: float = FS, fc: float = FC) -> np.ndarray:
    """Move baseband IQ (centred fc) to true frequency: complex128 output whose
    frequency f corresponds to real VLF f. ``n0`` is the absolute index of x[0]."""
    z = np.array(x, dtype=np.complex128)      # always a copy: the caller's x is not touched
    z *= _rotator(n0, len(z), fc, fs)
    return z


def design(fs: float = FS) -> dict:
    """Second-order sections for the three filters (all 4th-order Butterworth).
    'conf_lp' is the 7 kHz low-pass that becomes the 16-30 kHz analytic band
    after a shift to the 23 kHz band centre."""
    lo, hi = CONF_BAND
    return {
        "tone_hp": butter(FILTER_ORDER, TONE_HP_HZ, btype="highpass", fs=fs, output="sos"),
        "det_bp": butter(FILTER_ORDER, list(DET_BAND), btype="bandpass", fs=fs, output="sos"),
        "conf_lp": butter(FILTER_ORDER, (hi - lo) / 2.0, btype="lowpass", fs=fs, output="sos"),
        "conf_fc": (hi + lo) / 2.0,
    }


CHUNK = 1 << 17               # samples per block in the chunked offline helpers


def zero_phase(sos: np.ndarray, x: np.ndarray, inplace: bool = False) -> np.ndarray:
    """Forward-backward filtering, numerically the same as scipy's
    ``sosfiltfilt`` (odd extension, same pad length, same initial states) but
    run in blocks of ``CHUNK`` samples over one output buffer. scipy's version
    holds about four full-length copies at once; on a 30 s complex slice
    that pushed the offline front end to 278-284 MB RSS against a 300 MB
    target. This holds one (none with ``inplace=True`` on a float64 or
    complex128 array)."""
    x = np.asarray(x)
    nsec = sos.shape[0]
    pad = 3 * (2 * nsec + 1 - min((sos[:, 2] == 0).sum(), (sos[:, 5] == 0).sum()))
    n = len(x)
    if n <= pad:                    # scipy's default pad needs n > pad: shorter pad, never an error
        return sosfiltfilt(sos, x, padlen=max(0, n - 1)) if n else np.array(x, np.result_type(x.dtype, np.float64))
    dt = np.result_type(x.dtype, np.float64)
    y = x if (inplace and x.dtype == dt) else np.array(x, dtype=dt)
    zi0 = sosfilt_zi(sos).astype(dt)
    left = 2 * y[0] - y[pad:0:-1]
    right = 2 * y[-1] - y[-2:-(pad + 2):-1]
    # forward pass: left pad, signal (in place, block by block), right pad
    lf, zi = sosfilt(sos, left, zi=zi0 * left[0])
    for a in range(0, n, CHUNK):
        y[a:a + CHUNK], zi = sosfilt(sos, y[a:a + CHUNK], zi=zi)
    rf, zi = sosfilt(sos, right, zi=zi)
    # backward pass starts at the end of the forward-filtered right pad
    rb, zi = sosfilt(sos, rf[::-1], zi=zi0 * rf[-1])
    for b in range(n, 0, -CHUNK):
        a = max(0, b - CHUNK)
        seg, zi = sosfilt(sos, y[a:b][::-1], zi=zi)
        y[a:b] = seg[::-1]
    return y


def _rotate(z: np.ndarray, n0: int, f: float, fs: float, sign: int = 1) -> None:
    """In place: z *= exp(sign * j 2 pi f k / fs), k absolute, block by block
    (no full-length rotator array)."""
    for a in range(0, len(z), CHUNK):
        r = _rotator(n0 + a, len(z[a:a + CHUNK]), f, fs)
        z[a:a + CHUNK] *= r if sign > 0 else r.conj()


def remove_tone(z: np.ndarray, fs: float = FS, zero_phase: bool = True) -> np.ndarray:
    """High-pass the true-frequency complex signal at 300 Hz (4th order) to
    remove the ~0.5 Hz receiver tone and its skirt. Offline helper (zero phase
    by default); the stream keeps its own filter state."""
    sos = design(fs)["tone_hp"]
    return sosfiltfilt(sos, z) if zero_phase else sosfilt(sos, z)


def band_power_dbfs(y: np.ndarray, lo: float, hi: float, fs: float = FS, nperseg: int = 8192) -> float:
    """Welch band power in dBFS between lo and hi Hz (true frequency).

    Same measurement as the h2h spec check (Welch, 8192 points, power summed
    over the band). Complex input: two-sided spectrum, lo/hi signed. Real
    input: one-sided spectrum, so a real signal's band power is its full
    mean-square in that band."""
    y = np.asarray(y)
    if np.iscomplexobj(y):
        f, p = welch(y, fs, nperseg=nperseg, return_onesided=False)
    else:
        f, p = welch(y, fs, nperseg=nperseg)
    m = (f >= lo) & (f < hi)
    return float(10 * np.log10(p[m].sum() * (f[1] - f[0]) + 1e-300))


# ----------------------------------------------------------------------------- line excision

def _running_median(v: np.ndarray, size: int) -> np.ndarray:
    """``scipy.ndimage.median_filter(v, size, mode='nearest')`` for an odd size, by sorting the (n x size) sliding
    windows: the same values, without median_filter's per-call overhead (0.24 ms on 3125 blocks, 120 calls per
    minute of signal)."""
    h = size // 2
    pad = np.concatenate([np.full(h, v[0]), v, np.full(h, v[-1])])
    return np.sort(np.lib.stride_tricks.sliding_window_view(pad, size), axis=1)[:, h]


def _cut_mask(p: np.ndarray, fs: float) -> np.ndarray:
    """The line rule on one frame's bin powers ``p`` (FFT bin order). Returns
    the bool cut mask in the same order. See ``_cut_lines`` for the rule.

    The 20 Hz block median is taken with one ``np.partition`` on a
    (blocks x bins) view and the comparison runs on that view, so no
    full-length floor array is built (the engine calls this 120 times per
    minute of signal)."""
    L = len(p)
    ps = np.fft.fftshift(p)
    blk = max(1, int(round(LINE_BLOCK_HZ * L / fs)))
    nb = L // blk
    if nb == 0:
        return np.zeros(L, bool)
    v = ps[: nb * blk].reshape(nb, blk)
    h = blk // 2
    q = np.sort(v, axis=1)                    # rows of 20: a sort is 3.5x faster than a two-index partition
    med = q[:, h].astype(np.float64)
    if blk % 2 == 0:
        med = 0.5 * (q[:, h - 1] + med)
    fl = _running_median(med, LINE_SMOOTH_BLOCKS)
    core = np.empty(L, bool)
    near = np.empty(L, bool)
    core[: nb * blk] = (v > (LINE_K * fl)[:, None]).ravel()
    near[: nb * blk] = (v > (LINE_NEIGHBOUR_K * fl)[:, None]).ravel()
    core[nb * blk:] = ps[nb * blk:] > LINE_K * fl[-1]
    near[nb * blk:] = ps[nb * blk:] > LINE_NEIGHBOUR_K * fl[-1]
    cut = core.copy()
    cut[1:] |= core[:-1] & near[1:]
    cut[:-1] |= core[1:] & near[:-1]
    return np.fft.ifftshift(cut)


def _cut_lines(spec: np.ndarray, fs: float) -> tuple[np.ndarray, np.ndarray]:
    """Zero the narrow lines in one frame's FFT. Returns (spectrum, cut mask),
    both in the FFT's own bin order.

    Floor: power of each bin, sorted by frequency, median per 20 Hz block, then
    a 15-block running median. Cut: power > 8 x floor; a +-1 neighbour of a cut
    bin is cut too if it is itself > 4 x floor (the sine window's main lobe is
    3 bins wide, so a strong line leaks into its neighbours; a lone noise peak
    does not need them).

    Cost, measured (not the textbook 0.4 % chance rate): on 10-19 15Z (6 s at
    100 s) the rule cuts 2.2 % of the 2.5-9 kHz bins (3.8 % with an
    unconditional +-1 dilation, the first version), on 2026-11-14 02:12Z
    4.2 % (7.1 %). A sferic is broadband, so it loses about that share of
    its bins: zeroing a bin removes the pulse's component in it, and that
    energy is smeared over the 1 s frame instead of staying at the pulse.
    Injected 4.5/7 kHz sferic wavelets (60 per 10 s slice at 100 s) lose,
    as median energy within +-2 ms: clean 10-19 15Z 0.51 dB at 12 dB SNR and
    0.28 dB at 24 dB; storm 11-14 02:12Z 0.66 / 0.43 dB; hum 11-06 05Z
    0.71 / 0.43 dB, i.e. 6-15 % (first version with unconditional dilation:
    0.36-0.92 dB). The line removal is 20-60 dB. The det-band floor drops
    from -72.6 to about -74.1 dBFS on 10-19 15Z because the lines are gone
    (line-free input reference -73.9 to -74.1)."""
    cut = _cut_mask(np.abs(spec) ** 2, fs)
    out = spec.copy()
    out[cut] = 0
    return out, cut


def _frame_len(fs: float, frame_s: float = LINE_FRAME_S) -> int:
    L = int(round(frame_s * fs))
    return L + L % 2


def _sine_window(L: int) -> np.ndarray:
    """sin window; sin^2 overlap-adds to exactly 1 at half overlap."""
    return np.sin(np.pi * (np.arange(L) + 0.5) / L)


class LineExciser:
    """Streaming FFT line excision alone (1 s frames, half overlap, sine
    windows), zero-padded at both ends. A diagnostic and test tool: the front
    end itself uses ``_BandEngine``, which does the same excision on the same
    grid inside its own frames.

    Frame k covers ABSOLUTE samples [(k-1)H, (k+1)H), H = half a frame, so
    the grid does not depend on where the stream starts (``n0``) or on how
    the input is chunked. Samples before ``n0`` count as zeros. A sample is
    final once both frames that cover it are done, so output trails input by
    H to 2H samples. ``push`` returns (absolute index of the first returned
    sample, samples); ``flush`` ends the stream and returns the rest; after
    ``flush`` the object refuses more input (RuntimeError). ``cut_frac``
    tracks the share of bins cut."""

    def __init__(self, fs: float = FS, frame_s: float = LINE_FRAME_S, n0: int = 0):
        L = _frame_len(fs, frame_s)
        self.fs, self.L, self.H = fs, L, L // 2
        self.win = _sine_window(L)
        self.n0 = int(n0)
        k0 = self.n0 // self.H                       # first frame that touches n0
        self._k = k0                                 # next frame number
        self._buf = np.zeros(self.n0 - (k0 - 1) * self.H, dtype=np.complex128)  # from (k0-1)H
        self._ola = np.zeros(self.H, dtype=np.complex128)
        self._received = 0
        self._closed = False
        self.cut_bins = 0
        self.total_bins = 0

    @property
    def cut_frac(self) -> float:
        return self.cut_bins / max(1, self.total_bins)

    def _run(self) -> tuple[int, np.ndarray]:
        out = []
        start = None
        while len(self._buf) >= self.L:
            fr = np.fft.fft(self._buf[: self.L] * self.win)
            fr, cut = _cut_lines(fr, self.fs)
            self.cut_bins += int(cut.sum())
            self.total_bins += self.L
            y = np.fft.ifft(fr)
            y *= self.win
            done = self._ola + y[: self.H]          # samples [(k-1)H, kH)
            first = (self._k - 1) * self.H
            if first + self.H > self.n0:
                lo = max(0, self.n0 - first)
                if start is None:
                    start = first + lo
                out.append(done[lo:])
            self._ola = y[self.H:].copy()
            self._buf = self._buf[self.H:]
            self._k += 1
        if not out:
            return (max(self.n0, (self._k - 1) * self.H), np.zeros(0, dtype=np.complex128))
        return start, np.concatenate(out)

    def push(self, z: np.ndarray) -> tuple[int, np.ndarray]:
        """Add samples (true-frequency complex). Returns (absolute index of the
        first returned sample, finished samples), possibly empty."""
        if self._closed:
            raise RuntimeError("LineExciser.push after flush: make a new one")
        self._received += len(z)
        self._buf = np.concatenate([self._buf, np.asarray(z, dtype=np.complex128)])
        return self._run()

    def flush(self) -> tuple[int, np.ndarray]:
        """Pad with zeros to finish every received sample and return them.
        Can be called once."""
        if self._closed:
            raise RuntimeError("LineExciser.flush called twice")
        self._closed = True
        emitted = max(self.n0, (self._k - 1) * self.H)
        self._buf = np.concatenate([self._buf, np.zeros(self.L, dtype=np.complex128)])
        start, y = self._run()
        y = y[: max(0, self.n0 + self._received - start)]
        return (start if len(y) else emitted), y


def excise_lines(z: np.ndarray, fs: float = FS, inplace: bool = False, n0: int = 0) -> tuple[np.ndarray, float]:
    """Offline line excision of a whole slice on the absolute frame grid
    (identical result to ``LineExciser(n0=n0)``: zeros outside the slice),
    output aligned with the input. Returns (excised complex128 signal, share
    of bins cut).

    Works frame by frame in one buffer: frame k reads absolute samples
    [(k-1)H, (k+1)H) and then overwrites only [(k-1)H, kH), which no later
    frame reads, so with ``inplace=True`` (complex128 input) no second
    full-length copy is made."""
    L = _frame_len(fs)
    H = L // 2
    win = _sine_window(L)
    y = z if (inplace and isinstance(z, np.ndarray) and z.dtype == np.complex128) \
        else np.array(z, dtype=np.complex128)
    n = len(y)
    if n == 0:
        return y, 0.0
    n0 = int(n0)
    ola = np.zeros(H, dtype=np.complex128)
    seg = np.zeros(L, dtype=np.complex128)
    cut_bins = total = 0
    for k in range(n0 // H, (n0 + n - 1) // H + 2):   # first frame touching n0 .. last still covering n0+n-1
        a = (k - 1) * H - n0                          # frame start, slice-relative
        seg[:] = 0
        lo, hi = max(a, 0), min(a + L, n)
        if hi > lo:
            seg[lo - a: hi - a] = y[lo:hi]
        fr, cut = _cut_lines(np.fft.fft(seg * win), fs)
        cut_bins += int(cut.sum())
        total += L
        out = np.fft.ifft(fr)
        out *= win
        lo, hi = max(a, 0), min(a + H, n)            # final samples [a, a+H), clipped to the slice
        if hi > lo:
            y[lo:hi] = ola[lo - a:hi - a] + out[lo - a:hi - a]
        ola = out[H:].copy()
    return y, cut_bins / max(1, total)


# ----------------------------------------------------------------------------- the band engine (both paths)

class _Source:
    """Samples of a (possibly still growing) slice, extended beyond both ends
    by a reflection about the edge sample made at TRUE frequency (z = x times
    the mixer): mirror z(-m) = z(m) (EXT_MODE 'even', the default) or
    2 z(0) - z(m) ('odd', sosfiltfilt's pad), m clipped to the slice so any
    length >= 1 works. At true frequency the -22.6 dBFS receiver tone is a
    0.5 Hz drift and stays smooth across the edge; reflected in baseband,
    where it sits at -15 kHz, it put a +30 dB splash into the first ms of det.
    Mirror vs odd over 24 real slice edges: first 10 ms of det a median
    +1.0 / +1.2 dB over the slice level, first 2 ms +1.0 / +1.8 dB (old
    time-domain chain +0.6 / +0.4)."""

    def __init__(self, num: int = 0, den: int = 1):
        self.num, self.den = num, den           # mixer frequency as an exact fraction of fs
        self.head = np.zeros(0, np.complex64)   # the first samples (for the start reflection)
        self.buf = np.zeros(0, np.complex64)    # samples [b0, b0+len(buf))
        self.b0 = 0
        self.n = 0                              # samples received
        self.final = False

    def add(self, x: np.ndarray, keep_head: int) -> None:
        x = np.asarray(x, np.complex64)
        if len(self.head) < keep_head:
            self.head = np.concatenate([self.head, x[: keep_head - len(self.head)]])
        self.buf = np.concatenate([self.buf, x]) if len(self.buf) else x.copy()
        self.n += len(x)

    def trim(self, keep_from: int) -> None:
        keep_from = min(max(keep_from, self.b0), self.b0 + len(self.buf))
        if keep_from > self.b0:
            self.buf = self.buf[keep_from - self.b0:].copy()
            self.b0 = keep_from

    def _at(self, idx: np.ndarray) -> np.ndarray:
        out = np.empty(len(idx), np.complex64)
        h = idx < len(self.head)
        out[h] = self.head[idx[h]]
        out[~h] = self.buf[idx[~h] - self.b0]
        return out

    def segment(self, a: int, b: int) -> np.ndarray:
        """Samples [a, b) (slice-relative). Needs b <= n unless final."""
        n = self.n
        out = np.empty(b - a, np.complex64)
        lo, hi = max(a, 0), min(b, n)
        if hi > lo:
            out[lo - a:hi - a] = self.buf[lo - self.b0:hi - self.b0]
        if a < 0:
            i = np.arange(a, min(b, 0))
            out[:len(i)] = self._ext(i, 0)
        if b > n:
            i = np.arange(max(a, n), b)
            out[len(out) - len(i):] = self._ext(i, n - 1)
        return out

    def _ext(self, i: np.ndarray, e: int) -> np.ndarray:
        """Extension samples at slice-relative indices ``i`` reflected about
        the edge sample ``e``, built at TRUE frequency and mixed back to
        baseband (the mixer phase of index i): z(i) = z(j) (mirror, EXT_MODE
        'even') or 2 z(e) - z(j) ('odd'), z = x times the mixer."""
        n = self.n
        j = np.clip(2 * e - i, 0, n - 1)
        rel = lambda k: np.exp(2j * np.pi * ((k * self.num) % self.den) / self.den)   # noqa: E731
        y = self._at(j) * rel(j - i)
        if EXT_MODE == "odd":
            y = 2 * self._at(np.array([e]))[0] * rel(e - i) - y
        return y.astype(np.complex64)


class _ArraySource(_Source):
    """A whole slice already in memory (offline path): no copies."""

    def __init__(self, x: np.ndarray, num: int = 0, den: int = 1):
        super().__init__(num, den)
        self.buf = x
        self.head = x
        self.n = len(x)
        self.final = True

    def segment(self, a: int, b: int) -> np.ndarray:
        if a >= 0 and b <= self.n:
            return self.buf[a:b]
        return super().segment(a, b)


class _BandEngine:
    """The whole front end in one short-time FFT: frames of 1 s (1 Hz bins),
    half overlap, sine analysis and synthesis windows, on an ABSOLUTE sample
    grid (frame k covers absolute samples [(k-1)H, (k+1)H)), so a slice or a
    stream gives the same samples whatever its start or chunking.

    Per frame: window and mix to true VLF in one multiply (15 kHz is exactly
    15000 bins of a 62500-point frame, the frame's start phase is one exact
    scalar), FFT, the tone high-pass as its zero-phase response |H|^2 on the
    bins, the line rule (``_cut_mask``) on those bins, then the det band
    (|H_bp|^2, real part, via an inverse real FFT) and the analytic 16-30 kHz
    band (|H_lp(f - 23 kHz)|^2) by inverse FFT, synthesis window, overlap-add.
    ``det_raw`` is the det band from the same bins without the cut.

    The equivalence with the old time-domain chain and the reason for the
    change are in the module doc (relative rms difference about 4e-5 in det
    on real slices; 1.62 -> about 0.5 s of CPU per minute)."""

    def __init__(self, fs: float = FS, fc: float = FC, conf: bool = True, raw: bool = False):
        L = _frame_len(fs)
        self.fs, self.L, self.H = fs, L, L // 2
        self.want_conf, self.want_raw = conf, raw
        w = _sine_window(L)
        q = (Fraction(fc) / Fraction(fs)).limit_denominator(10**6)
        self.den, self.num = q.denominator, q.numerator % q.denominator
        m = np.arange(L, dtype=np.int64)
        self.wrot = (w * np.exp(2j * np.pi * ((m * self.num) % self.den) / self.den)).astype(np.complex64)
        self.win = w.astype(np.float32)
        s = design(fs)
        f = np.fft.fftfreq(L, 1.0 / fs)
        def amp2(sos, ff):
            return np.abs(sosfreqz(sos, worN=np.abs(ff), fs=fs)[1]) ** 2   # zero-phase amplitude |H|^2
        hp = amp2(s["tone_hp"], f)
        self.hp2 = (hp * hp).astype(np.float32)                      # power weight for the line rule
        half = np.arange(L // 2 + 1)
        # the tone high-pass is folded into both band masks (it is symmetric in f, like the det band)
        self.md = (0.5 * amp2(s["det_bp"], f[half]) * hp[half]).astype(np.float32)  # 0.5: Re z = (Z(f)+Z*(-f))/2
        fcn = (f - s["conf_fc"] + fs / 2) % fs - fs / 2               # offset from 23 kHz, aliased like the mixer
        self.mc = (amp2(s["conf_lp"], fcn) * hp).astype(np.float32)
        self.cut_bins = 0
        self.total_bins = 0
        self._ola_d = np.zeros(self.H, np.float32)
        self._ola_c = np.zeros(self.H, np.complex64)
        self._ola_r = np.zeros(self.H, np.float32)

    def frame(self, seg: np.ndarray, a_abs: int):
        """Process the frame starting at absolute index ``a_abs`` (samples
        ``seg``, length L). Returns the final first half: (det, conf, raw)
        for absolute samples [a_abs, a_abs + H); conf/raw None if not wanted."""
        H, L = self.H, self.L
        Z = sfft.fft(seg * self.wrot)
        ph = (a_abs * self.num) % self.den                            # 0 whenever H is a whole mixer period
        if ph:
            Z *= np.complex64(np.exp(2j * np.pi * ph / self.den))
        P = Z.real * Z.real + Z.imag * Z.imag
        P *= self.hp2                                                 # the line rule sees the tone-free bins
        Zr = Z.copy() if self.want_raw else None
        cut = _cut_mask(P, self.fs)
        self.cut_bins += int(np.count_nonzero(cut))
        self.total_bins += L
        Z[cut] = 0
        d = sfft.irfft(self.md * (Z[: H + 1] + np.concatenate((Z[:1], Z[:L - H - 1:-1])).conj()), L)
        d *= self.win
        det = self._ola_d + d[:H]
        self._ola_d = d[H:]
        conf = raw = None
        if self.want_conf:
            Z *= self.mc
            c = sfft.ifft(Z, overwrite_x=True)
            c *= self.win
            conf = self._ola_c + c[:H]
            self._ola_c = c[H:]
        if self.want_raw:
            r = sfft.irfft(self.md * (Zr[: H + 1] + np.concatenate((Zr[:1], Zr[:L - H - 1:-1])).conj()), L)
            r *= self.win
            raw = self._ola_r + r[:H]
            self._ola_r = r[H:]
        return det, conf, raw

    @property
    def cut_frac(self) -> float:
        return self.cut_bins / max(1, self.total_bins)


def _run_frames(eng: _BandEngine, src: _Source, n0: int, k: int, emit) -> int:
    """Run every frame from number ``k`` whose samples ``src`` can supply
    (all of them once ``src.final``) and hand each frame's final samples,
    clipped to the slice, to ``emit(rel_start, det, conf, raw)``. Returns the
    next frame number."""
    H = eng.H
    while True:
        a_abs = (k - 1) * H
        a, b = a_abs - n0, a_abs - n0 + eng.L                 # slice-relative frame span
        if src.final:
            if a >= src.n:                                    # every sample of the slice is final
                return k
        elif b > src.n:
            return k
        det, conf, raw = eng.frame(src.segment(a, b), a_abs)
        lo, hi = max(a, 0), min(a + H, src.n) if src.final else a + H
        if hi > lo:
            sl = slice(lo - a, hi - a)
            emit(lo, det[sl], None if conf is None else conf[sl], None if raw is None else raw[sl])
        k += 1


# ----------------------------------------------------------------------------- full chain

@dataclass
class FrontendOut:
    """One block of front-end output.

    n0:      absolute input sample index of det[0] / conf[0] / det_raw[0]
    det:     real 2.5-9 kHz detection band after line excision, float32
    conf:    analytic 16-30 kHz confirmation band after line excision (true
             frequency), complex64
    det_raw: the same 2.5-9 kHz band WITHOUT line excision, float32, same
             timing as det. Keeps the 120 Hz mains buzz that excision strips
             (fold contrast 13 dB here vs 1.6 dB in det on 11-14 02:12Z), for
             the hum flag and the phase-fold gate.
    cut_frac: share of FFT bins excised so far (diagnostic; census v1 found
             2-4 % on line-heavy captures)."""
    n0: int
    det: np.ndarray
    conf: np.ndarray
    cut_frac: float
    det_raw: np.ndarray | None = None


def frontend_offline(x: np.ndarray, n0: int = 0, fs: float = FS, fc: float = FC,
                     raw: bool = True) -> FrontendOut:
    """Front end for a whole slice of baseband IQ (complex64 from ``read_iq``).
    ``n0`` is the absolute index of x[0]: it sets the mixer phase AND the
    frame grid, so ``frontend_offline(x[s:], n0 + s)`` equals
    ``frontend_offline(x, n0)[s:]`` exactly once both are more than one frame
    (1 s) from their slice edges. Beyond the slice ends the input is
    extended by a mirror reflection at true frequency (``_Source``), so any
    length >= 0 works. ``raw=False`` skips det_raw (one inverse FFT per frame less)."""
    x = np.asarray(x, np.complex64)
    n = len(x)
    eng = _BandEngine(fs, fc, conf=True, raw=raw)
    det = np.zeros(n, np.float32)
    conf = np.zeros(n, np.complex64)
    draw = np.zeros(n, np.float32) if raw else None
    if n:
        def emit(lo, d, c, r):
            det[lo:lo + len(d)] = d
            conf[lo:lo + len(d)] = c
            if raw:
                draw[lo:lo + len(d)] = r
        _run_frames(eng, _ArraySource(x, eng.num, eng.den), int(n0), int(n0) // eng.H, emit)
    return FrontendOut(int(n0), det, conf, eng.cut_frac, draw)


class FrontendStream:
    """Live front end. Feed baseband IQ chunks of any size with ``push``;
    each call returns a ``FrontendOut`` (possibly empty) whose ``n0`` is the
    absolute input index of its first sample. ``flush`` drains the frames
    still open at the end; after it the stream refuses more input
    (RuntimeError).

    It runs the same ``_BandEngine`` frames on the same absolute grid as
    ``frontend_offline``, so its output EQUALS the offline output (unit test,
    to float32 rounding), not just its band power: the filters are zero
    phase in the frame and need no causal approximation. The cost is
    latency: a sample is out once the frame after it is complete, H to 2H
    samples (0.5-1 s) after it arrived, and the first output waits for 1 s
    of input (the start reflection)."""

    def __init__(self, fs: float = FS, fc: float = FC, n0: int = 0, raw: bool = True):
        self.fs, self.fc = fs, fc
        self.eng = _BandEngine(fs, fc, conf=True, raw=raw)
        self.n0 = int(n0)
        self._src = _Source(self.eng.num, self.eng.den)
        self._k = self.n0 // self.eng.H
        self._closed = False
        self._head_n = 2 * self.eng.H + 1         # start reflection reaches up to 2H samples in

    @property
    def latency_samples(self) -> int:
        """Worst-case delay from input to output (one frame)."""
        return self.eng.L

    @property
    def cut_frac(self) -> float:
        return self.eng.cut_frac

    def _drain(self) -> FrontendOut:
        out = []
        self._k = _run_frames(self.eng, self._src, self.n0, self._k, lambda lo, d, c, r: out.append((lo, d, c, r)))
        self._src.trim((self._k - 1) * self.eng.H - self.n0 - 2 * self.eng.H)   # keep 2H for the end reflection
        if not out:
            first = max(0, (self._k - 1) * self.eng.H - self.n0)
            return FrontendOut(self.n0 + first, np.zeros(0, np.float32), np.zeros(0, np.complex64),
                               self.eng.cut_frac, np.zeros(0, np.float32) if self.eng.want_raw else None)
        raw = np.concatenate([o[3] for o in out]).astype(np.float32) if self.eng.want_raw else None
        return FrontendOut(self.n0 + out[0][0], np.concatenate([o[1] for o in out]).astype(np.float32),
                           np.concatenate([o[2] for o in out]).astype(np.complex64), self.eng.cut_frac, raw)

    def push(self, x: np.ndarray) -> FrontendOut:
        """Process one chunk of baseband IQ."""
        if self._closed:
            raise RuntimeError("FrontendStream.push after flush: make a new stream")
        self._src.add(x, self._head_n)
        if self._src.n < self._head_n:            # start reflection not complete yet
            return self._drain_empty()
        return self._drain()

    def _drain_empty(self) -> FrontendOut:
        return FrontendOut(self.n0, np.zeros(0, np.float32), np.zeros(0, np.complex64), self.eng.cut_frac,
                           np.zeros(0, np.float32) if self.eng.want_raw else None)

    def flush(self) -> FrontendOut:
        """End of stream: return the samples still held. Once."""
        if self._closed:
            raise RuntimeError("FrontendStream.flush called twice")
        self._closed = True
        self._src.final = True
        if self._src.n == 0:
            return self._drain_empty()
        return self._drain()


def concat(blocks: list[FrontendOut]) -> FrontendOut:
    """Join consecutive stream blocks (empty ones allowed) into one."""
    blocks = [b for b in blocks if len(b.det)]
    if not blocks:
        return FrontendOut(0, np.zeros(0, np.float32), np.zeros(0, np.complex64), 0.0,
                           np.zeros(0, np.float32))
    raw = None
    if all(b.det_raw is not None for b in blocks):
        raw = np.concatenate([b.det_raw for b in blocks])
    return FrontendOut(blocks[0].n0, np.concatenate([b.det for b in blocks]),
                       np.concatenate([b.conf for b in blocks]), blocks[-1].cut_frac, raw)
