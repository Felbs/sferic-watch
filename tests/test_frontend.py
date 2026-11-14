"""Fast synthetic tests for sferic_watch.frontend (no captures, no radio)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sferic_watch import frontend as fe  # noqa: E402

FS = fe.FS
RNG = np.random.default_rng(1234)


def _noise(n, dbfs, rng=RNG):
    """Complex white noise with total power `dbfs` (over the full 62.5 kHz)."""
    a = 10 ** (dbfs / 20) / np.sqrt(2)
    return a * (rng.standard_normal(n) + 1j * rng.standard_normal(n))


def _tone(n, f_true, dbfs, n0=0, phase=0.3):
    """Complex baseband tone appearing at true frequency f_true."""
    k = np.arange(n0, n0 + n)
    return 10 ** (dbfs / 20) * np.exp(1j * (2 * np.pi * (f_true - fe.FC) / FS * k + phase))


def _db(x):
    return 10 * np.log10(np.mean(np.abs(x) ** 2))


def _scene(seconds=4.0, seed=7):
    """Like a clean capture: -22.6 dBFS receiver tone at 0.6 Hz true, -60 dBFS
    white floor, a few lines (4 kHz, 8 kHz, a 24 kHz transmitter) and some
    impulses ('sferics')."""
    rng = np.random.default_rng(seed)
    n = int(seconds * FS)
    x = _noise(n, -60.0, rng) + _tone(n, 0.6, -22.6)
    for f, d in [(4000.0, -55.0), (8000.0, -58.0), (24000.0, -45.0)]:
        x += _tone(n, f, d, phase=rng.uniform(0, 6))
    for t in rng.uniform(0.2, seconds - 0.2, 6):
        i = int(t * FS)
        x[i:i + 3] += 0.02 * np.array([1.0, -0.6, 0.3])
    return x.astype(np.complex64)


def test_read_iq_slice_and_scale(tmp_path):
    iq = np.arange(2000, dtype=np.int16) - 1000          # I,Q,I,Q,...
    p = tmp_path / "t.cs16"
    iq.tofile(p)
    x = fe.read_iq(p, start_s=10 / FS, dur_s=5 / FS)
    assert x.dtype == np.complex64 and len(x) == 5
    exp = (iq[20:30:2] + 1j * iq[21:30:2]) / 32768.0
    assert np.allclose(x, exp)
    assert len(fe.read_iq(p, start_s=995 / FS, dur_s=100 / FS)) == 5   # short at end of file


def test_rotator_is_exact_and_chunk_independent():
    a = fe._rotator(0, 1000, fe.FC, FS)
    b = np.concatenate([fe._rotator(0, 333, fe.FC, FS), fe._rotator(333, 667, fe.FC, FS)])
    assert np.array_equal(a, b)
    k = np.arange(1000)
    assert np.allclose(a, np.exp(2j * np.pi * fe.FC / FS * k), atol=1e-9)
    big = fe._rotator(37_500_000, 5, fe.FC, FS)   # 600 s in: still exact (period 25)
    assert np.allclose(big, fe._rotator(0, 5, fe.FC, FS))


def test_shift_puts_tone_at_true_frequency():
    n = 62500
    z = fe.shift_to_vlf(_tone(n, 5000.0, -30.0), 0)
    f = np.fft.fftfreq(n, 1 / FS)[np.argmax(np.abs(np.fft.fft(z)))]
    assert abs(f - 5000.0) < 1.5


def test_tone_removal_drops_rms_40db():
    n = int(2 * FS)
    x = _tone(n, 0.6, -22.6) + _noise(n, -75.0)
    z = fe.shift_to_vlf(x, 0)
    before = _db(z)
    after = _db(fe.remove_tone(z)[2000:-2000])
    assert before - after >= 40.0
    assert abs(after - (-75.0)) < 0.5   # what is left is the noise


def test_bands_select_right_frequencies():
    n = int(2 * FS)
    x = (_tone(n, 5000.0, -40.0) + _tone(n, 23000.0, -40.0) + _tone(n, 40000.0, -40.0)
         + _tone(n, 12000.0, -40.0)).astype(np.complex64)
    fe_out = fe.frontend_offline(x)
    # LineExciser would remove these pure tones; test the filters with excision off
    s = fe.design(FS)
    from scipy.signal import sosfiltfilt
    z = fe.shift_to_vlf(x, 0)
    det = sosfiltfilt(s["det_bp"], z.real)
    f = np.fft.rfftfreq(n, 1 / FS)
    D = np.abs(np.fft.rfft(det)) ** 2
    assert abs(f[np.argmax(D)] - 5000.0) < 1.5
    # 12 kHz and the 40 kHz image (folds to 22.5 kHz) are well down in det
    p5 = D[np.abs(f - 5000) < 3].sum()
    assert D[np.abs(f - 12000) < 3].sum() < p5 * 1e-3
    assert D[np.abs(f - 22500) < 3].sum() < p5 * 1e-6
    # confirmation band is analytic: 23 kHz in, 40 kHz (would alias in a real band) out
    rot = fe._rotator(0, n, s["conf_fc"], FS)
    conf = sosfiltfilt(s["conf_lp"], z * rot.conj()) * rot
    C = np.abs(np.fft.fft(conf)) ** 2
    fc = np.fft.fftfreq(n, 1 / FS)
    p23 = C[np.abs(fc - 23000) < 3].sum()
    assert C[np.abs(fc - 40000 + FS) < 3].sum() < p23 * 1e-6
    assert C[np.abs(fc - 5000) < 3].sum() < p23 * 1e-6
    assert fe_out.det.dtype == np.float32 and fe_out.conf.dtype == np.complex64


def test_line_excision_removes_lines_keeps_noise_and_impulses():
    rng = np.random.default_rng(3)
    n = int(4 * FS)
    noise = _noise(n, -60.0, rng)
    lines = _tone(n, 4000.0, -40.0) + _tone(n, 8031.5, -45.0) + _tone(n, 7968.5, -45.0)
    imp = np.zeros(n, complex)
    idx = (np.array([0.7, 1.9, 3.1]) * FS).astype(int)
    imp[idx] = 0.05
    y, frac = fe.excise_lines(noise + lines + imp)
    f = np.fft.fftfreq(n, 1 / FS)
    Y = np.abs(np.fft.fft(y)) ** 2
    Yn = np.abs(np.fft.fft(noise + imp)) ** 2
    for fl in (4000.0, 8031.5, 7968.5):
        m = np.abs(f - fl) < 1.0
        assert Y[m].sum() < 4 * Yn[m].sum()   # line gone to about the noise level
    # broadband power kept within 0.3 dB, impulses within 1 dB, few bins cut
    assert abs(_db(y) - _db(noise + imp)) < 0.3
    for i in idx:
        assert abs(20 * np.log10(abs(y[i]) / abs(noise[i] + imp[i]))) < 1.0
    assert frac < 0.02


def test_excision_perfect_reconstruction_when_nothing_cut(monkeypatch):
    monkeypatch.setattr(fe, "LINE_K", 1e30)
    x = _noise(int(2.3 * FS), -40.0)
    y, frac = fe.excise_lines(x)
    assert frac == 0.0
    assert np.allclose(y, x, atol=1e-12)


def test_stream_chunked_equals_one_shot():
    x = _scene(3.3)
    one = fe.FrontendStream()
    a = fe.concat([one.push(x), one.flush()])
    rng = np.random.default_rng(11)
    st = fe.FrontendStream()
    blocks, i = [], 0
    while i < len(x):
        k = int(rng.integers(1, 40000))
        blocks.append(st.push(x[i:i + k]))
        i += k
    blocks.append(st.flush())
    b = fe.concat(blocks)
    assert a.n0 == b.n0 == 0
    assert len(a.det) == len(b.det) == len(x)
    # consecutive blocks are contiguous
    pos = 0
    for blk in blocks:
        if len(blk.det):
            assert blk.n0 == pos
            pos += len(blk.det)
    sc = np.max(np.abs(a.det))
    assert np.max(np.abs(a.det - b.det)) < 1e-5 * sc
    assert np.max(np.abs(a.conf - b.conf)) < 1e-5 * np.max(np.abs(a.conf))
    assert len(b.det_raw) == len(x)
    assert np.max(np.abs(a.det_raw - b.det_raw)) < 1e-5 * np.max(np.abs(a.det_raw))


def test_stream_equals_offline():
    """Since the frame engine the stream runs the same zero-phase frames on the
    same absolute grid as the offline path: the outputs are the same samples
    (the old causal stream only matched in band power, 0.5 dB apart)."""
    x = _scene(4.0, seed=5)
    off = fe.frontend_offline(x)
    st = fe.FrontendStream()
    on = fe.concat([st.push(x[:70000]), st.push(x[70000:]), st.flush()])
    assert on.n0 == 0 and len(on.det) == len(off.det) == len(x)
    for a, b in ((off.det, on.det), (off.conf, on.conf), (off.det_raw, on.det_raw)):
        assert np.max(np.abs(a - b)) <= 1e-6 * np.max(np.abs(a))


def test_stream_absolute_index_offset():
    x = _scene(2.0)
    a = fe.FrontendStream(n0=0)
    ra = fe.concat([a.push(x), a.flush()])
    n0 = 2 * 31250                        # a whole mixer period (25) AND a whole frame hop: same output, shifted index
    b = fe.FrontendStream(n0=n0)
    rb = fe.concat([b.push(x), b.flush()])
    assert rb.n0 == n0
    assert np.allclose(ra.det, rb.det, atol=1e-7)


def test_frame_grid_is_absolute():
    """The review's probe: the same absolute samples processed from a different
    slice start (offset 31250 + 777, not on the grid) gave about 8 % rms
    difference in det when the frames started at each slice start. On the
    absolute grid they agree once both slices are a frame (1 s) in."""
    x = _scene(4.0, seed=8)
    n0, s = 123_456, 31250 + 777
    a = fe.frontend_offline(x, n0)
    b = fe.frontend_offline(x[s:], n0 + s)
    sl = slice(s + int(FS), len(x) - int(FS))
    for k in ("det", "conf", "det_raw"):
        A, B = getattr(a, k)[sl], getattr(b, k)[sl.start - s:sl.stop - s]
        assert np.sqrt(np.mean(np.abs(A - B) ** 2) / np.mean(np.abs(A) ** 2)) < 1e-6, k
    st1 = fe.FrontendStream(n0=n0)
    o1 = fe.concat([st1.push(x), st1.flush()])
    st2 = fe.FrontendStream(n0=n0 + s)
    o2 = fe.concat([st2.push(x[s:s + 40000]), st2.push(x[s + 40000:]), st2.flush()])
    assert o2.n0 == n0 + s
    A, B = o1.det[sl], o2.det[sl.start - s:sl.stop - s]
    assert np.sqrt(np.mean((A - B) ** 2) / np.mean(A ** 2)) < 1e-6


def test_short_and_empty_inputs_do_not_raise():
    """frontend_offline raised ValueError for 27 samples or fewer (sosfiltfilt's
    pad). Every length now works, stream included, and lengths are kept."""
    for n in (0, 1, 2, 27, 28, 1000, 31251):
        x = _scene(0.6)[:n]
        o = fe.frontend_offline(x)
        assert len(o.det) == len(o.conf) == len(o.det_raw) == n
        assert np.all(np.isfinite(o.det)) and np.all(np.isfinite(o.conf))
        st = fe.FrontendStream()
        r = fe.concat([st.push(x), st.flush()])
        assert len(r.det) == n
        if n:
            assert np.max(np.abs(r.det - o.det)) <= 1e-6 * max(np.max(np.abs(o.det)), 1e-30)
    s = fe.design(FS)
    for n in (1, 5, 27):
        assert len(fe.zero_phase(s["det_bp"], np.ones(n))) == n


def test_slice_edges_carry_no_tone_splash():
    """The -22.6 dBFS tone sits at -15 kHz in baseband. Reflecting the baseband
    samples at a slice edge (first try) put a +30 dB splash into the first
    ms of det; the reflection is done at TRUE frequency, where the tone is
    slow, and the first and last 10 ms stay at the noise level. In-band lines
    still splash a little at an edge whatever the reflection (the old chain
    too: +4-5 dB on this scene); bounded loosely here."""
    rng = np.random.default_rng(12)
    n = int(3 * FS)
    x = (_noise(n, -60.0, rng) + _tone(n, 0.6, -22.6) + _tone(n, 24000.0, -45.0)).astype(np.complex64)
    o = fe.frontend_offline(x[777:], 777)
    mid = _db(o.det[int(FS):int(2 * FS)])
    assert _db(o.det[:625]) - mid < 1.5
    assert _db(o.det[-625:]) - mid < 1.5
    o = fe.frontend_offline(_scene(3.0, seed=12)[777:], 777)
    mid = _db(o.det[int(FS):int(2 * FS)])
    assert _db(o.det[:625]) - mid < 10.0 and _db(o.det[-625:]) - mid < 10.0


def test_regression_guard_raw_floor_is_the_tone():
    """Bypassing the front end reproduces the OnAir-raw failure: a median of
    1 ms block powers reads the tone (-22.6 dBFS), not the sky. Through the
    front end it reads the band noise, ~50 dB lower."""
    x = _scene(3.0)
    blk = 62
    raw = np.abs(x[: len(x) // blk * blk]) ** 2
    raw_floor = 10 * np.log10(np.median(raw.reshape(-1, blk).mean(1)))
    assert abs(raw_floor - (-22.6)) < 0.5
    det = fe.frontend_offline(x).det.astype(float)
    p = det[: len(det) // blk * blk] ** 2
    det_floor = 10 * np.log10(np.median(p.reshape(-1, blk).mean(1)))
    # white -60 dBFS over 62.5 kHz -> real 2.5-9 kHz band about -60 + 10log10(6.5/62.5) = -69.8
    assert abs(det_floor - (-69.8)) < 1.5
    assert raw_floor - det_floor > 40


def test_band_power_dbfs_matches_known_noise():
    x = _noise(int(4 * FS), -50.0)
    expect = -50.0 + 10 * np.log10(6500.0 / FS)
    assert abs(fe.band_power_dbfs(x, 2500, 9000) - expect) < 0.3
    assert abs(fe.band_power_dbfs(x.real, 2500, 9000) - expect) < 0.4


def test_offline_excision_equals_stream_exciser():
    """The in-place offline excision uses the same frame grid as LineExciser."""
    x = fe.remove_tone(fe.shift_to_vlf(_scene(2.7, seed=3), 0))
    y_off, _ = fe.excise_lines(x)
    ex = fe.LineExciser()
    a0, a = ex.push(x)
    b0, b = ex.flush()
    y_st = np.concatenate([a, b])
    assert a0 == 0 and len(y_st) == len(x) == len(y_off)
    assert np.max(np.abs(y_off - y_st)) < 1e-12 * np.max(np.abs(x)) + 1e-15
    xc = x.copy()
    y_in, _ = fe.excise_lines(xc, inplace=True)
    assert y_in is xc and np.array_equal(y_in, y_off)


def test_stream_refuses_use_after_flush():
    st = fe.FrontendStream()
    st.push(_scene(0.5))
    st.flush()
    with pytest.raises(RuntimeError):
        st.push(_scene(0.5))
    with pytest.raises(RuntimeError):
        st.flush()
    ex = fe.LineExciser()
    ex.push(np.zeros(1000, complex))
    ex.flush()
    with pytest.raises(RuntimeError):
        ex.flush()


def _buzz(n, rate_hz=120.0, width_s=380e-6, dbfs=-55.0, rng=None):
    """Full-wave mains buzz: 380 us broadband bursts every 1/120 s, as real VLF
    (put at baseband the way the receiver sees it)."""
    rng = rng or np.random.default_rng(9)
    real = np.zeros(n)
    w = int(width_s * FS)
    shape = rng.standard_normal(w) * np.hanning(w)
    for t in np.arange(0.01, n / FS - 0.01, 1 / rate_hz):
        i = int(t * FS)
        real[i:i + w] += shape
    real *= 10 ** (dbfs / 20) / np.sqrt(np.mean(real[real != 0] ** 2))
    return real * fe._rotator(0, n, fe.FC, FS).conj()


def _fold_contrast_db(p, f=120.0, nbins=84):
    ph = ((np.arange(len(p)) / FS * f) % 1.0 * nbins).astype(int)
    h = np.bincount(ph, weights=p, minlength=nbins) / np.bincount(ph, minlength=nbins)
    return 10 * np.log10(h.max() / np.median(h))


def test_det_raw_keeps_mains_buzz_that_excision_strips():
    """Excision removes the 120 Hz comb (measured on 11-14 02:12Z: fold
    contrast 13.4 -> 1.6 dB); det_raw must keep it for the hum flag/gate."""
    n = int(4 * FS)
    x = (_noise(n, -60.0, np.random.default_rng(2)) + _tone(n, 0.6, -22.6) + _buzz(n)).astype(np.complex64)
    o = fe.frontend_offline(x)
    sl = slice(int(0.5 * FS), int(3.5 * FS))
    c_raw = _fold_contrast_db(o.det_raw[sl].astype(float) ** 2)
    c_det = _fold_contrast_db(o.det[sl].astype(float) ** 2)
    assert c_raw > 10.0
    assert c_raw - c_det > 3.0     # documents that det loses buzz; det_raw does not
    # with no lines and no buzz det_raw and det agree in power
    y = fe.frontend_offline(_noise(n, -60.0, np.random.default_rng(4)).astype(np.complex64))
    assert abs(_db(y.det_raw[sl]) - _db(y.det[sl])) < 0.3


def test_neighbour_rule_cuts_line_leakage():
    """A strong line between two bins leaks into its +-1 neighbours; they are
    cut, so the line is gone; a noise-only frame keeps > 99 % of its bins."""
    n = int(4 * FS)
    x = _noise(n, -60.0, np.random.default_rng(6)) + _tone(n, 5000.5, -30.0)
    y, _ = fe.excise_lines(x)
    f = np.fft.fftfreq(n, 1 / FS)
    m = np.abs(f - 5000.5) < 3
    floor_bin = np.median(np.abs(np.fft.fft(x)[np.abs(f - 5100) < 50]) ** 2)
    assert np.abs(np.fft.fft(y)[m]).max() ** 2 < 20 * floor_bin
    _, frac = fe.excise_lines(_noise(n, -60.0, np.random.default_rng(8)))
    assert frac < 0.01


def test_chunked_zero_phase_equals_scipy(monkeypatch):
    from scipy.signal import sosfiltfilt
    monkeypatch.setattr(fe, "CHUNK", 1000)          # many blocks, odd tail
    s = fe.design(FS)
    rng = np.random.default_rng(5)
    xr = rng.standard_normal(12345)
    xc = _noise(12345, -20.0, rng)
    for key, x in (("det_bp", xr), ("tone_hp", xc), ("conf_lp", xc)):
        ref = sosfiltfilt(s[key], x)
        got = fe.zero_phase(s[key], x)
        assert np.max(np.abs(got - ref)) < 1e-9 * np.max(np.abs(ref))
    y = xc.copy()
    assert fe.zero_phase(s["tone_hp"], y, inplace=True) is y
    z = xc.copy()
    fe._rotate(z, 777, fe.FC, FS)
    assert np.allclose(z, xc * fe._rotator(777, len(xc), fe.FC, FS), atol=1e-15)
