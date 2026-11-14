"""Fast tests for sferic_watch.detect on SYNTHETIC signals only (no banked captures)."""
import numpy as np
import pytest

import sferic_watch as sw
from sferic_watch import detect as dt
from sferic_watch import frontend as fe

FS = dt.FS


def noise_power(secs, seed=0, level=1.0):
    """Smoothed power of white real Gaussian noise (mean power ``level``), as the detector sees a quiet band."""
    rng = np.random.default_rng(seed)
    return dt.smooth_power(rng.standard_normal(int(secs * FS)) * np.sqrt(level))


def add_box(p, t, width_s, db, level=1.0):
    """Put a flat-topped power pulse ``db`` over ``level`` at time t (s), ``width_s`` wide (in place)."""
    a = int(t * FS)
    p[a:a + max(1, int(width_s * FS))] = level * 10 ** (db / 10)
    return a


def run(p, pc=None, **kw):
    return dt.events_from_power(p, pc, FS, **kw)


# ----------------------------------------------------------------------------- floor

def test_floor_is_median_of_block_means_and_robust_to_impulses():
    """30 % of 1 ms blocks carrying a 20 dB impulse move the median floor by well under 1 dB (a mean would move
    by 13 dB)."""
    p = noise_power(10).astype(np.float64)
    rng = np.random.default_rng(1)
    nb = len(p) // dt.BLOCK_N
    for b in rng.choice(nb, int(0.3 * nb), replace=False):
        p[b * dt.BLOCK_N + 20:b * dt.BLOCK_N + 30] = 100.0
    fl, ok = dt.sliding_floor(dt.block_means(p), dt.BLOCK_N / FS)
    assert ok[-1]
    clean = dt.block_means(noise_power(10))
    assert abs(10 * np.log10(fl[-1] / np.median(clean))) < 1.0


def test_floor_trust_backfill_and_trailing_window():
    """Untrusted steps (< 0.5 s of blocks) take the first trusted floor; after a +10 dB level step the floor follows
    within 4 s (trailing 4 s window, refreshed every 0.1 s) and not before about half the window has passed."""
    p = np.concatenate([noise_power(10, 2), noise_power(10, 3, level=10.0)])
    fl, ok = dt.sliding_floor(dt.block_means(p), dt.BLOCK_N / FS)
    hop = dt.BLOCK_N / FS * round(dt.FLOOR_HOP_S / (dt.BLOCK_N / FS))
    first = int(np.argmax(ok))
    assert 0.45 < first * hop < 0.6
    assert np.all(fl[:first] == fl[first])
    db = 10 * np.log10(fl / fl[first + 10])
    t = np.arange(len(fl)) * hop
    assert np.all(np.abs(db[(t > 1) & (t < 10)]) < 0.5)
    assert np.all(db[(t > 14.2)] > 9.0)
    assert np.all(db[(t > 10.2) & (t < 11.5)] < 1.0)


def test_floor_excluded_blocks_are_left_out():
    """Blocks a gate flags do not enter the median: marking every loud block excluded gives the quiet floor."""
    p = noise_power(6, 4).astype(np.float64)
    nb = len(p) // dt.BLOCK_N
    loud = np.zeros(nb, bool)
    loud[::2] = True                                   # half the blocks 20 dB up: the median would jump
    for b in np.flatnonzero(loud):
        p[b * dt.BLOCK_N:(b + 1) * dt.BLOCK_N] *= 100
    bm = dt.block_means(p)
    f_all, _ = dt.sliding_floor(bm, dt.BLOCK_N / FS)
    f_ex, _ = dt.sliding_floor(bm, dt.BLOCK_N / FS, exclude=loud)
    assert 10 * np.log10(f_all[-1]) > 5
    assert abs(10 * np.log10(f_ex[-1])) < 0.5


# ----------------------------------------------------------------------------- thresholds, hold, cap, merge

def test_start_threshold_is_10_db():
    p = noise_power(4, 5)
    a = add_box(p, 2.0, 1e-3, 9.0)
    b = add_box(p, 3.0, 1e-3, 11.0)
    ev, _ = run(p)
    assert len(ev["t_start"]) == 1
    assert abs(ev["t_start"][0] - b / FS) < 1e-4
    assert abs(ev["peak_db"][0] - 11.0) < 0.5
    del a


def test_end_needs_100us_hold_below_7_db():
    """A 64 us dip inside a pulse does not end it; a 160 us dip does (and then the 0.5 ms merge rejoins it, so check
    with merge off)."""
    p = noise_power(3, 6)
    add_box(p, 1.0, 2e-3, 20.0)
    p[int(1.001 * FS):int(1.001 * FS) + 4] = 1.0          # 64 us dip
    add_box(p, 2.0, 2e-3, 20.0)
    p[int(2.001 * FS):int(2.001 * FS) + 10] = 1.0         # 160 us dip
    s, e, _ = dt.find_impulses(p, np.ones(len(p)), FS, merge_s=0.0)
    assert np.sum((s > 0.9 * FS) & (s < 1.1 * FS)) == 1
    assert np.sum((s > 1.9 * FS) & (s < 2.1 * FS)) == 2
    w1 = (e[0] - s[0]) / FS
    assert abs(w1 - 2e-3) < 1e-4


def test_width_cap_drops_wide_impulses():
    p = noise_power(3, 7)
    add_box(p, 1.0, 8e-3, 20.0)
    add_box(p, 2.0, 4e-3, 20.0)
    ev, info = run(p)
    assert info["too_long"] == 1
    assert len(ev["t_start"]) == 1 and abs(ev["t_start"][0] - 2.0) < 1e-4
    assert abs(ev["width_s"][0] - 4e-3) < 2e-4


def test_width_cap_applies_to_merged_trains():
    """Verifier defect: 1 ms pulses with 0.3 ms gaps for 100 ms merged into one 99.8 ms 'sferic'. The cap now applies
    to the merged event, and the whole train is dropped (once, not one event every 5 ms). A short 3-pulse merge
    (1.6 ms) is still kept, and a lone pulse 2 ms after the train is still found."""
    p = noise_power(4, 17)
    t0 = 1.0
    for k in range(77):                                  # 1 ms on, 0.3 ms off, ~100 ms
        add_box(p, t0 + k * 1.3e-3, 1e-3, 20.0)
    add_box(p, t0 + 77 * 1.3e-3 + 2e-3, 0.3e-3, 20.0)    # separate pulse after the train
    for k in range(3):                                   # a short merged event: kept
        add_box(p, 3.0 + k * 0.6e-3, 0.3e-3, 20.0)
    ev, info = run(p)
    t = ev["t_start"]
    assert info["too_long"] == 1
    assert not np.any((t > t0 - 1e-3) & (t < t0 + 77 * 1.3e-3))
    assert np.sum(np.abs(t - (t0 + 77 * 1.3e-3 + 2e-3)) < 1e-4) == 1
    i = int(np.argmin(np.abs(t - 3.0)))
    assert abs(t[i] - 3.0) < 1e-4 and ev["width_s"][i] == pytest.approx(1.5e-3, abs=1e-4)
    assert np.all(ev["width_s"] <= dt.MAX_WIDTH_S)


def test_peak_time_is_middle_of_flat_top():
    """Verifier defect: a one-sample spike becomes a 16-sample flat top after the boxcar, and argmax took its first
    sample (7 samples = 112 us early). The peak is now the plateau middle: within one sample of the spike."""
    rng = np.random.default_rng(18)
    y = rng.standard_normal(int(2 * FS)) * 1e-3
    k = 62500
    y[k] = 1.0
    ev, _ = run(dt.smooth_power(y))
    assert len(ev["t_peak"]) == 1
    assert abs(ev["t_peak"][0] * FS - k) <= 1


def test_short_and_empty_and_zero_input_do_not_crash():
    """Verifier defect: input under one 62-sample block raised IndexError. Now it has no floor and no events, and
    says so; all-zero input gives a finite floor (clamped) and no events."""
    for n in (0, 1, 30, 61):
        ev, info = run(np.ones(n, np.float32))
        assert len(ev["t_start"]) == 0
        assert info["floor_none_s"] == pytest.approx(n / FS)
    ev, info = run(np.zeros(int(FS), np.float32))
    assert len(ev["t_start"]) == 0 and np.isfinite(info["floor_dbfs_min"])
    ev, info = dt.detect_iq(np.zeros(1000, np.complex64), FS)
    assert len(ev["t_start"]) == 0


def test_exclude_mask_length_is_forgiving():
    """Verifier defect: a mask shorter than the block count raised IndexError. Short masks are padded with False
    (usable), long ones cut; a short all-False mask equals no mask."""
    p = noise_power(3, 19)
    add_box(p, 2.0, 0.5e-3, 20.0)
    ev0, _ = run(p)
    nb = len(p) // dt.BLOCK_N
    for m in (np.zeros(10, bool), np.zeros(nb + 50, bool)):
        ev, _ = run(p, exclude_blocks=m)
        assert np.array_equal(ev["t_start"], ev0["t_start"])


def test_heavy_exclusion_holds_the_floor_and_reports_it():
    """Verifier defect: excluding every block after 5 s left the floor NaN from ~8.5 s on, and the detector went
    silently blind (12 of 15 pulses). The last trusted floor is now held, all 15 pulses are found, and info reports
    the held seconds."""
    p = noise_power(15, 20)
    times = 0.6 + np.arange(15) * 0.95
    for t in times:
        add_box(p, t, 0.4e-3, 20.0)
    nb = len(p) // dt.BLOCK_N
    ex = np.zeros(nb, bool)
    ex[int(5.0 / (dt.BLOCK_N / FS)):] = True
    ev, info = run(p, exclude_blocks=ex)
    assert all(np.any(np.abs(ev["t_start"] - t) < 1e-4) for t in times)
    assert 5.0 < info["floor_held_s"] < 11.0
    assert info["floor_none_s"] == 0


def test_merge_half_ms():
    """Pulses 0.3 ms apart are one event (width spans both); 1 ms apart are two."""
    p = noise_power(3, 8)
    add_box(p, 1.0, 0.3e-3, 20.0)
    add_box(p, 1.0006, 0.3e-3, 25.0)                     # gap 0.3 ms
    add_box(p, 2.0, 0.3e-3, 20.0)
    add_box(p, 2.0013, 0.3e-3, 20.0)                     # gap 1.0 ms
    ev, _ = run(p)
    t = ev["t_start"]
    assert np.sum(np.abs(t - 1.0) < 2e-3) == 1
    assert np.sum(np.abs(t - 2.0) < 3e-3) == 2
    i = int(np.argmin(np.abs(t - 1.0)))
    assert ev["width_s"][i] == pytest.approx(0.9e-3, abs=1e-4)
    assert ev["peak_db"][i] == pytest.approx(25.0, abs=0.5)
    assert ev["t_peak"][i] == pytest.approx(1.0006 + 0.15e-3, abs=1e-4)   # middle of the 25 dB flat top


def test_no_dead_time_under_a_dense_train():
    """A pulse every 2 ms (500/s, far denser than 120 Hz buzz) is detected pulse for pulse: there is no dead time.
    The July rule's 8 ms dead time would keep only every 5th."""
    p = noise_power(3, 9)
    times = 1.0 + np.arange(200) * 2e-3
    for t in times:
        add_box(p, t, 0.3e-3, 18.0)
    ev, _ = run(p)
    assert len(ev["t_start"]) == 200
    assert np.max(np.abs(ev["t_start"] - times)) < 5e-5


def test_rise_and_half_peak_onset_on_a_ramp():
    """Linear power ramp 0 -> 100 over 1 ms from t0: half peak is exactly 0.5 ms in; rise = start..peak."""
    p = np.full(int(2 * FS), 1.0, np.float32)
    a = int(1.0 * FS)
    n = int(1e-3 * FS)
    p[a:a + n] = np.linspace(1.0, 100.0, n)
    p[a + n:a + n + 20] = 100.0
    ev, _ = run(p)
    assert len(ev["t_start"]) == 1
    half_idx = a + (50.0 - 1.0) / (99.0 / (n - 1))
    assert ev["t_half"][0] == pytest.approx(half_idx / FS, abs=1 / FS)
    assert ev["rise_s"][0] == pytest.approx(ev["t_peak"][0] - ev["t_start"][0])
    assert ev["t_start"][0] < ev["t_half"][0] < ev["t_peak"][0]


def test_quiet_noise_gives_almost_nothing():
    """Gaussian noise, 16-sample smoothing: +10 dB is about 7 sigma of the smoothed power, so 60 s of pure noise
    gives (almost) no events."""
    p = noise_power(60, 10)
    ev, _ = run(p)
    assert len(ev["t_start"]) <= 1


# ----------------------------------------------------------------------------- full chain on IQ

def synth_iq(secs=12.0, seed=0):
    """Synthetic capture: clean-day band floor, the -22.6 dBFS receiver tone, broadband impulses (detected AND
    present in 16-30 kHz) and narrow 4-8 kHz-only bursts (detected, NOT in 16-30 kHz)."""
    import inject as inj
    x = inj.synthetic_host(secs, seed=seed).astype(np.complex128)
    floor_lin = 10 ** (-72.6 / 10)
    n = np.arange(int(2e-3 * FS))
    rng = np.random.default_rng(seed + 1)
    t_bb = 1.0 + np.arange(int(secs - 2)) * 1.0 + 0.137
    t_nb = t_bb + 0.4
    for t in t_bb:                                       # 3-sample spike: flat spectrum, all bands
        k = int(t * FS)
        x[k:k + 3] += np.sqrt(floor_lin * 10 ** 4.0) * 8 * np.exp(1j * rng.uniform(0, 6.28))
    for t in t_nb:                                       # 6 kHz RF tone burst (baseband -9 kHz), 1 ms, Hann
        k = int(t * FS)
        w = np.hanning(len(n) // 2)
        x[k:k + len(w)] += np.sqrt(floor_lin * 10 ** 3.0) * 6 * w * np.exp(-2j * np.pi * 9000 * n[:len(w)] / FS)
    return x.astype(np.complex64), t_bb, t_nb


def test_detect_iq_finds_pulses_and_confirms_only_broadband_ones():
    x, t_bb, t_nb = synth_iq()
    ev, info = dt.detect_iq(x, FS)
    t = ev["t_start"]
    # a 40 dB spike through the ZERO-PHASE band-pass rings before its true time (forward-backward filter), so the
    # +10 dB crossing comes up to ~0.7 ms early; the peak stays on time
    for tt in t_bb:
        assert np.any(np.abs(t - tt) < 1e-3), tt
        assert np.any(np.abs(ev["t_peak"] - tt) < 1e-4), tt
    for tt in t_nb:
        assert np.any(np.abs(t - tt) < 1.5e-3), tt
    assert len(t) <= len(t_bb) + len(t_nb) + 1
    hit_bb = [int(np.argmin(np.abs(t - tt))) for tt in t_bb]
    hit_nb = [int(np.argmin(np.abs(t - tt))) for tt in t_nb]
    assert ev["conf"][hit_bb].all()
    assert not ev["conf"][hit_nb].any()
    # the floor is the band floor (-72.6 dBFS in the real 2.5-9 kHz band is about -75.6 dBFS of real-part power
    # after the 16-sample smoothing, never near the -22.6 dBFS tone)
    assert -80 < info["floor_dbfs_min"] < -70
    assert info["cpu_frontend_s"] > 0 and info["cpu_detect_s"] >= 0


def test_band_powers_equal_the_front_end_and_any_slice_of_it():
    """band_powers runs the frame engine once over the whole input (no chunks, no guards): it equals the smoothed
    powers of frontend_offline, whatever chunk_s/guard_s say, and a slice starting off the frame grid gives the same
    powers as the same samples inside the longer run (the old 60 s chunks each started their own frame grid)."""
    x, _, _ = synth_iq(10.0, seed=3)
    p1, pc1 = dt.band_powers(x, FS, chunk_s=3.0, guard_s=2.0)
    o = fe.frontend_offline(x)
    p0, pc0 = dt.smooth_power(o.det), dt.smooth_power(o.conf)
    assert np.max(np.abs(p1 - p0)) <= 1e-5 * p0.max() and np.max(np.abs(pc1 - pc0)) <= 1e-5 * pc0.max()
    s = 31250 + 777
    p2, pc2 = dt.band_powers(x[s:], FS, n0=s)
    m = slice(s + int(FS), len(x) - int(FS))
    assert np.max(np.abs(p2[m.start - s:m.stop - s] / p1[m] - 1)) < 1e-4
    assert np.max(np.abs(pc2[m.start - s:m.stop - s] / pc1[m] - 1)) < 1e-4


def test_level_step_end_threshold_tracks_the_floor():
    """Verifier repro (step.py): a sustained +14..+18 dB level step made the first 'impulse' endless, because the end
    threshold was frozen at the floor of its start; 0 of 19 later pulses were found. The end threshold now tracks
    the floor: the step is one dropped too-long event and every later pulse is found. The health fields show it
    (level_steps, too_long_max_s about the time the floor needs to cross the new level)."""
    rng = np.random.default_rng(1)
    for step_db in (15, 18):
        lvl = 10 ** (step_db / 10)
        y = rng.standard_normal(int(30 * FS))
        y[int(5 * FS):] *= np.sqrt(lvl)
        p = dt.smooth_power(y)
        times = 10 + np.arange(19)
        for t in times:
            a = int(t * FS)
            p[a:a + 20] = lvl * 100
        ev, info = run(p)
        found = sum(np.any(np.abs(ev["t_start"] - t) < 2e-4) for t in times)
        assert found >= 18, (step_db, found)
        assert info["too_long"] >= 1 and info["level_steps"] == 1
        assert 0.5 < info["too_long_max_s"] < 4.0 and info["floor_rise_db_max"] > step_db - 1
    ev, info = run(noise_power(20, 2))                    # no step: quiet health
    assert info["level_steps"] == 0 and info["too_long_s"] == 0.0


def test_scan_cap_bounds_an_impulse_the_floor_cannot_follow():
    """With a floor that cannot follow (fixed here, as a held floor under heavy exclusion would be) a +15 dB level
    lasting 6 s is one impulse that never drops below the end threshold. The scan is closed every MAX_SCAN_S (4 s)
    and re-armed; the re-armed starts continue the dropped chain, so the whole 6 s counts as ONE too-long event
    (not one per 5 ms), the blind time is reported, and pulses after the level falls back are found."""
    p = noise_power(16, 21)
    p[int(3 * FS):int(9 * FS)] *= 10 ** 1.5
    times = 10 + np.arange(5) * 0.5
    for t in times:
        add_box(p, t, 0.3e-3, 20.0)
    st = {}
    s, e, n_long = dt.find_impulses(p, np.ones(len(p)), FS, stats=st)
    assert n_long == 1
    assert 5.9 < st["too_long_s"] < 6.1
    assert all(np.any(np.abs(s / FS - t) < 1e-4) for t in times)


def test_peak_db_is_the_true_peak():
    """Review defect: peak_db was read at t_peak, the middle of the run within 0.2 dB of the maximum, so it could
    read up to 0.2 dB low (17 events per 600 s under the 10 dB start). It is the maximum now; t_peak stays the
    plateau middle. No event can report a peak under the 10 dB start threshold."""
    p = np.full(int(2 * FS), 1.0, np.float32)
    a = int(1.0 * FS)
    p[a:a + 16] = 10 ** 1.1 * np.array([1.0] * 15 + [10 ** 0.019])   # flat top, max at the last sample
    ev, _ = run(p)
    assert ev["peak_db"][0] == pytest.approx(11.19, abs=0.01)
    assert ev["peak_db"].min() >= dt.START_DB
    ev, _ = run(noise_power(30, 23))
    assert len(ev["peak_db"]) == 0 or ev["peak_db"].min() > dt.START_DB


def test_fast_floor_helpers_equal_the_reference():
    """The vectorised floor (sliding windows, one partition per row), floor_at_steps (the 16-30 kHz floor only at
    event steps) and the np.repeat floor_per_sample give the same numbers as the plain definitions."""
    rng = np.random.default_rng(24)
    for nb in (5, 300, 503, 5000, 20000):
        bm = rng.random(nb) ** 3
        bs = dt.BLOCK_N / FS
        fl, ok = dt.sliding_floor(bm, bs)
        per_hop, win = round(dt.FLOOR_HOP_S / bs), round(dt.FLOOR_WIN_S / bs)
        for r in np.flatnonzero(ok):
            assert fl[r] == np.median(bm[max(0, r * per_hop - win):r * per_hop])
        st = rng.integers(0, len(fl) + 3, 40)
        assert np.array_equal(dt.floor_at_steps(bm, bs, st), fl[np.minimum(st, len(fl) - 1)])
        for n, hop in ((nb * 62, 6262), (7, 6262), (len(fl) * 6262 + 999, 6262)):
            ref = fl[np.minimum(np.arange(n) // hop, len(fl) - 1)]
            assert np.array_equal(dt.floor_per_sample(fl, n, hop), ref)
    a = rng.random((30, 4032))
    assert np.array_equal(dt._median_rows(a), np.median(a, axis=1))


def test_bypassing_the_front_end_puts_the_floor_on_the_tone():
    """Regression guard (the OnAir-raw failure): smoothed raw IQ power has its median floor at the -22.6 dBFS tone,
    and impulses 30 dB over the sferic band are invisible."""
    x, t_bb, _ = synth_iq(6.0, seed=4)
    ev, info = dt.events_from_power(dt.smooth_power(x), None, FS)
    assert abs(info["floor_dbfs_min"] - (-22.6)) < 0.5
    assert len(ev["t_start"]) == 0


# ----------------------------------------------------------------------------- July column and registry

def test_july_compat_tolerance():
    ev = np.array([1.0, 2.0, 3.0])
    tj = np.array([1.0015, 2.0031, 5.0])
    assert dt.july_compat(ev, tj).tolist() == [True, False, False]
    assert dt.july_compat(ev, np.array([])).tolist() == [False, False, False]


def test_july_times_dead_time():
    """The July rule's 8 ms dead time: spikes 3-5 ms apart (irregular) give at most every second one (rising edges are
    kept only more than 8 ms after the last kept one) - exactly the blindness v2 removes. The spacing is irregular on
    purpose: a strictly periodic train is a comb of lines and the step-3 line excision strips it (that is how it
    removes 120 Hz buzz; a 250/s regular train lost half its pulses here)."""
    x, _, _ = synth_iq(3.0, seed=5)
    x = x.astype(np.complex128)
    rng = np.random.default_rng(55)
    times = 1.0 + np.cumsum(rng.uniform(3e-3, 5e-3, 50))
    for t in times:
        k = int(t * FS)
        x[k:k + 3] += 0.03                               # about +30 dB in 2.5-9 kHz
    x = x.astype(np.complex64)
    win = (times[0] - 0.01, times[-1] + 0.01)
    tj, _ = dt.july_times(x, FS)
    assert np.sum((tj > win[0]) & (tj < win[1])) <= 26
    ev, _ = dt.detect_iq(x, FS)
    t = ev["t_start"]
    found = [np.any(np.abs(ev["t_peak"] - tt) < 2e-4) for tt in times]
    assert np.mean(found) >= 0.95


def test_registered_for_the_harness():
    e = sw.get_detector("v2_detect")
    assert e["ungated"] is None
    x, t_bb, _ = synth_iq(4.0, seed=6)
    t, pk, info = e["fn"](x, FS)
    assert len(t) == len(pk) and np.all(np.isfinite(pk))
    assert "too_long" in info
