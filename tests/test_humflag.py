"""Fast synthetic tests for sferic_watch.humflag (no captures read)."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sferic_watch import humflag as hf  # noqa: E402

FS = hf.FS
RNG_SEED = 1234


def noise_iq(seconds, rng, sigma=1e-3, tone_dbfs=-22.6):
    """Complex white noise plus the receiver's 0 Hz tone (which the band-pass must ignore)."""
    n = int(seconds * FS)
    x = (rng.standard_normal(n) + 1j * rng.standard_normal(n)).astype(np.complex64) * np.float32(sigma)
    x += np.complex64(10 ** (tone_dbfs / 20))
    return x


def add_pulses(x, times, amp, f_bb=-10000.0, width_s=1.6e-4):
    """Add Gaussian-windowed bursts at baseband f_bb (real 5 kHz, inside 2.5-9 kHz)."""
    half = int(4 * width_s * FS)
    k = np.arange(-half, half + 1)
    w = np.exp(-0.5 * (k / (width_s * FS)) ** 2)
    for t in times:
        c = int(round(t * FS))
        a, b = c - half, c + half + 1
        if a < 0 or b > len(x):
            continue
        n = np.arange(a, b)
        x[a:b] += (amp * w * np.exp(2j * np.pi * f_bb / FS * n)).astype(np.complex64)
    return x


def buzz_times(t0, t1, rng, f=119.98, jitter_s=40e-6, keep=1.0):
    """Pulse times on a f grid with rms jitter; keep < 1 drops cycles at random (sparse buzz)."""
    t = np.arange(np.ceil(t0 * f), np.floor(t1 * f)) / f + 0.0013
    t = t + rng.normal(0, jitter_s, len(t))
    if keep < 1:
        t = t[rng.random(len(t)) < keep]
    return t[(t > t0) & (t < t1)]


AMP = 0.02   # ~20x the per-component noise sigma: clear pulses, like the 15.7-17.8 dB buzz


# --------------------------------------------------------------- unit pieces

def test_interval_share_exact_grid_and_multiples():
    p = hf.MAINS_PERIOD
    t = np.cumsum([p, p, 2 * p, 3 * p, p])
    assert hf.interval_share(t) == pytest.approx(1.0)
    t2 = np.cumsum([0.0, p, 0.0031, p * 1.05])             # intervals p, 3.1 ms, 1.05p: 1 of 3 on grid
    assert hf.interval_share(t2) == pytest.approx(1 / 3)
    assert hf.interval_share(np.array([0.1])) == 0.0


def test_share_chance_stays_below_flag_level():
    rates = np.linspace(0.5, 400, 800)
    ch = np.array([hf.share_chance(r) for r in rates])
    assert ch.max() < 0.12
    assert ch.max() < hf.SHARE_FLAG / 2


def test_share_chance_matches_poisson_simulation():
    rng = np.random.default_rng(RNG_SEED)
    for rate in (20.0, 80.0):
        t = np.cumsum(rng.exponential(1 / rate, 200000))
        assert hf.interval_share(t) == pytest.approx(hf.share_chance(rate), abs=0.005)


def test_rayleigh_scan_finds_mains_frequency():
    rng = np.random.default_rng(RNG_SEED)
    t = buzz_times(0, 10, rng, f=119.93, jitter_s=40e-6)
    r, f = hf.rayleigh_scan(t)
    assert r > 0.9
    assert f == pytest.approx(119.93, abs=0.01)


def test_rayleigh_levels_random_times():
    rng = np.random.default_rng(RNG_SEED)
    n = 400
    rs = [hf.rayleigh_scan(np.sort(rng.uniform(0, 10, n)))[0] for _ in range(60)]
    chance, crit = hf.rayleigh_levels(n)
    assert chance == pytest.approx(0.05)
    assert not np.any(np.array(rs) > crit)              # p_fa 1e-4: none expected in 60 draws
    assert np.median(rs) < 3 * chance


@pytest.mark.parametrize("n", [50, 300])
def test_rayleigh_level_false_alarm_is_calibrated(n):
    """Measured false-alarm rate of the scanned R at a nominal 0.02 must be 0.02, not 2-3x it.

    1500 random blocks: binomial sd at p = 0.02 is 0.0036, so [0.011, 0.031] is about +-2.5 sd.
    The old exp(-u) = p / (1 + W T) level measured about 0.04 here (2-3x, two verifiers).
    """
    rng = np.random.default_rng(RNG_SEED + n)
    _, crit = hf.rayleigh_levels(n, p_fa=0.02)
    rs = np.array([hf.rayleigh_scan(np.sort(rng.uniform(0, 10, n)))[0] for _ in range(1500)])
    fa = float(np.mean(rs > crit))
    assert 0.011 <= fa <= 0.031


def test_locked_count_ignores_random_edges():
    rng = np.random.default_rng(RNG_SEED)
    b = buzz_times(0, 10, rng, f=119.99, keep=0.3)               # ~360 buzz pulses
    s = rng.uniform(0, 10, 2000)                                  # 200/s random edges
    t = np.sort(np.concatenate([b, s]))
    r, f = hf.rayleigh_scan(t)
    n_lock, _ = hf.locked_count(t, f)
    assert n_lock == pytest.approx(len(b), rel=0.1)
    r0, f0 = hf.rayleigh_scan(np.sort(s))
    assert abs(hf.locked_count(np.sort(s), f0)[0]) < 60            # chance part removed (sd ~13)


def test_debounce_keeps_first_and_spacing():
    idx = np.array([0, 10, 70, 140, 141, 300])
    assert hf.debounce(idx, 62).tolist() == [0, 70, 140, 300]


# --------------------------------------------------------------- block flags

def test_clean_noise_not_flagged():
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(20, rng)
    blocks = hf.flag_iq(x)
    assert len(blocks) == 2
    assert not any(b["hum"] for b in blocks)
    assert all(b["n_edges"] < hf.N_STAT for b in blocks)


def test_steady_buzz_flagged_with_frequency():
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(20, rng)
    add_pulses(x, buzz_times(0, 20, rng, f=119.98), AMP)
    blocks = hf.flag_iq(x)
    assert all(b["hum"] for b in blocks)
    for b in blocks:
        assert b["n_edges"] > 1000
        assert b["share"] > 0.9
        assert b["R"] > 0.9
        assert b["f_mains"] == pytest.approx(119.98, abs=0.02)


def test_poisson_impulses_at_80_per_s_not_flagged():
    """Dense random 'sferics' pass the N rule but must fail share and R."""
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(20, rng)
    t = np.sort(rng.uniform(0, 20, 1600))
    add_pulses(x, t, AMP)
    blocks = hf.flag_iq(x)
    for b in blocks:
        assert b["n_edges"] >= hf.N_MIN
        assert b["share"] < 0.2
        assert not b["hum"] and not b["hum_plan"]


def test_sparse_buzz_skipping_cycles_flagged():
    """Buzz keeping ~1 in 5 cycles (gaps of several half-cycles, like 11-05 20Z)."""
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(10, rng)
    add_pulses(x, buzz_times(0, 10, rng, keep=0.2), AMP)
    b = hf.flag_iq(x)[0]
    assert b["n_edges"] >= hf.N_MIN
    assert b["hum"]
    # a k = 1-only interval test would see well under the 0.3 flag level here
    assert hf.interval_share(np.sort(buzz_times(0, 10, rng, keep=0.2)), kmax=1) < 0.3


def test_too_few_buzz_pulses_not_flagged():
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(10, rng)
    add_pulses(x, buzz_times(3, 3.1, rng), AMP)           # ~12 pulses, under N_LOCK_MIN
    b = hf.flag_iq(x)[0]
    assert b["n_edges"] < hf.N_LOCK_MIN
    assert not b["hum"]


def test_buzz_plus_sferics_still_flagged():
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(10, rng)
    add_pulses(x, buzz_times(0, 10, rng), AMP)
    add_pulses(x, np.sort(rng.uniform(0, 10, 300)), AMP * 2)
    assert hf.flag_iq(x)[0]["hum"]


def test_buzz_under_storm_rate_flagged():
    """Full buzz under 200 random sferics/s: interval share drops below 0.3 (the plan
    rule misses it, verifier probe D) but the phase lock and locked count do not."""
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(10, rng)
    add_pulses(x, buzz_times(0, 10, rng), AMP)
    add_pulses(x, np.sort(rng.uniform(0, 10, 2000)), AMP * 2)
    b = hf.flag_iq(x)[0]
    assert b["share"] < hf.SHARE_FLAG
    assert not b["hum_plan"]
    assert b["hum"]
    assert b["n_locked"] > 800


def test_sparse_buzz_with_sferics_flagged():
    """Buzz at ~24/s plus 30 random edges/s (verifier probe E: the plan rule gave 0/50)."""
    rng = np.random.default_rng(RNG_SEED)
    t = np.sort(np.concatenate([buzz_times(0, 10, rng, keep=0.2), rng.uniform(0, 10, 300)]))
    st = hf.block_stats(t)
    assert st["hum"] and not st["hum_plan"]


def test_very_sparse_buzz_low_n_flagged():
    """~4 buzz pulses/s, N about 40 (like 11-05 20Z, N 15-98 with R 0.84-1.0)."""
    rng = np.random.default_rng(RNG_SEED)
    t = buzz_times(0, 10, rng, keep=0.035)
    st = hf.block_stats(t)
    assert st["n_edges"] < hf.N_MIN
    assert st["hum"] and not st["hum_plan"]


@pytest.mark.parametrize("rate", [3.0, 25.0, 80.0, 300.0])
def test_poisson_edges_never_flagged(rate):
    """Random edge times at any rate (debounced at 1 ms like real edges): 0 flags in 150 blocks."""
    rng = np.random.default_rng(int(rate))
    gap = int(round(hf.DEBOUNCE_S * FS))
    flags = 0
    for _ in range(150):
        t = np.sort(rng.uniform(0, 10, rng.poisson(rate * 10)))
        t = hf.debounce(np.rint(t * FS).astype(np.int64), gap) / FS
        flags += hf.block_stats(t)["hum"]
    assert flags == 0


# --------------------------------------------------------------- whole file

def test_flag_file_onset_partway(tmp_path):
    """Buzz switching on at 20 s of a 40 s file: first hum block starts at 20 s."""
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(40, rng)
    add_pulses(x, buzz_times(20, 40, rng), AMP)
    iq = np.empty(2 * len(x), dtype=np.int16)
    iq[0::2] = np.clip(np.rint(x.real * 32768), -32768, 32767)
    iq[1::2] = np.clip(np.rint(x.imag * 32768), -32768, 32767)
    p = tmp_path / "syn.cs16"
    iq.tofile(p)
    blocks = hf.flag_file(p)
    s = hf.summarize(blocks)
    assert [b["hum"] for b in blocks] == [False, False, True, True]
    assert s["status"] == "partial"
    assert s["first_hum_s"] == pytest.approx(20.0)
    assert s["hum_frac"] == pytest.approx(0.5)
    # file path and in-memory path agree
    mem = hf.flag_iq(x)
    assert [b["n_edges"] for b in mem] == pytest.approx([b["n_edges"] for b in blocks], abs=3)


def test_summarize_status():
    mk = lambda flags: [{"hum": f, "t0_s": 10.0 * i, "R": 0.0, "f_mains": 120.0} for i, f in enumerate(flags)]
    assert hf.summarize(mk([False] * 3))["status"] == "clean"
    assert hf.summarize(mk([True] * 3))["status"] == "hum"
    s = hf.summarize(mk([False, True, True]))
    assert s["status"] == "partial" and s["first_hum_s"] == 10.0 and s["last_hum_s"] == 20.0
    plan = [dict(b, hum_plan=False) for b in mk([True] * 3)]
    assert hf.summarize(plan, key="hum_plan")["status"] == "clean"


# --------------------------------------------------------------- two phase clusters, event source

def test_locked_count_two_clusters():
    """11-06 08:30Z shape: buzz in two clusters per cycle, 4.2 ms apart (about half a 120 Hz cycle), so the mean phase
    falls between them. Centring on the mean found almost none; the mode plus a second cluster counts both."""
    rng = np.random.default_rng(RNG_SEED + 7)
    a = buzz_times(0, 10, rng, f=120.0, keep=0.15)
    b = buzz_times(0, 10, rng, f=120.0, keep=0.12) + 4.2e-3
    t = np.sort(np.r_[a, b, rng.uniform(0, 10, 40)])
    n_lock, _ = hf.locked_count(t, 120.0)
    assert n_lock == pytest.approx(a.size + b.size, rel=0.2)
    # the old estimator, centred on the circular mean, for contrast
    ph = 2 * np.pi * 120.0 * (t - t[0])
    d = np.angle(np.exp(1j * (ph - np.angle(np.exp(1j * ph).sum())))) / (2 * np.pi * 120.0)
    assert np.sum(np.abs(d) <= hf.LOCK_WIN_S) < 0.5 * (a.size + b.size)


def test_locked_count_one_cluster_ignores_second_by_chance():
    """A single train plus random edges: no second cluster is added (its count is at chance)."""
    rng = np.random.default_rng(RNG_SEED + 8)
    b = buzz_times(0, 10, rng, f=119.99, keep=0.3)
    t = np.sort(np.r_[b, rng.uniform(0, 10, 300)])
    assert hf.locked_count(t, 119.99)[0] == pytest.approx(b.size, rel=0.12)


def test_event_source_flags_weak_buzz_the_edges_miss():
    """Detector starts with 3 buzz impulses/s plus 2 sferics/s in a block whose own edges saw nothing: the event
    source flags it (hum, hum_ev) while the edge flag (hum_edge) stays off; f_mains stays the edge value."""
    rng = np.random.default_rng(RNG_SEED + 9)
    ev = np.sort(np.r_[buzz_times(0, 10, rng, keep=3 / 120), rng.uniform(0, 10, 20)])
    blocks = [dict(hf.block_stats(np.sort(rng.uniform(0, 10, 5))), t0_s=0.0, dur_s=10.0)]
    hf.add_event_stats(blocks, ev)
    b = blocks[0]
    assert b["hum"] and b["hum_ev"] and not b["hum_edge"]
    assert b["f_ev"] == pytest.approx(119.98, abs=0.01) and not b["f_mains"] == b["f_ev"]


@pytest.mark.parametrize("rate", [1.0, 5.0, 30.0, 200.0])
def test_event_source_quiet_on_poisson_starts(rate):
    """Poisson detector starts (no mains), 200 blocks per rate: no event-source flag."""
    rng = np.random.default_rng(int(rate * 10) + 5)
    hits = sum(hf.event_stats(np.sort(rng.uniform(0, 10, rng.poisson(rate * 10))))["hum_ev"] for _ in range(200))
    assert hits == 0


def test_flag_iq_with_events_ors_the_sources():
    rng = np.random.default_rng(RNG_SEED)
    x = noise_iq(20, rng)
    ev = np.sort(np.r_[buzz_times(10, 20, rng, keep=3 / 120), rng.uniform(0, 20, 30)])
    blocks = hf.flag_iq(x, events=ev)
    assert [b["hum_edge"] for b in blocks] == [False, False]
    assert [b["hum"] for b in blocks] == [False, True]


def test_event_window_finds_sparse_source():
    """11-03 18Z shape: a mains-locked source at 0.9 starts/s (too few per 10 s block for the 10 s test) plus
    sferics at 0.5/s, switching on at 30 s of a 90 s run (about 65 % of the starts locked, as measured there).
    The 10 s test cannot see it; the 60 s window test flags the minute that holds it (the first 30 s may be
    flagged too: one window spans them). A capture shorter than one window gets the 10 s test only."""
    rng = np.random.default_rng(RNG_SEED + 11)
    src = buzz_times(30, 90, rng, f=119.99, keep=0.9 / 120)
    ev = np.sort(np.r_[src, rng.uniform(0, 90, 45)])
    blocks = [{"t0_s": 10.0 * i, "dur_s": 10.0, "hum": False} for i in range(9)]
    hf.add_event_stats(blocks, ev)
    assert not any(b["hum_ev10"] for b in blocks)
    assert all(b["hum"] and b["hum_ev_win"] for b in blocks[3:])
    short = [{"t0_s": 10.0 * i, "dur_s": 10.0, "hum": False} for i in range(2)]
    hf.add_event_stats(short, ev[ev < 20] + 0.0)
    assert not any(b["hum_ev_win"] for b in short)
