"""Fast tests for sferic_watch.pipeline on SYNTHETIC IQ only (noise host + receiver tone + synthetic pulses)."""
import numpy as np
import pytest

import sferic_watch as sw
from sferic_watch import detect as dt
from sferic_watch import pipeline as pl

import inject as inj                      # bench/inject.py: synthetic host and pulses (on the test path via conftest)
from harness import comb_share, hits

FS = pl.FS
SECS = 20.0


def damped(fs=FS, f0=6000.0, tau=0.15e-3, n=200):
    """A short damped 6 kHz pulse as complex baseband (centre 15 kHz), unit peak envelope."""
    t = np.arange(n) / fs
    return np.exp(-t / tau) * np.exp(2j * np.pi * (f0 - inj.FC) * t)


@pytest.fixture(scope="module")
def scene():
    """20 s synthetic host (clean-day floor, -22.6 dBFS tone) with
    * sparse buzz: 30 % of the 120 Hz grid from t = 0, 40 us jitter, +-2 dB amplitude scatter, ~20 dB peak;
    * sferics: Poisson 2/s, ~24 dB peak.
    Returns (x, buzz_times, sferic_times)."""
    rng = np.random.default_rng(11)
    x = inj.synthetic_host(SECS, seed=3)
    floor = inj.host_floor(x)
    pulse = damped()
    g0 = np.sqrt(floor / inj.snip_peak_power(pulse))
    k = np.sort(rng.choice(int(SECS * 120) - 2, int(0.3 * SECS * 120), replace=False)) + 1
    tb = 2.1e-3 + k / 120.0 + rng.normal(0, 40e-6, k.size)
    gb = g0 * 10 ** ((20 + rng.uniform(-2, 2, k.size)) / 20)
    ts = inj.poisson_starts(2.0, SECS, seed=12, guard=0.5)
    gs = np.full(ts.size, g0 * 10 ** (24 / 20))
    y = inj.inject(x, [pulse] * (k.size + ts.size), np.r_[tb, ts], np.r_[gb, gs])
    return y, tb, ts


@pytest.fixture(scope="module")
def runs(scene):
    x, _, _ = scene
    flags = pl.hf.flag_iq(x)
    return flags, pl.run(x, flags=flags), pl.run(x, gate=False, flags=flags)


def test_ungated_twin_equals_detect(scene, runs):
    """gate=False is exactly the step-4 detector (v2_detect), so the harness's false-rejection twin is honest."""
    x, _, _ = scene
    _, _, (ev0, _) = runs
    ev, _ = dt.detect_iq(x)
    np.testing.assert_array_equal(ev0["t_start"], ev["t_start"])


def test_buzz_flagged_gated_sferics_kept(scene, runs):
    """The synthetic buzz is flagged; after the gate the comb share drops to near chance (0.144) and >= 90 % of the
    sferics are still detected (the 24 dB pulses also escape when they land on the train)."""
    x, tb, ts = scene
    flags, (ev, info), (ev0, _) = runs
    assert sum(b["hum"] for b in flags) >= 1
    assert info["n_trains"] >= 1 and info["rejected"] > 0
    cs0, cs = comb_share(ev0["t_start"]), comb_share(ev["t_start"])
    assert cs0 > 0.5
    assert cs < 0.3
    assert hits(ev["t_start"], ts, 2e-3).mean() >= 0.9          # damped() starts at its first sample
    assert 0 < info["gated_frac"] < 0.1


def test_rejected_pulses_are_not_merge_targets():
    """A sferic 0.3 ms after a buzz pulse would merge into it (0.5 ms merge) and start at the buzz time. With the
    buzz pulse blanked first, the event starts at the sferic."""
    p = np.ones(int(2 * FS), np.float32)
    a = int(1.0 * FS)
    p[a:a + 20] = 30.0                                   # buzz pulse (rejected below)
    b = a + 20 + int(0.3e-3 * FS) + 6                    # sferic after the 100 us hold + 0.3 ms gap
    p[b:b + 20] = 100.0
    ev, _ = dt.events_from_power(p.copy(), None)
    assert ev["t_start"][0] == pytest.approx(a / FS)     # merged: starts at the buzz
    excl = np.zeros(len(p) // dt.BLOCK_N, bool)
    s, e, pk = pl.raw_impulses(p)
    assert len(s) == 2                                   # unmerged: two impulses
    pl.blank(p, s[:1], e[:1], excl)
    ev, _ = dt.events_from_power(p, None, exclude_blocks=excl)
    assert len(ev["t_start"]) == 1 and ev["t_start"][0] == pytest.approx(b / FS)
    assert excl.sum() >= 1


def test_raw_impulse_peaks_match_detector():
    """Single, well separated impulses: raw_impulses gives the same starts and peaks as the merged detector."""
    p = np.ones(int(3 * FS), np.float32)
    for t, db in ((1.0, 15.0), (1.5, 20.0), (2.2, 12.0)):
        p[int(t * FS):int(t * FS) + 25] = 10 ** (db / 10)
    s, e, pk = pl.raw_impulses(p)
    ev, _ = dt.events_from_power(p, None)
    np.testing.assert_allclose(s / FS, ev["t_start"])
    np.testing.assert_allclose(pk, ev["peak_db"], atol=1e-4)


def test_refractory_keeps_first_of_close_pairs():
    t = np.array([0.0, 0.001, 0.0025, 0.010, 0.0115])
    assert pl.refractory(t, 0.002).tolist() == [True, False, True, True, False]


def test_clean_scene_identical_with_and_without_gate():
    """No mains: nothing flagged, the gated chain returns exactly the ungated events (plan: clean within 5 %)."""
    x = inj.synthetic_host(8.0, seed=21)
    ts = inj.poisson_starts(5.0, 8.0, seed=22, guard=0.3)
    pulse = damped()
    g = np.sqrt(inj.host_floor(x) * 10 ** 2.0 / inj.snip_peak_power(pulse))
    y = inj.inject(x, [pulse] * ts.size, ts, np.full(ts.size, g))
    ev, info = pl.run(y)
    ev0, _ = pl.run(y, gate=False)
    assert info["hum_frac"] == 0 and info["rejected"] == 0 and info["gated_frac"] == 0
    np.testing.assert_array_equal(ev["t_start"], ev0["t_start"])


def test_registered_for_the_harness():
    assert sw.get_detector("v2")["ungated"] == "v2_nogate"
    assert "v2_nogate" in sw.detector_names()


def test_second_pass_peaks_use_the_first_pass_floor():
    """Verifier finding: the second pass found impulses on the re-estimated (lower) floor and measured their peaks
    there too, so the 6 dB escape loosened by the floor drop. ``peak_floor`` keeps the first-pass scale: a floor
    twice as high reads every peak 3.01 dB lower, while the impulses found stay the same."""
    p = np.ones(int(2 * FS), np.float32)
    p[int(1.0 * FS):int(1.0 * FS) + 25] = 100.0
    s, e, pk = pl.raw_impulses(p)
    s2, e2, pk2 = pl.raw_impulses(p, peak_floor=np.full(len(p), 2.0))
    np.testing.assert_array_equal(s, s2)
    np.testing.assert_allclose(pk - pk2, 10 * np.log10(2.0), atol=1e-4)


def test_plan_rule_variant_and_info(scene, runs):
    """scope="flagged", self_test=False is the plan's literal gate (flagged blocks only); it still gates this
    all-flagged scene, and the info dict says which variant ran and what kinds of trains it used."""
    x, tb, ts = scene
    flags, (ev, info), _ = runs
    evp, infp = pl.run(x, flags=flags, scope="flagged", self_test=False)
    assert info["scope"] == "all" and infp["scope"] == "flagged" and infp["self_test"] is False
    assert infp["rejected"] > 0 and sum(infp["train_kinds"].values()) == infp["n_trains"]
    assert comb_share(evp["t_start"]) < 0.3


def test_mains_free_sferics_with_ringing_tails_are_not_gated():
    """The review's high finding as a pipeline test: 70 s of the harness's synthetic host with synthetic-bank sferics
    (which ring for 1-3 ms and so make July edges on their own tails) at 5/s, no mains. Seven blocks reach the whole
    fold; the gate must reject nothing. (The old start-only guard also passes this particular scene: its false
    trains came in about 1 of 4 seeds of 120 s. The mechanism itself is pinned down in
    test_mainsgate.test_tail_edges_bias_start_free_but_not_extent_free; the seed scan is in the bench.)"""
    x = inj.synthetic_host(70.0, seed=5)
    syn = inj.synth_bank(200, seed=11)[0]
    cut = inj.poisson_starts(5.0, 70.0, seed=1001)
    rng = np.random.default_rng(2001)
    sn = [syn[k] for k in rng.integers(0, len(syn), len(cut))]
    y = inj.inject(x, sn, cut, inj.snr_gains(sn, inj.host_floor(x), 24.0))
    flags = pl.hf.flag_iq(y)
    ev, info = pl.run(y, flags=flags)
    ev0, _ = pl.run(y, gate=False, flags=flags)
    assert info["rejected"] == 0 and info["n_trains"] == 0
    np.testing.assert_array_equal(ev["t_start"], ev0["t_start"])
    assert all("hum_ev" in b and "hum_edge" in b for b in flags)      # caller's flags gained the event source
    assert info["hum_frac"] == 0
