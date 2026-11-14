"""Tests for sferic_watch.io and the bench harness pieces it feeds (scoring, injection, rule model).

All signals are synthetic and small; the whole file runs in a few seconds. (The harness/injection checks live here
because this step owns only tests/test_io.py; they test bench/harness.py and bench/inject.py.)"""
import json
import os

import numpy as np
import pytest

import sferic_watch as sw
from sferic_watch import io as swio

FS = 62500.0


# ---------------------------------------------------------------- io
def test_read_slice_exact_values_and_offset(tiny_capture):
    path, i, q = tiny_capture
    x = swio.read_slice(path, 0.5, 0.25, dc="none")
    s0, n = int(0.5 * FS), int(0.25 * FS)
    assert x.dtype == np.complex64 and len(x) == n
    assert np.array_equal(x.real * 32768, i[s0:s0 + n].astype(np.float32))
    assert np.array_equal(x.imag * 32768, q[s0:s0 + n].astype(np.float32))


def test_read_slice_dc_mean_removed(tiny_capture):
    path, _, _ = tiny_capture
    x = swio.read_slice(path, 0.0, 1.0)            # default dc="mean"
    assert abs(x.mean()) < 1e-7
    y = swio.read_slice(path, 0.0, 1.0, dc="none")
    assert np.isclose(y.imag.mean() * 32768, -7.0)


def test_read_slice_short_raises_unless_allowed(tiny_capture):
    path, _, _ = tiny_capture
    with pytest.raises(ValueError):
        swio.read_slice(path, 1.5, 1.0)
    x = swio.read_slice(path, 1.5, 1.0, allow_short=True)
    assert len(x) == int(0.5 * FS)
    with pytest.raises(ValueError):
        swio.read_slice(path, 0, 1, dc="bogus")


def test_cs16_round_trip_is_bit_exact(tiny_capture, tmp_path):
    path, _, _ = tiny_capture
    x = swio.read_slice(path, 0, 2.0, dc="none")
    p = tmp_path / "rt.cs16"
    swio.write_cs16(str(p), x, sidecar={"fs_hz": FS})
    assert np.array_equal(np.fromfile(p, np.int16), np.fromfile(path, np.int16))
    assert swio.read_sidecar(str(p))["fs_hz"] == FS


def test_sidecar_and_capture_info(tiny_capture):
    path, _, _ = tiny_capture
    s = swio.read_sidecar(path)
    assert s["antenna"] == "Antenna C" and s["utc_start"].hour == 4
    info = swio.capture_info(path)
    assert info["n_samples"] == int(2 * FS) and abs(info["secs"] - 2.0) < 1e-9 and info["fc"] == 15000.0
    assert swio.read_sidecar(path.replace("vlf_0000Z", "missing")) == {}


def test_iter_slices_complete_only(tiny_capture):
    path, _, _ = tiny_capture
    got = [(t0, len(x)) for t0, x in swio.iter_slices(path, 0.75, hop=0.5)]
    assert [g[0] for g in got] == [0.0, 0.5, 1.0]          # 1.5 + 0.75 > 2 s: not yielded
    assert all(g[1] == int(0.75 * FS) for g in got)


def test_list_corpus_filters(tiny_corpus):
    allc = swio.list_corpus(tiny_corpus)
    assert [e["id"] for e in allc] == ["tiny", "gone"]
    assert [e["id"] for e in swio.list_corpus(tiny_corpus, golden_only=True)] == ["tiny"]
    assert [e["id"] for e in swio.list_corpus(tiny_corpus, role="steady_hum")] == ["gone"]
    assert [e["id"] for e in swio.list_corpus(tiny_corpus, must_exist=True)] == ["tiny"]


def test_repo_corpus_is_well_formed_and_private():
    c = swio.load_corpus()
    golden = [e for e in c["captures"] if e.get("golden", True)]
    assert len(golden) == 14
    ij = c["injection"]
    assert len(ij["bank"]["cut_start_samples"]) == ij["bank"]["n"] == 386
    assert set(ij["pieces"]) <= {e["id"] for e in c["captures"]}
    txt = open(swio.CORPUS_JSON).read().lower()
    for word in ("latitude", "longitude", "\"lat\"", "\"lon\"", "grid_square", "maidenhead", "address"):
        assert word not in txt


# ---------------------------------------------------------------- registry
def test_registry_register_and_lookup():
    @sw.register("_test_det", ungated="_test_det_raw")
    def _d(x, fs):
        """Doc line."""
        return np.array([0.1]), np.array([3.0])
    e = sw.get_detector("_test_det")
    assert e["ungated"] == "_test_det_raw" and e["note"] == "Doc line."
    assert "_test_det" in sw.detector_names()
    with pytest.raises(KeyError):
        sw.get_detector("_nope_")


# ---------------------------------------------------------------- scoring
def test_corrected_recall_perfect_and_random():
    import harness as H
    truth = 1.0 + np.arange(80) * 1.43
    rec, ch, corr = H.corrected_recall(truth + 0.001, truth)
    assert rec == 1.0 and ch == 0.0 and corr == 1.0
    # a detector firing every 3 ms finds everything by luck: corrected recall must be ~0
    spam = np.arange(0, 120, 0.003)
    rec, ch, corr = H.corrected_recall(spam, truth)
    assert rec == 1.0 and ch == 1.0 and corr == 0.0
    # 1 ms late is a hit, 2.5 ms late is not
    assert H.hits([5.0025], [5.0]).sum() == 0 and H.hits([5.001], [5.0]).sum() == 1


def test_comb_share_chance_and_buzz():
    import harness as H
    rng = np.random.default_rng(0)
    pois = np.cumsum(rng.exponential(1 / 50, 20000))
    assert abs(H.comb_share(pois) - H.COMB_CHANCE) < 0.01          # 0.144 by construction
    buzz = np.arange(0, 60, 1 / 120.0) + rng.normal(0, 40e-6, 7200)
    assert H.comb_share(buzz) > 0.99
    assert H.comb_share([1, 2, 3]) is None


def _need_rule():
    import harness as H
    if H._rule_module() is None:
        pytest.skip("OnAir rule model is kept outside the repo (GPL-3); not installed")


def test_onair_rule_poisson_reproduces_mains_sim():
    _need_rule()
    """The onair-review simulation numbers (4.3 / 13.5 / 17.9 % at 40/60/80 per s) within 2 points."""
    import harness as H
    r = H.rule_poisson(rates=(5, 10, 20, 40, 60, 80))
    assert r["5"]["rejected_as_mains"] < 0.01
    for rate, ref in (("40", 0.043), ("60", 0.135), ("80", 0.179)):
        assert abs(r[rate]["rejected_as_mains"] - ref) <= 0.02, (rate, r[rate])


def test_onair_rule_rejects_a_train():
    _need_rule()
    import harness as H
    rng = np.random.default_rng(3)
    train = np.arange(0.0037, 20, 1 / 120.0) + rng.normal(0, 40e-6, 2400)
    kept, rej, on = H.onair_rule_reject(np.sort(train))
    assert rej.mean() > 0.95


def test_hum_blocks_intervals():
    import harness as H
    per = [10] * 9 + [900] * 6 + [8] * 3
    comb = [0.1] * 9 + [0.98] * 6 + [0.2] * 3
    assert H.hum_blocks(per, comb) == [[90, 150]]


# ---------------------------------------------------------------- injection
def test_inject_quantizes_like_int16_file():
    import inject as inj
    x = np.zeros(2000, np.complex64)
    s = np.full(100, (3.7 - 2.2j) / 32768)
    y = inj.inject(x, [s], [0.001])
    s0 = int(0.001 * FS)
    assert np.allclose(y[s0:s0 + 100] * 32768, 3 - 2j)          # truncation toward zero, as astype(int16)
    assert np.all(y[:s0] == 0) and np.all(y[s0 + 100:] == 0)


def test_snr_gain_hits_target():
    import inject as inj
    host = inj.synthetic_host(4.0, seed=1)
    floor = inj.host_floor(host)
    assert abs(10 * np.log10(floor) - (-72.6)) < 0.6            # in-band floor of the synthetic host
    w, kinds = inj.synth_bank(6, seed=2)
    assert set(kinds) <= {"damped", "dispersed"}
    for snr in (12, 30):
        g = inj.snr_gains(w, floor, snr)
        got = [10 * np.log10(inj.snip_peak_power(gi * wi) / floor) for gi, wi in zip(g, w)]
        assert np.allclose(got, snr, atol=1e-6)


def test_synthetic_sferic_onset_and_dispersion():
    import inject as inj
    rng = np.random.default_rng(4)
    s_on = int(inj.ONSET_S * FS)
    for kind in ("damped", "dispersed"):
        w, k = inj.synth_sferic(rng, kind=kind)
        assert k == kind
        p = np.abs(w) ** 2
        assert p[: s_on - 40].sum() < 1e-6 * p.sum()                 # nothing well before the onset
        assert np.argmax(p > 0.1 * p.max()) >= s_on - 40
    # dispersed: low frequencies arrive later than high ones
    w, _ = inj.synth_sferic(np.random.default_rng(5), kind="dispersed")
    lo = np.abs(inj.band_analytic(w, lo=1800, hi=3000)) ** 2
    hi = np.abs(inj.band_analytic(w, lo=12000, hi=20000)) ** 2
    t = np.arange(len(w))
    assert (t * lo).sum() / lo.sum() > (t * hi).sum() / hi.sum() + 30


def test_july_baseline_finds_isolated_pulses_and_is_blind_under_buzz():
    """Synthetic: clean host + 20 pulses at 24 dB -> found; add a 120 Hz train of strong pulses -> the 8 ms dead time
    hides most of them (the h2h finding, in miniature)."""
    import harness as H
    import inject as inj
    host = inj.synthetic_host(6.0, seed=7)
    floor = inj.host_floor(host)
    w, _ = inj.synth_bank(20, seed=8)
    cut = 0.2 + np.arange(20) * 0.28
    y = inj.inject(host, w, cut, inj.snr_gains(w, floor, 24))
    t, pk, _ = H.run_detector("july", y)
    rec, ch, corr = H.corrected_recall(t, cut + inj.ONSET_S)
    assert corr > 0.8
    bz = np.arange(0.003, 5.9, 1 / 120.0)
    bw = [inj.synth_sferic(np.random.default_rng(i), kind="damped")[0] for i in range(len(bz))]
    yb = inj.inject(y, bw, bz, inj.snr_gains(bw, floor, 18))
    tb, _, _ = H.run_detector("july", yb)
    assert H.comb_share(tb) > 0.8
    assert H.corrected_recall(tb, cut + inj.ONSET_S)[2] < corr - 0.3


# ---------------------------------------------------------------- fixes after verification
def test_corrected_recall_empty_truth_is_quiet_nan():
    import warnings
    import harness as H
    with warnings.catch_warnings():
        warnings.simplefilter("error")                         # no 'Mean of empty slice' warning
        r = H.corrected_recall(np.array([1.0, 2.0]), np.array([]))
    assert all(np.isnan(v) for v in r)


def test_inject_clips_pulses_off_either_end_and_skips_outside():
    import inject as inj
    x = np.zeros(1000, np.complex64)
    s = np.full(100, 8 / 32768, np.complex128)
    y = inj.inject(x, [s, s, s], [-50 / FS, 960 / FS, 2.0], quantize=False)   # half before, tail after, wholly after
    assert np.allclose(y[:50] * 32768, 8) and np.all(y[50:960] == 0) and np.allclose(y[960:] * 32768, 8)


def test_real_snr_bank_rejects_cuts_peaking_after_the_onset():
    """A cut whose strongest sferic is 5 ms after the labelled onset must not enter the SNR bank (its gain would be
    set by the later pulse and the labelled one injected weaker than stated)."""
    import inject as inj
    rng = np.random.default_rng(3)
    floor = 1e-10
    on = int(inj.ONSET_S * FS)

    def pulse(at, amp):
        w, _ = inj.synth_sferic(np.random.default_rng(int(rng.integers(1 << 30))), kind="damped")
        z = np.zeros(inj.SNIP_N, np.complex128)
        z[at - on:] = amp * w[: inj.SNIP_N - (at - on)]
        return z
    good = [pulse(on, 1.0) + pulse(on + 300, 0.2) for _ in range(3)]          # onset sferic dominates
    bad = [pulse(on, 0.2) + pulse(on + int(0.005 * FS), 1.0) for _ in range(3)]  # later sferic dominates
    for b in bad:
        assert inj.snip_peak_lag_s(inj.clean_snip(b)) > 0.004
    sel, nat = inj.real_snr_bank(np.array(good + bad), floor, min_count=6)
    assert len(sel) == 3                                       # only the three good cuts are eligible
    for sc in sel:
        assert -0.0005 <= inj.snip_peak_lag_s(sc) <= 0.001


def test_buzz_suspect_blocks_flags_sparse_buzz_only_when_sustained():
    import harness as H
    per = [8] * 10 + [40] * 6 + [900] * 3
    comb = [None] * 10 + [0.93] * 6 + [0.99] * 3
    assert H.hum_blocks(per, comb) == [[160, 190]]
    assert H.buzz_suspect_blocks(per, comb) == [[100, 160]]
    # two stray comb-heavy blocks on a busy clean sky are not enough
    assert H.buzz_suspect_blocks([33, 31, 20], [0.59, 0.63, 0.2]) == []


def test_onair_bin_lookup_is_not_a_session_path(monkeypatch):
    import harness as H
    c = swio.load_corpus()
    assert c["onair_bin"].startswith("~/") and "/tmp/" not in c["onair_bin"]
    monkeypatch.setenv("SFERIC_ONAIR_BIN", "/x/y/onair")
    assert H.onair_bin() == "/x/y/onair"
    monkeypatch.delenv("SFERIC_ONAIR_BIN")
    monkeypatch.setenv("PATH", "/nonexistent")
    assert H.onair_bin() == os.path.expanduser(c["onair_bin"])
