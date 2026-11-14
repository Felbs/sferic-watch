"""Fast synthetic tests for sferic_watch.tweek_audit (plan step 7). No captures are read."""
import json
import math

import numpy as np
import pytest

from sferic_watch import mainsgate as mg
from sferic_watch import tweek_audit as ta

FS = ta.FS


synth_tweek = ta.synth_tweek_iq


def noise(n, rms, seed=0):
    r = np.random.default_rng(seed)
    return ((r.standard_normal(n) + 1j * r.standard_normal(n)) * rms / math.sqrt(2)).astype(np.complex64)


def test_group_delay_and_height():
    assert math.isnan(ta.group_delay(1.0, 0, 1, 1.8))
    assert ta.group_delay(3.6, 1.0, 2.0, 1.8) == pytest.approx(1.0 + 2.0 * 3.6 / math.sqrt(3.6 ** 2 - 1.8 ** 2))
    assert float(ta.h_km(1.831)) == pytest.approx(81.87, abs=0.01)     # the published July number
    assert ta.ridge_step_khz() == pytest.approx(3 * 62.5 / 2048, rel=1e-9)   # 91.55 Hz grid


def test_h_text_always_has_ci():
    s = ta.h_text(1.6, 1.55, 1.65)
    assert "95 % CI" in s and s.startswith("h 93.7 km")
    lo, hi = s.split("CI ")[1].rstrip(")").split("-")
    assert float(lo) < 93.7 < float(hi)


def test_synthetic_tweek_2000km_recovered_within_100hz():
    """Ideal tweek fc 1.8 kHz, d = 2000 km: the July method lands within 0.1 kHz (measured 1.725, -75 Hz) and the
    fit is clear of every bound. The CI includes the 91.55 Hz grid quantisation."""
    n = int(0.2 * FS)
    x = synth_tweek(n, t_on=0.050, fc_khz=1.8, d_ms=6.67) + noise(n, 0.003)
    r = ta.fit_at(x, [0.050 + 6.67e-3], n_boot=60, seed=1)[0]
    assert r["fit"] and r["textbook"] and r["bound_free"]
    assert r["fc"] == pytest.approx(1.8, abs=0.10)
    lo, hi = r["fc_ci"]
    assert lo < r["fc_profile"] < hi and lo < r["fc"] < hi
    assert hi <= r["rmin"] + 1e-12
    assert not r["fit_not_global"]
    assert hi - lo >= 0.95 * ta.ridge_step_khz() - 1e-9 or r["ci_hi_at_bound"]


def test_july_method_bias_grows_with_range():
    """Documents a measured flaw: at d = 4000 km the +16 ms ridge window cuts the near-cutoff tail and the July fit
    puts fc 0.2-0.3 kHz too HIGH (1.6 -> 1.83, 1.8 -> 2.03) while every flag says the fit is clean (bound-free,
    textbook). The fitted d stays near 2300 km although the truth is 4000 km."""
    for fc in (1.6, 1.8):
        row = ta.synthetic_bias(fcs=(fc,), d_kms=(4000.0,), n_boot=40)[0]
        assert row["textbook"] and row["bound_free"]
        assert 0.15 < row["fc_err_khz"] < 0.35
        assert row["d_fit_km"] < 3000


def test_bound_flags_scalar_and_array():
    f = ta.bound_flags([1.831, 1.6, 1.405, 1.7], [1.8311, 1.74, 1.56, 1.74], [0.5, 0.5, 0.5, 0.004])
    assert list(f["at_fc_hi"]) == [True, False, False, False]
    assert list(f["at_fc_lo"]) == [False, False, True, False]
    assert list(f["at_d_floor"]) == [False, False, False, True]
    assert list(f["bound_free"]) == [False, True, False, False]


def test_noise_windows_rarely_textbook():
    n = int(2.0 * FS)
    x = noise(n, 0.01, seed=3)
    t = np.random.default_rng(4).uniform(0.05, 1.9, 40)
    res = ta.fit_at(x, t)
    assert sum(r["textbook"] for r in res) <= 2


def test_fit_at_skips_windows_off_the_end():
    x = noise(int(0.1 * FS), 0.01)
    r = ta.fit_at(x, [0.001, 0.09])
    assert all(v.get("skipped") for v in r)


def test_verdict_rules():
    v = ta.verdict(1037, 5, 300, 1, 3)                  # the plan's hum-free example: 0.48 % vs 0.33 %
    assert v["status"] == "insufficient" and len(v["reasons"]) == 2
    v = ta.verdict(500, 40, 300, 2, 12)                 # 8 % vs 0.67 %: 12x, 12 bound-free
    assert v["status"] == "above null" and v["ratio"] == pytest.approx(12.0)
    assert v["p_poisson"] < 1e-6
    v = ta.verdict(500, 40, 300, 0, 12)                 # empty null is taken as 1/300, not 0
    assert v["null_rate_rule"] == pytest.approx(1 / 300) and v["status"] == "above null"
    v = ta.verdict(10, 2, 300, 0, 12)                   # 20 % yield but only 10 candidates: still passes ratio...
    assert v["status"] == "above null"
    assert ta.verdict(0, 0, 0, 0, 0)["status"] == "insufficient"
    assert "no admissible null windows" in ta.verdict(5, 1, 0, 0, 20)["reasons"]


def test_median_ci_and_small_n_envelope():
    m, lo, hi = ta.median_ci(np.r_[np.full(20, 1.6), np.full(20, 1.7)])
    assert lo <= m <= hi
    assert all(math.isnan(v) for v in ta.median_ci([]))
    fits = [dict(textbook=True, bound_free=True, fit=True, fc=1.6, fc_ci=[1.5, 1.7])]
    s = ta.summarise(fits)
    assert s["fc_median_ci"] == [1.5, 1.7] and "envelope" in s["ci_method"]


def test_admissible_hum_and_train():
    flags = [dict(t0_s=0.0, dur_s=10.0, hum=False), dict(t0_s=10.0, dur_s=10.0, hum=True)]
    tr = mg.Train(f=120.0, t_ref=0.0, c=0.001, t_hi=10.0)
    t = np.array([0.001, 0.0015, 0.001 + 5 / 120.0, 0.005, 12.0])
    ok = ta.admissible(t, flags, [tr])
    assert list(ok) == [False, True, False, True, False]
    assert list(ta.hum_mask(t, flags)) == [False, False, False, False, True]


def test_candidates_need_conf_and_no_hum():
    ev = {"conf": np.array([True, True, False, True]), "hum": np.array([False, True, False, False])}
    assert list(ta.candidates(ev)) == [0, 3]


def test_july_reaudit_on_synthetic_json(tmp_path):
    tw = [dict(tag="a", t=1.0, fc=1.8311, fc_err=1e-5, D=0.0035, rmin=1.8311),     # pinned at rmin, tiny D
          dict(tag="a", t=2.0, fc=1.6, fc_err=1e-3, D=0.5, rmin=1.74),             # bound-free
          dict(tag="a", t=3.0, fc=1.4, fc_err=1e-3, D=0.5, rmin=1.56),             # at the 1.4 kHz floor
          dict(tag="a", t=4.0, fc=1.73, fc_err=1e-3, D=0.5, rmin=1.74)]            # 10 Hz from rmin
    p = tmp_path / "f.json"
    p.write_text(json.dumps(dict(fc_median_khz=1.7, h_median_km=88.0, tweeks=tw)))
    a = ta.july_reaudit(str(p))
    assert a["n"] == 4 and a["d_below_0p01"] == 1
    assert a["pinned"] == {"0.5Hz": 1, "5Hz": 1, "10Hz": 1, "20Hz": 2}
    assert a["bound_free_plan20Hz"]["n"] == 1 and a["bound_free_plan20Hz"]["fc_median"] == pytest.approx(1.6)
    assert a["bound_free_rule10Hz"]["n"] == 2
    assert "95 % CI" in a["all"]["h"]


def test_audit_capture_end_to_end_synthetic():
    """30 s of white noise with a tweek every 1-2 s: the gated audit finds them, and its null does not."""
    n = int(30 * FS)
    x = noise(n, 0.002, seed=5)
    tw = synth_tweek(int(0.2 * FS), t_on=0.02, fc_khz=1.75, d_ms=8.0, amp=0.2)
    # jittered spacing: a fixed 1.5 s pitch is exactly 180 mains periods and the mains gate (rightly) eats it
    on = 2.0 + np.cumsum(np.random.default_rng(9).uniform(1.0, 2.0, 18))
    on = on[on < 28.5]
    for t in on:
        s = int(t * FS)
        x[s:s + len(tw)] += tw
    r = ta.audit_capture(x, n_null=60, n_boot=20, seed=2)
    assert r["n_cand_total"] >= 0.8 * len(on)
    assert r["cand"]["n_textbook"] >= 0.6 * len(on)
    assert r["null"]["n_admissible"] == 60
    assert r["verdict"]["yield_rate"] > 3 * r["verdict"]["null_rate_rule"]
    assert r["cand"]["fc_median_bound_free"] == pytest.approx(1.75, abs=0.08)
    assert r["verdict"]["n_cand"] == r["n_cand_fitted"] == r["n_cand_total"] - r["cand"]["n_skipped"]
    assert r["phase"] == "july" and "stability" in r["mirror"]
    assert mg.gate.__name__ == "gate"            # the spy wrapper was removed


# ------------------------------------------------------------------ mixer phase (verifier finding 2026-11-14)

def _band_rms(xr, lo=1.5e3, hi=7e3):
    X = np.fft.rfft(xr)
    f = np.fft.rfftfreq(len(xr), 1 / FS)
    return float(np.sqrt((np.abs(X[(f > lo) & (f < hi)]) ** 2).sum()))


def test_window_real_phase_modes():
    """July's mixer restarts at every window; the absolute one counts from sample n0."""
    x = noise(1000, 0.1, seed=1)
    np.testing.assert_allclose(ta.window_real(x, 777, phase="july"), ta.window_real(x, 0, phase="absolute"))
    assert not np.allclose(ta.window_real(x, 777, phase="july"), ta.window_real(x, 777, phase="absolute"))
    with pytest.raises(ValueError):
        ta.window_real(x, 0, phase="local")


def test_real_rf_tweek_is_scaled_by_cos_phase():
    """A real-RF tweek (with its mirror image) comes out of .real scaled by cos(LO phase); an analytic one does not.
    86.4 deg per sample of window start is what made July's window-local mixer a random gain."""
    n = int(0.2 * FS)
    s = 3000
    ref = _band_rms(ta.window_real(ta.synth_tweek_iq(n, 0.05, real_rf=True, lo_phase_deg=0)[s:s + 2812], s,
                                   phase="absolute"))
    for ph in (30.0, 60.0, 90.0):
        x = ta.synth_tweek_iq(n, 0.05, real_rf=True, lo_phase_deg=ph)[s:s + 2812]
        assert _band_rms(ta.window_real(x, s, phase="absolute")) / ref == pytest.approx(abs(math.cos(math.radians(ph))),
                                                                                      abs=0.01)
    # July's mixer at window start s = 1 adds 86.4 deg: cos(86.4) = 0.063
    x = ta.synth_tweek_iq(n, 0.05, real_rf=True, lo_phase_deg=0)
    g = _band_rms(ta.window_real(x[s - 1:s - 1 + 2812], s - 1, phase="absolute") * 0 +
                  ta.window_real(x[s - 1:s - 1 + 2812], 1, phase="absolute"))
    g0 = _band_rms(ta.window_real(x[s - 1:s - 1 + 2812], s - 1, phase="absolute"))
    assert g / g0 == pytest.approx(abs(math.cos(math.radians(86.4 * (1 - (s - 1))))), abs=0.01)
    # analytic (no mirror): the phase does not change the band energy
    xa = ta.synth_tweek_iq(n, 0.05)[s:s + 2812]
    assert _band_rms(ta.window_real(xa, s, phase="absolute")) == pytest.approx(
        _band_rms(ta.window_real(xa, s, phase="july")), rel=1e-3)


def test_mirror_phase_recovers_lo_phase():
    n = int(0.5 * FS)
    for ph in (0.0, 30.0, 60.0, 120.0):
        x = ta.synth_tweek_iq(n, 0.05, real_rf=True, lo_phase_deg=ph) + noise(n, 0.003, seed=2)
        m = ta.mirror_phase(x, top_frac=0.1)
        d = (m["phi_deg"] - ph + 90.0) % 180.0 - 90.0
        assert abs(d) < 2.0 and m["stability"] > 0.9
    m = ta.mirror_phase(noise(n, 0.01, seed=4))                  # independent images: no coherence
    assert m["coherence"] < 0.3


# ------------------------------------------------------------------ profile refit = curve_fit's problem

def test_box_ls_matches_brute_force_with_active_t0_bound():
    rng = np.random.default_rng(5)
    g = rng.uniform(1, 3, (3, 12))
    rt = 20.0 + 0.5 * g[0] + rng.normal(0, 0.1, 12)               # free t0 ~ 20: outside [0, 15]
    w = rng.uniform(0.1, 1, 12)
    t0, D, sse = ta._box_ls(g, rt, w, 0.0, 15.0)
    T, DD = np.meshgrid(np.linspace(0, 15, 601), np.linspace(ta.D_LO, 20, 2001))
    for r in range(3):
        S = (((rt[None, None, :] - T[..., None] - DD[..., None] * g[r]) ** 2) * w).sum(-1)
        assert sse[r] <= S.min() + 1e-9
        assert 0 <= t0[r] <= 15 and ta.D_LO <= D[r]
    assert t0[0] == pytest.approx(15.0)


def test_profile_fit_agrees_with_curve_fit_on_synthetic_tweeks():
    """The bootstrap refit solves the same problem as ``fit``: weights rw, all bounds. Measured: 0.0-0.5 Hz apart on
    these tweeks (the old 1/rw-weighted, t0-unbounded version was up to 470 Hz apart)."""
    n = int(0.2 * FS)
    for fc, dk in ((1.8, 4000.0), (2.0, 1000.0), (1.6, 2000.0)):
        d_ms = dk / ta.C_KM_PER_MS
        x = synth_tweek(n, 0.05, fc, d_ms) + noise(n, 0.003, seed=6)
        s = int(round((0.05 + d_ms * 1e-3) * FS)) - int(round(ta.PRE_S * FS))
        ft = ta.fit(*ta.ridge(ta.window_real(x[s:s + int(round(ta.WIN_S * FS))], s)))
        pf = ta.profile_fit(ft["rf"], ft["rt"], ft["rw"])
        assert pf["fc"] == pytest.approx(ft["fc"], abs=0.0015)
        # 1 Hz grid: the profile SSE may exceed curve_fit's continuous optimum by a hair, never by more
        assert pf["sse"] <= ta.weighted_sse(ft["rf"], ft["rt"], ft["rw"], ft["t0"], ft["D"], ft["fc"]) * (1 + 1e-4)
        lo_t0, hi_t0 = ft["rt"].min() - ta.T0_PAD, ft["rt"].max() + ta.T0_PAD
        assert lo_t0 <= pf["t0"] <= hi_t0


def test_bootstrap_ci_is_about_profile_fc_and_clipped_at_bound():
    rf = np.r_[1.65, np.linspace(1.75, 6.4, 20)]
    rt = ta.group_delay(rf, 2.0, 0.8, 1.6) + np.random.default_rng(8).normal(0, 0.15, rf.size)
    ft = dict(fc=1.6, t0=2.0, D=0.8, rf=rf, rt=rt, rw=np.ones(rf.size))
    b = ta.bootstrap_fc(ft, 200, np.random.default_rng(0))
    lo, hi = b["fc_ci"]
    assert lo <= b["fc_profile"] <= hi
    assert hi <= rf.min() + 1e-12                                # never above the full ridge's fc bound
    far = dict(ft, fc=1.45, t0=0.0, D=5.0)                       # a "curve_fit" result far from the optimum
    assert ta.bootstrap_fc(far, 30, np.random.default_rng(0))["fit_not_global"]


def test_skipped_windows_are_counted_and_excluded():
    s = ta.summarise([{"fit": False, "textbook": False, "skipped": True}, {"fit": False, "textbook": False}])
    assert s["n_skipped"] == 1 and s["n"] == 2
