"""Tests for bench/glm_truth.py on SYNTHETIC lightning only (no GLM download, no capture, no station file)."""
import json

import numpy as np
import pytest

import glm_truth as gt                     # bench/glm_truth.py (bench/ is on the path via conftest)


# ---------------------------------------------------------------- helpers

def test_nearest_and_min_in_window():
    a = np.array([0.0, 1.0, 2.0, 5.0])
    i, d = gt.nearest(a, np.array([-1.0, 1.4, 1.6, 4.0, 9.0]))
    assert i.tolist() == [0, 1, 2, 3, 3]
    assert np.allclose(d, [1.0, 0.4, 0.4, 1.0, 4.0])
    i, d = gt.nearest(np.zeros(0), np.array([1.0]))
    assert i.tolist() == [-1] and np.isinf(d[0])

    rng = np.random.default_rng(1)
    t = np.sort(rng.uniform(0, 10, 500))
    v = rng.uniform(0, 100, 500)
    q = rng.uniform(0, 10, 300)
    best, arg = gt.min_in_window(t, v, q, 0.05, chunk=37)
    for k in range(q.size):
        m = np.abs(t - q[k]) <= 0.05
        if m.any():
            assert best[k] == v[m].min() and v[arg[k]] == best[k]
        else:
            assert np.isinf(best[k]) and arg[k] == -1


def test_lag_hist_and_peak_fit_recover_a_lag():
    rng = np.random.default_rng(2)
    ref = np.sort(rng.uniform(0, 600, 3000))
    true = rng.choice(ref, 800, replace=False) + 0.0253 + rng.normal(0, 0.0007, 800)
    det = np.sort(np.concatenate([true, rng.uniform(0, 600, 4000)]))
    h = gt.lag_hist(det, ref, chunk=500)
    assert h.size == 4001
    pf = gt.peak_fit(h)
    assert abs(pf["lag_s"] - 0.0253) < 0.0005
    assert 0.0005 <= pf["w_s"] <= 0.004
    assert pf["z"] > 20
    assert 600 < pf["excess_in_w"] < 900


def test_intervals():
    iv = gt.merge_intervals(np.array([[5, 7], [0, 2], [2, 3], [6, 9]]))
    assert iv.tolist() == [[0, 3], [5, 9]]
    assert gt.in_intervals(np.array([-1, 0, 2.5, 3, 4, 5, 8.9, 9]), iv).tolist() == \
        [False, True, True, False, False, True, True, False]
    assert not gt.in_intervals(np.array([1.0]), np.zeros((0, 2))).any()


def test_light_time_geometry():
    sub = gt.light_time_s(np.array([0.0]), np.array([-75.2]), np.array([-75.2]))[0]
    assert abs(sub - (gt.R_GEO_KM - gt.R_EARTH_EQ_KM) / gt.C_KM_S) < 1e-9          # ~119.3 ms
    off = gt.light_time_s(np.array([-20.0]), np.array([-45.0]), np.array([-75.2]))[0]
    assert sub < off < 0.140


def test_small_helpers():
    assert gt.corrected(60, 20, 100) == pytest.approx(0.5)
    assert np.isnan(gt.corrected(1, 1, 0))
    x = np.array([0.0, 1, 2, 3])
    assert gt.crossing(x, np.array([0.9, 0.7, 0.3, 0.1]), 0.5) == pytest.approx(1.5)
    assert gt.crossing(x, np.array([0.4, 0.3, 0.2, 0.1]), 0.5) is None
    s = gt.make_shifts(40)
    assert np.all(np.abs(s) >= 5) and np.all(np.abs(s) <= 58) and (s < 0).sum() == 20


# ---------------------------------------------------------------- end to end on a synthetic sky

LAG, V = 0.031, 0.99 * gt.C_KM_S


def _write_capture(gdir, edir, cid, seed, hum=False, satref=False):
    """One synthetic 600 s capture: GLM flashes 0-8000 km (1-3 groups each, 0-300 ms apart) and detector events.

    v2/v2_detect/onair_bp hear EVERY group of flashes within 2000 km and half of those at 2000-4000 km, at
    t_glm + d / v + LAG (+-0.3 ms), plus 40/min of noise. july hears only flashes within 1000 km, plus 2000/min noise.
    Lat/lon are random numbers (only reduced to a light time inside the module). ``satref``: the GLM group times are
    written as seen AT THE SATELLITE (true time + flash-to-GEO light time), for the light-time test."""
    rng = np.random.default_rng(seed)
    t0, secs = 1.8e9 + seed * 10000.0, 600.0
    nf = 3000
    ft = np.sort(rng.uniform(t0 - 60, t0 + secs + 60, nf))
    fd = rng.uniform(0, 8000, nf)
    ng = rng.integers(1, 4, nf)
    gpar = np.repeat(np.arange(nf), ng)
    gt_ = ft[gpar] + np.concatenate([np.r_[0, np.sort(rng.uniform(0, 0.3, k - 1))] for k in ng])
    o = np.argsort(gt_)
    gt_, gpar = gt_[o], gpar[o]
    gd = fd[gpar]
    flast = np.array([gt_[gpar == k].max() for k in range(nf)])
    uid = 19 * 2 ** 48 + np.arange(nf)
    glat, glon = rng.uniform(-30, 30, gt_.size), rng.uniform(-100, -50, gt_.size)
    heard = (gd < 2000) | ((gd < 4000) & (rng.random(gd.size) < 0.5))
    arr = gt_ + gd / V + LAG - t0 + rng.normal(0, 0.0003, gd.size)
    sig = arr[heard & (arr > 0) & (arr < secs)]
    noise = rng.uniform(0, secs, 400)
    v2t = np.sort(np.concatenate([sig, noise]))
    jh = gd < 1000
    jt = np.sort(np.concatenate([arr[jh & (arr > 0) & (arr < secs)], rng.uniform(0, secs, 20000)]))
    gt_glm = gt_ + (gt.light_time_s(glat, glon, np.full(gt_.size, -75.2)) if satref else 0.0)
    np.savez(gdir / f"glm_{cid}.npz", flash_t=ft, flash_t_last=flast, flash_dist_km=fd,
             flash_energy_J=rng.lognormal(-31, 1, nf), flash_area_m2=rng.uniform(1e7, 1e9, nf), flash_uid=uid,
             group_t=gt_glm, group_dist_km=gd, group_parent_uid=uid[gpar], group_sat=np.full(gd.size, 19, np.int8),
             group_lat=glat, group_lon=glon,
             t0=t0, missing_g19=np.zeros(0), missing_g18=np.zeros(0))
    hum_iv = np.array([[0.0, secs]]) if hum else np.zeros((0, 2))
    np.savez(edir / f"{cid}.npz", t0=t0, secs=secs, july_t=jt, july_pk=np.full(jt.size, 20.0),
             onair_bp_t=v2t, onair_bp_pk=np.full(v2t.size, 20.0), onair_bp_ok=True,
             v2_detect_t=v2t, v2_detect_pk=np.full(v2t.size, 20.0), v2_detect_conf=np.ones(v2t.size, bool),
             v2_t=v2t, v2_pk=np.full(v2t.size, 20.0), v2_conf=np.ones(v2t.size, bool), hum_iv=hum_iv,
             hum_frac=1.0 if hum else 0.0, gated_frac=0.0)
    return sig.size / v2t.size


@pytest.fixture
def sky(tmp_path, monkeypatch):
    gdir, edir = tmp_path / "glm", tmp_path / "ev"
    gdir.mkdir()
    edir.mkdir()
    monkeypatch.setattr(gt, "GLM_DIR", gdir)
    monkeypatch.setattr(gt, "EVENTS_DIR", edir)
    share = [_write_capture(gdir, edir, f"2026110{k}_0000Z", k, hum=(k == 2)) for k in range(1, 4)]
    return float(np.mean(share))


def test_end_to_end_synthetic(sky):
    ids = gt.capture_ids()
    assert len(ids) == 3
    clock = gt.clock_stage(ids, V, workers=1)
    assert abs(clock["pooled_group"]["lag_s"] - LAG) < 0.0005
    model = gt.lag_model(clock)
    assert model["mode"] == "crossfit"
    assert abs(model["pooled_s"] - LAG) < 0.0005
    for c in ids:                                   # cross-fitted lags land on the true lag too
        assert all(abs(x - LAG) < 0.001 for x in gt.lag_for(model, c))
    wg = max(clock["pooled_group"]["w_s"], 0.002)
    assert wg <= 0.004
    shifts = gt.make_shifts(10)
    ecut = np.array(clock["energy_terciles_J"])
    rows = [gt._match_one((c, gt.lag_for(model, c), model["pooled_s"], V, wg, 0.4, shifts, ecut)) for c in ids]
    P = gt.Pool(rows, len(shifts))
    ALL = np.arange(3)
    near = gt._cat_mask(dbins={0, 1, 2})
    far = gt._cat_mask(dbins={5})
    _, _, _, r_near = P.recall("v2", ALL, "g", near)
    _, _, _, r_mid = P.recall("v2", ALL, "g", gt._cat_mask(dbins={3, 4}))
    _, _, _, r_far = P.recall("v2", ALL, "g", far)
    assert r_near > 0.97
    assert 0.40 < r_mid < 0.75          # half of the groups heard: P(any of 1-3 groups) ~ 0.5-0.875
    assert abs(r_far) < 0.05
    _, _, _, rj, _ = P.precision("july", ALL, "g", dmax=1000)
    _, _, _, q, cpm = P.precision("v2", ALL, "g", dmax=4000)
    assert abs(q - sky) < 0.05            # the true share of v2 events that came from GLM groups
    assert rj < 0.1                       # july is mostly noise
    res = gt.summarize(P, clock, model, wg, 0.4, B=20)
    js = json.dumps(gt.jsonable(res))
    assert "lat" not in js and "lon" not in js          # aggregate output: nothing positional
    assert "mean_group_km" not in js
    assert abs(res["light_time"]["slope_within_capture_s_per_s"]) < 0.3     # GLM times at the flash
    assert set(res["detectors"]["v2"]["recall_by_distance"]["group"]["0-500"]) >= {"ci95_block8", "ci95_day",
                                                                                   "ci95_capture"}
    assert set(res["detectors"]["v2"]["by_day"]) == {"20261101", "20261102", "20261103"}


def test_light_time_satellite_referenced(tmp_path, monkeypatch):
    gdir, edir = tmp_path / "glm", tmp_path / "ev"
    gdir.mkdir()
    edir.mkdir()
    monkeypatch.setattr(gt, "GLM_DIR", gdir)
    monkeypatch.setattr(gt, "EVENTS_DIR", edir)
    for k in range(1, 4):
        _write_capture(gdir, edir, f"2026110{k}_0000Z", k, satref=True)
    ids = gt.capture_ids()
    clock = gt.clock_stage(ids, V, workers=1)
    model = gt.lag_model(clock, mode="pooled")
    shifts = gt.make_shifts(4)
    rows = [gt._match_one((c, gt.lag_for(model, c), model["pooled_s"], V, 0.008, 0.03, shifts,
                           np.array(clock["energy_terciles_J"]))) for c in ids]
    lt = gt.light_time_test(gt.Pool(rows, len(shifts)), 50)
    assert abs(lt["slope_within_capture_s_per_s"] + 1) < 0.3          # times at the satellite -> slope -1


def test_resample_and_adjust():
    rng = np.random.default_rng(0)
    i = gt.resample_idx(rng, "block", 20, L=8)
    assert i.size == 20 and np.all(np.diff(i[:8]) == 1)
    days = np.array(["a"] * 5 + ["b"] * 3)
    j = gt.resample_idx(rng, "day", 8, days)
    assert set(np.unique(days[j])) <= {"a", "b"} and j.size in (6, 8, 10)
    td = np.array([1.0, 25.0, 41.0])               # blocks 0 (even), 1 (odd), 2 (even)
    adj = gt.adjust_events(td, (0.030, 0.020), 0.025)
    assert np.allclose(adj, [0.995, 25.005, 40.995])
    mid = (np.arange(240) + 0.5) * 0.0005 - 0.06
    h = 10 + 200 * np.exp(-0.5 * ((mid - 0.004) / 0.0015) ** 2)
    pk, w = gt.centroid_peak(h, mid)
    assert abs(pk - 0.004) < 0.0003 and w > 100
