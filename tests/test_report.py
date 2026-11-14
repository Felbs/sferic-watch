"""Fast tests for sferic_watch.report, sferic_watch.cli and tools/storm_panel.py on SYNTHETIC captures only."""
import importlib.util
import json
import os

import numpy as np
import pytest

from sferic_watch import cli
from sferic_watch import mainsgate as mg
from sferic_watch import io as sio
from sferic_watch import report as rp

import inject as inj                      # bench/inject.py: synthetic host and pulses (on the path via conftest)

FS = 62500.0
SECS = 20.0
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def damped(f0=6000.0, tau=0.15e-3, n=200):
    """A short damped 6 kHz pulse as complex baseband (centre 15 kHz)."""
    t = np.arange(n) / FS
    return np.exp(-t / tau) * np.exp(2j * np.pi * (f0 - inj.FC) * t)


def broadband(n=120):
    """A sferic-like pulse that also reaches 16-30 kHz: a short damped click (2 kHz and 22 kHz parts)."""
    t = np.arange(n) / FS
    env = np.exp(-t / 0.1e-3)
    return env * (np.exp(2j * np.pi * (6000.0 - inj.FC) * t) + np.exp(2j * np.pi * (22000.0 - inj.FC) * t))


def scene(buzz_from=None, seed=3):
    """SECS of synthetic host with Poisson 2/s broadband sferics at ~24 dB, and (optionally) dense 120 Hz buzz
    (60 % of the pulses, 40 us jitter, ~20 dB, 6 kHz only) from ``buzz_from`` s on."""
    rng = np.random.default_rng(seed)
    x = inj.synthetic_host(SECS, seed=seed)
    floor = inj.host_floor(x)
    sf = broadband()
    ts = inj.poisson_starts(2.0, SECS, seed=seed + 1, guard=0.5)
    gs = np.full(ts.size, np.sqrt(floor / inj.snip_peak_power(sf)) * 10 ** (24 / 20))
    snips, times, gains = [sf] * ts.size, list(ts), list(gs)
    if buzz_from is not None:
        bz = damped()
        g0 = np.sqrt(floor / inj.snip_peak_power(bz))
        k = np.arange(int(buzz_from * 120) + 1, int(SECS * 120) - 2)
        k = np.sort(rng.choice(k, int(0.6 * k.size), replace=False))
        tb = 2.1e-3 + k / 120.0 + rng.normal(0, 40e-6, k.size)
        snips += [bz] * k.size
        times += list(tb)
        gains += list(g0 * 10 ** ((20 + rng.uniform(-2, 2, k.size)) / 20))
    return inj.inject(x, snips, np.array(times), np.array(gains)), ts


def write_capture(tmp_path, x, name="vlf_0430Z", folder="20261105", extra=None):
    d = tmp_path / folder
    d.mkdir(parents=True, exist_ok=True)
    side = {"freq_hz": 15000.0, "fs_hz": FS, "secs": len(x) / FS, "utc": "2026-11-05T04:30:00+00:00",
            "antenna": "Antenna C", "agc": False, "IFGR": 40.0, "RFGR": 0.0}
    side.update(extra or {})
    p = d / f"{name}.cs16"
    sio.write_cs16(str(p), x, side)
    return str(p)


@pytest.fixture(scope="module")
def clean_cap(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("clean")
    x, ts = scene()
    return write_capture(tmp, x, extra={"latitude": 12.5, "lon": 3.0, "grid": "XX00"}), ts


@pytest.fixture(scope="module")
def clean_run(clean_cap):
    path, _ = clean_cap
    return rp.analyze_file(path)


# ----------------------------------------------------------------------------- measures

def test_comb_share_chance_and_buzz():
    """Poisson events read near the 0.144 chance level; a pure 120 Hz train reads 1; fewer than 10 events: None."""
    rng = np.random.default_rng(0)
    t = np.cumsum(rng.exponential(1 / 50.0, 20000))
    assert rp.comb_share(t) == pytest.approx(rp.COMB_CHANCE, abs=0.01)
    assert rp.comb_share(np.arange(200) / 120.0) == 1.0
    assert rp.comb_share(np.arange(9) / 120.0) is None
    assert rp.COMB_CHANCE == pytest.approx(0.144)


def test_coincidence_chance_formula_matches_random_trains():
    """Two independent Poisson trains: the measured coincidence rate matches n_lo n_hi 2w / T^2 within 10 %."""
    rng = np.random.default_rng(1)
    T = 2000.0
    lo = np.sort(rng.uniform(0, T, 4000))
    hi = np.sort(rng.uniform(0, T, 20000))
    meas = rp.coincident(lo, hi).sum() / T
    assert meas == pytest.approx(rp.chance_rate(lo.size, hi.size, T), rel=0.1)
    assert rp.coincident(np.array([1.0, 2.0]), np.array([1.0004, 5.0])).tolist() == [True, False]
    assert rp.coincident(np.array([1.0]), np.zeros(0)).tolist() == [False]


def test_reject_union_single_train_and_open_span():
    """A 120 Hz train over 30-90 s: +-0.2 ms windows cover 2 x 0.2 ms x 120 = 4.8 % of each second it spans; an
    open-ended span counts to the capture end."""
    tr = [mg.Train(f=120.0, t_ref=0.0, c=1e-3, t_lo=30.0, t_hi=90.0)]
    lo, hi = rp.reject_intervals(tr, 600.0)
    assert rp.covered_seconds(lo, hi, 0, 60) == pytest.approx(30 * 0.048, rel=1e-3)
    assert rp.covered_seconds(lo, hi, 60, 120) == pytest.approx(30 * 0.048, rel=1e-3)
    lo, hi = rp.reject_intervals([mg.Train(f=120.0, t_ref=0.0, c=1e-3, t_lo=0.0, t_hi=float("inf"))], 600.0)
    assert rp.covered_seconds(lo, hi, 540, 600) == pytest.approx(60 * 0.048, rel=1e-3)


def test_reject_union_counts_overlap_once():
    """Overlapping trains (the verifier's case): five identical trains blank 4.8 %, not 24 %; two trains 0.1 ms
    apart blank 0.5 ms per pulse (0.06), not 0.8 ms (0.096); the old sum is kept only as gated_frac_sum."""
    five = [mg.Train(f=120.0, t_ref=0.0, c=2e-3, t_lo=0.0, t_hi=60.0) for _ in range(5)]
    lo, hi = rp.reject_intervals(five, 600.0)
    assert rp.covered_seconds(lo, hi, 0, 60) / 60 == pytest.approx(0.048, rel=1e-3)
    assert rp.gated_seconds_sum(five, 0, 60, 600) / 60 == pytest.approx(0.24)
    assert rp.trust(0.1, rp.covered_seconds(lo, hi, 0, 60) / 60) == []
    two = [mg.Train(f=120.0, t_ref=0.0, c=c, t_lo=0.0, t_hi=60.0) for c in (2e-3, 2.1e-3)]
    lo, hi = rp.reject_intervals(two, 600.0)
    assert rp.covered_seconds(lo, hi, 0, 60) / 60 == pytest.approx(0.5e-3 * 120, rel=1e-3)


def test_reject_union_matches_judge_with_track():
    """The union is the set mainsgate.judge rejects a weak impulse in: random probe times inside the intervals are
    rejected, those outside are not (a tracked phase that wanders 0.3 ms, two overlapping trains, one 1 s burst)."""
    tt = np.arange(0.5, 60, 1.0)
    trs = [mg.Train(f=120.004, t_ref=0.0, c=3e-3, track_t=tt, track_off=3e-4 * np.sin(tt / 7), median_db=20,
                    t_lo=0.0, t_hi=60.0),
           mg.Train(f=120.0, t_ref=0.0, c=3.15e-3, median_db=20, t_lo=10.0, t_hi=40.0),
           mg.Train(f=120.0, t_ref=0.0, c=6e-3, median_db=20, t_lo=20.0, t_hi=21.0, kind="burst")]
    lo, hi = rp.reject_intervals(trs, 60.0)
    t = np.random.default_rng(5).uniform(0, 60, 200000)
    rej, _ = mg.judge(t, np.zeros(t.size), trs)
    k = np.searchsorted(lo, t, side="right") - 1
    inside = (k >= 0) & (t < hi[np.clip(k, 0, None)])
    assert np.mean(inside != rej) < 1e-3          # disagreements only within ~us of a window edge
    assert rp.covered_seconds(lo, hi, 0, 60) / 60 == pytest.approx(rej.mean(), abs=0.002)


def test_keep_trains_restores_gate():
    store = []
    orig = mg.gate
    with rp.keep_trains(store):
        assert mg.gate is not orig
    assert mg.gate is orig


def test_event_blocks_outside_flag():
    """An event after the last flag block gets block -1 and R NaN, not the last block's R."""
    flags = [{"t0_s": 0.0, "dur_s": 10.0, "R": 0.1}, {"t0_s": 10.0, "dur_s": 10.0, "R": 0.9}]
    blk, R = rp.event_blocks(np.array([0.5, 10.0, 19.9, 20.5]), flags)
    assert blk.tolist() == [0, 1, 1, -1]
    assert R[:3].tolist() == [0.1, 0.9, 0.9] and np.isnan(R[3])


def test_trust_rule():
    assert rp.trust(0.20, 0.10) == []
    assert rp.trust(0.26, 0.10) == ["comb_share"]
    assert rp.trust(None, 0.21) == ["gated_frac"]
    assert rp.trust(0.9, 0.5) == ["comb_share", "gated_frac"]


def test_minute_spans_and_capture_id():
    assert rp.minute_spans(600.0)[-1] == (540.0, 600.0) and len(rp.minute_spans(600.0)) == 10
    assert rp.minute_spans(125.0) == [(0.0, 60.0), (60.0, 120.0)]
    assert rp.minute_spans(20.0) == [(0.0, 20.0)]
    assert rp.capture_id("/x/20261105/vlf_0430Z.cs16") == "vlf_20261105_0430Z"
    assert rp.capture_id("/x/storm_live/vlf_20261114_0212Z.cs16") == "vlf_20261114_0212Z"


def test_tail_score():
    """A pulse with a 4 ms ringing tail scores higher than a bare 0.4 ms pulse; off the end it is NaN."""
    p = np.ones(int(1 * FS), np.float32)
    k1, k2 = int(0.2 * FS), int(0.5 * FS)
    p[k1:k1 + 25] = 100.0
    p[k2:k2 + 25] = 100.0
    p[k2 + 25:k2 + 25 + int(4e-3 * FS)] = 10.0
    tl = rp.tail_db(p, np.array([k1 / FS, k2 / FS, 0.999]), np.zeros(3))
    assert tl[0] == pytest.approx(0.0, abs=0.01) and tl[1] > 8.0 and np.isnan(tl[2])


# ----------------------------------------------------------------------------- whole synthetic capture

def test_session_validates_and_is_private(clean_run):
    """The session validates against schema.json; no sidecar position key gets through; the strict privacy grep
    finds nothing in the session, the catalogue or the schema itself."""
    doc, csv_text = clean_run
    schema = rp.load_schema()
    assert rp.validate(doc, schema) == []
    text = rp.dumps(doc)
    assert rp.privacy_hits(text) == []
    assert rp.privacy_hits(csv_text) == []
    with open(rp.SCHEMA_PATH) as f:
        assert rp.privacy_hits(f.read()) == []
    assert doc["capture"]["id"] == "vlf_20261105_0430Z"
    assert doc["capture"]["file"] == "20261105/vlf_0430Z.cs16"


def test_validator_catches_errors(clean_run):
    doc, _ = clean_run
    schema = rp.load_schema()
    bad = json.loads(rp.dumps(doc))
    bad["capture"]["position"] = "x"
    del bad["summary"]["hum_frac"]
    bad["minutes"][0]["gated_frac"] = 1.5
    err = rp.validate(bad, schema)
    assert any("unexpected key position" in e for e in err)
    assert any("missing hum_frac" in e for e in err)
    assert any("1.5 > 1" in e for e in err)


def test_clean_scene_numbers(clean_cap, clean_run):
    """Clean synthetic scene: no hum, nothing gated, comb share near chance, most sferics found and seen in the
    16-30 kHz band too (coincidence), with a chance coincidence rate far below the measured one."""
    _, ts = clean_cap
    doc, csv_text = clean_run
    s = doc["summary"]
    assert s["hum_frac"] == 0 and s["gated_frac"] == 0 and s["trusted"]
    assert s["n_events"] >= 0.9 * ts.size
    assert s["v2_n"] >= 0.8 * ts.size
    assert s["v2_chance_per_min"] < 0.05 * s["v2_per_min"]
    rows = csv_text.strip().split("\n")
    assert rows[0].split(",") == list(rp.EVENT_COLUMNS)
    assert len(rows) - 1 == doc["events"]["n"] == s["n_events"]
    assert len(doc["minutes"]) == 1 and doc["minutes"][0]["dur_s"] == SECS


def test_two_runs_byte_identical(clean_cap, tmp_path):
    """Deterministic: two analyze runs write byte-identical session and catalogue files."""
    path, _ = clean_cap
    outs = []
    for k in range(2):
        d = tmp_path / f"run{k}"
        assert cli.main(["analyze", path, "--out", str(d)]) == 0
        outs.append([(d / f).read_bytes() for f in sorted(os.listdir(d))])
    assert outs[0] == outs[1] and len(outs[0]) == 2


def test_hum_switch_on_rate_stable(tmp_path):
    """Buzz switching on at 10 s (about 70 pulses/s): hum_frac goes 0 -> ~1 between the halves, but the gated
    rate and the two-band rate stay within 1.6x of the clean half (the plan's hum-boundary test, in miniature)."""
    x, ts = scene(buzz_from=10.0, seed=7)
    rep, ev = rp.analyze_iq(x)
    t = ev["t_start_s"]
    halves = [(0.0, 10.0), (10.0, 20.0)]
    n_true = [((ts >= a) & (ts < b)).sum() for a, b in halves]
    n = [((t >= a) & (t < b)).sum() for a, b in halves]
    blocks = rep["hum_blocks"]
    assert [b["hum"] for b in blocks] == [False, True]
    assert rep["summary"]["hum_frac"] == pytest.approx(0.5)
    r = [n[i] / max(n_true[i], 1) for i in range(2)]
    assert max(r) / min(r) < 1.6
    v2 = [(ev["coinc_hi"] & (t >= a) & (t < b)).sum() / max(k, 1) for (a, b), k in zip(halves, n_true)]
    assert max(v2) / max(min(v2), 1e-9) < 1.6
    assert rep["summary"]["comb_share"] < 0.25


# ----------------------------------------------------------------------------- CLI census and the panel

def test_census_cli(clean_cap, tmp_path):
    path, _ = clean_cap
    out = tmp_path / "census.csv"
    sess = tmp_path / "sess"
    assert cli.main(["census", os.path.dirname(os.path.dirname(path)), "--out", str(out),
                     "--sessions", str(sess)]) == 0
    lines = out.read_text().strip().split("\n")
    assert lines[0].split(",") == list(rp.CENSUS_COLUMNS) and len(lines) == 2
    assert lines[1].startswith("vlf_20261105_0430Z,20261105/vlf_0430Z.cs16,")
    assert sorted(os.listdir(sess)) == ["vlf_20261105_0430Z.events.csv", "vlf_20261105_0430Z.session.json"]


def _panel():
    spec = importlib.util.spec_from_file_location("storm_panel", os.path.join(ROOT, "tools", "storm_panel.py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_panel_reads_sessions_and_greys(clean_run, tmp_path):
    """The panel reads session files into per-minute legs, greys an untrusted minute, carries no tweek ranges, and
    the jump alert ignores untrusted minutes."""
    doc, csv_text = clean_run
    d2 = json.loads(rp.dumps(doc))
    d2["capture"]["id"] = "vlf_20261105_0500Z"
    d2["capture"]["utc_start"] = "2026-11-05T05:00:00+00:00"
    d2["minutes"][0].update(rate_per_min=999.0, comb_share=0.6, trusted=False, untrusted_because=["comb_share"])
    rp.write_outputs(doc, csv_text, str(tmp_path))
    rp.write_outputs(d2, csv_text, str(tmp_path))
    pn = _panel()
    legs = pn.load_session(tmp_path)
    assert [g["hhmm"] for g in legs] == ["04:30", "05:00"]
    assert legs[0]["trusted"] and not legs[1]["trusted"]
    assert legs[1]["why"] == ["comb_share"]
    assert all(g["d_km"] == [] and g["h_km"] is None for g in legs)
    assert pn.jump_alert(legs) is None                  # the 999/min minute is untrusted
    page = pn.served_page()
    assert b"sw-trust" in page and b"</body>" in page
    assert page.count(b"sw-trust") == 1 and page.index(b"sw-trust") < page.rindex(b"</body>")


def test_panel_sky_share_and_dated_labels(clean_run, tmp_path):
    """SKY % is chance-corrected (chance comb share 0.144 reads 100 %, all-comb reads 0); labels get MM-DD when the
    folder spans two UTC dates."""
    pn = _panel()
    assert pn.sky_share(0.144) == pytest.approx(1.0, abs=0.002)
    assert pn.sky_share(1.0) == 0.0 and pn.sky_share(None) is None
    assert pn.sky_share(0.572) == pytest.approx(0.5, abs=0.002)
    doc, csv_text = clean_run
    d2 = json.loads(rp.dumps(doc))
    d2["capture"]["id"] = "vlf_20261106_0100Z"
    d2["capture"]["utc_start"] = "2026-11-06T01:00:00+00:00"
    rp.write_outputs(doc, csv_text, str(tmp_path))
    rp.write_outputs(d2, csv_text, str(tmp_path))
    legs = pn.load_session(tmp_path)
    assert [g["hhmm"] for g in legs] == ["11-05 04:30", "11-06 01:00"]
