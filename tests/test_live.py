"""Fast tests for sferic_watch.live on SYNTHETIC IQ (noise host + receiver tone + pulses) through MockRadio only.

Minutes are shortened to 20 s (two 10 s hum blocks) so a few minutes of chain run in seconds; the real daemon uses
60 s. The real SDR is never opened."""
import json
import os
import subprocess
import sys
import time

import numpy as np
import pytest

from sferic_watch import io as sio
from sferic_watch import live as lv
from sferic_watch import radio as rd
from sferic_watch import report as rp

import inject as inj                      # bench/inject.py (on the test path via conftest)

FS = lv.FS
MIN_S = 20.0
SECS = 60.0
T0 = 1_794_621_600.0                      # 2026-11-14T02:00:00Z, a whole UTC minute


def damped(fs=FS, f0=6000.0, tau=0.15e-3, n=200):
    t = np.arange(n) / fs
    return np.exp(-t / tau) * np.exp(2j * np.pi * (f0 - inj.FC) * t)


@pytest.fixture(scope="module")
def scene():
    """60 s synthetic host with sparse 120 Hz buzz (30 % of the grid, ~20 dB) and Poisson sferics 3/s (~24 dB),
    quantised to int16 the way the recorder writes it. Returns (int16 interleaved, sferic times)."""
    rng = np.random.default_rng(5)
    x = inj.synthetic_host(SECS, seed=4)
    floor = inj.host_floor(x)
    pulse = damped()
    g0 = np.sqrt(floor / inj.snip_peak_power(pulse))
    k = np.sort(rng.choice(int(SECS * 120) - 2, int(0.3 * SECS * 120), replace=False)) + 1
    tb = 2.1e-3 + k / 120.0 + rng.normal(0, 40e-6, k.size)
    gb = g0 * 10 ** ((20 + rng.uniform(-2, 2, k.size)) / 20)
    ts = inj.poisson_starts(3.0, SECS, seed=6, guard=0.5)
    gs = np.full(ts.size, g0 * 10 ** (24 / 20))
    y = inj.inject(x, [pulse] * (k.size + ts.size), np.r_[tb, ts], np.r_[gb, gs])
    return sio.to_cs16(y), ts


@pytest.fixture(scope="module")
def batch(scene):
    """The batch engine on the same int16 samples (io.read_slice's scaling and DC rule)."""
    iq, _ = scene
    x = (iq[0::2] + 1j * iq[1::2]).astype(np.complex64) / np.float32(32768.0)
    x -= x.mean(dtype=np.complex128).astype(np.complex64)
    rep, ev = rp.analyze_iq(x)
    return rep, ev


def run_chain(iq, chunks, **kw):
    ch = lv.Chain(minute_s=MIN_S, **kw)
    out, pos, i = [], 0, 0
    n = len(iq) // 2
    while pos < n:
        k = chunks[i % len(chunks)]
        out += ch.push(iq[2 * pos:2 * min(n, pos + k)])
        pos += k
        i += 1
    out += ch.end()
    return out


@pytest.fixture(scope="module")
def full_window(scene):
    """Chain whose window always spans the whole 60 s (history and lookahead >= 3 minutes): must equal batch."""
    iq, _ = scene
    return run_chain(iq, [65536], history_min=5, lookahead_min=5)


def test_chain_with_full_window_equals_batch(full_window, batch):
    rep, ev = batch
    t = np.concatenate([r["events"]["t_start"] for r in full_window])
    assert len(full_window) == 3 and len(t) == len(ev["t_start_s"])
    assert np.abs(t - ev["t_start_s"]).max() < 1e-9
    conf = np.concatenate([r["events"]["conf"] for r in full_window])
    assert np.array_equal(conf, ev["conf"])
    # minute rows: rp.rate_row over the 20 s minutes of the batch events
    secs = SECS
    for r in full_window:
        a = r["t0_s"]
        n_b = int(((ev["t_start_s"] >= a) & (ev["t_start_s"] < a + MIN_S)).sum())
        assert r["row"]["n_events"] == n_b
    assert sum(r["row"]["n_conf"] for r in full_window) == int(ev["conf"].sum())


def test_chain_does_not_depend_on_chunking(scene, full_window):
    iq, _ = scene
    other = run_chain(iq, [1000, 77777, 4096, 333], history_min=5, lookahead_min=5)
    for a, b in zip(full_window, other):
        assert np.array_equal(a["events"]["t_start"], b["events"]["t_start"])
        assert a["row"] == b["row"]


def test_chain_finds_the_sferics_and_gates_the_buzz(full_window, scene):
    _, ts = scene
    t = np.concatenate([r["events"]["t_start"] for r in full_window])
    j = np.searchsorted(t, ts)
    d = np.minimum(np.abs(t[np.clip(j - 1, 0, len(t) - 1)] - ts), np.abs(t[np.clip(j, 0, len(t) - 1)] - ts))
    assert (d < 2e-3).mean() > 0.85                         # injected sferics found
    assert len(t) < 1.3 * len(ts)                           # 2160 buzz pulses mostly gated


def test_commit_is_causal_with_stated_latency(scene):
    """lookahead 0: minute m is committed as soon as m plus the 2 s guard is in; not one sample earlier."""
    iq, _ = scene
    N, G = int(MIN_S * FS), int(lv.GUARD_S * FS)
    ch = lv.Chain(minute_s=MIN_S, history_min=2, lookahead_min=0, warmup=False)
    assert ch.push(iq[:2 * (N + G - 1)]) == []
    r = ch.push(iq[2 * (N + G - 1):2 * (N + G)])
    assert [x["minute"] for x in r] == [0]
    ch2 = lv.Chain(minute_s=MIN_S, history_min=2, lookahead_min=1, warmup=False)
    assert ch2.push(iq[:2 * (2 * N + G - 1)]) == []
    assert [x["minute"] for x in ch2.push(iq[2 * (2 * N + G - 1):2 * (2 * N + G)])] == [0]


def test_warmup_waits_for_a_full_window_then_runs_at_the_stated_latency(scene, full_window):
    """warm-up (default): the first minutes of a segment wait until the whole W-minute window is in, so a segment of
    up to W minutes is gated exactly as offline (here W = 1 + 1 + 1 = 3 = the whole 60 s scene)."""
    iq, _ = scene
    N, G = int(MIN_S * FS), int(lv.GUARD_S * FS)
    ch = lv.Chain(minute_s=MIN_S, history_min=1, lookahead_min=1)
    assert ch.window(0, 0, 2) == (0, 2) and ch.window(1, 0, 2) == (0, 2)
    assert ch.window(5, 0, 6) == (4, 6)                    # steady state: c - H .. c + L
    assert ch.window(6, 0, 6) == (4, 6)                    # segment end: reaches back to keep W minutes
    assert ch.push(iq[:2 * (2 * N + G)]) == []             # minutes 0, 1 in: still waiting for minute 2
    out = ch.push(iq[2 * (2 * N + G):]) + ch.end()
    assert [r["minute"] for r in out] == [0, 1, 2]
    for a, b in zip(out, full_window):                     # the whole-segment window = batch
        assert np.array_equal(a["events"]["t_start"], b["events"]["t_start"])
        assert a["row"] == b["row"]


def test_dead_or_clipped_front_end_is_never_trusted():
    """All-zero input, a stuck full-scale value and clipped noise gave trusted rate-0 (or noise-stroke) minutes."""
    fs, n = FS, int(MIN_S * FS)
    rng = np.random.default_rng(3)
    cases = {"zeros": np.zeros(2 * n * 3, np.int16),
             "stuck": np.full(2 * n * 3, 32767, np.int16),
             "clipped": np.clip(rng.standard_normal(2 * n * 3) * 40000, -32768, 32767).astype(np.int16)}
    for name, iq in cases.items():
        out = lv.Chain(minute_s=MIN_S, history_min=1, lookahead_min=0).push(iq)
        assert out, name
        for r in out:
            assert r["row"]["trusted"] is False, name
            assert any(w.startswith("frontend_") for w in r["row"]["untrusted_because"]), name
    assert lv.frontend_faults({"floor_dbfs": -75.0, "clipped": 0}, n) == []


def test_chain_memory_is_bounded(scene):
    """Only the gate window's minutes stay cached and only the raw samples the next minute needs."""
    iq, _ = scene
    ch = lv.Chain(minute_s=10.0, history_min=1, lookahead_min=1)
    for k in range(0, len(iq) // 2, 65536):
        ch.push(iq[2 * k:2 * (k + 65536)])
        assert len(ch.cache) <= 1 + 1 + 1 + 1 + 1          # window + the one a segment-end window reaches back
        assert ch.raw.size // 2 <= (10.0 + 2 * lv.GUARD_S) * FS + 65536


# ----------------------------------------------------------------------------- alert

def row(rate, trusted=True):
    return {"conf_per_min": rate, "trusted": trusted}


def feed(al, t0, rates, trusted=True):
    recs = []
    for i, r in enumerate(rates):
        a = al.feed(t0 + 60.0 * i, row(r, trusted))
        if a:
            recs.append(a)
    return recs


def test_alert_learns_then_rises_with_hysteresis_and_falls():
    got = []
    al = lv.Alert(notifier=got.append)
    t = T0
    assert feed(al, t, [12.0] * 120) == []                   # 2 h of quiet: baseline 12/min
    assert al.last["status"] == "ok" and al.last["baseline_conf_per_min"] == pytest.approx(12.0)
    t += 120 * 60
    up = feed(al, t, [100.0] * 20)
    assert [a["to"] for a in up] == ["storm"]               # 100 >= 6 x 12 and >= 80: straight to "storm"
    assert got == up                                        # the notifier saw it
    t += 20 * 60
    down = feed(al, t, [12.0] * 60)
    assert [a["to"] for a in down] == ["active", "quiet"]
    assert al.level == 0


def test_alert_never_rises_on_untrusted_minutes():
    al = lv.Alert()
    feed(al, T0, [12.0] * 120)
    assert feed(al, T0 + 7200, [5000.0] * 30, trusted=False) == []
    assert al.level == 0 and al.last["status"] == "too_few_trusted_minutes"


def test_alert_needs_a_baseline_first():
    al = lv.Alert()
    assert feed(al, T0, [500.0] * 30) == []
    assert al.last["status"] == "learning_baseline"


def test_alert_short_spike_is_ignored():
    al = lv.Alert()
    feed(al, T0, [12.0] * 120)
    assert feed(al, T0 + 7200, [12.0] * 5 + [300.0] * 2 + [12.0] * 10) == []   # a 2-minute spike: median holds


# ----------------------------------------------------------------------------- files

def test_store_rotates_daily_and_prunes(tmp_path):
    st = lv.Store(str(tmp_path), keep_days=3)
    day = 86400.0
    st.append("minutes", [{"utc": lv.iso(T0), "x": 1}], T0)
    st.append("minutes", [{"utc": lv.iso(T0 + day), "x": 2}], T0 + day)
    assert (tmp_path / "archive" / "minutes-20261114.jsonl").exists()
    assert json.loads((tmp_path / "minutes.jsonl").read_text())["x"] == 2
    st.append("minutes", [{"utc": lv.iso(T0 + 4 * day), "x": 3}], T0 + 4 * day)   # 4 days on: day 1 pruned
    names = os.listdir(tmp_path / "archive")
    assert "minutes-20261114.jsonl" not in names and "minutes-20261115.jsonl" in names
    st2 = lv.Store(str(tmp_path))
    assert st2.day == "20261118"                            # a restart picks up the day on disk


def test_store_disk_cap_stops_raw_then_jsonl(tmp_path, monkeypatch):
    st = lv.Store(str(tmp_path), raw=True, min_free_gb=5.0)
    import shutil

    class U:
        def __init__(self, free):
            self.free, self.total, self.used = free, 0, 0
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(3e9))
    st.check_disk()
    assert not st.raw_on and st.jsonl_on and "raw recording off" in st.warning
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(1e8))
    st.check_disk()
    assert not st.jsonl_on
    st.append("events", [{"a": 1}], T0)
    assert not (tmp_path / "events.jsonl").exists()
    monkeypatch.setattr(shutil, "disk_usage", lambda p: U(50e9))
    st.check_disk()
    assert st.raw_on and st.jsonl_on and st.warning is None


def test_state_json_is_written_atomically(tmp_path):
    st = lv.Store(str(tmp_path))
    st.write_state({"a": 1})
    st.write_state({"a": 2})
    assert json.loads((tmp_path / "state.json").read_text()) == {"a": 2}
    assert [f for f in os.listdir(tmp_path) if ".tmp" in f] == []


# ----------------------------------------------------------------------------- radio lock

def test_radio_lock_is_exclusive(tmp_path):
    p = str(tmp_path / "radio.lock")
    a = lv.RadioLock(p, check_foreign=False)
    a.acquire()
    b = lv.RadioLock(p, check_foreign=False)
    with pytest.raises(lv.RadioBusy, match="held"):
        b.acquire()
    a.release()
    b.acquire()
    b.release()


def test_radio_lock_refuses_while_dlayer_run_is_running(tmp_path):
    fake = tmp_path / "dlayer_run.py"
    fake.write_text("import time\ntime.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, str(fake)])
    try:
        t = time.time()
        while proc.pid not in lv.foreign_radio_users() and time.time() - t < 5:
            time.sleep(0.05)
        with pytest.raises(lv.RadioBusy, match="dlayer_run.py is running"):
            lv.RadioLock(str(tmp_path / "radio.lock")).acquire()
    finally:
        proc.kill()
        proc.wait()


# ----------------------------------------------------------------------------- daemon end to end (mock, inline)

def run_daemon(tmp_path, iq, faults=(), align="start", t_start=T0 + 0.5, **kw):
    radio = rd.MockRadio(arrays=[iq], speed=0, t_start=t_start, settle=False, chunk=8192, faults=faults)
    store = lv.Store(str(tmp_path / "state"))
    d = lv.Daemon(radio, store, align=align, minute_s=MIN_S, inline=True, log=lambda m: None, **kw)
    assert d.run() == 0
    rd_ = lambda n: [json.loads(l) for l in open(tmp_path / "state" / n)] if (tmp_path / "state" / n).exists() else []
    return d, rd_("minutes.jsonl"), rd_("events.jsonl"), json.loads((tmp_path / "state" / "state.json").read_text())


def test_daemon_end_to_end(tmp_path, scene, full_window):
    iq, _ = scene
    d, mins, evs, state = run_daemon(tmp_path, iq, history_min=5, lookahead_min=5)
    assert state["status"] == "stopped" and state["chain"]["minutes"] == 3 == len(mins)
    assert state["radio"]["kind"] == "mock" and "heartbeat_utc" in state and state["alert"]["level"] == 0
    t_expect = np.concatenate([r["events"]["t_start"] for r in full_window]) + T0 + 0.5
    assert np.abs(np.array([e["t"] for e in evs]) - t_expect).max() < 2e-6
    assert [m["n_events"] for m in mins] == [r["row"]["n_events"] for r in full_window]
    m = mins[0]
    for k in ("rate_per_min", "conf_per_min", "hum_frac", "gated_frac", "comb_share", "trusted", "health"):
        assert k in m
    assert m["health"]["floor_dbfs"] < -40 and "overflows" in m["health"]
    assert evs[0]["utc"].endswith("Z") and {"peak_db", "conf", "hum", "coinc_hi"} <= set(evs[0])
    text = (tmp_path / "state" / "state.json").read_text() + (tmp_path / "state" / "minutes.jsonl").read_text()
    assert rp.privacy_hits(text) == []


def test_daemon_aligns_to_utc_minutes(tmp_path, scene):
    iq, _ = scene
    d, mins, evs, state = run_daemon(tmp_path, iq, align="utc", t_start=T0 - 7.0, history_min=1, lookahead_min=0)
    assert [m["utc"] for m in mins] == [lv.iso(T0), lv.iso(T0 + MIN_S)]     # starts at the next whole minute
    assert all(e["t"] >= T0 for e in evs)


def test_daemon_overflow_splits_segments(tmp_path, scene):
    iq, _ = scene
    d, mins, evs, state = run_daemon(tmp_path, iq, faults=[dict(kind="overflow", at_s=25.0, n=1)],
                                     history_min=1, lookahead_min=0)
    assert state["chain"]["segments"] == 2
    assert state["radio"]["stats"]["overflows"] == 1
    gaps = [m for m in mins if m.get("gap")]
    mins = [m for m in mins if not m.get("gap")]
    assert {m["segment"] for m in mins} == {0, 1}
    assert sum(m["health"]["overflows"] for m in mins) == 1
    ts = [m["t"] for m in mins]
    assert ts == sorted(ts)
    # the gap is an explicit row, between the last minute before it and the first after it
    assert len(gaps) == 1 and state["chain"]["gaps"] == 1
    g = gaps[0]
    assert "overflows" in g["reason"] and g["t"] < g["t_end"]
    before = max(m["t"] + m["dur_s"] for m in mins if m["t"] < g["t_end"])
    after = min(m["t"] for m in mins if m["t"] >= g["t_end"])
    assert abs(g["t"] - before) < 1e-3 and abs(g["t_end"] - after) < 1e-3


def test_cli_live_mock_runs(tmp_path, scene):
    """The command line, end to end in a subprocess: --radio mock --speed 0 on a .cs16 file."""
    iq, _ = scene
    p = tmp_path / "vlf_0200Z.cs16"
    iq[:2 * int(25 * FS)].tofile(p)
    (tmp_path / "vlf_0200Z.json").write_text('{"utc": "2026-11-14T02:00:00+00:00"}')
    sd = tmp_path / "st"
    r = subprocess.run([sys.executable, "-m", "sferic_watch", "live", "--radio", "mock", "--mock-files", str(p),
                        "--speed", "0", "--state-dir", str(sd), "--inline", "--lookahead-min", "0"],
                       capture_output=True, text=True, timeout=120, cwd=os.path.dirname(os.path.dirname(lv.__file__)))
    assert r.returncode == 0, r.stdout + r.stderr
    st = json.loads((sd / "state.json").read_text())
    assert st["status"] == "stopped"
    assert st["chain"]["history_min"] == lv.HISTORY_MIN    # 60 s minutes here: 25 s of signal gives no minute
    assert st["chain"]["minutes"] == 0


def test_panel_live_view_reads_the_daemon_files(tmp_path, scene):
    import importlib.util
    iq, _ = scene
    d, mins, evs, state = run_daemon(tmp_path, iq, history_min=1, lookahead_min=0)
    root = os.path.dirname(os.path.dirname(os.path.abspath(lv.__file__)))
    spec = importlib.util.spec_from_file_location("storm_panel", os.path.join(root, "tools", "storm_panel.py"))
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    legs, st = sp.load_live(tmp_path / "state", max_minutes=2)
    assert len(legs) == 2 and legs[-1]["rate"] == mins[-1]["rate_per_min"]
    assert legs[-1]["trusted"] == mins[-1]["trusted"] and st["status"] == "stopped"
    a = sp.live_alert(st, legs)
    assert a is not None and "WATCHER STOPPED" in a["msg"]
    st["status"], st["heartbeat_t"] = "running", time.time()
    st["alert"] = {"level": 2, "name": "storm", "stat_conf_per_min": 90.0, "baseline_conf_per_min": 12.0,
                   "baseline_source": "rolling_6h"}
    assert "STORM LEVEL STORM" in sp.live_alert(st, legs)["msg"]


def test_daemon_survives_a_chain_error(tmp_path, scene, monkeypatch):
    """A commit that raises drops its segment; the daemon starts a new one and reports the error (exit code 1)."""
    iq, _ = scene
    orig = lv.Chain._commit
    calls = {"n": 0}

    def flaky(self, c, last):
        calls["n"] += 1
        if calls["n"] == 1:
            raise RuntimeError("synthetic chain failure")
        return orig(self, c, last)
    monkeypatch.setattr(lv.Chain, "_commit", flaky)
    radio = rd.MockRadio(arrays=[iq], speed=0, t_start=T0, settle=False, chunk=8192)
    d = lv.Daemon(radio, lv.Store(str(tmp_path / "s")), align="start", minute_s=MIN_S, inline=True,
                  history_min=1, lookahead_min=0, log=lambda m: None, warmup=False)
    assert d.run() == 1
    st = json.loads((tmp_path / "s" / "state.json").read_text())
    assert st["status"] == "stopped" and st["exit_reason"] == "source ended"
    assert "synthetic chain failure" in st["chain"]["errors"][0]
    assert st["chain"]["segments"] == 2 and st["chain"]["minutes"] >= 1


def test_daemon_survives_a_full_disk(tmp_path, scene, monkeypatch):
    """A .jsonl append that fails (ENOSPC) loses those records, is counted in state.json, and the watch goes on."""
    import errno
    iq, _ = scene
    orig = lv.Store.append
    calls = {"n": 0}

    def full(self, name, recs, t):
        calls["n"] += 1
        if calls["n"] <= 2:
            raise OSError(errno.ENOSPC, "No space left on device")
        return orig(self, name, recs, t)
    monkeypatch.setattr(lv.Store, "append", full)
    d, mins, evs, state = run_daemon(tmp_path, iq, history_min=1, lookahead_min=0)
    assert state["status"] == "stopped" and state["exit_reason"] == "source ended" and not state["chain"]["errors"]
    assert state["disk"]["write_errors"] == 2 and "No space" in state["disk"]["last_write_error"]
    assert state["chain"]["minutes"] == 3 and len(mins) == 2      # the first minute's lines were lost, not later ones


def test_store_repairs_a_torn_last_line(tmp_path):
    """What SIGKILL / a power cut mid-append leaves: a last line without newline. The next start cuts it off."""
    d = tmp_path / "st"
    d.mkdir()
    good = '{"utc": "2026-11-14T02:00:00Z", "t": 1.0}\n'
    (d / "minutes.jsonl").write_text(good + good[:17])
    (d / "events.jsonl").write_text(good)
    s = lv.Store(str(d))
    assert s.repaired == {"minutes.jsonl": 17}
    s.append("minutes", [{"utc": "2026-11-14T02:01:00Z", "t": 2.0}], 1_794_621_660.0)
    lines = (d / "minutes.jsonl").read_text().splitlines()
    assert len(lines) == 2 and all(json.loads(x) for x in lines)
    assert (d / "events.jsonl").read_text() == good
