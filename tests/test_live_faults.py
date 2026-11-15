"""Regression tests for the live watcher's failure paths (verifier findings of the stress round), on SYNTHETIC data and
the MockRadio only; the real SDR is never opened."""
import json
import os
import signal
import subprocess
import sys
import textwrap
import time

import numpy as np
import pytest

from sferic_watch import live as lv
from sferic_watch import radio as rd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FS = rd.FS
T0 = 1_794_621_600.0


def _alive(pid: int) -> bool:
    try:
        with open(f"/proc/{pid}/stat") as f:
            return f.read().rsplit(")", 1)[1].split()[0] != "Z"
    except OSError:
        return False


def _run_py(code: str, timeout: float):
    return subprocess.run([sys.executable, "-c", textwrap.dedent(code)], capture_output=True, text=True,
                          timeout=timeout, cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})


# ----------------------------------------------------------------------------- the chain worker process

KILL_PARENT = """
    import os, signal, sys, time, threading
    import numpy as np
    from sferic_watch import live as lv
    w = lv.ProcessWorker({"fs": 62500.0, "minute_s": 20.0, "history_min": 1, "lookahead_min": 0}, maxsize=50)
    print(w.proc.pid, flush=True)
    w.put(("start", 0))
    x = (np.random.default_rng(0).standard_normal(2 * 65536 * 4) * 300).astype(np.int16)   # 1 MB messages
    def killer():
        time.sleep(float(sys.argv[1]) if len(sys.argv) > 1 else 2.5)
        os.kill(os.getpid(), signal.SIGKILL)
    threading.Thread(target=killer, daemon=True).start()
    while True:
        try:
            w.put(("data", 0, x), timeout=0.2)
        except Exception:
            pass
"""


@pytest.mark.parametrize("trial", range(3))
def test_worker_exits_when_the_daemon_is_killed_mid_message(trial):
    """SIGKILL of the daemon while its feeder is part-way through a 1 MB message: the worker used to block in that
    pipe read for ever (orphaned with 0.2-0.6 GB). Now its parent watch / parent-death signal / EOF end it."""
    p = subprocess.Popen([sys.executable, "-c", textwrap.dedent(KILL_PARENT)], stdout=subprocess.PIPE, text=True,
                         cwd=ROOT, env={**os.environ, "PYTHONPATH": ROOT})
    wpid = int(p.stdout.readline())
    p.wait(30)
    assert p.returncode == -signal.SIGKILL
    t = time.time()
    while _alive(wpid) and time.time() - t < 10:
        time.sleep(0.1)
    alive = _alive(wpid)
    if alive:
        os.kill(wpid, signal.SIGKILL)
    assert not alive, "chain worker outlived its daemon"


KILL_WORKER = """
    import os, signal, sys, threading, time
    import numpy as np
    from sferic_watch import live as lv, radio as rd
    arr = (np.random.default_rng(0).standard_normal(2 * 62500 * 30) * 300).astype(np.int16)
    r = rd.MockRadio(arrays=[arr], speed=0, loop=True, t_start=1.7e9, settle=False)
    d = lv.Daemon(r, lv.Store(sys.argv[1]), align="start", minute_s=20.0, history_min=1, lookahead_min=0,
                  log=lambda m: None, restore=False)
    def killer():
        while d.worker is None:
            time.sleep(0.05)
        time.sleep(float(sys.argv[2]))
        os.kill(d.worker.proc.pid, signal.SIGKILL)
    threading.Thread(target=killer, daemon=True).start()
    rc = d.run()
    print("RC", rc, d.exit_reason, flush=True)
"""


@pytest.mark.parametrize("kill_s", [1.5, 4.0])
def test_daemon_exits_when_its_worker_dies(tmp_path, kill_s):
    """The worker killed (OOM) while the queue is full or a message is part-sent: the daemon used to spin for ever in
    _send, or hang at interpreter exit joining the queue feeder. It must return worker_died and the process exit."""
    sd = tmp_path / "s"
    code = KILL_WORKER.replace("sys.argv[1]", repr(str(sd))).replace("sys.argv[2]", repr(str(kill_s)))
    r = _run_py(code, timeout=90)
    assert "RC 1 worker_died" in r.stdout, r.stdout + r.stderr[-2000:]
    assert r.returncode == 0
    st = json.loads((sd / "state.json").read_text())
    assert st["status"] == "stopped" and st["exit_reason"] == "worker_died"


# ----------------------------------------------------------------------------- the storm level

def row(v, tr=True):
    return {"trusted": tr, "conf_per_min": v}


def test_alert_needs_new_trusted_minutes_to_rise():
    """4 trusted high minutes in every 10 (STAT_NEED is 5), the rest untrusted: re-checks of the same stale trusted
    values used to count towards UP_HOLD and raised "severe"."""
    a, t = lv.Alert(), T0
    for _ in range(420):
        a.feed(t, row(5))
        t += 60
    ups = []
    for i in range(180):
        r = a.feed(t, row(1000, (i % 10) < 4))
        t += 60
        if r:
            ups.append(r)
    assert ups == [] and a.level == 0


def test_alert_steps_down_when_nothing_can_be_read():
    """A storm level must not stand for ever on untrusted minutes: one level down per BLIND_HOLD minutes."""
    a, t = lv.Alert(), T0
    for _ in range(420):
        a.feed(t, row(5))
        t += 60
    for _ in range(30):
        a.feed(t, row(1000))
        t += 60
    top = a.level
    assert top == 3
    recs = []
    for _ in range(24 * 60):
        r = a.feed(t, row(5, False))
        t += 60
        if r:
            recs.append(r)
    assert a.level == 0
    assert [r["reason"] for r in recs] == ["no_trusted_minutes"] * top
    assert recs[0]["t"] - (T0 + 450 * 60) <= (lv.BLIND_HOLD + lv.STAT_MIN) * 60


def test_alert_restore_equals_a_continuous_run():
    rng = np.random.default_rng(1)
    rows = []
    t = T0
    for i in range(3 * 1440):
        v = float(rng.poisson(8 if (i // 600) % 3 else 400 if 200 < i % 600 < 260 else 8))
        rows.append({"t": t, "trusted": bool(rng.random() > 0.1), "conf_per_min": v})
        t += 60
    a = lv.Alert()
    for r in rows:
        a.feed(r["t"], r)
    b = lv.Alert()
    assert b.restore(rows[:2000] + [{"gap": True, "t": 0}]) == 2000
    for r in rows[2000:]:
        b.feed(r["t"], r)
    assert (a.level, a.since, a.ht, a.hv, a.last) == (b.level, b.since, b.ht, b.hv, b.last)


def test_daemon_restores_the_level_and_writes_a_restart_gap_row(tmp_path):
    sd = tmp_path / "s"
    sd.mkdir()
    base = [{"utc": lv.iso(T0 - 7200 + 60 * i), "t": T0 - 7200 + 60 * i, "dur_s": 60.0, "trusted": True,
             "conf_per_min": 9.0} for i in range(100)]
    (sd / "minutes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in base))
    x = (np.random.default_rng(2).standard_normal(2 * int(62500 * 25)) * 300).astype(np.int16)
    radio = rd.MockRadio(arrays=[x], speed=0, t_start=T0, settle=False)
    d = lv.Daemon(radio, lv.Store(str(sd)), align="start", minute_s=10.0, inline=True, history_min=1,
                  lookahead_min=0, log=lambda m: None)
    assert d.restored_minutes == 100 and len(d.alert.ht) == 100
    assert d.run() == 0
    rows = [json.loads(l) for l in open(sd / "minutes.jsonl")]
    gaps = [r for r in rows if r.get("gap")]
    assert len(gaps) == 1 and gaps[0]["reason"] == ["watcher_not_running"]
    assert gaps[0]["t"] == pytest.approx(base[-1]["t"] + 60) and gaps[0]["t_end"] == pytest.approx(T0)


# ----------------------------------------------------------------------------- files

def test_store_drops_a_nul_block_inside_the_file(tmp_path):
    """ext4 after a power cut: a NUL-filled block, then later complete lines. The corrupt line went unnoticed and the
    rotation day was lost (yesterday's lines then archived under today)."""
    d = tmp_path / "st"
    d.mkdir()
    good = lambda day, t: json.dumps({"utc": f"2026-11-{day}T23:58:00Z", "t": t}) + "\n"
    (d / "minutes.jsonl").write_bytes((good(13, 1.0) + good(13, 2.0)).encode() + b"\0" * 300 + b"\n"
                                      + good(13, 3.0).encode())
    s = lv.Store(str(d))
    assert s.corrupt_dropped == {"minutes.jsonl": 1}
    assert s.day == "20261113"
    lines = (d / "minutes.jsonl").read_text().splitlines()
    assert [json.loads(x)["t"] for x in lines] == [1.0, 2.0, 3.0]
    s.append("minutes", [{"utc": "2026-11-14T00:00:00Z", "t": 4.0}], T0 - 2 * 3600 * 1)   # 2026-11-14 00:00Z
    assert (d / "archive" / "minutes-20261113.jsonl").exists()
    assert [json.loads(x)["t"] for x in (d / "minutes.jsonl").read_text().splitlines()] == [4.0]


# ----------------------------------------------------------------------------- radio

def coded(secs: float) -> np.ndarray:
    n = int(secs * FS)
    pos = np.arange(n)
    a = np.empty(2 * n, np.int16)
    a[0::2] = pos % 30000
    a[1::2] = pos // 30000
    return a


def true_t0(b, t0):
    return t0 + (int(b.iq[1]) * 30000 + int(b.iq[0])) / FS


def drain(r, limit_s=60.0):
    out = []
    r.start()
    t_end = time.time() + limit_s
    while time.time() < t_end:
        b = r.get(0.2)
        if b is None:
            if r.finished or not r.alive:
                break
            continue
        out.append(b)
    r.stop()
    return out


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_gap_anchor_is_refined_on_the_lag_minimum(seed):
    """With read-return jitter (10 ms mean) the anchor after a gap came from one read and was late by its jitter
    (median 8 ms, p90 24 ms). The refinement on the next 16-read lag minimum brings it within a few ms."""
    sig = coded(40.0)
    r = rd.MockRadio(arrays=[sig], speed=0, t_start=T0, settle=False, chunk=4096, jitter_s=0.010, seed=seed,
                     faults=[dict(kind="overflow", at_s=10.0, n=1)])
    bl = drain(r)
    gi = [i for i, b in enumerate(bl) if b.gap_before][1]
    raw_err = bl[gi].t0 - true_t0(bl[gi], T0)
    shift = sum(b.anchor_shift_s for b in bl[gi + 1:])
    assert r.stats.anchor_refines == 1
    assert abs(raw_err + shift) < 0.003, (raw_err, shift)
    assert abs(bl[-1].t0 - true_t0(bl[-1], T0)) < 0.003


def test_slow_walk_reanchor_keeps_the_drift_estimate(monkeypatch):
    """A slow host-vs-sample walk of 1000 ppm (5 ms per 5 s): the re-anchor moves the time line by the walk measured
    on the lag minimum, and clock_drift_ppm keeps reading the walk (it used to fall back toward 0)."""
    monkeypatch.setattr(rd, "DRIFT_MIN_S", 5.0)
    sig = coded(60.0)
    faults = [dict(kind="clock", at_s=5.0 * k, dur_s=0.005) for k in range(1, 12)]
    r = rd.MockRadio(arrays=[sig], speed=0, t_start=T0, settle=False, chunk=4096, faults=faults)
    bl = drain(r)
    assert r.stats.reanchors >= 1 and r.stats.clock_steps == 0
    assert any(b.reanchored for b in bl)
    assert r.stats.clock_drift_ppm == pytest.approx(1000.0, rel=0.25)
    # the time line follows the host clock within the 20 ms re-anchor band
    last = bl[-1]
    host_lead = sum(f["dur_s"] for f in faults if f["at_s"] <= 60.0)
    assert abs(last.t0 - true_t0(last, T0) - host_lead) < rd.SLIP_S + 0.002


def test_stop_reports_a_reader_that_does_not_stop():
    class Hung(rd.RadioSource):
        def _open(self):
            pass

        def _read(self, buf):
            time.sleep(3.0)                    # a driver call that does not return in time
            return -1
    r = Hung()
    r.start()
    time.sleep(0.2)
    assert r.stop(timeout=0.5) is False and r.stats.state == "stop_timeout"


def test_empty_looping_mock_ends_instead_of_spinning():
    r = rd.MockRadio(arrays=[np.zeros(0, np.int16)], speed=0, loop=True, t_start=T0)
    t = time.time()
    bl = drain(r, limit_s=10)
    assert bl == [] and r.finished and time.time() - t < 5


def test_mock_reads_live_raw_sidecars_as_sample0(tmp_path):
    x = coded(2.0)
    for name, side in (("rec", {"utc": "2026-11-14T02:00:00+00:00"}),
                       ("live", {"utc": "2026-11-14T02:00:00+00:00", "utc_is_sample0": True})):
        x.tofile(tmp_path / f"{name}.cs16")
        (tmp_path / f"{name}.json").write_text(json.dumps(side))
    rec = rd.MockRadio(files=[str(tmp_path / "rec.cs16")], speed=0)
    live = rd.MockRadio(files=[str(tmp_path / "live.cs16")], speed=0)
    assert live.t_start == pytest.approx(T0) and rec.t_start == pytest.approx(T0 - rd.STAMP_OFFSET_S)


# ----------------------------------------------------------------------------- one radio, one program

@pytest.mark.parametrize("argv,hit", [
    (["python3", "/x/dlayer_run.py"], True), (["python3", "-u", "dlayer_run.py"], True),
    (["python3", "-X", "dev", "-u", "/home/u/dlayer_run/dlayer_run.py", "--x"], True),
    (["/usr/bin/python3.14", "-W", "ignore", "dlayer_run.py"], True), (["./dlayer_run.py"], True),
    (["nvim", "dlayer_run.py"], False), (["less", "/x/dlayer_run.py"], False),
    (["python3", "-c", "print('dlayer_run.py')"], False), (["python3", "other.py", "dlayer_run.py"], False)])
def test_dlayer_run_detection(argv, hit):
    assert lv.runs_script(argv) is hit


def test_watcher_yields_the_radio_to_a_later_dlayer_run(tmp_path, monkeypatch):
    """dlayer_run takes no lock, so one started after the watcher was not noticed. The watcher now looks every
    FOREIGN_CHECK_S, closes the radio and exits 3 (systemd does not restart on 3)."""
    monkeypatch.setattr(lv, "FOREIGN_CHECK_S", 0.2)
    lock = lv.RadioLock(str(tmp_path / "radio.lock"))
    lock.acquire()
    x = (np.random.default_rng(3).standard_normal(2 * 62500 * 30) * 300).astype(np.int16)
    radio = rd.MockRadio(arrays=[x], speed=1.0, loop=True, t_start=T0, settle=False)
    d = lv.Daemon(radio, lv.Store(str(tmp_path / "s")), align="start", minute_s=10.0, inline=True,
                  log=lambda m: None, lock=lock, duration_s=30)
    fake = tmp_path / "dlayer_run.py"
    fake.write_text("import time\ntime.sleep(30)\n")
    proc = subprocess.Popen([sys.executable, "-X", "dev", str(fake)])
    try:
        t = time.time()
        rc = d.run()
        assert rc == 3 and time.time() - t < 10
        assert "yielded the radio" in d.exit_reason and radio.stats.state == "stopped"
        assert lock.fd is None                                  # released
    finally:
        proc.kill()
        proc.wait()


def test_panel_skips_gap_rows(tmp_path):
    import importlib.util
    spec = importlib.util.spec_from_file_location("storm_panel", os.path.join(ROOT, "tools", "storm_panel.py"))
    sp = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sp)
    rows = [{"utc": lv.iso(T0), "t": T0, "dur_s": 60.0, "rate_per_min": 5.0, "trusted": True},
            {"gap": True, "utc": lv.iso(T0 + 60), "t": T0 + 60, "t_end": T0 + 300, "reason": ["overflows"]},
            {"utc": lv.iso(T0 + 300), "t": T0 + 300, "dur_s": 60.0, "rate_per_min": 7.0, "trusted": True}]
    (tmp_path / "minutes.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    legs, _ = sp.load_live(tmp_path, max_minutes=10)
    assert [g["rate"] for g in legs] == [5.0, 7.0]


def test_failed_append_leaves_no_torn_line(tmp_path, monkeypatch):
    """ENOSPC half way through an append: the file is cut back so the next append starts on a clean line."""
    import builtins
    import json as _json
    s = lv.Store(str(tmp_path))
    s.append("events", [{"a": 1}], 1.7e9)
    real_open = builtins.open

    class Half:
        def __init__(self, f):
            self.f = f
        def __enter__(self):
            return self
        def __exit__(self, *a):
            self.f.close()
        def write(self, text):
            self.f.write(text[: len(text) // 2])
            self.f.flush()
            raise OSError(28, "No space left on device")

    monkeypatch.setattr(builtins, "open", lambda p, m="r", *a, **k: Half(real_open(p, m, *a, **k))
                        if str(p).endswith("events.jsonl") and m == "a" else real_open(p, m, *a, **k))
    with pytest.raises(OSError):
        s.append("events", [{"b": 2}, {"c": 3}], 1.7e9)
    monkeypatch.setattr(builtins, "open", real_open)
    s.append("events", [{"d": 4}], 1.7e9)
    lines = open(s.path("events.jsonl")).read().splitlines()
    assert [_json.loads(x) for x in lines] == [{"a": 1}, {"d": 4}]
