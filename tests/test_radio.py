"""Fast tests for sferic_watch.radio on SYNTHETIC int16 arrays (MockRadio only; the real SDR is never opened).

The test signal encodes each sample's position (I = pos mod 30000, Q = pos div 30000), so every block's true
first-sample time is known and the time stamps can be checked exactly."""
import os
import sys
import time

import numpy as np
import pytest

from sferic_watch import radio as rd

FS = rd.FS
T0 = 1_000_000.0


def coded(secs: float) -> np.ndarray:
    n = int(secs * FS)
    pos = np.arange(n)
    a = np.empty(2 * n, np.int16)
    a[0::2] = pos % 30000
    a[1::2] = pos // 30000
    return a


def true_t0(b: rd.Block) -> float:
    return T0 + (int(b.iq[1]) * 30000 + int(b.iq[0])) / FS


def drain(r: rd.RadioSource, limit_s: float = 60.0) -> list[rd.Block]:
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


@pytest.fixture(scope="module")
def sig():
    return coded(40.0)


def mock(sig, faults=(), **kw):
    kw.setdefault("chunk", 4096)
    return rd.MockRadio(arrays=[sig], speed=0, t_start=T0, faults=faults, **kw)


def test_clean_replay_is_exact_and_stamped(sig):
    r = mock(sig, settle=False)
    bl = drain(r)
    got = np.concatenate([b.iq for b in bl])
    assert np.array_equal(got, sig)                       # every sample, in order, nothing lost or added
    assert [b.gap_before for b in bl].count(True) == 1 and bl[0].gap_before
    err = np.array([b.t0 - true_t0(b) for b in bl])
    assert np.abs(err).max() < 1e-9                       # stamp offset corrected exactly on the virtual clock
    assert r.stats.lag_min_ms == pytest.approx(rd.STAMP_OFFSET_S * 1e3)
    assert r.stats.overflows == r.stats.timeouts == r.stats.clock_steps == 0


def test_settle_read_is_thrown_away(sig):
    bl = drain(mock(sig, settle=True))
    first = int(bl[0].iq[1]) * 30000 + int(bl[0].iq[0])
    assert first >= FS * rd.SETTLE_S                      # the first second never reaches the consumer
    assert abs(bl[0].t0 - true_t0(bl[0])) < 1e-9


@pytest.mark.parametrize("fault, counter", [
    (dict(kind="overflow", at_s=5, n=2), "overflows"),
    (dict(kind="stall", at_s=5, dur_s=2), "overflows"),
    (dict(kind="timeout", at_s=5, dur_s=1.5), "timeout_gaps"),
    (dict(kind="vanish", at_s=5, dur_s=3), "read_errors"),
    (dict(kind="drop", at_s=5, n=3000), "clock_steps"),
    (dict(kind="dup", at_s=5, n=3000), "clock_steps"),
])
def test_faults_start_a_new_segment_with_true_time(sig, fault, counter):
    r = mock(sig, faults=[fault], settle=False)
    bl = drain(r)
    assert getattr(r.stats, counter) >= 1
    gaps = [i for i, b in enumerate(bl) if b.gap_before]
    assert len(gaps) == 2                                 # the start, and one after the fault
    g = bl[gaps[1]]
    assert g.t0 - T0 > 5.0
    assert abs(g.t0 - true_t0(g)) < 1e-9                  # re-anchored on the true time
    after = np.array([b.t0 - true_t0(b) for b in bl[gaps[1]:]])
    assert np.abs(after).max() < 1e-9
    # inside each segment the samples are contiguous (no silent hole after the gap is declared)
    seg = bl[gaps[1]:]
    pos = np.array([int(b.iq[1]) * 30000 + int(b.iq[0]) for b in seg])
    assert np.array_equal(np.diff(pos), [b.n for b in seg[:-1]])


def test_vanish_reopens_with_backoff_and_settles_again(sig):
    r = mock(sig, faults=[dict(kind="vanish", at_s=5, dur_s=3)], settle=True)
    bl = drain(r)
    assert r.stats.reopens >= 1 and r.stats.open_failures >= 1
    assert "not found" in r.stats.last_error or "vanished" in r.stats.last_error
    g = [b for b in bl if b.gap_before][1]
    assert g.t0 - T0 >= 5.0 + 3.0 + rd.SETTLE_S - 0.1     # gone for 3 s, then a 1 s settle


def test_small_slip_is_undetectable_and_documented(sig):
    """100 samples (1.6 ms) slip under the 20 ms SLIP_S: no gap, later times shift by the slip (module doc)."""
    r = mock(sig, faults=[dict(kind="drop", at_s=5, n=100)], settle=False)
    bl = drain(r)
    assert r.stats.clock_steps == 0 and sum(b.gap_before for b in bl) == 1
    err = bl[-1].t0 - true_t0(bl[-1])
    assert err == pytest.approx(-100 / FS, abs=1e-9)


def test_host_clock_step_is_a_gap_and_follows_the_host_clock(sig):
    r = mock(sig, faults=[dict(kind="clock", at_s=5, dur_s=4.0)], settle=False)
    bl = drain(r)
    assert r.stats.clock_steps == 1
    g = [b for b in bl if b.gap_before][1]
    assert g.t0 - true_t0(g) == pytest.approx(4.0, abs=1e-6)   # the host clock is the time reference


def test_ring_drops_oldest_and_marks_gap(sig):
    r = rd.MockRadio(arrays=[sig], speed=0, t_start=T0, settle=False, chunk=4096, lossless=False, ring_s=1.0)
    r.start()
    while not r.finished:
        time.sleep(0.05)
    time.sleep(0.2)
    bl = []
    while True:
        b = r.get(0)
        if b is None:
            break
        bl.append(b)
    r.stop()
    assert r.stats.ring_drops > 0
    assert bl[0].gap_before
    assert sum(b.n for b in bl) <= FS * 1.0
    assert r.stats.ring_dropped_samples + sum(b.n for b in bl) == len(sig) // 2


def test_lossless_mock_waits_instead_of_dropping(sig):
    r = rd.MockRadio(arrays=[sig], speed=0, t_start=T0, settle=False, chunk=4096, ring_s=1.0)
    bl = drain(r)
    assert r.stats.ring_drops == 0 and sum(b.n for b in bl) == len(sig) // 2


def test_paced_replay_runs_at_the_asked_speed():
    a = coded(2.0)
    r = rd.MockRadio(arrays=[a], speed=10.0, t_start=T0, settle=False, chunk=4096)
    t = time.monotonic()
    bl = drain(r)
    dt = time.monotonic() - t
    assert sum(b.n for b in bl) == len(a) // 2
    assert 0.15 <= dt <= 1.5                               # 2 s of signal at 10x: about 0.2 s


def test_paced_reopen_skips_the_outage_instead_of_bursting():
    """Wall time spent closed (backoff) is lost stream time; the pacer restarts, so no full-speed catch-up burst."""
    r = rd.MockRadio(arrays=[coded(30.0)], speed=10.0, t_start=T0, settle=False, chunk=4096)
    r._wall0 = (time.monotonic() - 0.5, 0)               # 0.5 s of wall time passed since the last read
    r._open()
    assert r._emitted / FS == pytest.approx(5.0, abs=0.5)   # 0.5 s wall at 10x = 5 s of signal lost
    assert r._wall0 is None
    r2 = rd.MockRadio(arrays=[coded(3.0)], speed=0, t_start=T0, settle=False)
    r2._wall0 = (time.monotonic() - 0.5, 0)
    r2._open()
    assert r2._emitted == 0                               # unpaced replays lose nothing


def test_loop_restarts_and_files(tmp_path):
    a = coded(1.0)
    p = tmp_path / "vlf_0000Z.cs16"
    a.tofile(p)
    (tmp_path / "vlf_0000Z.json").write_text('{"utc": "2026-11-14T02:00:00+00:00"}')
    r = rd.MockRadio(files=[str(p)], speed=0, loop=True, settle=False, chunk=4096)
    assert r.t_start == pytest.approx(1794621600.0 - rd.STAMP_OFFSET_S)   # sidecar utc minus the recorder's 26 ms
    r.start()
    got = 0
    while got < 3 * len(a) // 2:
        b = r.get(1.0)
        assert b is not None
        got += b.n
    r.stop()
    assert r.stats.ring_drops == 0


def test_restart_reopens(sig):
    r = rd.MockRadio(arrays=[sig], speed=20.0, t_start=T0, settle=False, chunk=4096)
    r.start()
    time.sleep(0.2)
    r.restart()
    time.sleep(0.3)
    r.stop()
    assert r.stats.restarts == 1 and r.stats.reopens >= 1


def test_soapy_radio_builds_without_soapysdr():
    """Constructing (and describing) the real radio must not import SoapySDR or touch hardware."""
    before = "SoapySDR" in sys.modules
    r = rd.SoapyRadio(antenna="Antenna C")
    d = r.describe()
    assert d["antenna"] == "Antenna C" and d["ifgr_db"] == 40 and d["rfgr_db"] == 0 and d["agc"] is False
    assert d["fs_hz"] == 62500.0 and d["fc_hz"] == 15000.0 and d["format"] == "CS16"
    assert ("SoapySDR" in sys.modules) == before
