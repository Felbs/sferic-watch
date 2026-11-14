"""Synthetic-signal tests for sferic_watch.health (no real captures)."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sferic_watch import health as H  # noqa: E402

RNG = np.random.default_rng(8)


def msk(n, f_base, clock_ppm=0.0, lo_hz=0.0, amp=1e-3, baud=200.0, seed=1):
    """MSK at baseband offset f_base (RF = f_base + FC) as seen by a receiver
    whose sample clock and LO share one crystal off by clock_ppm (positive =
    fast) and whose synthesiser adds a further lo_hz.

    A fast clock means sample n is taken at true time n / (FS*(1+eps)); the
    transmitter's carrier and symbol timing live on true time.  The LO error
    is added in receiver time, as a mixer would.
    """
    eps = clock_ppm * 1e-6
    t = np.arange(n) / H.FS / (1 + eps)
    nsym = int(t[-1] * baud) + 2
    b = np.random.default_rng(seed).choice([-1.0, 1.0], nsym)
    k = (t * baud).astype(np.int64)
    frac = t * baud - k
    cum = np.concatenate([[0.0], np.cumsum(b)])[:nsym]
    # the carrier lives on true time; our LO (same crystal) on receiver time
    t_rx = np.arange(n) / H.FS
    ph = 2 * np.pi * ((f_base + H.FC) * t - H.FC * t_rx) + np.pi / 2 * (cum[k] + b[k] * frac)
    ph += 2 * np.pi * lo_hz * np.arange(n) / H.FS
    return amp * np.exp(1j * ph)


def write_cs16(path, x):
    iq = np.empty(2 * x.size, np.int16)
    iq[0::2] = np.clip(np.round(x.real * 32768), -32767, 32767)
    iq[1::2] = np.clip(np.round(x.imag * 32768), -32767, 32767)
    iq.tofile(path)


def test_shoulder_level_ignores_centre_line():
    """A strong narrow line exactly on 24.000 kHz must not read as NAA."""
    n = int(10 * H.FS)
    t = np.arange(n) / H.FS
    noise = 1e-4 * (RNG.standard_normal(n) + 1j * RNG.standard_normal(n))
    line = 1e-2 * np.exp(2j * np.pi * (24000 - H.FC) * t)
    r = H.block_levels(noise + line)
    assert abs(r["NAA"][0]) < 1.0
    r2 = H.block_levels(noise + msk(n, 24000 - H.FC, amp=3e-3))
    assert r2["NAA"][0] > 10.0


def test_dc_tone_at_zero_rf():
    """The 0 Hz RF tone sits at -FC in IQ; a -22.6 dBFS tone reads as such."""
    n = int(10 * H.FS)
    t = np.arange(n) / H.FS
    a = 10 ** (-22.6 / 20)
    x = a * np.exp(2j * np.pi * (-H.FC + 0.5) * t) + 1e-4 * RNG.standard_normal(n)
    assert H.block_levels(x)["DC"] == pytest.approx(-22.6, abs=0.3)


@pytest.mark.parametrize("ppm,lo", [(-0.7, 0.506), (2.0, -0.3)])
def test_msk_clock_recovers_ppm_and_lo(ppm, lo):
    """Recover a known clock error and LO offset to well inside the 0.5 ppm
    and 0.02 Hz quarantine limits from 120 s of strong MSK."""
    n = int(120 * H.FS)
    x = msk(n, 24000 - H.FC, clock_ppm=ppm, lo_hz=lo, amp=2e-3)
    x = x + 3e-4 * (RNG.standard_normal(n) + 1j * RNG.standard_normal(n))
    m = H._Mixer(24000 - H.FC)
    step = int(60 * H.FS)
    for n0 in range(0, n, step):
        m.push(n0, x[n0:n0 + step].astype(np.complex64))
    r = H.msk_clock(m.series())
    assert r["line_db"] > 20
    # the carrier offset includes the clock's share: -24 kHz * eps
    assert r["lo_hz"] == pytest.approx(lo - 24000 * ppm * 1e-6, abs=0.002)
    assert r["clock_ppm"] == pytest.approx(ppm, abs=0.15)


def test_mixer_is_seamless():
    """Decimating in slices gives the same series as one long slice."""
    n = int(4 * H.FS)
    x = (RNG.standard_normal(n) + 1j * RNG.standard_normal(n)).astype(np.complex64)
    a, b = H._Mixer(9000.0), H._Mixer(9000.0)
    a.push(0, x)
    h = n // 2
    b.push(0, x[:h])
    b.push(h, x[h:])
    assert np.allclose(a.series(), b.series(), atol=1e-5)


def msk500(secs, clock_ppm=-0.7, off_hz=0.5, amp=1.0, noise=0.0, seed=1,
           t_jump=None, dt_jump=0.0, dph_jump=0.0, fs2=500.0, baud=200.0):
    """MSK already at ~0 Hz and 500 S/s (what _Mixer hands msk_clock), so a
    600 s file costs 300 k samples.  Optional glitch at t_jump: symbol
    timing moves by dt_jump seconds and the carrier phase by dph_jump rad,
    the kind of event measured on 20261020 0800Z at 51 s."""
    rng = np.random.default_rng(seed)
    eps = clock_ppm * 1e-6
    n = int(secs * fs2)
    t_rx = np.arange(n) / fs2
    t = t_rx / (1 + eps)
    ph0 = np.zeros(n)
    if t_jump is not None:
        late = t_rx >= t_jump
        t = t + dt_jump * late
        ph0 = dph_jump * late
    nsym = int(t[-1] * baud) + 2
    b = rng.choice([-1.0, 1.0], nsym)
    k = (t * baud).astype(np.int64)
    frac = t * baud - k
    cum = np.concatenate([[0.0], np.cumsum(b)])[:nsym]
    ph = 2 * np.pi * off_hz * t_rx + np.pi / 2 * (cum[k] + b[k] * frac) + ph0
    y = amp * np.exp(1j * ph)
    if noise:
        y = y + noise * (rng.standard_normal(n) + 1j * rng.standard_normal(n)) / np.sqrt(2)
    return y.astype(np.complex64)


def test_msk_clock_survives_timing_glitch():
    """A 2 ms symbol-timing slip plus a 1 rad carrier jump at 51 s of a
    600 s file must not move the clock: the first version read +0.91 ppm
    on such a file (20261020 0800Z) whose true clock was -0.74."""
    y = msk500(600, clock_ppm=-0.74, noise=0.05, t_jump=51.0, dt_jump=2e-3, dph_jump=1.0)
    r = H.msk_clock(y)
    assert r["glitches"] >= 1
    assert r["clock_ppm"] == pytest.approx(-0.74, abs=0.05)
    assert abs(r["clock_ppm"] + 0.74) < 4 * r["clock_err_ppm"] + 0.02
    clean = H.msk_clock(msk500(600, clock_ppm=-0.74, noise=0.05))
    assert clean["glitches"] == 0
    assert clean["clock_ppm"] == pytest.approx(-0.74, abs=0.05)


def test_msk_clock_error_is_honest():
    """Over 16 noisy weak-line files the scatter of (estimate - truth) /
    clock_err_ppm must be about 1 (0.5..2): the first version's error sat
    4-8 sigma from the truth after slips."""
    pulls = []
    for seed in range(16):
        y = msk500(600, clock_ppm=-0.7, noise=1.2, seed=100 + seed)
        r = H.msk_clock(y)
        assert r["line_db"] > 15
        pulls.append((r["clock_ppm"] + 0.7) / r["clock_err_ppm"])
    sd = float(np.std(pulls))
    assert 0.5 < sd < 2.0, pulls


def test_find_step_abrupt_vs_gradual():
    """A 13 dB switch is a step; a 10 dB dip over 8 minutes is not."""
    s = np.full(60, -76.0) + 0.3 * RNG.standard_normal(60)
    s[25:] += 13
    d, t = H.find_step(s)
    assert d == pytest.approx(13, abs=1.0) and 240 <= t <= 260
    g = -76 - np.linspace(0, 10, 60) + 0.3 * RNG.standard_normal(60)
    assert H.find_step(g)[0] == 0.0
    # a 20 s burst (e.g. a sferic cluster) that comes back is not a state
    # change; find_step's 3-block medians need a level to hold >= 30 s
    b = np.full(60, -76.0)
    b[30:32] = -60
    assert H.find_step(b)[0] == 0.0


@pytest.mark.parametrize("sl", [slice(30, 32), slice(30, 33), slice(28, 32), slice(30, 35),
                                slice(57, 59), slice(1, 3), slice(0, 2), slice(58, 60)])
def test_find_step_rejects_bursts(sl):
    """20-50 s bursts mid-file and 20 s bursts at either edge are not
    steps (verifier probes: the first version called a 20 s burst in the
    last or first minute a 13 dB step, and a 40 s mid-file burst too)."""
    s = np.full(60, -76.0) + 0.3 * RNG.standard_normal(60)
    s[sl] += 13
    assert H.find_step(s)[0] == 0.0


def test_find_step_near_edges_and_nan():
    """A real switch 35 s before the file ends (11-06 0630Z at 565 s) is
    found; a NaN block next to a real step does not hide it."""
    s = np.full(60, -76.0)
    s[57:] += 13
    d, t = H.find_step(s)
    assert d == pytest.approx(13, abs=0.5) and t == pytest.approx(565, abs=10)
    for nan_at in (25, 33, 28):
        s = np.full(60, -76.0) + 0.3 * RNG.standard_normal(60)
        s[30:] += 13
        s[nan_at] = np.nan
        d, t = H.find_step(s)
        # the switch is on the 29/30 block boundary: either block is "k"
        assert d == pytest.approx(13, abs=1.0) and 290 <= t <= 310


def test_block_secs_is_used(tmp_path):
    """Times follow scan's block_secs, not the 10 s module constant."""
    n = 60
    blocks = {s: (np.full(n, 12.0), np.full(n, -76.0)) for s in H.LEVEL_STATIONS}
    blocks["NAA"][1][30:] += 13
    blocks["NAA"][1][9:12] -= 20        # 3-block dip, minimum at block 10
    blocks["NML"][1][30:] += 13
    blocks["NAU"][1][30:] += 13
    nanmsk = {"line_db": np.nan, "clock_ppm": np.nan, "clock_err_ppm": np.nan,
              "lo_hz": np.nan, "glitches": 0}
    sc = {"secs": 300.0, "clipped": 0, "blocks": blocks, "dc": np.full(n, -22.6),
          "msk": {"NAA": nanmsk, "NML": nanmsk}, "block_secs": 5.0}
    r = H.row_from_scan(tmp_path / "vlf_0000Z.cs16", {}, sc)
    assert r["NAA_step_t_s"] == pytest.approx(147.5, abs=5)
    assert 45 <= r["NAA_min_t_s"] <= 60      # blocks 9-11 at 5 s, centres
    assert r["step_kind"] == "common"


def test_quarantine_flags():
    row = {"clock_ppm": -0.7, "lo_hz": 0.506, "secs": 600.0, "clipped": 0,
           "dc_tone_dbfs": -22.6, "NAA_db": 15.0, "spur_minus_naa_db": -6.3,
           "NAU_db": 1.0, "NAA_step_db": 0.0}
    meta = {"fs_hz": 62500.0, "freq_hz": 15000.0, "agc": False, "IFGR": 40.0,
            "biasT_ctrl": "false", "overflows": 0}
    assert H.quarantine_flags(row, meta) == ([], [])
    bad = dict(row, clock_ppm=-1.3, lo_hz=0.53, secs=420.0, dc_tone_dbfs=-30.0,
               spur_minus_naa_db=-10.0, NAU_db=4.0)
    q, info = H.quarantine_flags(bad, dict(meta, agc=True))
    assert set(q) == {"clock", "lo", "short", "dc_tone", "sidecar_agc"}
    assert set(info) == {"spur_off", "nau_night"}
    q, info = H.quarantine_flags(dict(row, clock_ppm=np.nan, lo_hz=np.nan), meta)
    assert "no_clock" in info and "clock" not in q


def test_corpus_states():
    rows = [{"NAA_dbfs": v, "NAA_step_db": 0.0, "NAA_min_dbfs": v} for v in
            (-76, -75.5, -76.5, -77, -63, -64)]
    rows.append({"NAA_dbfs": -80, "NAA_step_db": 0.0, "NAA_min_dbfs": -92})
    ref = H.corpus_states(rows)
    assert ref == pytest.approx(-76.0, abs=0.6)
    assert [r["naa_flag"] for r in rows] == [0, 0, 0, 0, 1, 1, 0]
    assert [r["naa_fade"] for r in rows] == [0, 0, 0, 0, 0, 0, 1]


def test_capture_health_end_to_end(tmp_path):
    """Full path: a 90 s synthetic file with NAA, NML, a 0 Hz tone and a
    13 dB NAA step comes back with the right clock, LO, tone and step,
    and is flagged short (it is not 600 s)."""
    n = int(90 * H.FS)
    t = np.arange(n) / H.FS
    naa = msk(n, 24000 - H.FC, clock_ppm=-0.7, lo_hz=0.5, amp=1e-3, seed=2)
    naa[int(40 * H.FS):] *= 10 ** (13 / 20)
    nml = msk(n, 25200 - H.FC, clock_ppm=-0.7, lo_hz=0.5, amp=4e-4, seed=3)
    tone = 10 ** (-22.6 / 20) * np.exp(2j * np.pi * (-H.FC + 0.5) * t)
    x = naa + nml + tone + 2e-4 * (RNG.standard_normal(n) + 1j * RNG.standard_normal(n))
    p = tmp_path / "vlf_0000Z.cs16"
    write_cs16(p, x)
    p.with_suffix(".json").write_text(json.dumps({"fs_hz": 62500.0, "freq_hz": 15000.0,
                                                  "agc": False, "IFGR": 40.0}))
    r = H.capture_health(p)
    assert r["clock_src"] == "NAA"
    assert r["clock_ppm"] == pytest.approx(-0.7, abs=0.5)
    assert r["lo_hz"] == pytest.approx(0.5 + 24000 * 0.7e-6, abs=0.005)
    assert r["dc_tone_dbfs"] == pytest.approx(-22.6, abs=0.5)
    assert r["NAA_step_db"] == pytest.approx(13, abs=1.5)
    assert r["NAA_step_t_s"] == pytest.approx(35, abs=10)
    assert "short" in r["quarantine"].split(";")
    assert "naa_step" in r["info"].split(";")


def test_absent_station_makes_no_step_or_fade():
    """NAA off the air reads -300 dBFS; that must be NaN, not a 200 dB step."""
    over = np.r_[np.full(30, 12.0), np.full(30, 0.0)]
    dbfs = np.r_[np.full(30, -76.0), np.full(30, -300.0)]
    p = H.present_dbfs(over, dbfs)
    assert np.isnan(p[30:]).all()
    assert H.find_step(p)[0] == 0.0


def test_step_kind():
    assert H.step_kind(13.0, 12.0, 11.0) == "common"
    assert H.step_kind(13.0, 0.5, -1.0) == "naa_only"
    # one present witness that moved with NAA is enough for 'common'
    assert H.step_kind(13.0, np.nan, 12.0) == "common"
    assert H.step_kind(13.0, np.nan, 1.0) == "unclear"
    assert H.step_kind(0.0, 1.0, 1.0) == ""


def test_noisy_clock_is_not_quarantined():
    """A 1.3 ppm deviation with a 1 ppm standard error is noise, not a fault;
    the same deviation measured to 0.1 ppm is quarantined."""
    row = {"clock_ppm": -2.0, "clock_err_ppm": 1.0, "lo_hz": 0.506, "secs": 600.0,
           "clipped": 0, "dc_tone_dbfs": -22.6, "NAA_db": 15.0,
           "spur_minus_naa_db": -6.3, "NAU_db": 1.0, "NAA_step_db": 0.0}
    meta = {"agc": False, "IFGR": 40.0}
    q, info = H.quarantine_flags(row, meta)
    assert "clock" not in q and "clock_noisy" in info
    q, info = H.quarantine_flags(dict(row, clock_err_ppm=0.1), meta)
    assert "clock" in q
