"""Fast tests for sferic_watch.mainsgate on SYNTHETIC impulse times only (no captures)."""
import numpy as np
import pytest

from sferic_watch import mainsgate as mg

F = 120.0
P = 1 / F


def buzz_times(rng, secs, rate, phase_s, jitter_s=40e-6, t0=0.0, drift=None, f=F):
    """Times of a sparse mains train: ``rate`` pulses/s picked at random from the 120 Hz grid starting t0 + phase_s,
    each with Gaussian jitter (default 40 us rms, the measured 37-44 us). ``drift(t)`` adds a slow phase wander (s)."""
    k = np.arange(int(secs * f))
    k = np.sort(rng.choice(k, int(rate * secs), replace=False))
    t = t0 + phase_s + k / f + rng.normal(0, jitter_s, k.size)
    if drift is not None:
        t = t + drift(t)
    return t


def poisson(rng, secs, rate, t0=0.0):
    return np.sort(t0 + rng.uniform(0, secs, rng.poisson(rate * secs)))


def block(t0=0.0, dur=10.0, f=F, hum=True):
    return {"t0_s": t0, "dur_s": dur, "hum": hum, "f_mains": f}


# ----------------------------------------------------------------------------- building blocks

def test_busiest_window_wraps_round_the_circle():
    """Phases straddling 0 / P are one window, not two half windows."""
    ph = np.array([P - 50e-6, P - 10e-6, 20e-6, 60e-6, 3e-3])
    c, n = mg.busiest_window(ph, P)
    assert n == 4
    d = (c + P / 2) % P - P / 2
    assert abs(d) < 0.2e-3


def test_hot_p_is_a_tail_probability():
    """Expected count gives p near 1; 10 of 30 in one 0.4 ms window is far beyond chance (expected 1.4)."""
    assert mg.hot_p(2, 30, F) == 1.0
    assert mg.hot_p(10, 30, F) < 1e-4
    assert mg.n_positions(P) == 84


@pytest.mark.parametrize("n", [20, 100, 600])
def test_scan_false_alarm_is_at_or_under_the_stated_level(n):
    """Verifier finding: the old 20.8-window factor under-stated the busiest-window scan 3-7x. With the union bound over
    every tested position, uniform phases must pass a nominal 1e-2 at most 1e-2 of the time (4000 trials: allow 1.5e-2
    for sampling scatter; the old code measured 2.3-5.3e-2 here)."""
    rng = np.random.default_rng(n)
    hits = 0
    for _ in range(4000):
        ph = rng.uniform(0, P, n)
        c, cnt = mg.busiest_window(ph, P)
        hits += mg.hot_p(cnt, n, F) < 1e-2
    assert hits / 4000 <= 1.5e-2


def test_peeled_arc_raises_q():
    """Verifier finding: after a train's arc is removed the free starts are denser on what is left, so the same count
    must be less surprising."""
    assert mg.hot_p(8, 40, F, removed=0.8e-3) > mg.hot_p(8, 40, F)


def test_refine_f_finds_the_train_frequency():
    rng = np.random.default_rng(1)
    t = buzz_times(rng, 10, 3, 2e-3, f=120.004)
    f, r = mg.refine_f(t, 120.0)
    assert abs(f - 120.004) <= 0.0006 and r > 0.9


# ----------------------------------------------------------------------------- false alarms (no mains at all)

def test_random_blocks_rarely_make_a_train():
    """Uniformly random starts in 3000 flagged-looking blocks (N 8-60): the family-wise level is 1e-4 per fold search,
    so 3000 blocks should give about 0.3 false first trains; allow up to 3."""
    rng = np.random.default_rng(2)
    false = 0
    for i in range(3000):
        t = poisson(rng, 10, rng.uniform(0.8, 6.0))
        false += len(mg.find_trains(t, np.full(t.size, 12.0), 0, 10, 120.0)) > 0
    assert false <= 3


@pytest.mark.parametrize("rate", [40, 80, 120])
def test_poisson_false_rejection_with_the_gate_forced_on(rate):
    """Plan TEST: Poisson-only impulses at 40/80/120 per s lose <= 2 %, even if every block is (wrongly) flagged."""
    rng = np.random.default_rng(rate)
    t = poisson(rng, 60, rate)
    blocks = [block(a) for a in range(0, 60, 10)]
    rej, esc, trains = mg.gate(t, rng.normal(13, 3, t.size), blocks)
    assert rej.mean() <= 0.02


# ----------------------------------------------------------------------------- trains

def test_single_train_rejected_sferics_kept():
    """A sparse train (3/s, 40 us jitter) inside 5/s Poisson sferics: >= 90 % of the train rejected, and only the
    sferics that land inside the +-0.2 ms window (4.8 %) may be lost."""
    rng = np.random.default_rng(3)
    tb = buzz_times(rng, 10, 3, 2.5e-3)
    ts = poisson(rng, 10, 5)
    t = np.concatenate([tb, ts])
    is_b = np.r_[np.ones(tb.size, bool), np.zeros(ts.size, bool)]
    o = np.argsort(t)
    t, is_b = t[o], is_b[o]
    pk = np.where(is_b, 11.0, 14.0)
    rej, esc, trains = mg.gate(t, pk, [block()])
    assert len(trains) >= 1
    assert rej[is_b].mean() >= 0.9
    assert rej[~is_b].mean() <= 0.12


def test_second_train_is_peeled():
    """11-04 01Z shape: two trains 4.4 ms apart (3/s and 1.2/s). Both are found and both are rejected."""
    rng = np.random.default_rng(4)
    a = buzz_times(rng, 10, 3, 3.0e-3)
    b = buzz_times(rng, 10, 1.2, 7.4e-3)
    s = poisson(rng, 10, 1.0)
    t = np.concatenate([a, b, s])
    lab = np.r_[np.zeros(a.size), np.ones(b.size), 2 * np.ones(s.size)]
    o = np.argsort(t)
    t, lab = t[o], lab[o]
    rej, _, trains = mg.gate(t, np.full(t.size, 11.0), [block()])
    assert len(trains) >= 2
    for phase in (3.0e-3, 7.4e-3):                    # a train sits on each true phase (checked at t = 5 s)
        t_true = np.array([5.0 + phase])
        assert any(abs(tr.residual(t_true)[0]) < 0.2e-3 for tr in trains)
    assert rej[lab == 0].mean() >= 0.9 and rej[lab == 1].mean() >= 0.8


def test_phase_wander_is_tracked():
    """A 0.45 ms phase wander over the 10 s block (as measured on 11-04 01Z) is followed by the 1 s tracker:
    >= 90 % of the train is still rejected."""
    rng = np.random.default_rng(5)
    tb = buzz_times(rng, 10, 3, 1.0e-3, drift=lambda t: 0.45e-3 * np.sin(2 * np.pi * t / 20))
    rej, _, trains = mg.gate(tb, np.full(tb.size, 11.0), [block()])
    assert trains and rej.mean() >= 0.9


def test_amplitude_escape():
    """An impulse on the train is kept when its peak is >= 6 dB over the train's median, rejected when 4 dB."""
    rng = np.random.default_rng(6)
    tb = buzz_times(rng, 10, 3, 2e-3)
    pk = np.full(tb.size, 11.0)
    pk[5], pk[9] = 17.5, 15.0
    rej, esc, _ = mg.gate(tb, pk, [block()])
    assert not rej[5] and esc[5]
    assert rej[9] and not esc[9]


def test_unflagged_blocks_are_untouched_without_self_test():
    """The plan's rule (self_test=False): an unflagged block is never searched."""
    rng = np.random.default_rng(7)
    tb = buzz_times(rng, 20, 3, 2e-3)
    rej, _, trains = mg.gate(tb, np.full(tb.size, 11.0), [block(0, hum=False), block(10, hum=True)], self_test=False)
    assert not rej[tb < 10].any()
    assert rej[tb >= 10].mean() >= 0.9
    assert all(tr.t_ref == 10 for tr in trains)


def test_self_test_catches_buzz_the_flag_missed():
    """11-05 04:30Z shape: 2.5 buzz impulses/s kept by the detector in a block humflag did not flag (frequency
    unknown, 120.004 Hz here). The self test finds it and rejects >= 90 %; Poisson sferics may lose the window
    share (4.8 %, allow 12 % on 34 sferics)."""
    rng = np.random.default_rng(17)
    tb = buzz_times(rng, 10, 2.5, 3.1e-3, f=120.004)
    ts = poisson(rng, 10, 3)
    t = np.concatenate([tb, ts])
    is_b = np.r_[np.ones(tb.size, bool), np.zeros(ts.size, bool)]
    o = np.argsort(t)
    t, is_b = t[o], is_b[o]
    rej, _, trains = mg.gate(t, np.full(t.size, 11.0), [block(0, hum=False, f=np.nan)])
    assert trains and all(tr.kind == "self" for tr in trains)
    assert rej[is_b].mean() >= 0.9 and rej[~is_b].mean() <= 0.12


def test_self_test_quiet_on_random_blocks():
    """2000 unflagged blocks of Poisson starts at 1-30/s: the self test (level 1e-5) finds no train."""
    rng = np.random.default_rng(18)
    false = 0
    for i in range(2000):
        t = poisson(rng, 10, rng.uniform(1, 30))
        false += len(mg.find_trains(t, np.full(t.size, 12.0), 0, 10, mg.SELF_F0, p_hot=mg.P_SELF,
                                    span=mg.SELF_SPAN, step=mg.SELF_STEP)) > 0
    assert false == 0


def test_partial_last_block_is_gated_to_its_end():
    """Verifier finding: coverage used to stop at the last 1 s track step + 0.5 s, so a 7.3 s block lost its last
    0.3 s. Train impulses in 7.0-7.3 s are now rejected too."""
    rng = np.random.default_rng(19)
    tb = buzz_times(rng, 7.3, 4, 2e-3)
    tb = tb[tb < 7.3]
    rej, _, trains = mg.gate(tb, np.full(tb.size, 11.0), [block(0, dur=7.3)])
    late = tb >= 7.0
    assert late.any() and rej[late].all()


def test_too_few_impulses_no_fold():
    t = np.array([1.0, 1 + P, 1 + 2 * P])
    assert mg.find_trains(t, np.full(3, 11.0), 0, 10, 120.0) == []


def test_gated_seconds_is_window_share():
    rng = np.random.default_rng(8)
    tb = buzz_times(rng, 10, 4, 2e-3)
    _, _, trains = mg.gate(tb, np.full(tb.size, 11.0), [block()])
    assert len(trains) == 1
    assert mg.gated_seconds(trains) == pytest.approx(10 * 0.4e-3 * trains[0].f, rel=1e-6)


# ----------------------------------------------------------------------------- edge-clocked mode (pipeline default)

def dense_edges(rng, secs, phase_s, drift, rate=100.0):
    """The hum flag's edges: a dense train (``rate``/s) on the same mains clock, 40 us jitter, plus 1/s random."""
    return np.sort(np.r_[buzz_times(rng, secs, rate, phase_s, drift=drift), poisson(rng, secs, 1.0)])


def test_segments_split_runs_and_cap_length():
    bl = [block(10 * i, hum=h) for i, h in enumerate([1, 1, 0, 1, 1, 1, 1, 1, 1, 1, 1])]
    segs = mg.segments(bl, max_blocks=6)
    assert [len(s) for s in segs] == [2, 6, 2]


def test_sparse_train_found_with_edge_clock():
    """11-03 05Z shape: only 0.5 buzz impulses/s survive excision (5 per 10 s block: no block can pass P_HOT on its
    own), with a 0.45 ms phase wander. With the dense edge clock over 60 s the train is found and >= 85 % of it is
    rejected; the detector train sits 0.15 ms after the edges (different thresholds), which the fold absorbs."""
    rng = np.random.default_rng(9)
    drift = lambda t: 0.45e-3 * np.sin(2 * np.pi * t / 25)
    tb = buzz_times(rng, 60, 0.5, 2.15e-3, drift=drift)
    ts = poisson(rng, 60, 1.0)
    t = np.concatenate([tb, ts])
    is_b = np.r_[np.ones(tb.size, bool), np.zeros(ts.size, bool)]
    o = np.argsort(t)
    t, is_b = t[o], is_b[o]
    blocks = [block(a) for a in range(0, 60, 10)]
    rej0, _, tr0 = mg.gate(t, np.full(t.size, 11.0), blocks)              # no edges: per-block, too thin
    edges = dense_edges(rng, 60, 2.0e-3, drift)
    rej, _, trains = mg.gate(t, np.full(t.size, 11.0), blocks, edges=edges)
    assert rej0[is_b].mean() < 0.5
    assert trains and rej[is_b].mean() >= 0.85
    assert rej[~is_b].mean() <= 0.1


def test_clocked_false_trains_at_or_under_level():
    """Verifier finding (7e-4 measured for a stated 1e-4): pure Poisson starts against a real edge clock, 60 s
    segments at 3-40/s, searched at a loose level 1e-2 so that false trains can be counted: at most 1e-2 of 600
    searches may find one (allow 12 for scatter)."""
    rng = np.random.default_rng(20)
    edges = dense_edges(rng, 60, 1e-3, None)
    clocks = [mg.clock_from_edges(edges, a, a + 10, 120.0) for a in range(0, 60, 10)]
    false = 0
    for i in range(600):
        t = poisson(rng, 60, rng.uniform(3, 40))
        false += len(mg.find_trains_clocked(t, np.full(t.size, 12.0), clocks, p_hot=1e-2)) > 0
    assert false <= 12


def test_high_rate_poisson_does_not_spawn_extra_trains():
    """Verifier adversarial case: one real train (2.5/s) plus Poisson 300/s over a 60 s edge-clocked segment used to
    give 2.2 trains per block and 8.5 % sferic loss. Now: one train, sferic loss about one window share (4.8 %)."""
    rng = np.random.default_rng(21)
    tb = buzz_times(rng, 60, 2.5, 2.0e-3)
    ts = poisson(rng, 60, 300)
    t = np.concatenate([tb, ts])
    is_b = np.r_[np.ones(tb.size, bool), np.zeros(ts.size, bool)]
    o = np.argsort(t)
    t, is_b = t[o], is_b[o]
    edges = dense_edges(rng, 60, 2.0e-3, None)
    rej, _, trains = mg.gate(t, np.full(t.size, 11.0), [block(a) for a in range(0, 60, 10)], edges=edges)
    seg = [tr for tr in trains if tr.kind == "segment"]
    assert len(seg) == 6                                  # one train, one copy per block
    assert rej[~is_b].mean() <= 0.06


def burst_scene(rng, secs=60.0, every=3.0, n_pulse=8, rate_s=5.0):
    """11-05 12:30Z shape: bursts of ``n_pulse`` pulses at 2 x 8.33 ms, each tight (25 us) but at its own phase on
    the mains clock (measured there: 5.5-7.5 ms, moving burst to burst; here anywhere on the circle, so that no fixed
    60 s fold window can hold it), one every ~``every`` s; plus Poisson sferics. Returns (t, is_burst)."""
    tb = []
    for t0 in np.arange(0.5, secs - 1, every) + rng.uniform(0, 1, int(np.ceil((secs - 1.5) / every))):
        ph = rng.uniform(0, P)
        k0 = np.floor(t0 * F)
        tb.append((k0 + 2 * np.arange(n_pulse)) / F + ph + rng.normal(0, 25e-6, n_pulse))
    tb = np.concatenate(tb)
    ts = poisson(rng, secs, rate_s)
    t = np.concatenate([tb, ts])
    is_b = np.r_[np.ones(tb.size, bool), np.zeros(ts.size, bool)]
    o = np.argsort(t)
    return t[o], is_b[o]


def test_bursts_at_wandering_phase_are_gated():
    """The 60 s fold cannot hold a source whose phase moves from burst to burst; the 1 s burst pass does. Without it
    under half the burst pulses go; with it >= 80 %, and Poisson sferics lose <= 5 %."""
    rng = np.random.default_rng(22)
    t, is_b = burst_scene(rng)
    edges = dense_edges(rng, 60, 1e-3, None)
    blocks = [block(a) for a in range(0, 60, 10)]
    rej0, _, _ = mg.gate(t, np.full(t.size, 11.0), blocks, edges=edges, bursts=False)
    rej, _, trains = mg.gate(t, np.full(t.size, 11.0), blocks, edges=edges)
    assert rej0[is_b].mean() < 0.5
    assert rej[is_b].mean() >= 0.8
    assert rej[~is_b].mean() <= 0.05


@pytest.mark.parametrize("rate", [40, 80, 120])
def test_poisson_false_rejection_edge_clock(rate):
    """Poisson-only detector impulses, every block flagged and a real edge clock present: <= 2 % rejected."""
    rng = np.random.default_rng(100 + rate)
    t = poisson(rng, 60, rate)
    edges = dense_edges(rng, 60, 1e-3, None)
    rej, _, _ = mg.gate(t, rng.normal(13, 3, t.size), [block(a) for a in range(0, 60, 10)], edges=edges)
    assert rej.mean() <= 0.02


# ----------------------------------------------------------------------------- clock independence, scope, whole fold

def test_start_free_drops_edges_on_starts():
    e = np.array([1.0, 2.0, 3.0, 4.0005, 5.0])
    t = np.array([2.0004, 4.0])
    np.testing.assert_array_equal(mg.start_free(e, t), [1.0, 3.0, 5.0])


def test_clocked_scopes():
    flagged = {"hum": True}
    locked = {"hum": False, "n_edges": 40, "R": 0.5, "r_crit": 0.3}
    quiet = {"hum": False, "n_edges": 40, "R": 0.2, "r_crit": 0.3}
    assert [mg.clocked(b, "flagged") for b in (flagged, locked, quiet)] == [True, False, False]
    assert [mg.clocked(b, "locked") for b in (flagged, locked, quiet)] == [True, True, False]
    assert [mg.clocked(b, "all") for b in (flagged, locked, quiet)] == [True, True, True]


def test_scope_all_quiet_when_no_mains_even_if_edges_are_the_sferics():
    """No mains anywhere: the edges are the sferics themselves (every start also makes an edge) plus other random
    edges, as on a clean day. With start_free the clock is independent of the starts, so 100 x 60 s of 2-40/s Poisson
    starts give no train at the default levels (expected about 100 x 17 searches x 1e-4 = 0.2)."""
    rng = np.random.default_rng(30)
    false = 0
    for i in range(100):
        t = poisson(rng, 60, rng.uniform(2, 40))
        e = np.sort(np.r_[t + rng.normal(0, 20e-6, t.size), poisson(rng, 60, 5)])
        blocks = [block(a, hum=False, f=np.nan) for a in range(0, 60, 10)]
        _, _, trains = mg.gate(t, np.full(t.size, 12.0), blocks, edges=e)
        false += len(trains) > 0
    assert false <= 1


def test_clock_fitted_to_the_starts_would_flatter_them():
    """Why start_free exists: if the edges ARE the starts (a sferic makes both), a clock fitted to them puts its phase
    on the busiest start cluster and the fold test is no longer honest. Measured here: with the starts left in the
    edges, sparse blocks (1.5/s) give clearly more false trains at a loose level than with start_free."""
    rng = np.random.default_rng(31)
    hits_raw = hits_free = 0
    for i in range(300):
        t = poisson(rng, 60, 1.5)
        e = np.sort(np.r_[t, poisson(rng, 60, 1.0)])
        clocks_raw = [mg.clock_from_edges(e, a, a + 10, 120.0) for a in range(0, 60, 10)]
        clocks_free = [mg.clock_from_edges(mg.start_free(e, t), a, a + 10, 120.0) for a in range(0, 60, 10)]
        pk = np.full(t.size, 12.0)
        hits_raw += len(mg.find_trains_clocked(t, pk, [c for c in clocks_raw if c], p_hot=0.05)) > 0
        hits_free += len(mg.find_trains_clocked(t, pk, [c for c in clocks_free if c], p_hot=0.05)) > 0
    assert hits_free <= 0.05 * 300 * 1.6
    assert hits_raw > 2 * max(hits_free, 1)


def test_whole_fold_finds_a_very_sparse_train():
    """11-05 20Z shape: 0.06 buzz impulses/s over 300 s (under 4 per 60 s segment) among 1/s Poisson sferics. Only the
    whole-capture fold finds it; >= 80 % of the train goes, and the sferics lose about one window share."""
    rng = np.random.default_rng(32)
    tb = buzz_times(rng, 300, 0.06, 3.3e-3)
    ts = poisson(rng, 300, 1.0)
    t = np.concatenate([tb, ts])
    is_b = np.r_[np.ones(tb.size, bool), np.zeros(ts.size, bool)]
    o = np.argsort(t)
    t, is_b = t[o], is_b[o]
    edges = dense_edges(rng, 300, 3.1e-3, None)
    blocks = [block(a) for a in range(0, 300, 10)]
    rej0, _, tr0 = mg.gate(t, np.full(t.size, 11.0), blocks, edges=edges, whole=False)
    rej, _, trains = mg.gate(t, np.full(t.size, 11.0), blocks, edges=edges)
    assert rej0[is_b].mean() < 0.3
    assert any(tr.kind == "whole" for tr in trains)
    assert rej[is_b].mean() >= 0.8
    assert rej[~is_b].mean() <= 0.08


# ----------------------------------------------------------------------------- honest clock: edges on sferic tails

def tail_scene(rng, rate, secs=60.0, delays=None, noise_rate=0.0):
    """No mains: Poisson sferic starts, each impulse 3.2 ms long, each making an edge at its start and one on its
    tail at a delay fixed by its waveform (20 shapes, 1.2-2.8 ms, like the synthetic bank), plus independent noise
    edges at ``noise_rate``/s. Returns (starts, ends, edges)."""
    if delays is None:
        delays = np.random.default_rng(99).uniform(1.2e-3, 2.8e-3, 20)
    t = poisson(rng, secs, rate)
    k = rng.integers(0, delays.size, t.size)
    e = np.sort(np.r_[t + 20e-6, t + delays[k], poisson(rng, secs, noise_rate)])
    return t, t + 3.2e-3, e


def test_extent_free_drops_edges_inside_impulses():
    e = np.array([0.5, 1.0005, 1.0025, 1.0045, 1.0065, 2.0])
    t, te = np.array([1.0]), np.array([1.0045])
    np.testing.assert_array_equal(mg.extent_free(e, t, te), [0.5, 1.0065, 2.0])
    # without ends each impulse counts as EXTENT_DEFAULT_S (1 ms) long, plus the 1 ms guard: [0.999, 1.002]
    np.testing.assert_array_equal(mg.extent_free(e, t), [0.5, 1.0025, 1.0045, 1.0065, 2.0])


def test_tail_edges_bias_start_free_but_not_extent_free():
    """The review's high finding, on synthetic times: edges more than 1 ms from a start but on the sferic's own tail
    survive ``start_free``, and a clock fitted to them over-states the fold (harness Poisson 5/s lost 9 %). Searched
    at a loose 0.05 over 6 x 10 s, the start_free clock gives clearly more false segments than 0.05 x 200; with
    ``extent_free`` (independent noise edges left only) the count stays at the level (allow 1.6x for scatter)."""
    rng = np.random.default_rng(40)
    n_old = n_new = 0
    for i in range(200):
        t, te, e = tail_scene(rng, rng.uniform(1.5, 10), noise_rate=20.0)
        pk = np.full(t.size, 12.0)
        for free, add in ((mg.start_free(e, t), 0), (mg.extent_free(e, t, te), 1)):
            cks = [mg.clock_from_edges(free, a, a + 10, 120.0, span=mg.SELF_SPAN) for a in range(0, 60, 10)]
            hit = len(mg.find_trains_clocked(t, pk, [c for c in cks if c], p_hot=0.05)) > 0
            n_old += hit and not add
            n_new += hit and add
    assert n_new <= 0.05 * 200 * 1.6
    assert n_old > 1.5 * max(n_new, 1)


@pytest.mark.parametrize("rate", [2, 5, 10, 20, 40, 80, 120, 200])
def test_gate_quiet_on_poisson_with_tail_edges(rate):
    """Default gate (scope all, bursts, whole fold, self test) on 120 s of mains-free Poisson starts whose edges are
    their own tails plus 5/s noise, 3 seeds per rate: <= 1 % of starts rejected in every run."""
    for seed in range(3):
        rng = np.random.default_rng(1000 * rate + seed)
        t, te, e = tail_scene(rng, rate, secs=120.0, noise_rate=5.0)
        blocks = [block(a, hum=False, f=np.nan) for a in range(0, 120, 10)]
        rej, _, _ = mg.gate(t, rng.normal(14, 3, t.size), blocks, edges=e, t_end=te)
        assert rej.mean() <= 0.01


def test_free_search_on_the_scan_edge_gives_no_train():
    """A periodic source at 120.08 Hz is not mains here (measured 119.96-120.008 Hz): the self test's best f lands on
    the 120.05 Hz edge of its grid and no train is made; the same source at 120.004 Hz is found."""
    rng = np.random.default_rng(41)
    for f, want in ((120.08, False), (120.004, True)):
        tb = buzz_times(rng, 10, 3, 2e-3, f=f)
        tr = mg.find_trains(tb, np.full(tb.size, 11.0), 0, 10, mg.SELF_F0, p_hot=mg.P_SELF, span=mg.SELF_SPAN,
                            step=mg.SELF_STEP)
        assert bool(tr) is want


def test_whole_train_needs_starts_in_three_blocks():
    """A cluster confined to 2 of 12 clocked blocks is not a 'whole' train (min_blocks=3); spread over 6 it is."""
    rng = np.random.default_rng(42)
    edges = dense_edges(rng, 120, 1e-3, None)
    clocks = [mg.clock_from_edges(edges, a, a + 10, 120.0) for a in range(0, 120, 10)]
    for span_blocks, want in ((2, False), (6, True)):
        tb = buzz_times(rng, 10 * span_blocks, 1.5, 3e-3)
        t = np.sort(np.r_[tb, poisson(rng, 120, 1.0)])
        tr = mg.find_trains_clocked(t, np.full(t.size, 11.0), clocks, min_blocks=mg.WHOLE_MIN_BLOCKS, kind="whole")
        assert bool(tr) is want


def test_unlocked_noise_edges_fall_back_to_the_self_test():
    """An un-flagged block whose edges are only noise (not phase locked) gets no clock, so its starts still get the
    self test: 2.5 buzz impulses/s at 120.004 Hz are found and rejected."""
    rng = np.random.default_rng(43)
    tb = buzz_times(rng, 10, 2.5, 3.1e-3, f=120.004)
    t = np.sort(np.r_[tb, poisson(rng, 10, 2.0)])
    e = poisson(rng, 10, 30.0)
    assert mg.clock_from_edges(mg.extent_free(e, t), 0, 10, mg.SELF_F0, span=mg.SELF_SPAN, p_lock=mg.P_CLOCK) is None
    rej, _, trains = mg.gate(t, np.full(t.size, 11.0), [block(0, hum=False, f=np.nan)], edges=e)
    assert trains and all(tr.kind == "self" for tr in trains)
    assert rej[np.isin(t, tb)].mean() >= 0.9


def test_event_flagged_block_never_uses_the_event_frequency():
    """A block the hum flag marked from the detector's own starts (hum_edge False) has f fitted to those starts:
    ``edge_flagged`` is False, so the gate searches it with the self test's frequency trials."""
    b = dict(block(0, hum=True, f=120.0), hum_edge=False)
    assert not mg.edge_flagged(b)
    assert mg.edge_flagged(block(0, hum=True)) and not mg.edge_flagged(dict(block(0), hum_edge=False, hum=False))
