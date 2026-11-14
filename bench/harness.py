"""One scoreboard for sferic detectors on the frozen corpus (bench/corpus.json).

Usage (heavy runs: wrap in systemd-run --user --scope -q -p MemoryMax=4G nice -n 10):
  python bench/harness.py inject  --detector july onair_bp [--set native real_snr synth] [--snr 12 18 24 30]
  python bench/harness.py poisson --detector july onair_bp [--rates 5 10 20 40 60 80 120 200]
  python bench/harness.py rule                         # model of OnAir's mains rule on pure Poisson times (no signal)
  python bench/harness.py corpus  --detector july onair_bp [--workers 2]
  python bench/harness.py all     --detector july onair_bp --out results.json
  python bench/harness.py label   --write              # rough hum on/off blocks into corpus.json (from the july pass)
  python bench/harness.py list                         # registered detectors

What it measures, and why each number exists:
  inject  chance-corrected recall within +-2 ms of injected pulses on six 120 s pieces of real air (200-320 s of six
          captures: 1 clean, 3 steady hum, 2 intermittent). Chance = the same hit test at 4 offsets with no pulse
          (+37.3, +211.1, +529.7, -314.1 ms, as h2h), corrected = (recall - chance)/(1 - chance). Without the
          correction a detector that fires 100/s on mains buzz scores 0.4 "recall" by luck (h2h: A on hum, chance 0.39).
          Sets: native (h2h reproduction), real_snr and synth at 12/18/24/30 dB peak SNR.
  poisson synthetic noise host (clean-day floor, receiver tone) with pulses at Poisson rates and NO mains. Anything a
          gated detector loses relative to its ungated twin is a false mains rejection.
  rule    a behavioural model of OnAir's train-vote mains rule (written from a prose description, not from its
          source) on bare Poisson times: 4.2/13.3/17.7% rejected at 40/60/80 per s (the earlier mains_sim.py study
          read 4.3/13.5/17.9%).
  corpus  every capture end to end: events/min, residual 120 Hz comb share (share of inter-event intervals within
          0.6 ms of a multiple of 8.333 ms; chance 1.2/8.333 = 0.144), 10-14 kHz cross-band confirmation share, CPU s
          per minute of signal.

Detectors plug in through sferic_watch.register (see sferic_watch/__init__.py). Baselines registered here:
  july             the July sferic_rate rule (A in h2h): 30 s chunks, shift to real, 4th-order 2.5-9 kHz sosfiltfilt,
                   |y| over median + 8 MAD-sigma, rising edges, 8 ms dead time.
  onair_bp         OnAir's SfericsReceiver (C++ libdect2core.a, the h2h harness binary) behind our causal 4th-order
                   2.5-9 kHz band-pass, mains filter on, default threshold - exactly the h2h "OnAir + our BP" run.
  onair_bp_nomains the same with OnAir's mains filter off (the ungated twin of onair_bp).
  onair_raw        the binary as shipped (complex IQ, no band selection): the regression guard that should score ~0.
The binary is GPL (OnAir); it is built outside this repo and run as an external program only. Path: $SFERIC_ONAIR_BIN,
else onair_sferics_h2h on $PATH, else corpus.json "onair_bin". When it is absent the onair_* baselines are skipped with
a message and everything else still runs.
Temporary .cs16 files go to $SFERIC_TMP (default /var/tmp: /tmp is tmpfs on this box and would hold 150 MB in RAM).
"""
import argparse
import importlib
import json
import os
import pkgutil
import resource
import shutil
import subprocess
import sys
import tempfile
import time

import numpy as np
from scipy.signal import butter, sosfilt, sosfiltfilt

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)

import sferic_watch as sw                       # noqa: E402
from sferic_watch import io as swio             # noqa: E402
import inject as inj                            # noqa: E402

FS, FC = 62500.0, 15000.0
TOL = 0.002                                     # hit window, s
CHANCE_OFFSETS = (0.0373, 0.2111, 0.5297, -0.3141)
MAINS_HALF = 1 / 120.0
COMB_WIN = 0.0006
COMB_CHANCE = 2 * COMB_WIN / MAINS_HALF         # 0.144
CHUNK_S = 30.0


# ================================================================ scoring
def hits(det, truth, tol=TOL):
    """Bool per truth time: is any detection within +-tol?"""
    det = np.sort(np.asarray(det, float))
    truth = np.asarray(truth, float)
    if det.size == 0:
        return np.zeros(truth.size, bool)
    i = np.searchsorted(det, truth)
    d = np.full(truth.size, np.inf)
    m = i < det.size
    d[m] = np.abs(det[i[m]] - truth[m])
    m = i > 0
    d[m] = np.minimum(d[m], np.abs(det[i[m] - 1] - truth[m]))
    return d <= tol


def corrected_recall(det, truth, tol=TOL, offsets=CHANCE_OFFSETS):
    """(recall, chance, corrected): chance = mean hit rate at the truth times shifted by ``offsets`` (no pulse there)."""
    if len(truth) == 0:
        return float("nan"), float("nan"), float("nan")       # nothing injected: recall is undefined, say so quietly
    rec = float(hits(det, truth, tol).mean())
    ch = float(np.mean([hits(det, np.asarray(truth) + o, tol).mean() for o in offsets]))
    corr = (rec - ch) / (1 - ch) if ch < 1 else 0.0
    return rec, ch, corr


def extra_rate(det, truth, secs, tol=0.003):
    """Detections per second not within tol (3 ms) of any injected pulse."""
    det = np.asarray(det, float)
    if det.size == 0:
        return 0.0
    return float(np.sum(~hits(np.asarray(truth, float), det, tol)) / secs)


def comb_share(times, min_n=10):
    """Share of inter-event intervals within 0.6 ms of a multiple of 8.333 ms (storm_watch's powerline fraction).

    Independent impulses give 2*0.6/8.333 = 0.144; buzz-dominated output gives 0.9-1.0. None if < min_n events."""
    t = np.sort(np.asarray(times, float))
    if t.size < min_n:
        return None
    ph = np.diff(t) / MAINS_HALF
    return float(np.mean(np.abs(ph - np.round(ph)) * MAINS_HALF < COMB_WIN))


_SOS_A = butter(4, [2500, 9000], btype="band", fs=FS, output="sos")
_SOS_X = butter(4, [10000, 14000], btype="band", fs=FS, output="sos")


def _real_vlf(x, fs=FS, fc=FC):
    """Shift complex baseband to real VLF (RF f -> f), float64. Phase restarts at x[0] (as h2h did per 30 s chunk)."""
    n = np.arange(len(x))
    return (x.astype(np.complex128) * np.exp(1j * 2 * np.pi * (fc / fs) * n)).real


def confirm_share(x, times, fs=FS, k=6.0, half_win_s=0.002):
    """Share of events whose +-2 ms neighbourhood in the independent 10-14 kHz band peaks over median + k MAD-sigma.

    A sferic is broadband, so its onset shows in 10-14 kHz too; mains buzz is 0-2 dB above 16 kHz and weak there.
    Same recipe as h2h/ours.py (causal 4th-order band, 30 s chunks). None if no events."""
    times = np.sort(np.asarray(times, float))
    if times.size == 0:
        return None
    nch = int(CHUNK_S * fs)
    hw = int(half_win_s * fs)
    ok = 0
    for c0 in range(0, len(x), nch):
        seg_t = times[(times >= c0 / fs) & (times < (c0 + nch) / fs)]
        if seg_t.size == 0:
            continue
        e2 = np.abs(sosfilt(_SOS_X, _real_vlf(x[c0:c0 + nch], fs)))
        m2 = np.median(e2)
        s2 = np.median(np.abs(e2 - m2)) * 1.4826 + 1e-15
        for t in seg_t:
            e = int(round(t * fs)) - c0
            w = e2[max(0, e - hw):e + hw]
            ok += int(w.size > 0 and w.max() > m2 + k * s2)
    return ok / times.size


# ================================================================ baselines
@sw.register("july", note="July sferic_rate rule: 2.5-9 kHz sosfiltfilt, median + 8 MAD-sigma, 8 ms dead time")
def july(x, fs=FS):
    """The July sferic_rate.py edge rule, chunked like h2h's ours.py (30 s chunks; a final chunk >= 1 s is used too).

    Peak = 20*log10(max |y| in the 8 ms after the edge / median |y|). The 8 ms dead time is the known flaw: buzz pulses
    arrive every 8.33 ms, so under hum each detection blinds the next 96% of the time (h2h)."""
    nch = int(CHUNK_S * fs)
    g = int(0.008 * fs)
    times, peaks = [], []
    for c0 in range(0, len(x), nch):
        xc = x[c0:c0 + nch]
        if len(xc) < fs:
            break
        ew = np.abs(sosfiltfilt(_SOS_A, _real_vlf(xc, fs)))
        med = np.median(ew)
        sig = np.median(np.abs(ew - med)) * 1.4826 + 1e-15
        above = ew > med + 8 * sig
        edges = np.flatnonzero(~above[:-1] & above[1:])
        kept = []
        for e in edges:
            if not kept or e - kept[-1] > g:
                kept.append(e)
        for e in kept:
            times.append((c0 + e) / fs)
            peaks.append(20 * np.log10(ew[e:e + g].max() / (med + 1e-30)))
    return np.array(times), np.array(peaks)


def onair_bin():
    """Path of the OnAir h2h binary: $SFERIC_ONAIR_BIN, else ``onair_sferics_h2h`` on $PATH, else corpus.json
    "onair_bin" (a home-relative install location, ~/.local/share/sferic-watch/; see bench/README.md for the build).

    The binary is GPL (it links OnAir's libdect2core.a), so it is built locally and never committed."""
    p = os.environ.get("SFERIC_ONAIR_BIN") or shutil.which("onair_sferics_h2h") \
        or swio.load_corpus().get("onair_bin") or ""
    return os.path.expanduser(p)


def onair_available():
    """True when the OnAir h2h binary exists and is executable (the onair_* baselines are optional)."""
    b = onair_bin()
    return bool(b) and os.path.isfile(b) and os.access(b, os.X_OK)


def drop_missing_onair(detectors, log=sys.stderr):
    """Remove onair_* detectors when the binary is absent, saying so once. Returns the remaining list."""
    if onair_available():
        return list(detectors)
    gone = [d for d in detectors if d.startswith("onair_")]
    if gone:
        print(f"OnAir binary not found (set SFERIC_ONAIR_BIN; build steps in bench/README.md): skipping {gone}",
              file=log)
    return [d for d in detectors if not d.startswith("onair_")]


def tmp_root():
    """Directory for the temporary 150 MB .cs16 the OnAir binary reads: $SFERIC_TMP, else /var/tmp (disk-backed on
    this box), else the system temp dir. /tmp is tmpfs here, where a 600 s file would count against RAM."""
    for d in (os.environ.get("SFERIC_TMP"), "/var/tmp"):
        if d and os.path.isdir(d) and os.access(d, os.W_OK):
            return d
    return None


def run_onair(x, fs=FS, mains=1, front="bp"):
    """Feed x to the OnAir h2h binary through a temporary .cs16. Returns (times, peak_dB, info).

    The binary assumes 62.5 kS/s and reads int16 IQ; x from read_slice(dc="none") round-trips bit-exactly, so this is
    the same input h2h used. info carries the binary's own CPU (its loop only) and the mains/tooLong counters."""
    if abs(fs - 62500.0) > 1e-6:
        raise ValueError("the OnAir h2h binary is hard-wired to 62.5 kS/s")
    b = onair_bin()
    if not os.path.exists(b):
        raise FileNotFoundError(f"OnAir binary not found at {b!r}; build it (bench/README.md) and set SFERIC_ONAIR_BIN")
    d = tempfile.mkdtemp(prefix="onair_", dir=tmp_root())
    f = os.path.join(d, "x.cs16")
    try:
        swio.to_cs16(x).tofile(f)
        r = subprocess.run([b, f, os.path.join(d, "o"), str(mains), "0", front], capture_output=True, text=True, check=True)
        info = json.loads(r.stdout.strip().splitlines()[-1])
        a = np.loadtxt(os.path.join(d, "o.times"), ndmin=2) if os.path.getsize(os.path.join(d, "o.times")) else np.zeros((0, 3))
    finally:
        for p in os.listdir(d):
            os.remove(os.path.join(d, p))
        os.rmdir(d)
    return a[:, 0], a[:, 1], {"cpu_self_s": info["cpu_s"], "mains": info["mains"], "tooLong": info["tooLong"],
                              "floorDb_min": info["floorDb_min"], "floorDb_max": info["floorDb_max"]}


@sw.register("onair_bp", ungated="onair_bp_nomains", note="OnAir C++ SfericsReceiver behind our 2.5-9 kHz band-pass (h2h)")
def onair_bp(x, fs=FS):
    """OnAir's detector (binary) behind our causal 4th-order 2.5-9 kHz band-pass, mains filter on."""
    return run_onair(x, fs, 1, "bp")


@sw.register("onair_bp_nomains", note="onair_bp with OnAir's mains filter off")
def onair_bp_nomains(x, fs=FS):
    """OnAir's detector behind our band-pass with its mains filter off."""
    return run_onair(x, fs, 0, "bp")


@sw.register("onair_raw", note="OnAir as shipped (no band selection): expected blind, floor -22.6 dBFS")
def onair_raw(x, fs=FS):
    """OnAir's detector on the raw complex IQ (its default): regression guard, h2h measured 0.0-0.1/min."""
    return run_onair(x, fs, 1, "raw")


def load_plugins():
    """Import every sferic_watch submodule so their @register calls run. Failures are reported, not fatal."""
    bad = {}
    for m in pkgutil.iter_modules(sw.__path__):
        try:
            importlib.import_module(f"sferic_watch.{m.name}")
        except Exception as e:                       # a broken plug-in must not hide the baselines
            bad[m.name] = repr(e)
    return bad


def run_detector(name, x, fs=FS):
    """Run a registered detector; returns (times, peaks, info) with info['cpu_s'] = process + child CPU seconds."""
    fn = sw.get_detector(name)["fn"]
    c0 = time.process_time()
    r0 = resource.getrusage(resource.RUSAGE_CHILDREN)
    out = fn(x, fs)
    r1 = resource.getrusage(resource.RUSAGE_CHILDREN)
    cpu = time.process_time() - c0 + (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    times, peaks = np.asarray(out[0], float), np.asarray(out[1], float)
    info = dict(out[2]) if len(out) > 2 and out[2] else {}
    info["cpu_s"] = cpu
    return times, peaks, info


# ================================================================ OnAir-style mains rule on bare times (optional)
# A behavioural model of OnAir's train-vote mains rule follows that GPL-3 rule step for step, so it is NOT kept in
# this MIT repo. If you have one, put it at $SFERIC_ONAIR_RULE (default ~/.local/share/sferic-watch/onair_rule_model.py)
# exposing onair_rule_reject(times) -> (kept, rejected, state_on). Without it the 'rule' mode is skipped.

def _rule_module():
    path = os.path.expanduser(os.environ.get("SFERIC_ONAIR_RULE", "~/.local/share/sferic-watch/onair_rule_model.py"))
    if not os.path.exists(path):
        return None
    import importlib.util
    spec = importlib.util.spec_from_file_location("onair_rule_model", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def onair_rule_reject(times, **kw):
    """Run the optional external OnAir rule model; raises FileNotFoundError when it is not installed."""
    mod = _rule_module()
    if mod is None:
        raise FileNotFoundError("OnAir rule model not installed (see the comment above _rule_module)")
    return mod.onair_rule_reject(times, **kw)


def rule_poisson(rates=(5, 10, 20, 40, 60, 80, 120, 200), secs=120.0, seed=1):
    """Share of pure Poisson impulse times (no mains at all) the external OnAir rule model throws away, and the share
    counted while it believed a train was on. Measured with the model: 4.21/13.32/17.72 % at 40/60/80 per s
    (earlier mains_sim.py study: 4.3/13.5/17.9 %)."""
    rng = np.random.default_rng(seed)
    out = {}
    for r in rates:
        t = np.cumsum(rng.exponential(1 / r, int(r * secs)))
        _, rej, on = onair_rule_reject(t)
        out[str(r)] = {"rejected_as_mains": round(float(rej.mean()), 4), "mains_on_share": round(float(on.mean()), 4)}
    return out


# ================================================================ injection runs
def _pieces(corpus):
    """(tag, path, class) of the injection pieces, in corpus order."""
    caps = {c["id"]: c for c in corpus["captures"]}
    return [(t, os.path.expanduser(caps[t]["path"]), corpus["injection"]["piece_class"][t])
            for t in corpus["injection"]["pieces"]]


_BANK = {}


def real_bank(corpus):
    """The frozen h2h bank (386 cuts), cached in memory; optionally from an .npy cache at $SFERIC_BANK_NPY."""
    if "real" not in _BANK:
        b = corpus["injection"]["bank"]
        cache = os.environ.get("SFERIC_BANK_NPY")
        if cache and os.path.exists(cache):
            _BANK["real"] = np.load(cache)
        else:
            src = os.path.expanduser(b["source"])
            _BANK["real"] = inj.real_bank(src, b["cut_start_samples"], swio.read_slice)
            if cache:
                np.save(cache, _BANK["real"])
        src = os.path.expanduser(b["source"])
        _BANK["src_floor"] = inj.host_floor(swio.read_slice(src, 0, 120, dc="none"))
    return _BANK["real"]


def build_injections(corpus, sets=("native",), snrs=(12, 18, 24, 30)):
    """Yield (tag, class, set, snr, orig_piece, injected_piece, truth, extra) for every requested case.

    native: h2h's exact schedule/picks (rng 7). real_snr / synth: rng(100 + piece index) schedule shared by every SNR,
    so SNRs differ only in gain."""
    ij = corpus["injection"]
    t0, secs = ij["piece_t0_s"], ij["piece_secs"]
    bank = real_bank(corpus) if any(s in sets for s in ("native", "real_snr")) else None
    h2h = inj.h2h_schedule(len(bank)) if "native" in sets else None
    snr_bank = None
    if "real_snr" in sets:
        snr_bank, nat = inj.real_snr_bank(bank, _BANK["src_floor"])
    syn = inj.synth_bank(200, seed=11)[0] if "synth" in sets else None
    for pi, (tag, path, cls) in enumerate(_pieces(corpus)):
        x = swio.read_slice(path, t0, secs, dc="none")
        if "native" in sets:
            cut, pick = h2h[tag]
            yield tag, cls, "native", None, x, inj.inject(x, bank[pick], cut), cut + inj.ONSET_S, {}
        floor = inj.host_floor(x)
        for set_name, b in (("real_snr", snr_bank), ("synth", syn)):
            if set_name not in sets:
                continue
            rng = np.random.default_rng(100 + pi)
            cut = inj.schedule(rng, secs)
            pick = rng.integers(0, len(b), len(cut))
            sn = [b[k] for k in pick]
            for snr in snrs:
                g = inj.snr_gains(sn, floor, snr)
                extra = {"host_floor_dbfs": round(10 * np.log10(floor), 2)}
                if set_name == "real_snr":
                    extra["bank_n"] = len(b)
                    extra["bank_native_db_min"] = round(float(nat.min()), 1)
                yield tag, cls, set_name, snr, x, inj.inject(x, sn, cut, g), cut + inj.ONSET_S, extra


def score_injections(detectors, sets=("native",), snrs=(12, 18, 24, 30), corpus=None):
    """Rows of {piece, class, set, snr, detector, bg_per_min, recall, chance, corr, extra_per_s, n_det, cpu_s_per_min}."""
    corpus = corpus or swio.load_corpus()
    rows, bg = [], {}
    for tag, cls, set_name, snr, x, y, truth, extra in build_injections(corpus, sets, snrs):
        secs = len(x) / FS
        for d in detectors:
            if (tag, d) not in bg:
                to, _, io_ = run_detector(d, x)
                bg[(tag, d)] = (len(to) / (secs / 60), io_["cpu_s"] / (secs / 60))
            ti, _, ii = run_detector(d, y)
            rec, ch, corr = corrected_recall(ti, truth)
            rows.append(dict(piece=tag, cls=cls, set=set_name, snr=snr, detector=d, bg_per_min=round(bg[(tag, d)][0], 1),
                             recall=round(rec, 4), chance=round(ch, 4), corr=round(corr, 4),
                             extra_per_s=round(extra_rate(ti, truth, secs), 3), n_det=len(ti),
                             cpu_s_per_min=round(ii["cpu_s"] / (secs / 60), 4), **extra))
    return rows


def summarize_injections(rows):
    """Mean corrected recall per (detector, set, snr) overall and per piece class."""
    out = {}
    for r in rows:
        k = f"{r['detector']}|{r['set']}|{r['snr'] if r['snr'] is not None else 'native'}"
        o = out.setdefault(k, {"per_piece": {}, "by_class": {}})
        o["per_piece"][r["piece"]] = r["corr"]
        o["by_class"].setdefault(r["cls"], []).append(r["corr"])
    for o in out.values():
        o["mean_corr"] = round(float(np.mean(list(o["per_piece"].values()))), 3)
        o["by_class"] = {c: [round(min(v), 2), round(max(v), 2), round(float(np.mean(v)), 3)] for c, v in o["by_class"].items()}
    return out


# ================================================================ Poisson (signal) false-rejection mode
def score_poisson(detectors, rates=(5, 10, 20, 40, 60, 80, 120, 200), secs=120.0, snr_db=24.0, seed=5):
    """Synthetic host (no mains) + synthetic sferics at Poisson rates and snr_db. Per detector and rate: corrected
    recall and, when the detector names an ungated twin, the false mains rejection = share of pulses the twin found
    within 2 ms that the gated detector did not."""
    syn = inj.synth_bank(200, seed=11)[0]
    host = inj.synthetic_host(secs, seed=seed)
    floor = inj.host_floor(host)
    out = {}
    for ri, r in enumerate(rates):
        cut = inj.poisson_starts(r, secs, seed=1000 + ri)
        rng = np.random.default_rng(2000 + ri)
        sn = [syn[k] for k in rng.integers(0, len(syn), len(cut))]
        y = inj.inject(host, sn, cut, inj.snr_gains(sn, floor, snr_db))
        truth = cut + inj.ONSET_S
        cache = {}
        for d in detectors:
            for name in (d, sw.get_detector(d)["ungated"]):
                if name and name not in cache:
                    cache[name] = run_detector(name, y)[0]
            rec, ch, corr = corrected_recall(cache[d], truth)
            row = {"n_pulses": len(truth), "recall": round(rec, 4), "corr": round(corr, 4),
                   "det_per_s": round(len(cache[d]) / secs, 2)}
            ug = sw.get_detector(d)["ungated"]
            if ug:
                h_g, h_u = hits(cache[d], truth), hits(cache[ug], truth)
                row["false_reject"] = round(float(np.sum(h_u & ~h_g) / max(1, h_u.sum())), 4)
                row["ungated_recall"] = round(float(h_u.mean()), 4)
            out.setdefault(d, {})[str(r)] = row
    return out


# ================================================================ corpus pass
def _corpus_one(args):
    """Worker: one capture end to end through every detector. Returns a row per detector."""
    entry, detectors = args
    load_plugins()
    x = swio.read_slice(entry["path"], 0, swio.capture_info(entry["path"])["secs"], dc="none")
    secs = len(x) / FS
    rows = []
    for d in detectors:
        t, pk, info = run_detector(d, x)
        cs, cf = comb_share(t), confirm_share(x, t)
        rows.append({"capture": entry["id"], "detector": d, "secs": secs, "per_min": round(len(t) / (secs / 60), 2),
                     "comb_share": None if cs is None else round(cs, 3),
                     "comb_chance": round(COMB_CHANCE, 3),
                     "confirm_10_14k": None if cf is None else round(cf, 3),
                     "cpu_s_per_min": round(info["cpu_s"] / (secs / 60), 4),
                     "per_10s": np.histogram(t, bins=np.arange(0, secs + 10, 10))[0].tolist(),
                     "comb_10s": [None if (c := comb_share(t[(t >= a) & (t < a + 10)])) is None else round(c, 2)
                                  for a in np.arange(0, secs, 10)],
                     **{k: v for k, v in info.items() if k != "cpu_s"}})
    del x
    return rows


def score_corpus(detectors, workers=2, golden_only=False):
    """All corpus captures through every detector, ``workers`` processes (2 max on the shared box)."""
    entries = swio.list_corpus(golden_only=golden_only, must_exist=True)
    jobs = [(e, list(detectors)) for e in entries]
    if workers <= 1:
        res = [_corpus_one(j) for j in jobs]
    else:
        import multiprocessing as mp
        with mp.get_context("spawn").Pool(workers) as pool:
            res = pool.map(_corpus_one, jobs, chunksize=1)
    return [r for rr in res for r in rr]


def hum_blocks(per_10s, comb_10s, rate_thr=20.0, comb_thr=0.5):
    """Rough hum on/off intervals [t_on, t_off] (s) from the july rule's per-10 s counts and comb shares.

    Under buzz the July rule fires ~90/s with >0.9 of intervals on the comb; clean air gives ~1/s (census). The 20/s
    and 0.5 cut sits in the empty gap between the two groups. Plan step 2's hum flag replaces this."""
    on = [(n / 10.0 > rate_thr) and (c is not None and c > comb_thr) for n, c in zip(per_10s, comb_10s)]
    iv, start = [], None
    for i, h in enumerate(on + [False]):
        if h and start is None:
            start = i * 10
        if not h and start is not None:
            iv.append([start, i * 10])
            start = None
    return iv


def buzz_suspect_blocks(per_10s, comb_10s, min_n=20, comb_thr=0.5, rate_thr=20.0, min_total_s=30):
    """Sparse-buzz suspect intervals: 10 s blocks BELOW the dense-hum rate (<= 20 edges/s) whose July intervals still
    sit on the 120 Hz comb (share > 0.5 with >= 20 edges; chance is 0.144).

    Why: hum_blocks misses buzz that is sparse (11-05 20Z: 33-58 ms gaps, 3-5 edges/s, comb 0.95) or intermittent
    (11-06 10Z: 9-16 edges/s, comb 0.54-0.68). Busy clean skies can reach comb 0.5-0.6 in a stray block (11-03 21Z:
    2 blocks; 11-06 08:30Z: 1 block), so suspects are only returned when they total >= ``min_total_s`` (3 blocks);
    measured on the july pass this flags exactly 11-05 20Z (530 s), 11-05 04:30Z (250 s, before its dense switch-on)
    and 11-06 10Z (250 s)."""
    on = [(n >= min_n) and (n / 10.0 <= rate_thr) and (c is not None and c > comb_thr) for n, c in zip(per_10s, comb_10s)]
    if 10 * sum(on) < min_total_s:
        return []
    iv, start = [], None
    for i, h in enumerate(on + [False]):
        if h and start is None:
            start = i * 10
        if not h and start is not None:
            iv.append([start, i * 10])
            start = None
    return iv


def label_corpus(rows, write=False):
    """Put rough hum labels from a july corpus pass into corpus.json.

    hum_on_s / hum_frac: dense buzz (hum_blocks). buzz_suspect_s: sparse/intermittent buzz the dense rule cannot see
    (buzz_suspect_blocks). hum_label_complete: False when suspects exist - then hum_on_s is a LOWER bound on the buzz,
    not ground truth for 'clean'."""
    path = swio.CORPUS_JSON
    c = swio.load_corpus()
    by = {r["capture"]: r for r in rows if r["detector"] == "july"}
    for e in c["captures"]:
        r = by.get(e["id"])
        if r is None:
            continue
        iv = hum_blocks(r["per_10s"], r["comb_10s"])
        e["hum_on_s"] = iv
        e["hum_frac"] = round(sum(b - a for a, b in iv) / r["secs"], 3)
        e["buzz_suspect_s"] = buzz_suspect_blocks(r["per_10s"], r["comb_10s"])
        e["hum_label_complete"] = not e["buzz_suspect_s"]
    c["hum_label_method"] = ("rough: july rule per 10 s block. hum_on_s = dense buzz (> 20 edges/s AND comb share > 0.5, "
                             "bench/harness.py hum_blocks). buzz_suspect_s = blocks <= 20 edges/s with >= 20 edges and "
                             "comb share > 0.5, kept only when they total >= 30 s (buzz_suspect_blocks). "
                             "hum_label_complete false => hum_on_s is a lower bound, not truth. Plan step 2 replaces this")
    if write:
        with open(path, "w") as f:
            json.dump(c, f, indent=1)
    return {e["id"]: (e.get("hum_on_s"), e.get("hum_frac"), e.get("buzz_suspect_s")) for e in c["captures"]}


# ================================================================ CLI
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("mode", choices=["inject", "poisson", "rule", "corpus", "all", "label", "list"])
    ap.add_argument("--detector", nargs="+", default=["july", "onair_bp"])
    ap.add_argument("--set", nargs="+", default=["native", "real_snr", "synth"])
    ap.add_argument("--snr", nargs="+", type=float, default=[12, 18, 24, 30])
    ap.add_argument("--rates", nargs="+", type=float, default=[5, 10, 20, 40, 60, 80, 120, 200])
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--golden-only", action="store_true")
    ap.add_argument("--rows", help="label: corpus rows JSON from an earlier corpus/all run")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--out", help="write results JSON here")
    a = ap.parse_args(argv)
    bad = load_plugins()
    if bad:
        print("plug-ins that failed to import:", bad, file=sys.stderr)
    a.detector = drop_missing_onair(a.detector)
    res = {"detectors": a.detector}
    w0 = time.time()
    if a.mode == "list":
        for n in sw.detector_names():
            e = sw.get_detector(n)
            print(f"{n:20s} ungated={e['ungated']!s:18s} {e['note']}")
        return
    if a.mode in ("rule", "all"):
        if _rule_module() is None:
            print("rule: OnAir rule model not installed, skipped", file=sys.stderr)
        else:
            res["rule_poisson"] = rule_poisson()
    if a.mode in ("inject", "all"):
        rows = score_injections(a.detector, a.set, a.snr)
        res["inject_rows"] = rows
        res["inject_summary"] = summarize_injections(rows)
    if a.mode in ("poisson", "all"):
        res["poisson"] = score_poisson(a.detector, a.rates)
    if a.mode in ("corpus", "all"):
        c0 = time.time()
        res["corpus_rows"] = score_corpus(a.detector, a.workers, a.golden_only)
        res["corpus_wall_s"] = round(time.time() - c0, 1)
    if a.mode == "label":
        rows = json.load(open(a.rows))["corpus_rows"]
        res["labels"] = label_corpus(rows, a.write)
    res["wall_s"] = round(time.time() - w0, 1)
    txt = json.dumps(res, indent=1, default=lambda o: o.tolist() if hasattr(o, "tolist") else str(o))
    if a.out:
        with open(a.out, "w") as f:
            f.write(txt)
    brief = {k: v for k, v in res.items() if k not in ("inject_rows", "corpus_rows")}
    print(json.dumps(brief, indent=1, default=str)[:20000])


if __name__ == "__main__":
    main()
