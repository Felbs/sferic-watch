#!/usr/bin/env python3
"""Plan step 7 on real data: the tweek honesty audit over the banked captures (reads captures, never the radio).

Modes (all write JSON into --out, one file per capture plus summaries):
  july      re-audit dlayer-diary tweeks_fits.json (read only); prints the pin counts and bound-free medians.
  synth     the July ridge + fit on ideal synthetic tweeks (fc 1.6/1.8/2.0 kHz x d 1000/2000/4000 km).
  accept    the step-7 acceptance checks:
              11-06 04:00Z: random-window textbook count with the gate (bar <= 2/300) and without (plan: 57/300,
              which used the prototype's draw order - reproduced here as well as a fresh seed); plus the gate
              decomposition (the same 300 ungated windows split by the hum-block gate and the mains-train gate),
              because the full gate admits no window there (hum_frac 1.00) and the bar would pass vacuously;
              11-05 04:30Z first 450 s: gated audit must print 'insufficient' (under both mixers).
  phase     real-data mixer-phase check on --path: 300 random windows fitted with July's and the absolute mixer,
            each window's own mirror phase measured, textbook rate by |cos(phase)| gain.
  capture   one capture: --path P [--t-stop S] [--nogate] [--phase july|absolute]
  all       every capture in ~/dlayer_run/data/vlf plus ~/storm_live, resumable, --workers (<= 2) or --shard i/n;
            then pooled verdicts for the corpus and per UTC hour. --phase july (default, the published mixer) writes
            captures_july/ and summary_all_july.json; --phase absolute writes the *_absolute variants.

Run with the box limits, for example:
  systemd-run --user --scope -q -p MemoryMax=4G nice -n 10 ~/Projects/.venv/bin/python bench/tweek_audit_all.py \
      all --out /path/to/build/tweek
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import resource
import sys
import time
from multiprocessing import get_context

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from sferic_watch import io as swio            # noqa: E402
from sferic_watch import tweek_audit as ta     # noqa: E402

VLF = os.path.expanduser("~/dlayer_run/data/vlf")
LIVE = os.path.expanduser("~/storm_live")
JULY = "~/Projects/dlayer-diary/tweeks/data/tweeks_fits.json"


def _dump(obj, path):
    with open(path, "w") as fh:
        json.dump(obj, fh, indent=1, default=lambda v: v.item() if hasattr(v, "item") else str(v))


def cap_id(path: str) -> str:
    """20261106_vlf_0400Z from .../20261106/vlf_0400Z.cs16 (live files already carry the date)."""
    b = os.path.splitext(os.path.basename(path))[0]
    d = os.path.basename(os.path.dirname(path))
    return f"{d}_{b}" if d.isdigit() else b


def utc_hour(path: str) -> int:
    """UTC hour from the vlf_HHMMZ / vlf_YYYYMMDD_HHMMZ file name."""
    b = os.path.splitext(os.path.basename(path))[0]
    return int(b.split("_")[-1][:2])


def audit_path(path: str, t_stop: float | None = None, gate: bool = True, seed: int = 7, phase: str = "july") -> dict:
    """Read [0, t_stop) of a capture (complex64, DC mean removed) and run ``tweek_audit.audit_capture``."""
    info = swio.capture_info(path)
    secs = info["secs"] if t_stop is None else min(t_stop, info["secs"])
    c0, w0 = time.process_time(), time.time()
    x = swio.read_slice(path, 0, secs)
    r = ta.audit_capture(x, info["fs"], seed=seed, gate=gate, phase=phase)
    del x
    r.update(id=cap_id(path), path=path.replace(os.path.expanduser("~"), "~"), utc_hour=utc_hour(path),
             cpu_s=round(time.process_time() - c0, 1), wall_s=round(time.time() - w0, 1),
             maxrss_mb=round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024))
    return r


def _line(r: dict) -> str:
    v, c, n = r["verdict"], r["cand"], r["null"]
    h = c.get("h_bound_free", "no bound-free fit")
    return (f"{r['id']:<24} {r.get('phase', '?'):<8} gate={r['gate']!s:<5} hum {r['hum_frac']:.2f} cand {v['n_cand']:5d} tb {v['n_textbook']:3d} "
            f"bf {v['n_bound_free']:3d} | null {v['n_null_textbook']}/{v['n_null']} (drawn {n['n_drawn']}) | "
            f"{v['status']}: {'; '.join(v['reasons']) or 'ok'} | {h}")


def mode_july(out: str) -> dict:
    a = ta.july_reaudit(JULY)
    _dump(a, os.path.join(out, "july_reaudit.json"))
    print(f"July JSON: n={a['n']}, published fc {a['published_fc_median']:.3f} kHz / h {a['published_h_median']:.1f} km")
    print(f"  fc within 0.5/5/10/20 Hz of rmin: {a['pinned']}")
    print(f"  D < 0.01 ms: {a['d_below_0p01']}; d = cD pct 10/25/50/75/90 = {a['d_km_pct_10_25_50_75_90']} km; "
          f"in 1000-5000 km: {a['d_in_1000_5000']}; median published fc_err {a['fc_err_median_hz']:.1f} Hz")
    print(f"  all 77: median fc {a['all']['fc_median']:.3f} kHz, {a['all']['h']}")
    for k in ("bound_free_plan20Hz", "bound_free_rule10Hz"):
        b = a[k]
        print(f"  {k}: n={b['n']} median fc {b['fc_median']:.3f} kHz [{b['fc_ci'][0]:.3f}, {b['fc_ci'][1]:.3f}], {b['h']}")
    for p in a["with_ridge"]:
        print(f"  ridge-shipped fit t={p['t']:.2f}: fc {p['fc']:.4f} (published fc_err {p['fc_err_published_hz']:.3f} Hz); "
              f"equal-weight profile fc {p['fc_profile_equal_w']:.4f}, bootstrap+grid CI [{p['fc_ci'][0]:.3f}, "
              f"{p['fc_ci'][1]:.3f}]{' (upper end at the rmin bound)' if p['ci_hi_at_bound'] else ''} {p['h']} "
              f"bound_free={p['bound_free']}")
    return a


def mode_synth(out: str) -> list:
    rows = ta.synthetic_bias()
    _dump(rows, os.path.join(out, "synthetic_bias.json"))
    for r in rows:
        if not r["fit"]:
            print(f"  fc {r['fc_true']} d {r['d_true_km']:.0f}: no fit")
            continue
        ci = r.get("fc_ci")
        cis = f"CI [{ci[0]:.3f}, {ci[1]:.3f}] covers={r['covers']}" if ci else "no CI (not textbook)"
        if ci:
            cis += f" profile fc {r['fc_profile']:.4f} not_global={r['fit_not_global']}"
        print(f"  fc {r['fc_true']} d {r['d_true_km']:.0f} km: fit {r['fc_fit']:.3f} (err {r['fc_err_khz']*1e3:+.0f} Hz) "
              f"d_fit {r['d_fit_km']:.0f} km textbook={r['textbook']} bound_free={r['bound_free']} {cis}")
    return rows


def null_prototype_order(path: str, n: int = 300, phase: str = "absolute") -> dict:
    """ourchain/nulltest.py's draw for 11-06 04:00Z: default_rng(7), 300 draws on [1, 449) discarded (its clean
    capture came first), then 300 on [1, 599). Ungated, July window and fit. nulltest.py used the absolute mixer;
    ``phase`` picks it."""
    rng = np.random.default_rng(7)
    rng.uniform(1, 449, n)
    t = rng.uniform(1, 599, n)
    res = []
    for tt in t:
        s = int(tt * ta.FS) - int(ta.PRE_S * ta.FS)
        x = swio.read_slice(path, s / ta.FS, ta.WIN_S, dc="none")
        r = ta.analyse_window(ta.window_real(x, s, phase=phase))
        res.append(r)
    tb = [r for r in res if r["textbook"]]
    return dict(n=n, n_textbook=len(tb), fc_median=float(np.median([r["fc"] for r in tb])) if tb else None,
                n_at_fc_hi=sum(r["at_fc_hi"] for r in tb))


def gate_decomposition(path: str, n: int = 300, seed: int = 7, phase: str = "july") -> dict:
    """Do the gates remove textbook-looking buzz windows? The same ``n`` random windows (seed as audit_capture's
    null) are all fitted, then counted inside each gate's admitted subset: none, hum-block gate only, mains-train
    gate only, both. A gate that works lowers the textbook RATE of what it admits, not just the count."""
    from sferic_watch import mainsgate as mg
    info = swio.capture_info(path)
    x = swio.read_slice(path, 0, info["secs"])
    ev, pinfo, flags, trains = ta.capture_trains(x, info["fs"])
    t = np.random.default_rng(seed).uniform(1.0, info["secs"] - 1.0, n)
    fits = ta.fit_at(x, t, info["fs"], phase=phase)
    del x
    tb = np.array([f["textbook"] for f in fits])
    ok_h = ~ta.hum_mask(t, flags)
    ok_t = ~mg.judge(t, np.full(t.size, -np.inf), trains)[0] if trains else np.ones(n, bool)
    out = dict(path=path.replace(os.path.expanduser("~"), "~"), phase=phase, n=n, hum_frac=pinfo.get("hum_frac"),
               n_trains=len(trains))
    for k, m in (("none", np.ones(n, bool)), ("hum_gate", ok_h), ("train_gate", ok_t), ("both", ok_h & ok_t)):
        out[k] = dict(admitted=int(m.sum()), textbook=int(tb[m].sum()),
                      rate=float(tb[m].mean()) if m.any() else float("nan"))
        rej = ~m
        out[k]["rejected"] = int(rej.sum())
        out[k]["textbook_in_rejected"] = int(tb[rej].sum())
    return out


def _gd_line(d: dict) -> str:
    return "  gate decomposition (" + d["phase"] + "): " + "; ".join(
        f"{k}: {d[k]['textbook']}/{d[k]['admitted']} admitted, {d[k]['textbook_in_rejected']}/{d[k]['rejected']} rejected"
        for k in ("none", "hum_gate", "train_gate", "both"))


def mode_accept(out: str) -> dict:
    res = {}
    hum = f"{VLF}/20261106/vlf_0400Z.cs16"
    c = f"{VLF}/20261105/vlf_0430Z.cs16"
    for ph in ("july", "absolute"):
        g = audit_path(hum, gate=True, phase=ph)
        u = audit_path(hum, gate=False, phase=ph)
        proto = null_prototype_order(hum, phase=ph)
        gd = gate_decomposition(hum, phase=ph)
        gd_c = gate_decomposition(c, phase=ph)
        res[f"hum_1106_0400_{ph}"] = dict(gated=g, ungated=u, ungated_prototype_draws=proto, gate_decomposition=gd)
        print(_line(g)); print(_line(u))
        print(f"  prototype draw order, ungated: {proto['n_textbook']}/300 textbook, median fc {proto['fc_median']}, "
              f"{proto['n_at_fc_hi']} at fc upper bound")
        print(_gd_line(gd))
        print("  11-05 04:30Z full 600 s:", _gd_line(gd_c))
        res[f"acc_gated_null_le_2_{ph}"] = dict(
            measured=f"{g['null']['n_textbook']}/{g['null']['n_admissible']} admissible of 300 drawn",
            pass_=g["null"]["n_textbook"] <= 2, vacuous=g["null"]["n_admissible"] == 0)
        k = audit_path(c, t_stop=450.0, gate=True, phase=ph)
        ku = audit_path(c, t_stop=450.0, gate=False, phase=ph)
        res[f"clean_1105_0430_450s_{ph}"] = k
        res[f"clean_1105_0430_450s_ungated_{ph}"] = ku
        res[f"gate_decomposition_1105_0430_{ph}"] = gd_c
        print(_line(k)); print(_line(ku))
        res[f"acc_insufficient_{ph}"] = dict(measured=k["verdict"]["status"],
                                             pass_=k["verdict"]["status"] == "insufficient")
    _dump(res, os.path.join(out, "acceptance.json"))
    return res


def mode_phase(out: str, path: str, n: int = 300, seed: int = 11) -> dict:
    """Real-data check of the mixer-phase mechanism (``tweek_audit.window_real``). For n random 45 ms windows: fit with
    July's mixer and the absolute one, and measure the window's own mirror phase phi_j under July's mixer
    (``mirror_phase`` on the slice, all blocks); the absolute mixer's phase is phi_j - 86.4 deg x start sample. If
    the mechanism is right, each mixer's textbook rate rises with its gain |cos(phi)|."""
    rng = np.random.default_rng(seed)
    info = swio.capture_info(path)
    t = rng.uniform(1.0, info["secs"] - 1.0, n)
    rows = []
    for tt in t:
        s = int(round(tt * ta.FS)) - int(round(ta.PRE_S * ta.FS))
        x = swio.read_slice(path, s / ta.FS, ta.WIN_S)
        mp = ta.mirror_phase(x, nfft=512, top_frac=1.0)
        pj = math.radians(mp["phi_deg"])
        pa = pj - 2 * math.pi * (ta.FC / ta.FS) * s
        rj = ta.analyse_window(ta.window_real(x, s, phase="july"))
        ra = ta.analyse_window(ta.window_real(x, s, phase="absolute"))
        rows.append(dict(t=float(tt), coh=mp["coherence"], g_july=abs(math.cos(pj)), g_abs=abs(math.cos(pa)),
                         tb_july=rj["textbook"], tb_abs=ra["textbook"], fc_july=rj.get("fc"), fc_abs=ra.get("fc")))
    res = dict(path=path.replace(os.path.expanduser("~"), "~"), n=n, seed=seed,
               textbook_july=sum(r["tb_july"] for r in rows), textbook_abs=sum(r["tb_abs"] for r in rows),
               status_differs=sum(r["tb_july"] != r["tb_abs"] for r in rows),
               coherence_median=float(np.median([r["coh"] for r in rows])))
    both = [r for r in rows if r["fc_july"] is not None and r["fc_abs"] is not None]
    res["fc_abs_diff_median_hz"] = float(np.median([abs(r["fc_july"] - r["fc_abs"]) * 1e3 for r in both])) if both else None
    bins = [(0.0, 0.33), (0.33, 0.67), (0.67, 1.01)]
    for mix in ("july", "abs"):
        res[f"rate_by_gain_{mix}"] = []
        for lo, hi in bins:
            sel = [r for r in rows if lo <= r[f"g_{mix}"] < hi]
            res[f"rate_by_gain_{mix}"].append(dict(gain=f"{lo:.2f}-{min(hi, 1):.2f}", n=len(sel),
                                                    textbook=sum(r[f"tb_{mix}"] for r in sel)))
    # paired: in windows where exactly one mixer says textbook, does the higher-gain mixer win?
    dis = [r for r in rows if r["tb_july"] != r["tb_abs"]]
    res["discordant_won_by_higher_gain"] = sum((r["tb_july"] and r["g_july"] > r["g_abs"]) or
                                               (r["tb_abs"] and r["g_abs"] > r["g_july"]) for r in dis)
    res["discordant"] = len(dis)
    _dump(dict(summary=res, rows=rows), os.path.join(out, f"phase_{cap_id(path)}.json"))
    print(json.dumps(res, indent=1))
    return res


def all_paths() -> list[str]:
    p = sorted(glob.glob(f"{VLF}/*/vlf_*.cs16")) + sorted(glob.glob(f"{LIVE}/vlf_*.cs16"))
    return [q for q in p if os.path.getsize(q) > 0]


def _job(args):
    path, out, phase = args
    dst = os.path.join(out, f"captures_{phase}", cap_id(path) + ".json")
    if os.path.exists(dst):
        return json.load(open(dst))
    try:
        r = audit_path(path, phase=phase)
    except Exception as e:                          # report, never hide
        r = dict(id=cap_id(path), path=path, error=repr(e))
    _dump(r, dst)
    print(_line(r) if "error" not in r else f"{r['id']}: ERROR {r['error']}", flush=True)
    return r


def pooled(rows: list[dict]) -> dict:
    """Pooled verdict over captures: sums of candidates, textbook, admissible null windows, null textbook, bound-free;
    median fc of all bound-free textbook events with a CI."""
    ok = [r for r in rows if "verdict" in r]
    s = lambda k: int(sum(r["verdict"][k] for r in ok))   # noqa: E731
    v = ta.verdict(s("n_cand"), s("n_textbook"), s("n_null"), s("n_null_textbook"), s("n_bound_free"))
    fits = [f for r in ok for f in r["textbook_events"]]
    v["captures"] = len(ok)
    v["summary"] = ta.summarise(fits)
    v["null_textbook_fc"] = [f["fc"] for r in ok for f in r["textbook_null"]]
    return v


DAY_UTC = range(13, 21)                       # local daytime at this station in Oct-Nov (no tweeks expected)
NIGHT_UTC = list(range(22, 24)) + list(range(0, 12))


def day_night(rows: list[dict]) -> dict:
    """Physical negative control. Tweeks need a night-time D region; in daytime the waveguide absorbs them, so
    textbook fits on daytime events measure how often the gates call an ordinary sferic a tweek. Pools day
    (13-20Z) and night (22-11Z) captures and compares textbook yields with a Fisher exact test."""
    from scipy.stats import fisher_exact
    out = {}
    for k, hrs in (("day", DAY_UTC), ("night", NIGHT_UTC)):
        rs = [r for r in rows if "verdict" in r and r["utc_hour"] in hrs]
        g = lambda f: int(sum(r["verdict"][f] for r in rs))   # noqa: E731
        c, t = g("n_cand"), g("n_textbook")
        out[k] = dict(hours=f"{min(hrs):02d}-{max(hrs):02d}Z" if k == "day" else "22-11Z", captures=len(rs),
                      n_cand=c, n_textbook=t, **{"yield": t / c if c else float("nan")},
                      n_null=g("n_null"), n_null_textbook=g("n_null_textbook"),
                      summary=ta.summarise([f for r in rs for f in r["textbook_events"]]))
    d, n = out["day"], out["night"]
    if d["n_cand"] and n["n_cand"]:
        orr, p = fisher_exact([[n["n_textbook"], n["n_cand"] - n["n_textbook"]],
                               [d["n_textbook"], d["n_cand"] - d["n_textbook"]]])
        out.update(odds_ratio=float(orr), fisher_p=float(p))
    else:
        out.update(odds_ratio=float("nan"), fisher_p=float("nan"))
    return out


def mode_all(out: str, workers: int, shard: str | None = None, phase: str = "july") -> dict | None:
    """Audit every capture (resumable: a capture whose JSON exists is read back), then pool. One worker peaks at
    about 2.0 GB RSS on a 600 s capture, so under one 4 GB scope use --workers 1, or run two shards (--shard 0/2 and
    --shard 1/2) each in its own scope; a shard pools only once every capture file exists."""
    os.makedirs(os.path.join(out, f"captures_{phase}"), exist_ok=True)
    paths = all_paths()
    todo = paths
    if shard:
        i, n = (int(v) for v in shard.split("/"))
        todo = paths[i::n]
    print(f"{len(todo)} of {len(paths)} captures, {workers} workers", flush=True)
    if workers > 1:
        with get_context("spawn").Pool(workers, maxtasksperchild=4) as pool:
            list(pool.imap(_job, [(p, out, phase) for p in todo]))
    else:
        for p in todo:
            _job((p, out, phase))
    files = [os.path.join(out, f"captures_{phase}", cap_id(p) + ".json") for p in paths]
    if not all(os.path.exists(f) for f in files):
        print("not every capture is done yet; pooling skipped", flush=True)
        return None
    rows = [json.load(open(f)) for f in files]
    errs = [r for r in rows if "error" in r]
    corpus = pooled(rows)
    by_hour = {h: pooled([r for r in rows if r.get("utc_hour") == h]) for h in range(24)}
    summ = dict(phase=phase, n_captures=len(rows), errors=[(r["id"], r["error"]) for r in errs], corpus=corpus,
                by_utc_hour={h: {k: v for k, v in b.items() if k != "null_textbook_fc"} for h, b in by_hour.items()},
                captures_above_null=[r["id"] for r in rows if r.get("verdict", {}).get("status") == "above null"],
                pinned_textbook=dict(fc_hi=sum(r["cand"]["n_at_fc_hi"] for r in rows if "cand" in r),
                                     fc_lo=sum(r["cand"]["n_at_fc_lo"] for r in rows if "cand" in r),
                                     d_floor=sum(r["cand"]["n_at_d_floor"] for r in rows if "cand" in r)),
                maxrss_mb_max=max((r.get("maxrss_mb", 0) for r in rows), default=0),
                mirror_stability_pct_5_50_95=[float(v) for v in np.percentile(
                    [r["mirror"]["stability"] for r in rows if "mirror" in r] or [np.nan], [5, 50, 95])],
                textbook_fit_not_global=sum(r["cand"].get("n_fit_not_global", 0) for r in rows if "cand" in r))
    summ["day_night_control"] = day_night(rows)
    _dump(summ, os.path.join(out, f"summary_all_{phase}.json"))
    c = corpus
    print(f"\nCORPUS [{phase} mixer] ({c['captures']} captures): {c['status']} ({'; '.join(c['reasons']) or 'ok'}); "
          f"cand {c['n_cand']} textbook {c['n_textbook']} bound-free {c['n_bound_free']}; "
          f"null {c['n_null_textbook']}/{c['n_null']}; ratio {c['ratio']:.2f}; p {c['p_poisson']:.3g}")
    if "h_bound_free" in c["summary"]:
        print(f"  pooled bound-free: {c['summary']['h_bound_free']} (n={c['summary']['n_bound_free']})")
    for h, b in by_hour.items():
        if b["captures"]:
            hb = b["summary"].get("h_bound_free", "-")
            print(f"  {h:02d}Z: {b['captures']:2d} caps cand {b['n_cand']:5d} tb {b['n_textbook']:3d} bf {b['n_bound_free']:3d} "
                  f"null {b['n_null_textbook']}/{b['n_null']} -> {b['status']} | {hb}")
    print(f"  captures 'above null': {summ['captures_above_null']}; errors: {len(errs)}; max RSS {summ['maxrss_mb_max']} MB")
    dn = summ["day_night_control"]
    for k in ("day", "night"):
        b = dn[k]
        print(f"  {k} {b['hours']}: {b['captures']} caps, yield {b['n_textbook']}/{b['n_cand']} = {b['yield']:.4f}, "
              f"null {b['n_null_textbook']}/{b['n_null']}, bound-free {b['summary']['n_bound_free']}, "
              f"{b['summary'].get('h_bound_free', '-')}")
    print(f"  night/day yield odds ratio {dn['odds_ratio']:.2f}, Fisher p {dn['fisher_p']:.3g}")
    return summ


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["july", "synth", "accept", "phase", "capture", "all"])
    ap.add_argument("--phase", choices=list(ta.PHASES), default="july", help="window mixer (see window_real)")
    ap.add_argument("--out", default=os.environ.get("SFERIC_OUT", "tweek_audit_out"))
    ap.add_argument("--path")
    ap.add_argument("--t-stop", type=float)
    ap.add_argument("--nogate", action="store_true")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--shard", help="i/n: audit only every n-th capture starting at i (then pool if all are done)")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.mode == "july":
        mode_july(a.out)
    elif a.mode == "synth":
        mode_synth(a.out)
    elif a.mode == "accept":
        mode_accept(a.out)
    elif a.mode == "capture":
        r = audit_path(os.path.expanduser(a.path), a.t_stop, gate=not a.nogate, phase=a.phase)
        _dump(r, os.path.join(a.out, cap_id(a.path) + ("_nogate" if a.nogate else "") + f"_{a.phase}.json"))
        print(_line(r), f"cpu {r['cpu_s']} s, maxrss {r['maxrss_mb']} MB")
    elif a.mode == "phase":
        mode_phase(a.out, os.path.expanduser(a.path))
    else:
        mode_all(a.out, min(2, a.workers), a.shard, a.phase)


if __name__ == "__main__":
    main()
