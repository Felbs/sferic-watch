"""Plan step 6 on real data: the report over every banked capture (+ tonight's live one) and its acceptance checks.

    systemd-run --user --scope -q -p MemoryMax=4G nice -n 10 python bench/report_all.py --out DIR [--workers 2]
    python bench/report_all.py --out DIR --check-only        # re-check an existing run

Writes DIR/sessions/<id>.session.json + <id>.events.csv, DIR/census.csv, DIR/run.log and DIR/acceptance.json.

The checks, as the plan states them:
1. Deterministic: two captures (one clean day, one steady hum) are analyzed again into DIR/det_check and must be
   byte-identical to the corpus run's files.
2. Hum boundaries (11-05 12:30/13:00Z, 11-06 19:30/20:00Z, 20:00/20:30Z, 21:30/22:00Z): across each pair the gated
   rate and the two-band (v2) rate change by less than 1.6x, while hum_frac goes from 0 to above 0.9. Both parts are
   reported per pair: if hum_frac does not actually step from 0 to > 0.9 the pair is said to be no hum boundary under
   the step-2 flag, rather than silently dropped.
3. v2 chance below 5 % of the v2 rate on at least 90 % of slots (reported per capture and per minute).
4. Every session validates against sferic_watch/schema.json.
5. Privacy grep (lat|lon|latitude|longitude|grid, any case, as substrings) finds nothing in any output file.
No number here is tuned to pass; a failing check is written as pass=false with its measured value.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))

from sferic_watch import cli                    # noqa: E402
from sferic_watch import report as rp            # noqa: E402

CORPUS_DIRS = ["~/dlayer_run/data/vlf"]
LIVE = ["~/storm_live/vlf_20261114_0212Z.cs16"]
BOUNDARIES = [("vlf_20261105_1230Z", "vlf_20261105_1300Z"), ("vlf_20261106_1930Z", "vlf_20261106_2000Z"),
              ("vlf_20261106_2000Z", "vlf_20261106_2030Z"), ("vlf_20261106_2130Z", "vlf_20261106_2200Z")]
DET_CHECK = ["~/dlayer_run/data/vlf/20261019/vlf_1500Z.cs16", "~/dlayer_run/data/vlf/20261105/vlf_1230Z.cs16"]
RATIO_MAX = 1.6
CHANCE_SHARE_MAX = 0.05
SLOT_SHARE_MIN = 0.90


def ratio(a, b):
    """max/min of two rates; inf when one is zero and the other is not."""
    if a is None or b is None:
        return None
    lo, hi = sorted((a, b))
    return hi / lo if lo > 0 else (1.0 if hi == 0 else float("inf"))


def load_docs(sess_dir):
    docs = {}
    for f in sorted(os.listdir(sess_dir)):
        if f.endswith(".session.json"):
            with open(os.path.join(sess_dir, f)) as fh:
                d = json.load(fh)
            docs[d["capture"]["id"]] = d
    return docs


def check(out, det_dir_ok):
    """All acceptance checks on an existing run in ``out``; returns the acceptance dict."""
    sess = os.path.join(out, "sessions")
    docs = load_docs(sess)
    res = {"n_sessions": len(docs)}
    # 4. schema
    schema = rp.load_schema()
    bad = {k: rp.validate(d, schema)[:3] for k, d in docs.items()}
    bad = {k: v for k, v in bad.items() if v}
    res["schema"] = {"invalid": len(bad), "examples": dict(list(bad.items())[:3]), "pass": not bad and len(docs) > 0}
    # 5. privacy grep on every output file
    # Only the engine's outputs are grepped: sessions/, det_check/, census.csv, run.log. Not acceptance*.json (this
    # check's own report, which quotes earlier hits back) nor any scratch the operator keeps beside the run (a
    # headless-browser profile left there once tripped the grep on Chromium's own files).
    hits = {}
    for root, dirs, files in os.walk(out):
        if root == out:
            dirs[:] = [d for d in dirs if d in ("sessions", "det_check")]
            files = [f for f in files if f in ("census.csv", "run.log")]
        for f in files:
            if f.endswith((".json", ".csv", ".log")):
                with open(os.path.join(root, f), errors="replace") as fh:
                    h = rp.privacy_hits(fh.read())
                if h:
                    hits[os.path.relpath(os.path.join(root, f), out)] = h
    with open(rp.SCHEMA_PATH) as fh:
        sh = rp.privacy_hits(fh.read())
    if sh:
        hits["sferic_watch/schema.json"] = sh
    res["privacy"] = {"files_with_hits": len(hits), "examples": dict(list(hits.items())[:5]), "pass": not hits}
    # 2. hum boundaries
    pairs = []
    for a, b in BOUNDARIES:
        if a not in docs or b not in docs:
            pairs.append({"pair": [a, b], "missing": True, "pass": False})
            continue
        sa, sb = docs[a]["summary"], docs[b]["summary"]
        h = sorted((sa["hum_frac"], sb["hum_frac"]))
        is_boundary = h[0] == 0 and h[1] > 0.9
        r_g = ratio(sa["rate_per_min"], sb["rate_per_min"])
        r_o = ratio(sa["rate_open_per_min"], sb["rate_open_per_min"])
        r_v = ratio(sa["v2_per_min"], sb["v2_per_min"])
        pairs.append({"pair": [a, b], "hum_frac": [sa["hum_frac"], sb["hum_frac"]], "hum_boundary": is_boundary,
                      "rate_per_min": [sa["rate_per_min"], sb["rate_per_min"]], "ratio_gated": r_g,
                      "rate_open_per_min": [sa["rate_open_per_min"], sb["rate_open_per_min"]], "ratio_open": r_o,
                      "v2_per_min": [sa["v2_per_min"], sb["v2_per_min"]], "ratio_v2": r_v,
                      "comb_share": [sa["comb_share"], sb["comb_share"]],
                      "gated_frac": [sa["gated_frac"], sb["gated_frac"]],
                      "pass": bool(is_boundary and r_g < RATIO_MAX and r_v < RATIO_MAX)})
    res["hum_boundaries"] = {"pairs": pairs, "n_pass": sum(p["pass"] for p in pairs),
                             "pass": all(p["pass"] for p in pairs)}
    # 3. v2 chance share
    def share_ok(rows):
        ok = [r["v2_per_min"] > 0 and r["v2_chance_per_min"] < CHANCE_SHARE_MAX * r["v2_per_min"] for r in rows]
        return sum(ok), len(ok)
    k_c, n_c = share_ok([d["summary"] for d in docs.values()])
    k_m, n_m = share_ok([m for d in docs.values() for m in d["minutes"]])
    res["v2_chance"] = {"captures_ok": k_c, "captures": n_c, "capture_share": round(k_c / max(n_c, 1), 4),
                        "minutes_ok": k_m, "minutes": n_m, "minute_share": round(k_m / max(n_m, 1), 4),
                        "pass": k_c >= SLOT_SHARE_MIN * n_c and k_m >= SLOT_SHARE_MIN * n_m}
    res["determinism"] = det_dir_ok
    # context numbers
    ss = [d["summary"] for d in docs.values()]
    res["context"] = {"captures_untrusted": sum(not s["trusted"] for s in ss),
                      "minutes_untrusted": sum(s["n_minutes_untrusted"] for s in ss),
                      "minutes": sum(s["n_minutes"] for s in ss),
                      "captures_with_hum": sum(s["hum_frac"] > 0 for s in ss)}
    return res


def det_check(out, workers):
    """Analyze the DET_CHECK captures again into out/det_check and compare bytes with the corpus run."""
    dd = os.path.join(out, "det_check")
    cli.run_many([os.path.expanduser(p) for p in DET_CHECK], dd, workers, log=lambda m: None)
    rows = []
    for f in sorted(os.listdir(dd)):
        with open(os.path.join(dd, f), "rb") as a, open(os.path.join(out, "sessions", f), "rb") as b:
            rows.append({"file": f, "identical": a.read() == b.read()})
    return {"files": rows, "pass": bool(rows) and all(r["identical"] for r in rows)}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--check-only", action="store_true")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if not a.check_only:
        paths = rp.find_captures(*CORPUS_DIRS) + [os.path.expanduser(p) for p in LIVE]
        with open(os.path.join(a.out, "run.log"), "w") as log:
            def say(m):
                print(m, flush=True)
                log.write(m + "\n")
                log.flush()
            res = cli.run_many(paths, os.path.join(a.out, "sessions"), a.workers, log=say)
            bad = cli.write_census(res, os.path.join(a.out, "census.csv"))
            say(f"{len(res)} captures, {bad} failed")
    det = det_check(a.out, a.workers)
    acc = check(a.out, det)
    with open(os.path.join(a.out, "acceptance.json"), "w") as f:
        json.dump(acc, f, indent=1)
    print(json.dumps(acc, indent=1))


if __name__ == "__main__":
    main()
