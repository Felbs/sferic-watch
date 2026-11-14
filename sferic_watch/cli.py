"""Command line for the report (plan step 6).

    python -m sferic_watch analyze FILE.cs16 [FILE.cs16 ...] --out DIR
        one <id>.session.json + <id>.events.csv per capture
    python -m sferic_watch census DIR_OF_CAPTURES [DIR ...] --out CSV [--sessions DIR] [--workers 2]
        one census row per capture (every vlf_*.cs16 under the folders); --sessions also keeps the per-capture files

``sferic-watch ...`` (the installed console script) and ``python -m sferic_watch.cli ...`` take the same arguments.

Run heavy jobs capped, on the shared box:
    systemd-run --user --scope -q -p MemoryMax=4G nice -n 10 python -m sferic_watch census ...
Each worker holds one capture: about 300 MB of complex64 IQ plus two 150 MB float32 power arrays. Two workers at
most (6-core box shared with other jobs).
"""
from __future__ import annotations

import argparse
import os
import sys
from multiprocessing import Pool

from sferic_watch import report as rp

MAX_WORKERS = 2


def _one(args: tuple[str, str | None]) -> tuple[str, dict | None, str]:
    """Analyze one capture; write its session files if out_dir is given. Returns (path, census row or None, msg)."""
    path, out_dir = args
    try:
        doc, csv_text = rp.analyze_file(path)
    except Exception as e:                       # one bad file must not stop a corpus run; it is reported
        return path, None, f"error: {type(e).__name__}: {e}"
    if out_dir:
        rp.write_outputs(doc, csv_text, out_dir)
    s = doc["summary"]
    msg = (f"{doc['capture']['id']}: {s['rate_per_min']}/min (open {s['rate_open_per_min']}), hum {s['hum_frac']}, "
           f"gated {s['gated_frac']}, comb {s['comb_share']} (chance {s['comb_chance']}), "
           f"v2 {s['v2_per_min']} (chance {s['v2_chance_per_min']}), "
           f"{s['n_minutes_untrusted']}/{s['n_minutes']} min untrusted")
    return path, rp.census_row(doc), msg


def run_many(paths: list[str], out_dir: str | None, workers: int = 1, log=print) -> list[tuple[str, dict | None]]:
    """Analyze ``paths`` with up to two worker processes, in input order; returns [(path, census row or None)]."""
    workers = max(1, min(MAX_WORKERS, workers))
    jobs = [(p, out_dir) for p in paths]
    res = []
    if workers == 1 or len(jobs) <= 1:
        it = map(_one, jobs)
        for path, row, msg in it:
            log(msg)
            res.append((path, row))
        return res
    with Pool(workers, maxtasksperchild=4) as pool:
        for path, row, msg in pool.imap(_one, jobs):
            log(msg)
            res.append((path, row))
    return res


def write_census(rows: list[tuple[str, dict | None]], out_csv: str) -> int:
    """Write the census CSV (failed captures get a row with status 'error'); returns the number of failures."""
    out, bad = [], 0
    for path, row in rows:
        if row is None:
            bad += 1
            row = {"id": rp.capture_id(path), "file": f"{os.path.basename(os.path.dirname(path))}/"
                                                         f"{os.path.basename(path)}", "status": "error"}
        out.append(row)
    d = os.path.dirname(os.path.abspath(out_csv))
    os.makedirs(d, exist_ok=True)
    with open(out_csv, "w") as f:
        f.write(rp.census_csv(out))
    return bad


def main(argv: list[str] | None = None) -> int:
    """Entry point; returns a process exit code (1 when any capture failed)."""
    ap = argparse.ArgumentParser(prog="sferic_watch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    a = sub.add_parser("analyze", help="session JSON + event catalogue per capture")
    a.add_argument("files", nargs="+")
    a.add_argument("--out", required=True, help="output folder")
    a.add_argument("--workers", type=int, default=1)
    c = sub.add_parser("census", help="one CSV row per capture under the folders")
    c.add_argument("dirs", nargs="+")
    c.add_argument("--out", required=True, help="census CSV path")
    c.add_argument("--sessions", default=None, help="also write each capture's session files here")
    c.add_argument("--workers", type=int, default=1)
    args = ap.parse_args(argv)
    if args.cmd == "analyze":
        res = run_many([os.path.expanduser(f) for f in args.files], args.out, args.workers)
        return int(any(r is None for _, r in res))
    paths = rp.find_captures(*args.dirs)
    if not paths:
        print("no vlf_*.cs16 captures found", file=sys.stderr)
        return 1
    res = run_many(paths, args.sessions, args.workers)
    return int(write_census(res, args.out) > 0)


if __name__ == "__main__":
    sys.exit(main())
