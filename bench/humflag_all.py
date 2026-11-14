"""Relabel every banked VLF capture with the per-10 s hum flag and check it.

Reads (read-only) ~/dlayer_run/data/vlf/*/vlf_*Z.cs16 (or --vlf DIR) plus any extra .cs16
given with --extra, and the census table given with --census (a CSV with columns file,
hum_status, solar_class; file as YYYYMMDD/vlf_HHMMZ.cs16). Writes into --out (required;
nothing is written inside the repo unless you point it there):

  <out>/hum_blocks.csv     one row per 10 s block
  <out>/hum_captures.csv   one row per capture (hum_frac, first hum block, status, census status)
  <out>/acceptance.json    the plan step 2 acceptance numbers

Acceptance (plan step 2): 11-05 04:30Z first hum block at 440-460 s; 11-06 02:30Z at
80-100 s; capture status agrees with census.csv on >= 148/150; no block flagged in
the clean-day captures; at most 2 % of blocks with an interval share in the 0.1-0.7 gap.

Every number is reported twice: for the module's flag (``hum``: phase lock over the
Rice level and >= 20 buzz edges per block) and for the plan's literal rule
(``hum_plan``: N >= 100, share > 0.3, same R level), so the change can be judged on
the corpus. "Clean-day captures" is read two ways: the golden set's clean roles in
bench/corpus.json (clean_day, clean_evening: 3 captures) and, more broadly, every
capture the census calls clean in daytime. The golden set's hum_on_s intervals are
also scored block by block (they were made with a rough 20/s rate cut, and the
corpus notes say 11-05 20Z and 11-06 10Z hold buzz below that cut).

The census labelled each capture from three 20 s slices (60, 300, 540 s) and flagged
a slice when its 1 ms-debounced 120 Hz interval share was above 0.3: hum = 3/3, partial
= 1-2, clean = 0. Two agreements are reported: whole-file status (what this module
says over all 60 blocks) and slice-matched status (our flags only on the census slice
blocks, either of the two 10 s blocks counts), which isolates rule differences from
coverage differences.

With --events the detector's raw impulse starts are computed per capture (front end +
detector, about 16 s of CPU per 600 s capture) and fed to the flag's second edge source
(``humflag.add_event_stats``); ``hum`` is then edges OR events and the per-block CSV
also carries hum_edge, hum_ev and the event statistics.

Usage: python bench/humflag_all.py --out DIR --census CSV [--events] [--vlf DIR] [--extra FILE.cs16 ...] [--workers 2]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from multiprocessing import Pool
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sferic_watch import humflag as hf  # noqa: E402

VLF = Path.home() / "dlayer_run/data/vlf"
CENSUS_SLICES = (60, 300, 540)
CORPUS = Path(__file__).resolve().parent / "corpus.json"
BLOCK_COLS = ["file", "block", "t0_s", "dur_s", "n_edges", "rate_hz", "share", "share_chance",
              "R", "r_chance", "r_crit", "f_mains", "n_locked", "buzz_rate_hz", "hum", "hum_edge", "hum_plan",
              "n_ev", "R_ev", "r_crit_ev", "f_ev", "n_locked_ev", "hum_ev10", "hum_ev_win", "hum_ev"]
RULES = ("hum", "hum_plan")


def label(path: Path) -> str:
    return f"{path.parent.name}/{path.name}"


def detector_starts(path: Path) -> "np.ndarray":
    """Raw impulse starts (s) of the v2 detector on a whole capture (read-only use of detect / pipeline)."""
    from sferic_watch import detect as dt, io as swio, pipeline as pl
    x = swio.read_slice(str(path), 0, swio.capture_info(str(path))["secs"], dc="none")
    p, _ = dt.band_powers(x, pl.FS)
    del x
    return pl.raw_impulses(p, pl.FS)[0] / pl.FS


def run_one(job):
    path, events = job
    t0 = time.time()
    blocks = hf.flag_file(path, events=detector_starts(path) if events else None)
    return label(path), blocks, time.time() - t0


def slice_status(blocks: list[dict], key: str = "hum") -> str:
    """Status from our flags on the census slice positions only (two 10 s blocks per 20 s slice)."""
    by_t = {round(b["t0_s"]): b[key] for b in blocks}
    hits = sum(bool(by_t.get(s) or by_t.get(s + 10)) for s in CENSUS_SLICES)
    return "hum" if hits == 3 else ("partial" if hits else "clean")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output folder (created)")
    ap.add_argument("--census", required=True, help="census CSV: file, hum_status, solar_class columns")
    ap.add_argument("--vlf", default=str(VLF), help="folder of YYYYMMDD/vlf_HHMMZ.cs16 captures")
    ap.add_argument("--extra", nargs="*", default=[], help="more .cs16 captures (e.g. a live night)")
    ap.add_argument("--events", action="store_true", help="also use the detector's starts as an edge source")
    ap.add_argument("--workers", type=int, default=2)
    a = ap.parse_args()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    files = sorted(Path(a.vlf).glob("*/vlf_*Z.cs16")) + [Path(p) for p in a.extra if Path(p).exists()]
    t_start = time.time()
    with Pool(min(2, a.workers)) as pool:
        res = pool.map(run_one, [(f, a.events) for f in files], chunksize=1)
    wall = time.time() - t_start

    census = {r["file"]: r for r in csv.DictReader(open(a.census))}

    with open(out / "hum_blocks.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, BLOCK_COLS)
        w.writeheader()
        for lab, blocks, _ in res:
            for b in blocks:
                row = {k: b[k] for k in BLOCK_COLS if k in b}
                row["file"] = lab
                for k in ("share", "share_chance", "R", "r_chance", "r_crit", "n_locked", "buzz_rate_hz"):
                    row[k] = round(row[k], 4) if row[k] == row[k] else ""
                row["f_mains"] = round(b["f_mains"], 4) if b["f_mains"] == b["f_mains"] else ""
                row["rate_hz"] = round(b["rate_hz"], 2)
                for k in ("hum", "hum_edge", "hum_plan", "hum_ev10", "hum_ev_win", "hum_ev"):
                    if k in b:
                        row[k] = int(b[k])
                for k in ("R_ev", "r_crit_ev", "f_ev", "n_locked_ev"):
                    if k in b:
                        row[k] = round(b[k], 4) if b[k] == b[k] else ""
                w.writerow(row)

    caps = []
    for lab, blocks, secs in res:
        c = census.get(lab)
        row = {"file": lab}
        for key in RULES:
            sfx = "" if key == "hum" else "_plan"
            sm = hf.summarize(blocks, key)
            row.update({f"hum_blocks{sfx}": sm["hum_blocks"], f"hum_frac{sfx}": round(sm["hum_frac"], 4),
                        f"first_hum_s{sfx}": sm["first_hum_s"], f"last_hum_s{sfx}": sm["last_hum_s"],
                        f"status{sfx}": sm["status"], f"slice_status{sfx}": slice_status(blocks, key)})
        row.update({"n_blocks": len(blocks), "census_status": c["hum_status"] if c else "",
                    "census_solar": c["solar_class"] if c else "", "cpu_s": round(secs, 2)})
        caps.append(row)
    with open(out / "hum_captures.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, list(caps[0]))
        w.writeheader()
        w.writerows(caps)

    blocks_by = {lab: bl for lab, bl, _ in res}
    acc = {"rules": {key: acceptance(caps, blocks_by, key) for key in RULES}}
    blocks_all = [b for bl in blocks_by.values() for b in bl]
    gap = [b for b in blocks_all if 0.1 <= b["share"] <= 0.7]
    big = [b for b in blocks_all if b["n_edges"] >= hf.N_MIN]
    gap_big = [b for b in big if 0.1 <= b["share"] <= 0.7]
    acc.update({
        "n_blocks": len(blocks_all),
        "gap_frac_all_blocks": len(gap) / len(blocks_all),
        "gap_blocks_flagged_hum": sum(b["hum"] for b in gap),
        "n_blocks_N_ge_100": len(big),
        "gap_frac_N_ge_100": len(gap_big) / max(1, len(big)),
        "wall_s": round(wall, 1),
        "cpu_s_per_capture_median": sorted(c["cpu_s"] for c in caps)[len(caps) // 2],
    })
    (out / "acceptance.json").write_text(json.dumps(acc, indent=1))
    print(json.dumps(acc, indent=1))


def golden():
    """Golden captures from bench/corpus.json keyed by 'YYYYMMDD/vlf_HHMMZ.cs16'."""
    out = {}
    for c in json.loads(CORPUS.read_text())["captures"]:
        p = Path(c["path"])
        out[f"{p.parent.name}/{p.name}"] = c
    return out


def acceptance(caps: list[dict], blocks_by: dict, key: str) -> dict:
    """The step-2 acceptance numbers for one rule (key 'hum' or 'hum_plan')."""
    sfx = "" if key == "hum" else "_plan"
    byf = {c["file"]: c for c in caps}
    inc = [c for c in caps if c["census_status"]]
    gold = golden()
    clean_gold = [f for f, g in gold.items() if set(g["roles"]) & {"clean_day", "clean_evening"}]
    clean_day = [c for c in inc if c["census_status"] == "clean" and c["census_solar"] == "day"]
    # golden hum_on_s intervals scored per 10 s block
    g_tot = g_ok = 0
    g_rows = []
    for f, g in gold.items():
        if f not in blocks_by:
            continue
        on = g.get("hum_on_s") or []
        bl = blocks_by[f]
        lab = [any(a <= b["t0_s"] < z for a, z in on) for b in bl]
        ok = sum(bool(b[key]) == l for b, l in zip(bl, lab))
        g_tot += len(bl)
        g_ok += ok
        g_rows.append((f, g["roles"][0], ok, len(bl)))
    return {
        "onset_1105_0430": byf.get("20261105/vlf_0430Z.cs16", {}).get("first_hum_s" + sfx),
        "onset_1106_0230": byf.get("20261106/vlf_0230Z.cs16", {}).get("first_hum_s" + sfx),
        "n_census": len(inc),
        "agree_whole_file": sum(c["status" + sfx] == c["census_status"] for c in inc),
        "agree_slice_matched": sum(c["slice_status" + sfx] == c["census_status"] for c in inc),
        "disagree_whole_file": [(c["file"], c["census_status"], c["status" + sfx], c["hum_frac" + sfx])
                                for c in inc if c["status" + sfx] != c["census_status"]],
        "disagree_slice_matched": [(c["file"], c["census_status"], c["slice_status" + sfx])
                                   for c in inc if c["slice_status" + sfx] != c["census_status"]],
        "golden_clean_captures": clean_gold,
        "golden_clean_flagged_blocks": sum(byf[f]["hum_blocks" + sfx] for f in clean_gold if f in byf),
        "census_clean_day_captures": len(clean_day),
        "census_clean_day_flagged_blocks": sum(c["hum_blocks" + sfx] for c in clean_day),
        "census_clean_day_captures_with_flags": [(c["file"], c["hum_blocks" + sfx]) for c in clean_day
                                                 if c["hum_blocks" + sfx]],
        "golden_block_agreement": f"{g_ok}/{g_tot}",
        "golden_block_agreement_by_capture": g_rows,
        "status_counts_whole": {s: sum(c["status" + sfx] == s for c in inc) for s in ("hum", "partial", "clean")},
        "census_counts": {s: sum(c["census_status"] == s for c in inc) for s in ("hum", "partial", "clean")},
        "tonight": [{k: c[k] for k in ("file", "n_blocks", "hum_blocks" + sfx, "status" + sfx)}
                    for c in caps if "20261114" in c["file"]],
    }


if __name__ == "__main__":
    main()
