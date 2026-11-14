"""Run sferic_watch.health over the banked corpus and check plan step 8.

Usage:
    python bench/health_all.py OUT_DIR [capture.cs16 ...]

Scans every ~/dlayer_run/data/vlf/YYYYMMDD/vlf_HHMMZ.cs16 plus any extra
files given (tonight's ~/storm_live capture), two worker processes, and
writes OUT_DIR/health.csv (one row per capture) and OUT_DIR/health_blocks.csv
(10 s NAA / NAU / NML / spur dBFS, for checking steps by eye).  Then prints
the acceptance checks of plan step 8:

  * LO 0.505 +- 0.01 Hz and clock -1.2..-0.3 ppm wherever NAA's squared
    line is above 20 dB;
  * NAA high-state / step flags at 10-20 08-10Z and 11-06 04:00-05:00 and
    07:30-10:30Z, never on the 11-03..05 night plateaus.  The plateau is
    taken from the census: the sunset dip ends at 22:00Z and the sunrise
    dip starts near 10:30Z, so 22:00-10:30Z (the first run used 23:00,
    which left out a flagged 11-03 2230Z step; the 23:00 count is still
    printed for comparison);
  * sunrise fades near 11:00Z (about -91 dBFS) on 11-03 and 11-05.
Read-only on the data; never touches the radio.
"""
from __future__ import annotations

import csv
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sferic_watch import health as H  # noqa: E402

VLF = Path.home() / "dlayer_run/data/vlf"


def _key(path):
    """(YYYYMMDD, HHMM) from either naming scheme."""
    p = Path(path)
    stem = p.stem.replace("vlf_", "").rstrip("Z")
    if "_" in stem:                      # storm_live: vlf_YYYYMMDD_HHMMZ
        d, t = stem.split("_")
    else:
        d, t = p.parent.name, stem
    return d, t


CACHE = None


def _scan_cached(path):
    """scan_capture() with a per-capture .npz cache in OUT_DIR/cache, so
    re-running the analysis (flags, steps) does not re-read 150 MB files."""
    d, t = _key(path)
    c = CACHE / f"{d}_{t}.npz"
    if c.exists():
        z = np.load(c)
        msk = json.loads(str(z["msk"]))
        blocks = {s: (z[f"{s}_over"], z[f"{s}_dbfs"]) for s in H.LEVEL_STATIONS}
        return {"secs": float(z["secs"]), "clipped": int(z["clipped"]), "blocks": blocks,
                "dc": z["dc"], "msk": msk, "block_secs": float(z["block_secs"])}
    sc = H.scan_capture(path)
    arrs = {f"{s}_over": sc["blocks"][s][0] for s in H.LEVEL_STATIONS}
    arrs.update({f"{s}_dbfs": sc["blocks"][s][1] for s in H.LEVEL_STATIONS})
    np.savez(c, secs=sc["secs"], clipped=sc["clipped"], dc=sc["dc"],
             msk=json.dumps(sc["msk"]), block_secs=sc["block_secs"], **arrs)
    return sc


def _init(cache):
    global CACHE
    CACHE = Path(cache)


def _one(path):
    sc = _scan_cached(path)
    row = H.row_from_scan(path, H.read_sidecar(path), sc)
    d, t = _key(path)
    row["date"], row["hhmm"] = d, t
    bs = float(sc["block_secs"])
    blocks = [(d, t, (i + 0.5) * bs) + tuple(round(float(sc["blocks"][s][1][i]), 2)
                                               for s in ("NAA", "NAU", "NML", "SPUR38"))
              for i in range(sc["blocks"]["NAA"][1].size)]
    return row, blocks


def _in(row, date, t0, t1):
    return row["date"] == date and t0 <= row["hhmm"] <= t1


def main():
    out = Path(sys.argv[1])
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(VLF.glob("*/vlf_*Z.cs16")) + [Path(a) for a in sys.argv[2:]]
    (out / "cache").mkdir(exist_ok=True)
    with Pool(2, initializer=_init, initargs=(str(out / "cache"),)) as pool:
        res = pool.map(_one, [str(f) for f in files], chunksize=1)
    rows = [r for r, _ in res]
    bank = [r for r in rows if r["date"] < "20261114"]
    low_ref = H.corpus_states(bank)
    H.corpus_states([r for r in rows if r not in bank], low_ref=low_ref)
    keys = ["date", "hhmm"] + [k for k in rows[0] if k not in ("date", "hhmm")]
    with open(out / "health.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        w.writerows(rows)
    with open(out / "health_blocks.csv", "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["date", "hhmm", "t_mid_s", "NAA_dbfs", "NAU_dbfs", "NML_dbfs", "SPUR38_dbfs"])
        for _, b in res:
            w.writerows(b)

    acc = {"n_captures": len(rows), "low_ref_dbfs": round(low_ref, 2)}
    # 1. clock / LO wherever NAA line > 20 dB
    good = [r for r in rows if r["NAA_line_db"] > 20]
    lo = np.array([r["NAA_lo_hz"] for r in good])
    ck = np.array([r["NAA_clock_ppm"] for r in good])
    acc["naa_line_gt20"] = len(good)
    acc["lo_min_max"] = [round(float(lo.min()), 5), round(float(lo.max()), 5)]
    acc["lo_in_tol"] = int(np.sum(np.abs(lo - 0.505) <= 0.01))
    acc["clock_min_med_max"] = [round(float(ck.min()), 3), round(float(np.median(ck)), 3), round(float(ck.max()), 3)]
    acc["clock_in_range"] = int(np.sum((ck >= -1.2) & (ck <= -0.3)))
    sel = np.array([r["clock_ppm"] for r in rows if r["clock_src"]])
    acc["selected_clock_n_in_range_min_max"] = [len(sel), int(np.sum((sel >= -1.2) & (sel <= -0.3))),
                                                round(float(sel.min()), 3), round(float(sel.max()), 3)]
    acc["clock_err_med"] = round(float(np.median([r["NAA_clock_err_ppm"] for r in good])), 3)
    both = [r for r in rows if np.isfinite(r["clock_agree_ppm"])]
    acc["naa_nml_both"] = len(both)
    if both:
        acc["naa_minus_nml_ppm_med_absmax"] = [round(float(np.median([r["clock_agree_ppm"] for r in both])), 3),
                                               round(float(np.max(np.abs([r["clock_agree_ppm"] for r in both]))), 3)]
    pull = np.array([r["clock_pull"] for r in both])
    acc["clock_pull_n_std_mad_absmax"] = [int(pull.size), round(float(np.std(pull)), 2),
                                         round(float(1.4826 * np.median(np.abs(pull - np.median(pull)))), 2),
                                         round(float(np.max(np.abs(pull))), 2)] if pull.size else None
    acc["clock_out_of_range"] = [(r["date"], r["hhmm"], r["NAA_line_db"], r["NAA_clock_ppm"], r["NAA_clock_err_ppm"],
                                  r["NAA_glitches"], r["clock_src"], r["clock_ppm"], r["clock_err_ppm"])
                                 for r in good if not -1.2 <= r["NAA_clock_ppm"] <= -0.3]
    acc["naa_out_of_range_within_3sigma_of_median"] = int(sum(
        abs(r["NAA_clock_ppm"] - float(np.median(ck))) <= 3 * r["NAA_clock_err_ppm"]
        for r in good if not -1.2 <= r["NAA_clock_ppm"] <= -0.3))
    acc["clock_glitch_captures"] = [(r["date"], r["hhmm"], r["clock_src"], r["clock_glitches"], r["clock_ppm"])
                                    for r in rows if r["clock_src"] and r["clock_glitches"] > 0]
    acc["quarantined"] = [(r["date"], r["hhmm"], r["quarantine"]) for r in rows if r["quarantine"]]
    # 2. state / step flags
    want = [r for r in rows if _in(r, "20261020", "0800", "1000") or _in(r, "20261106", "0400", "0500")
            or _in(r, "20261106", "0730", "1030")]
    acc["flag_wanted_hit"] = f"{sum(r['naa_flag'] for r in want)}/{len(want)}"
    acc["flag_wanted_missed"] = [(r["date"], r["hhmm"]) for r in want if not r["naa_flag"]]
    for start in ("2200", "2300"):
        plateau = [r for r in rows if r["date"] in ("20261103", "20261104", "20261105")
                   and (r["hhmm"] >= start or r["hhmm"] <= "1030")]
        sfx = "" if start == "2200" else "_from2300"
        acc["flag_on_plateau" + sfx] = f"{sum(r['naa_flag'] for r in plateau)}/{len(plateau)}"
        acc["flag_on_plateau_list" + sfx] = [(r["date"], r["hhmm"], r["NAA_dbfs"], r["NAA_step_db"],
                                              r["NAA_step_t_s"], r["NML_at_step_db"], r["NAU_at_step_db"],
                                              r["step_kind"]) for r in plateau if r["naa_flag"]]
    acc["all_flagged"] = [(r["date"], r["hhmm"], r["NAA_state"], r["NAA_step_db"],
                           r["NAU_at_step_db"], r["NML_at_step_db"], r["step_kind"]) for r in rows if r["naa_flag"]]
    acc["steps"] = [(r["date"], r["hhmm"], r["NAA_step_db"], r["NAA_step_t_s"], r["step_kind"])
                    for r in rows if r["NAA_step_db"] != 0.0]
    acc["naa_off"] = [(r["date"], r["hhmm"]) for r in rows if r["NAA_state"] == "off"]
    # 3. sunrise fades
    for d in ("20261103", "20261105"):
        r = next((r for r in rows if r["date"] == d and r["hhmm"] == "1100"), None)
        acc[f"fade_{d}_1100"] = None if r is None else (r["naa_fade"], r["NAA_min_dbfs"])
    acc["all_fades"] = [(r["date"], r["hhmm"], r["NAA_min_dbfs"]) for r in rows if r["naa_fade"]]
    # extras
    sp = [r["spur_minus_naa_db"] for r in rows if r["NAA_db"] >= 10]
    acc["spur_minus_naa_med_std_n"] = [round(float(np.median(sp)), 2), round(float(np.std(sp)), 2), len(sp)]
    acc["dc_tone_med_min_max"] = [round(float(np.median([r["dc_tone_dbfs"] for r in rows])), 2),
                                  round(float(min(r["dc_tone_dbfs"] for r in rows)), 2),
                                  round(float(max(r["dc_tone_dbfs"] for r in rows)), 2)]
    acc["nau_night_n"] = sum("nau_night" in r["info"] for r in rows)
    acc["tonight"] = [{k: r[k] for k in ("date", "hhmm", "NAA_dbfs", "NAA_line_db", "clock_ppm", "lo_hz",
                                         "NAA_state", "quarantine", "info")} for r in rows if r not in bank]
    (out / "acceptance.json").write_text(json.dumps(acc, indent=1, default=str))
    print(json.dumps(acc, indent=1, default=str))


if __name__ == "__main__":
    main()
