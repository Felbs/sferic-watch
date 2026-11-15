"""Fetch GOES GLM L2 LCFA lightning for every banked VLF capture, decode it, and save per-capture npz files.

Usage (heavy runs: wrap in systemd-run --user --scope -q -p MemoryMax=4G nice -n 10):
  python bench/glm_fetch.py fetch                       # all captures under ~/dlayer_run/data/vlf + ~/storm_live
  python bench/glm_fetch.py fetch --capture PATH.cs16   # one or more given captures
  python bench/glm_fetch.py validate                    # decode checks on the raw files already downloaded
  python bench/glm_fetch.py prune --max-raw-gb 3        # drop raw .nc if they exceed the cap (npz are kept)
  python bench/glm_fetch.py lighttime                   # G18 vs G19 twin groups: are GLM times at the flash?
  python bench/glm_fetch.py retag                       # rewrite time_ref in npz files written before 2026-11-14

Data source: the public NOAA buckets noaa-goes19 (GOES-East) and noaa-goes18 (GOES-West), plain HTTPS, no
credentials. Files are chosen by TIME only (the capture window from its JSON sidecar, padded 60 s each side); no
position is ever sent anywhere.

Decoding (verified against the file metadata, see `validate`):
  * time variables are int16 with _Unsigned=true, scale_factor ~0.00038148 s and add_offset -5 s; their `units`
    attribute names the absolute reference, "seconds since <time_coverage_start>", which equals product_time
    (seconds since 2000-01-01 12:00:00, counted without leap seconds). Times are stored here as POSIX seconds (UTC).
  * GLM L2 times are referenced to the flash (the optical emission), not to the satellite: the ~0.12 s
    flash-to-GEO light time is already removed by the ground processing. Measured two ways: the same group seen by
    G18 and G19 (light times differing by -4..+9 ms) has the same time in both (`lighttime` below), and the
    VLF-minus-GLM residual does not depend on the light time (bench/glm_truth.py). Any lag against the VLF clock is
    left to a lag scan downstream.
  * flash/group lat/lon are float32 degrees (lightning-ellipsoid parallax already applied by NOAA).
  * energy (J) and area (m^2) are unsigned int16 with scale/offset and _FillValue -1 (decoded as NaN).

Merge: G19 and G18 overlap over the central US. A G18 flash that matches a G19 flash within 50 ms and 50 km is
dropped (G19 preferred) together with all of its groups.

Output, one file per capture: $SFERIC_GLM_DIR (default ~/.local/share/sferic-watch/glm)/glm_<capture-id>.npz with
  flash_t, flash_t_last (POSIX s), flash_lat, flash_lon, flash_energy_J, flash_area_m2, flash_quality, flash_uid,
  flash_sat, flash_dist_km; group_t, group_lat, group_lon, group_energy_J, group_area_m2, group_quality,
  group_parent_uid, group_sat, group_dist_km; and capture metadata (t0, secs, window, missing file slots).
  uid = sat * 2**48 + (product start in s since J2000 // 20) * 2**16 + per-file id, so uids are unique across files.

The station position is read only inside this process, from $SFERIC_STATION (default
~/.local/share/sferic-watch/station.json), to compute great-circle distances. It is never printed or written
anywhere except as distances inside the private npz files. Everything printed is an aggregate.
"""
import argparse
import concurrent.futures as cf
import datetime as dt
import glob
import json
import os
import re
import sys
import time
import urllib.parse
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

HOME = Path.home()
GLM_DIR = Path(os.environ.get("SFERIC_GLM_DIR", HOME / ".local/share/sferic-watch/glm"))
RAW_DIR = GLM_DIR / "raw"
STATION = Path(os.environ.get("SFERIC_STATION", HOME / ".local/share/sferic-watch/station.json"))
CAPTURE_GLOBS = [str(HOME / "dlayer_run/data/vlf/*/vlf_*.cs16"), str(HOME / "storm_live/vlf_*.cs16")]
SATS = {19: "noaa-goes19", 18: "noaa-goes18"}
PAD_S = 60.0
MAX_CONN = 8
J2000 = dt.datetime(2000, 1, 1, 12, tzinfo=dt.timezone.utc)
EARTH_R_KM = 6371.0088
MATCH_DT_S, MATCH_KM = 0.050, 50.0
TIME_REF = "POSIX UTC seconds; GLM optical time at the flash (light time to GEO already removed by NOAA)"
KEY_RE = re.compile(r"OR_GLM-L2-LCFA_G(\d+)_s(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_e(\d{4})(\d{3})(\d{2})(\d{2})(\d{2})(\d)_c")


# ---------------------------------------------------------------- captures

def capture_id(p: Path) -> str:
    """vlf_HHMMZ.cs16 in a YYYYMMDD dir -> YYYYMMDD_HHMMZ; vlf_YYYYMMDD_HHMMZ.cs16 -> YYYYMMDD_HHMMZ."""
    stem = p.stem[4:] if p.stem.startswith("vlf_") else p.stem
    return stem if "_" in stem else f"{p.parent.name}_{stem}"


def load_captures(paths=None):
    if not paths:
        paths = sorted(set(sum((glob.glob(g) for g in CAPTURE_GLOBS), [])))
    caps = []
    for p in map(Path, paths):
        side = json.loads(p.with_suffix(".json").read_text())
        t0 = dt.datetime.fromisoformat(side["utc"]).timestamp()
        secs = float(side.get("secs") or p.stat().st_size / 4 / float(side["fs_hz"]))
        caps.append({"id": capture_id(p), "path": str(p), "t0": t0, "secs": secs,
                     "ntp_synced": side.get("ntp_synced"), "w0": t0 - PAD_S, "w1": t0 + secs + PAD_S})
    return caps


# ---------------------------------------------------------------- listing / download

def _http(url, tries=4):
    for k in range(tries):
        try:
            with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "glm-fetch/1"}),
                                        timeout=60) as r:
                return r.read()
        except Exception:
            if k == tries - 1:
                raise
            time.sleep(2 * (k + 1))


def list_hour(sat, hour):
    """All (key, size, t_start, t_end) in one bucket hour directory."""
    base = f"https://{SATS[sat]}.s3.amazonaws.com/"
    prefix = f"GLM-L2-LCFA/{hour:%Y/%j/%H}/"
    out, token = [], None
    ns = "{http://s3.amazonaws.com/doc/2006-03-01/}"
    while True:
        q = f"?list-type=2&prefix={prefix}" + (f"&continuation-token={urllib.parse.quote(token)}" if token else "")
        root = ET.fromstring(_http(base + q))
        for c in root.findall(f"{ns}Contents"):
            key = c.find(f"{ns}Key").text
            m = KEY_RE.search(key)
            if not m:
                continue
            out.append((key, int(c.find(f"{ns}Size").text), _keytime(m.groups()[1:7]), _keytime(m.groups()[7:13])))
        if root.findtext(f"{ns}IsTruncated") == "true":
            token = root.findtext(f"{ns}NextContinuationToken")
        else:
            return out


def _keytime(g):
    y, j, h, mi, s, tenth = map(int, g)
    return (dt.datetime(y, 1, 1, tzinfo=dt.timezone.utc) + dt.timedelta(days=j - 1, hours=h, minutes=mi, seconds=s,
                                                                         milliseconds=100 * tenth)).timestamp()


def raw_path(sat, key):
    return RAW_DIR / f"G{sat}" / key.split("/", 1)[1]


def download(sat, key, size):
    p = raw_path(sat, key)
    if p.exists() and p.stat().st_size == size:
        return p, 0
    p.parent.mkdir(parents=True, exist_ok=True)
    data = _http(f"https://{SATS[sat]}.s3.amazonaws.com/{key}")
    if len(data) != size:
        raise IOError(f"short read {key}: {len(data)} != {size}")
    tmp = p.with_suffix(".part")
    tmp.write_bytes(data)
    tmp.rename(p)
    return p, len(data)


# ---------------------------------------------------------------- decoding

def _attr(v, name, default=None):
    a = v.attrs.get(name, default)
    if isinstance(a, bytes):
        a = a.decode()
    if isinstance(a, np.ndarray) and a.size == 1:
        a = a.item()
    return a


def _decode(v):
    """netCDF CF decoding by hand: _Unsigned, _FillValue -> NaN, scale_factor, add_offset."""
    raw = v[()]
    raw = np.atleast_1d(raw)
    fill = _attr(v, "_FillValue")
    isfill = (raw == fill) if fill is not None else np.zeros(raw.shape, bool)
    if str(_attr(v, "_Unsigned", "false")).lower() == "true" and raw.dtype.kind == "i":
        raw = raw.view(raw.dtype.str.replace("i", "u"))
    sf, ao = _attr(v, "scale_factor"), _attr(v, "add_offset")
    if sf is None and ao is None:
        return raw.astype(np.float64) if isfill.any() else raw
    out = raw.astype(np.float64) * (1.0 if sf is None else float(sf)) + (0.0 if ao is None else float(ao))
    out[isfill] = np.nan
    return out


def _units_ref(v):
    """'seconds since 2026-10-19 15:00:00.000' -> POSIX seconds of the reference."""
    u = _attr(v, "units")
    m = re.match(r"seconds since (\d{4}-\d\d-\d\d)[ T](\d\d:\d\d:\d\d(?:\.\d+)?)", u)
    return dt.datetime.fromisoformat(f"{m.group(1)}T{m.group(2)}+00:00").timestamp()


def read_glm(path, sat):
    import h5py
    with h5py.File(path, "r") as f:
        pt = float(f["product_time"][()])
        cov0 = dt.datetime.fromisoformat(_attr(f, "time_coverage_start").replace("Z", "+00:00")).timestamp()
        cov1 = dt.datetime.fromisoformat(_attr(f, "time_coverage_end").replace("Z", "+00:00")).timestamp()
        ref = _units_ref(f["flash_time_offset_of_first_event"])
        file_key = int(pt // 20) << 16
        d = {"pt_posix": J2000.timestamp() + pt, "ref": ref, "cov0": cov0, "cov1": cov1}
        nf = int(f["flash_count"][()]) if "flash_count" in f else len(f["flash_lat"])
        ng = int(f["group_count"][()]) if "group_count" in f else len(f["group_lat"])
        if nf <= 0 or "flash_lat" not in f:
            d["flash"], d["group"] = _cat([], "flash"), _cat([], "group")
            return d
        base = sat * (1 << 48) + file_key
        d["flash"] = {
            "t": ref + _decode(f["flash_time_offset_of_first_event"]),
            "t_last": ref + _decode(f["flash_time_offset_of_last_event"]),
            "lat": _decode(f["flash_lat"]).astype(np.float64), "lon": _decode(f["flash_lon"]).astype(np.float64),
            "energy_J": _decode(f["flash_energy"]), "area_m2": _decode(f["flash_area"]),
            "quality": _decode(f["flash_quality_flag"]).astype(np.int16),
            "uid": base + _decode(f["flash_id"]).astype(np.int64),
        }
        gref = _units_ref(f["group_time_offset"])
        d["group"] = {
            "t": gref + _decode(f["group_time_offset"]),
            "lat": _decode(f["group_lat"]).astype(np.float64), "lon": _decode(f["group_lon"]).astype(np.float64),
            "energy_J": _decode(f["group_energy"]), "area_m2": _decode(f["group_area"]),
            "quality": _decode(f["group_quality_flag"]).astype(np.int16),
            "parent_uid": base + _decode(f["group_parent_flash_id"]).astype(np.int64),
        }
        assert len(d["flash"]["t"]) == nf and len(d["group"]["t"]) == ng
    return d


# ---------------------------------------------------------------- geometry / merge

def gc_km(lat1, lon1, lat2, lon2):
    p1, p2 = np.radians(lat1), np.radians(lat2)
    dp, dl = p2 - p1, np.radians(np.asarray(lon2) - np.asarray(lon1))
    a = np.sin(dp / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * EARTH_R_KM * np.arcsin(np.sqrt(np.clip(a, 0, 1)))


def _station():
    s = json.loads(STATION.read_text())
    return float(s["lat"]), float(s["lon"])


def dup_mask(t19, la19, lo19, t18, la18, lo18):
    """True for each G18 flash that has a G19 flash within MATCH_DT_S and MATCH_KM."""
    o = np.argsort(t19)
    t19, la19, lo19 = t19[o], la19[o], lo19[o]
    lo_i = np.searchsorted(t19, t18 - MATCH_DT_S, "left")
    hi_i = np.searchsorted(t19, t18 + MATCH_DT_S, "right")
    dup = np.zeros(len(t18), bool)
    for k in np.nonzero(hi_i > lo_i)[0]:
        sl = slice(lo_i[k], hi_i[k])
        dup[k] = bool((gc_km(la18[k], lo18[k], la19[sl], lo19[sl]) <= MATCH_KM).any())
    return dup


FKEYS = ("t", "t_last", "lat", "lon", "energy_J", "area_m2", "quality", "uid")
GKEYS = ("t", "lat", "lon", "energy_J", "area_m2", "quality", "parent_uid")


def _cat(parts, sub):
    keys = FKEYS if sub == "flash" else GKEYS
    ints = {"quality": np.int16, "uid": np.int64, "parent_uid": np.int64}
    return {k: np.concatenate([np.asarray(p[sub][k]) for p in parts] + [np.zeros(0, ints.get(k, np.float64))])
            for k in keys}


# ---------------------------------------------------------------- main steps

def plan(caps):
    hours = set()
    for c in caps:
        h = int(c["w0"] // 3600) * 3600
        while h < c["w1"] + 30:          # +30: a file whose name starts inside the window but ends after it
            hours.add(h)
            h += 3600
    jobs = [(s, dt.datetime.fromtimestamp(h, dt.timezone.utc)) for s in SATS for h in sorted(hours)]
    listing = {}
    with cf.ThreadPoolExecutor(MAX_CONN) as ex:
        for (s, h), res in zip(jobs, ex.map(lambda j: list_hour(*j), jobs)):
            listing.setdefault(s, []).extend(res)
    need, per_cap = {}, {}
    for c in caps:
        per_cap[c["id"]] = {}
        for s, files in listing.items():
            sel = [f for f in files if f[3] > c["w0"] and f[2] < c["w1"]]
            per_cap[c["id"]][s] = sel
            for f in sel:
                need[(s, f[0])] = f[1]
    return listing, need, per_cap


def missing_slots(files, w0, w1):
    """20 s slots inside [w0, w1) with no file start (GLM files start on :00/:20/:40)."""
    have = {int(round(f[2])) // 20 for f in files}
    slots = range(int(w0 // 20), int(np.ceil(w1 / 20)))
    return [s * 20 for s in slots if s not in have]


def build_capture(c, files_by_sat, st, cache):
    parts = {}
    for s, files in files_by_sat.items():
        parts[s] = []
        for key, size, _, _ in files:
            ck = (s, key)
            if ck not in cache:
                cache[ck] = read_glm(raw_path(s, key), s)
            parts[s].append(cache[ck])
    fl = {s: _cat(p, "flash") for s, p in parts.items()}
    gr = {s: _cat(p, "group") for s, p in parts.items()}
    for s in fl:   # keep the padded window only (by flash first-event time; groups follow their flash)
        kf = (fl[s]["t"] >= c["w0"]) & (fl[s]["t"] < c["w1"])
        keep_uid = fl[s]["uid"][kf]
        fl[s] = {k: v[kf] for k, v in fl[s].items()}
        kg = np.isin(gr[s]["parent_uid"], keep_uid)
        gr[s] = {k: v[kg] for k, v in gr[s].items()}
    n18_raw = len(fl[18]["t"])
    ndup = 0
    if len(fl[18]["t"]) and len(fl[19]["t"]):
        dup = dup_mask(fl[19]["t"], fl[19]["lat"], fl[19]["lon"], fl[18]["t"], fl[18]["lat"], fl[18]["lon"])
        ndup = int(dup.sum())
        drop_uid = fl[18]["uid"][dup]
        fl[18] = {k: v[~dup] for k, v in fl[18].items()}
        kg = ~np.isin(gr[18]["parent_uid"], drop_uid)
        gr[18] = {k: v[kg] for k, v in gr[18].items()}
    F = {k: np.concatenate([fl[s][k] for s in (19, 18)]) for k in FKEYS}
    G = {k: np.concatenate([gr[s][k] for s in (19, 18)]) for k in GKEYS}
    F["sat"] = np.concatenate([np.full(len(fl[s]["t"]), s, np.int8) for s in (19, 18)])
    G["sat"] = np.concatenate([np.full(len(gr[s]["t"]), s, np.int8) for s in (19, 18)])
    o = np.argsort(F["t"], kind="stable"); F = {k: v[o] for k, v in F.items()}
    o = np.argsort(G["t"], kind="stable"); G = {k: v[o] for k, v in G.items()}
    F["dist_km"] = gc_km(st[0], st[1], F["lat"], F["lon"])
    G["dist_km"] = gc_km(st[0], st[1], G["lat"], G["lon"])
    miss = {s: missing_slots(files_by_sat.get(s, []), c["w0"], c["w1"]) for s in SATS}
    out = {f"flash_{k}": v for k, v in F.items()}
    out.update({f"group_{k}": v for k, v in G.items()})
    out.update(capture_id=c["id"], t0=c["t0"], secs=c["secs"], w0=c["w0"], w1=c["w1"],
               n_files_g19=len(files_by_sat.get(19, [])), n_files_g18=len(files_by_sat.get(18, [])),
               missing_g19=np.array(miss[19], float), missing_g18=np.array(miss[18], float),
               n_g18_dupes_dropped=ndup, time_ref=TIME_REF)
    GLM_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(GLM_DIR / f"glm_{c['id']}.npz", **out)
    core = (F["t"] >= c["t0"]) & (F["t"] < c["t0"] + c["secs"])
    d = F["dist_km"][core]
    return {"id": c["id"], "n_flash_win": int(len(F["t"])), "n_group_win": int(len(G["t"])),
            "n_flash_core": int(core.sum()), "n_g18_raw": n18_raw, "n_g18_dupes": ndup,
            "within": {r: int((d <= r).sum()) for r in (500, 1000, 2000, 4000)},
            "missing": {s: len(miss[s]) for s in SATS}}


def _q(a):
    a = np.asarray(a, float)
    return {"min": float(a.min()), "p25": float(np.percentile(a, 25)), "median": float(np.median(a)),
            "p75": float(np.percentile(a, 75)), "max": float(a.max()), "mean": round(float(a.mean()), 1)}


def cmd_fetch(a):
    caps = load_captures(a.capture)
    print(f"captures: {len(caps)}", flush=True)
    listing, need, per_cap = plan(caps)
    tot = sum(need.values())
    print(f"files needed: {len(need)} ({tot / 1e6:.0f} MB): G19 {sum(1 for s, _ in need if s == 19)}, "
          f"G18 {sum(1 for s, _ in need if s == 18)}", flush=True)
    got, fails = 0, []
    with cf.ThreadPoolExecutor(MAX_CONN) as ex:
        futs = {ex.submit(download, s, k, n): (s, k) for (s, k), n in need.items()}
        for i, fu in enumerate(cf.as_completed(futs)):
            try:
                got += fu.result()[1]
            except Exception as e:
                fails.append((futs[fu], repr(e)))
            if (i + 1) % 1000 == 0:
                print(f"  {i + 1}/{len(futs)} files, {got / 1e6:.0f} MB new", flush=True)
    print(f"downloaded new: {got / 1e6:.0f} MB; failures: {len(fails)}", flush=True)
    for f in fails[:10]:
        print("  FAIL", f)
    failed = set(k for k, _ in fails)
    st = _station()
    rows, cache = [], {}
    for c in sorted(caps, key=lambda c: c["t0"]):
        fb = {s: [f for f in per_cap[c["id"]][s] if (s, f[0]) not in failed] for s in SATS}
        rows.append(build_capture(c, fb, st, cache))
        if len(cache) > 400:
            cache.clear()
    summary(rows, len(need), tot, fails)


def summary(rows, nfiles, tot, fails):
    nc = np.array([r["n_flash_core"] for r in rows])
    res = {
        "captures": len(rows), "files": nfiles, "total_MB": round(tot / 1e6, 1), "download_failures": len(fails),
        "captures_with_missing_slots": {f"G{s}": sum(1 for r in rows if r["missing"][s]) for s in SATS},
        "missing_slots_total": {f"G{s}": sum(r["missing"][s] for r in rows) for s in SATS},
        "flashes_per_capture_600s_merged": _q(nc),
        "flashes_per_capture_padded_window": _q([r["n_flash_win"] for r in rows]),
        "groups_per_capture_padded_window": _q([r["n_group_win"] for r in rows]),
        "g18_dupes_dropped_total": sum(r["n_g18_dupes"] for r in rows),
        "g18_flashes_before_merge_total": sum(r["n_g18_raw"] for r in rows),
        "within_km_per_capture_600s": {str(k): _q([r["within"][k] for r in rows]) for k in (500, 1000, 2000, 4000)},
        "captures_with_any_flash_within_km": {str(k): sum(1 for r in rows if r["within"][k]) for k in
                                              (500, 1000, 2000, 4000)},
    }
    print(json.dumps(res, indent=1))
    (GLM_DIR / "fetch_summary.json").write_text(json.dumps({"aggregate": res}, indent=1))


def cmd_validate(a):
    """Decode checks on a sample of raw files: times inside coverage, reference == product_time, lat/lon, energy."""
    files = sorted(RAW_DIR.glob("G*/**/*.nc"))
    if not files:
        sys.exit("no raw files")
    rng = np.random.default_rng(0)
    pick = [files[i] for i in rng.choice(len(files), min(a.n, len(files)), replace=False)]
    agg = {s: {"files": 0, "flashes": 0, "groups": 0, "ref_minus_cov0": [], "pt_minus_cov0": [],
               "flash_in_cov": 0, "flash_before_cov": [], "group_in_cov": 0, "flash_last_le_cov1": 0,
               "lat": [], "lon": [], "fe": [], "ge": [], "fa": [], "nan_energy": 0,
               "orphan_groups": 0, "first_le_last": 0, "groups_within_parent_span": 0}
           for s in SATS}
    import h5py
    for p in pick:
        s = int(p.parts[-5][1:])
        d = read_glm(p, s)
        A = agg[s]
        A["files"] += 1
        A["ref_minus_cov0"].append(d["ref"] - d["cov0"])
        A["pt_minus_cov0"].append(d["pt_posix"] - d["cov0"])
        F, G = d["flash"], d["group"]
        A["flashes"] += len(F["t"]); A["groups"] += len(G["t"])
        if len(F["t"]):
            A["flash_in_cov"] += int(((F["t_last"] >= d["cov0"] - 1e-3) & (F["t_last"] <= d["cov1"] + 1e-3)).sum())
            A["flash_before_cov"].extend((d["cov0"] - F["t"]).tolist())
            A["first_le_last"] += int((F["t"] <= F["t_last"] + 1e-6).sum())
            A["lat"].extend([F["lat"].min(), F["lat"].max()]); A["lon"].extend([F["lon"].min(), F["lon"].max()])
            A["fe"].extend(F["energy_J"].tolist()); A["fa"].extend(F["area_m2"].tolist())
            A["ge"].extend(G["energy_J"].tolist())
            A["nan_energy"] += int(np.isnan(F["energy_J"]).sum())
            A["orphan_groups"] += int((~np.isin(G["parent_uid"], F["uid"])).sum())
            idx = {u: i for i, u in enumerate(F["uid"])}
            pi = np.array([idx.get(u, -1) for u in G["parent_uid"]])
            ok = pi >= 0
            A["groups_within_parent_span"] += int(((G["t"][ok] >= F["t"][pi[ok]] - 2e-3) &
                                                   (G["t"][ok] <= F["t_last"][pi[ok]] + 2e-3)).sum())
            A["group_in_cov"] += int(((G["t"] >= d["cov0"] - 5.01) & (G["t"] <= d["cov1"] + 1e-3)).sum())
    out = {}
    for s, A in agg.items():
        if not A["files"]:
            continue
        fe, ge, fa = np.array(A["fe"]), np.array(A["ge"]), np.array(A["fa"])
        out[f"G{s}"] = {
            "files": A["files"], "flashes": A["flashes"], "groups": A["groups"],
            "units_ref_minus_coverage_start_s": [min(A["ref_minus_cov0"]), max(A["ref_minus_cov0"])],
            "product_time_naive_J2000_minus_coverage_start_s": [min(A["pt_minus_cov0"]), max(A["pt_minus_cov0"])],
            "flash_last_event_inside_coverage_frac": A["flash_in_cov"] / max(A["flashes"], 1),
            "flash_first_event_before_coverage_start_s_max": float(max(A["flash_before_cov"] or [0])),
            "group_inside_[cov0-5s,cov1]_frac": A["group_in_cov"] / max(A["groups"], 1),
            "flash_first_le_last_frac": A["first_le_last"] / max(A["flashes"], 1),
            "orphan_groups": A["orphan_groups"],
            "groups_inside_parent_flash_span_frac": A["groups_within_parent_span"] / max(A["groups"], 1),
            "flash_lat_range": [float(min(A["lat"])), float(max(A["lat"]))],
            "flash_lon_range": [float(min(A["lon"])), float(max(A["lon"]))],
            "flash_energy_J_p1_p50_p99": [float(np.nanpercentile(fe, q)) for q in (1, 50, 99)],
            "group_energy_J_p1_p50_p99": [float(np.nanpercentile(ge, q)) for q in (1, 50, 99)],
            "flash_area_km2_p1_p50_p99": [float(np.nanpercentile(fa, q) / 1e6) for q in (1, 50, 99)],
            "flash_energy_fill_nan": A["nan_energy"],
        }
    print(json.dumps(out, indent=1))


def raw_bytes():
    return sum(p.stat().st_size for p in RAW_DIR.glob("G*/**/*.nc"))


def cmd_prune(a):
    b = raw_bytes()
    print(f"raw .nc total: {b / 1e9:.2f} GB (cap {a.max_raw_gb} GB)")
    if b > a.max_raw_gb * 1e9:
        for p in RAW_DIR.glob("G*/**/*.nc"):
            p.unlink()
        print("raw deleted (npz kept)")


def _sat_light_time(lat, lon, sat):
    """Flash-to-GEO light time (s), spherical Earth (same model as bench/glm_truth.light_time_s)."""
    sl = {19: -75.2, 18: -137.0}[sat]
    la, dl = np.radians(lat), np.radians(np.asarray(lon) - sl)
    r = np.sqrt(42164.0 ** 2 + 6378.137 ** 2 - 2 * 42164.0 * 6378.137 * np.cos(la) * np.cos(dl))
    return r / 299792.458


def cmd_lighttime(a):
    """Same lightning group seen by both satellites: is t18 - t19 ~ 0 (times at the flash) or ~ lt18 - lt19 (times at
    the satellite)? Candidate pairs: a G18 and a G19 group within ``--km`` km and +-25 ms (wider than any light-time
    difference, so neither hypothesis is truncated). The pair-difference histograms of dt and of dt - dlt are compared
    (a sharp peak at 0 marks the right reference), and the peak of dt is fitted per 1 ms bin of dlt (slope 0 = flash
    referenced, 1 = satellite referenced). No station is used; lat/lon stay in this process; output is aggregate."""
    hours = sorted({p.parent.relative_to(RAW_DIR / "G19") for p in (RAW_DIR / "G19").glob("*/*/*/*.nc")})
    hours = [h for h in hours if (RAW_DIR / "G18" / h).is_dir()]
    if not hours:
        sys.exit("no raw hours with both satellites")
    pick = hours[:: max(1, len(hours) // a.hours)][: a.hours]
    HALF, STEP = 0.025, 0.0002
    edges = np.arange(-HALF, HALF + STEP / 2, STEP)
    dlt_edges = np.arange(-0.006, 0.0111, 0.001)
    h_dt = np.zeros(len(edges) - 1)
    h_res = np.zeros(len(edges) - 1)
    h_bin = np.zeros((len(dlt_edges) - 1, len(edges) - 1))
    n_pairs, dlts = 0, []

    def load(sat, hour):
        G = [read_glm(p, sat)["group"] for p in sorted((RAW_DIR / f"G{sat}" / hour).glob("*.nc"))]
        G = _cat([{"group": g} for g in G], "group")
        o = np.argsort(G["t"])
        return G["t"][o], G["lat"][o], G["lon"][o]

    for h in pick:
        t9, a9, o9 = load(19, h)
        t8, a8, o8 = load(18, h)
        if not len(t9) or not len(t8):
            continue
        lo = np.searchsorted(t9, t8 - HALF)
        hi = np.searchsorted(t9, t8 + HALF)
        cnt = hi - lo
        rep = np.repeat(np.arange(len(t8)), cnt)
        idx = np.arange(cnt.sum()) - np.repeat(np.cumsum(cnt) - cnt, cnt) + np.repeat(lo, cnt)
        km = gc_km(a8[rep], o8[rep], a9[idx], o9[idx])
        k = km <= a.km
        rep, idx = rep[k], idx[k]
        d_t = t8[rep] - t9[idx]
        d_lt = _sat_light_time(a8[rep], o8[rep], 18) - _sat_light_time(a9[idx], o9[idx], 19)
        h_dt += np.histogram(d_t, edges)[0]
        h_res += np.histogram(d_t - d_lt, edges)[0]
        b = np.clip(np.searchsorted(dlt_edges, d_lt, "right") - 1, 0, len(dlt_edges) - 2)
        for j in range(len(dlt_edges) - 1):
            h_bin[j] += np.histogram(d_t[b == j], edges)[0]
        n_pairs += int(k.sum())
        dlts.append(d_lt)
    mid = (edges[:-1] + edges[1:]) / 2

    def peak(hh):
        bg = float(np.median(hh[np.abs(mid) > 0.015])) if hh.sum() else 0.0
        k = int(np.argmax(hh))
        j = slice(max(k - 3, 0), k + 4)
        w = np.clip(hh[j] - bg, 0, None)
        return (float((mid[j] * w).sum() / w.sum()) if w.sum() else float("nan")), float(hh[k] / max(bg, 1.0))
    xs, ys, ws = [], [], []
    for j in range(len(dlt_edges) - 1):
        if h_bin[j].sum() >= 200:
            pk, ratio = peak(h_bin[j])
            if np.isfinite(pk) and ratio > 3:
                xs.append((dlt_edges[j] + dlt_edges[j + 1]) / 2)
                ys.append(pk)
                ws.append(h_bin[j].sum())
    slope = float(np.polyfit(xs, ys, 1, w=np.sqrt(ws))[0]) if len(xs) >= 3 else None
    p_dt, r_dt = peak(h_dt)
    p_res, r_res = peak(h_res)
    dl = np.concatenate(dlts) if dlts else np.zeros(1)
    print(json.dumps({
        "hours_used": len(pick), "candidate_pairs": n_pairs,
        "dlt_ms_p5_p50_p95": [round(float(x) * 1e3, 2) for x in np.percentile(dl, [5, 50, 95])],
        "peak_of_dt_ms": round(p_dt * 1e3, 3), "peak_over_background_dt": round(r_dt, 1),
        "peak_of_dt_minus_dlt_ms": round(p_res * 1e3, 3), "peak_over_background_dt_minus_dlt": round(r_res, 1),
        "dlt_bins_used": len(xs), "slope_peak_dt_on_dlt": None if slope is None else round(slope, 3),
        "reading": "slope 0 and a sharp dt peak at 0 => GLM times are at the flash; slope 1 => at the satellite"},
        indent=1))


def cmd_retag(a):
    """Rewrite the time_ref field of existing npz files (older ones say, wrongly, 'at satellite'). Mode is kept."""
    n = 0
    for p in sorted(GLM_DIR.glob("glm_*.npz")):
        with np.load(p, allow_pickle=False) as z:
            if str(z["time_ref"]) == TIME_REF:
                continue
            d = {k: z[k] for k in z.files}
        d["time_ref"] = np.array(TIME_REF)
        mode = p.stat().st_mode & 0o777
        tmp = p.with_name(p.stem + ".retag.npz")
        np.savez_compressed(tmp, **d)
        os.chmod(tmp, mode)
        os.replace(tmp, p)
        n += 1
    print(f"retagged {n} files")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("fetch"); f.add_argument("--capture", nargs="*")
    v = sub.add_parser("validate"); v.add_argument("--n", type=int, default=60)
    p = sub.add_parser("prune"); p.add_argument("--max-raw-gb", type=float, default=3.0)
    p = sub.add_parser("lighttime"); p.add_argument("--hours", type=int, default=24)
    p.add_argument("--km", type=float, default=15.0)
    sub.add_parser("retag")
    a = ap.parse_args()
    {"fetch": cmd_fetch, "validate": cmd_validate, "prune": cmd_prune, "lighttime": cmd_lighttime,
     "retag": cmd_retag}[a.cmd](a)


if __name__ == "__main__":
    main()
