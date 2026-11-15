"""Stress test of the live watcher (``sferic_watch.live``) on the MOCK radio only. It never opens the SDR: the
daemon child it starts poisons ``SoapySDR`` in ``sys.modules`` before anything else, and the lock tests replace
``radio.SoapyRadio`` by a replaying mock.

Run it capped on the shared box:
    systemd-run --user --scope -q -p MemoryMax=4G nice -n 10 python bench/stress_live.py SUB --out DIR [...]

Subcommands (each writes DIR/<name>/... and DIR/<name>.json):
  throughput  unpaced replay of a few captures: signal minutes per wall minute (to pick the soak speed)
  soak        A. the whole banked corpus in time order through the daemon (``--speed``), each capture its own
              segment at its true UTC time (``file_gaps``: the mock jumps its clock to each sidecar time and marks
              a gap, as the real radio would show the 20-min gaps between banked captures). Samples every second
              RSS / CPU / open fds / threads of the daemon and its chain worker; analyses lag, memory trend, CPU
              per signal minute and events against the batch census (``--census``).
  realtime    B. ``--minutes`` wall minutes at 1x on a buzz-heavy capture loop: CPU share, processing latency,
              ring high-water, reader lateness, overflows.
  faults      C. each fault separately on the same captures as a no-fault baseline, then mixed; SIGTERM, SIGKILL +
              restart, torn lines, slow disk, disk nearly full, ENOSPC, the radio lock and the dlayer_run guard.
  alerts      D. every alert of the soak run against GOES GLM (aggregate counts only; reads flash times and
              distances from the private GLM npz files, never positions).
  equiv       E. batch equivalence of the soak run per capture and per minute (``--census``).
  child       internal: one daemon run from a JSON config (instrumented MockRadio + optional Store faults).

Batch reference (``--census DIR``): a census run of the current engine over the same captures, i.e. DIR/census.csv
and DIR/census_sessions/<id>.session.json + <id>.events.csv (``python -m sferic_watch census ... --sessions``).
"""
from __future__ import annotations

import argparse
import errno
import glob
import json
import os
import random
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
if REPO not in sys.path:
    sys.path.insert(0, REPO)

PY = sys.executable
CORPUS_GLOB = os.path.join("~", "dlayer_run", "data", "vlf", "*", "vlf_*.cs16")
GLM_DIR = os.path.join("~", ".local", "share", "sferic-watch", "glm")
STAMP_OFFSET_S = 0.026
PAGE_MB = os.sysconf("SC_PAGE_SIZE") / 2**20
TICK = os.sysconf("SC_CLK_TCK")


def corpus() -> list[str]:
    return sorted(glob.glob(os.path.expanduser(CORPUS_GLOB)))


def cap_id(path: str) -> str:
    return f"vlf_{os.path.basename(os.path.dirname(path))}_{os.path.basename(path)[4:-5]}"


def sidecar_t0(path: str) -> float:
    """Epoch of sample 0 (the sidecar utc is stamped STAMP_OFFSET_S late)."""
    with open(os.path.splitext(path)[0] + ".json") as f:
        return datetime.fromisoformat(json.load(f)["utc"]).timestamp() - STAMP_OFFSET_S


def wjson(path: str, doc) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w") as f:
        json.dump(doc, f, indent=1, default=float)
    os.replace(tmp, path)


# ============================================================================================= the daemon child

def child(cfg_path: str) -> int:
    sys.modules["SoapySDR"] = None            # hard guard: any attempt to import the SDR driver fails here
    os.environ["SFERIC_WATCH_NO_SDR"] = "1"   # ...and SoapyRadio refuses to open (radio.NO_SDR_ENV)
    with open(cfg_path) as f:
        cfg = json.load(f)
    import shutil
    from collections import namedtuple
    from sferic_watch import live as lv
    from sferic_watch import radio as rd

    out = cfg["out"]
    instr = {"ring_hw_samples": 0, "ring_hw_t": None, "pace_late_max_s": 0.0, "pace_late_p99_s": None,
             "file_switches": 0, "short_injected": 0, "disk_blocks": [], "enospc_raised": 0, "free_calls": 0,
             "append_calls": 0, "pid": os.getpid()}
    late = []
    t_start_wall = time.time()

    class StressMock(rd.MockRadio):
        def __init__(self, *a, **kw):
            kw["settle"] = cfg.get("settle", True)
            super().__init__(*a, **kw)
            self._rng = random.Random(cfg.get("seed", 1))
            self._utc = [sidecar_t0(f) for f in self.files] if cfg.get("file_gaps") else None

        def _src_read(self, n):
            i0 = self._src_i
            raw = super()._src_read(n)
            if self._utc is not None and raw.size and self._src_i != i0:
                # a new banked capture: its own sidecar time, and a gap (the recorder stopped in between)
                self.t_start = self._utc[self._src_i] - self._emitted / self.fs - self._clock_off
                self._mark_gap()
                instr["file_switches"] += 1
                instr.setdefault("switch_wall_s", []).append(round(time.time() - t_start_wall, 2))
            return raw

        def _read(self, buf):
            p = cfg.get("short_p", 0.0)
            if p and self._rng.random() < p:
                c = self.chunk
                self.chunk = self._rng.randint(1, c - 1)
                try:
                    r = super()._read(buf)
                finally:
                    self.chunk = c
                instr["short_injected"] += 1
                return r
            return super()._read(buf)

        def _pace(self):
            if self.speed > 0 and self._wall0 is not None:
                due = self._wall0[0] + (self._emitted - self._wall0[1]) / (self.fs * self.speed)
                lt = time.monotonic() - due
                if lt > instr["pace_late_max_s"]:
                    instr["pace_late_max_s"] = lt
                late.append(max(lt, 0.0))
                if len(late) > 200000:
                    del late[:100000]
            super()._pace()

        def _put(self, b):
            super()._put(b)
            if self._ring_n > instr["ring_hw_samples"]:
                instr["ring_hw_samples"] = self._ring_n
                instr["ring_hw_t"] = round(time.time() - t_start_wall, 1)

    rd.MockRadio = StressMock
    if cfg.get("fake_soapy"):
        def fake_soapy(antenna=None, **kw):
            return StressMock(files=cfg["files"], speed=cfg.get("speed", 1.0), loop=True)
        rd.SoapyRadio = fake_soapy

    # -- Store faults (the main thread is the only writer)
    orig_append = lv.Store.append
    blocks = list(cfg.get("slow_disk", []))          # [{"at_append": k, "block_s": 30}]
    enospc = cfg.get("enospc")                       # {"from_append": k, "n": m}

    def append(self, name, recs, t):
        instr["append_calls"] += 1
        k = instr["append_calls"]
        for b in blocks:
            if not b.get("done") and k >= b["at_append"]:
                b["done"] = True
                w0 = time.time()
                time.sleep(b["block_s"])
                instr["disk_blocks"].append({"append": k, "wall_start_s": round(w0 - t_start_wall, 1),
                                             "blocked_s": round(time.time() - w0, 1)})
        if enospc and enospc["from_append"] <= k < enospc["from_append"] + enospc["n"]:
            instr["enospc_raised"] += 1
            raise OSError(errno.ENOSPC, "No space left on device (injected)")
        return orig_append(self, name, recs, t)
    lv.Store.append = append

    sched = cfg.get("free_gb")                        # [[from_call, free_gb], ...] by disk_usage call index
    if sched:
        real = shutil.disk_usage
        DU = namedtuple("usage", "total used free")

        def fake_usage(p):
            instr["free_calls"] += 1
            k = instr["free_calls"]
            v = None
            for c, g in sched:
                if k >= c:
                    v = g
            if v is None:
                return real(p)
            return DU(int(100e9), int(100e9 - v * 1e9), int(v * 1e9))
        lv.shutil.disk_usage = fake_usage

    stop = threading.Event()

    def dump():
        doc = dict(instr)
        if late:
            s = sorted(late)
            doc["pace_late_p99_s"] = s[int(0.99 * (len(s) - 1))]
            doc["pace_late_n"] = len(s)
        wjson(os.path.join(out, "instr.json"), doc)

    def dumper():
        while not stop.wait(2.0):
            try:
                dump()
            except Exception:
                pass
    threading.Thread(target=dumper, daemon=True).start()
    rc = None
    try:
        rc = lv.main(cfg["argv"])
    finally:
        import resource
        stop.set()
        s, c = resource.getrusage(resource.RUSAGE_SELF), resource.getrusage(resource.RUSAGE_CHILDREN)
        instr["rusage"] = {"self_cpu_s": s.ru_utime + s.ru_stime, "children_cpu_s": c.ru_utime + c.ru_stime,
                           "self_maxrss_mb": s.ru_maxrss / 1024, "children_maxrss_mb": c.ru_maxrss / 1024}
        instr["rc"] = rc
        instr["wall_s"] = time.time() - t_start_wall
        dump()
    return rc


# ============================================================================================= process sampling

def _read(p):
    with open(p, "rb") as f:
        return f.read()


def proc_info(pid: int) -> dict | None:
    try:
        rest = _read(f"/proc/{pid}/stat").decode().rsplit(")", 1)[1].split()
        rss = int(_read(f"/proc/{pid}/statm").split()[1]) * PAGE_MB
        cmd = _read(f"/proc/{pid}/cmdline").replace(b"\0", b" ").decode(errors="replace")
        fds = len(os.listdir(f"/proc/{pid}/fd"))
    except (OSError, IndexError, ValueError):
        return None
    role = ("tracker" if "resource_tracker" in cmd else "worker" if "spawn_main" in cmd or "multiprocessing" in cmd
            else "main")
    return {"pid": pid, "role": role, "cpu_s": (int(rest[11]) + int(rest[12])) / TICK, "rss_mb": round(rss, 1),
            "threads": int(rest[17]), "fds": fds}


def descendants(pid: int) -> list[int]:
    out, todo = [], [pid]
    while todo:
        q = todo.pop()
        try:
            for t in os.listdir(f"/proc/{q}/task"):
                kids = _read(f"/proc/{q}/task/{t}/children").split()
                for k in kids:
                    out.append(int(k))
                    todo.append(int(k))
        except OSError:
            pass
    return out


def read_state(sd: str) -> dict | None:
    try:
        with open(os.path.join(sd, "state.json")) as f:
            return json.load(f)
    except (OSError, ValueError):
        return None


class Run:
    """One daemon child: config, process, sampler."""

    def __init__(self, out: str, files: list[str], speed: float = 10.0, argv_extra=(), **cfg):
        self.out = out
        os.makedirs(out, exist_ok=True)
        self.sd = cfg.pop("state_dir", os.path.join(out, "state"))
        radio = "soapy" if cfg.get("fake_soapy") else "mock"
        argv = ["--radio", radio, "--state-dir", self.sd]
        if radio == "mock":
            argv += ["--mock-files", *files, "--speed", str(speed)]
        argv += list(argv_extra)
        self.cfg = {"out": out, "argv": argv, "files": files, "speed": speed, **cfg}
        wjson(os.path.join(out, "cfg.json"), self.cfg)
        self.samples = []
        self.p = None

    def start(self):
        self.log = open(os.path.join(self.out, "daemon.log"), "a")
        self.t0 = time.time()
        self.p = subprocess.Popen([PY, os.path.abspath(__file__), "child", os.path.join(self.out, "cfg.json")],
                                  stdout=self.log, stderr=subprocess.STDOUT, cwd=self.out)
        return self

    def sample(self):
        rec = {"t": round(time.time() - self.t0, 2), "procs": []}
        for pid in [self.p.pid] + descendants(self.p.pid):
            pi = proc_info(pid)
            if pi:
                rec["procs"].append(pi)
        st = read_state(self.sd)
        if st:
            rec.update(pid=st.get("pid"), status=st["status"], minutes=st["chain"]["minutes"], segments=st["chain"]["segments"],
                       hb_age=round(time.time() - st["heartbeat_t"], 1), ring_fill=st["radio"]["ring_fill"],
                       ring_drops=st["radio"]["stats"]["ring_drops"])
        self.samples.append(rec)
        return rec

    def wait(self, max_s: float = 1e9, every: float = 1.0, on_tick=None) -> int | None:
        """Sample until the child exits (returns rc) or max_s (returns None, child still running)."""
        while self.p.poll() is None and time.time() - self.t0 < max_s:
            self.sample()
            if on_tick is not None and on_tick(self):
                break
            time.sleep(every)
        return self.p.poll()

    def term(self, sig=signal.SIGTERM, timeout=300):
        t = time.time()
        self.p.send_signal(sig)
        try:
            rc = self.p.wait(timeout)
        except subprocess.TimeoutExpired:
            self.p.kill()
            rc = "timeout-killed"
        return rc, round(time.time() - t, 2)

    def finish(self):
        with open(os.path.join(self.out, "samples.jsonl"), "w") as f:
            for s in self.samples:
                f.write(json.dumps(s) + "\n")
        self.log.close()


# ============================================================================================= output checks

def jsonl_check(path: str) -> dict:
    data = _read(path) if os.path.exists(path) else b""
    recs, bad = [], 0
    for line in data.split(b"\n"):
        if not line.strip():
            continue
        try:
            recs.append(json.loads(line))
        except ValueError:
            bad += 1
    return {"n": len(recs), "bad": bad, "torn_tail": bool(data) and not data.endswith(b"\n"), "recs": recs}


def all_jsonl(sd: str, name: str) -> dict:
    """Records of one output over the archive (rotated days) and the current file, in file order."""
    files = sorted(glob.glob(os.path.join(sd, "archive", f"{name}-*.jsonl"))) + [os.path.join(sd, f"{name}.jsonl")]
    out = {"n": 0, "bad": 0, "torn_tail": 0, "recs": [], "files": 0}
    for p in files:
        if not os.path.exists(p):
            continue
        c = jsonl_check(p)
        out["files"] += 1
        out["n"] += c["n"]
        out["bad"] += c["bad"]
        out["torn_tail"] += int(c["torn_tail"])
        out["recs"] += c["recs"]
    return out


def minutes_check(mins: list[dict], gaps: list[dict] = ()) -> dict:
    """Minute rows (gap rows apart): order, duplicates, holes; and whether every hole has an explicit gap row that
    spans it (t .. t_end within 1 s)."""
    ts = [m["t"] for m in mins]
    back = sum(1 for a, b in zip(ts, ts[1:]) if b < a)
    dup = len(ts) - len({round(t, 3) for t in ts})
    holes, hole_min, explained = 0, 0, 0
    for a, b in zip(ts, ts[1:]):
        if b - a > 60.5:
            holes += 1
            hole_min += int(round((b - a) / 60.0)) - 1
            if any(abs(g["t"] - (a + 60.0)) < 1.0 and abs(g["t_end"] - b) < 1.0 for g in gaps):
                explained += 1
    reasons = {}
    for g in gaps:
        for r in g.get("reason") or []:
            reasons[r] = reasons.get(r, 0) + 1
    return {"n": len(mins), "non_monotonic": back, "duplicate_t": dup, "holes": holes,
            "missing_minute_slots": hole_min, "explicit_gap_rows": len(gaps), "holes_with_gap_row": explained,
            "gap_minutes_in_rows": round(sum(g.get("missing_min") or 0 for g in gaps), 2),
            "gap_reasons": reasons, "segments": len({m.get("segment") for m in mins})}


def match_share(a, b, tol=1e-3):
    import numpy as np
    a, b = np.asarray(a, float), np.sort(np.asarray(b, float))
    if a.size == 0:
        return None, np.zeros(0, bool)
    if b.size == 0:
        return 0.0, np.zeros(a.size, bool)
    j = np.searchsorted(b, a)
    d = np.minimum(np.abs(b[np.clip(j - 1, 0, b.size - 1)] - a), np.abs(b[np.clip(j, 0, b.size - 1)] - a))
    ok = d <= tol
    return float(ok.mean()), ok


def outputs_summary(sd: str) -> dict:
    mins = all_jsonl(sd, "minutes")
    gaps = [m for m in mins["recs"] if m.get("gap")]
    mins["recs"] = [m for m in mins["recs"] if not m.get("gap")]
    evs = all_jsonl(sd, "events")
    al = all_jsonl(sd, "alerts")
    st = read_state(sd)
    return {"minutes": minutes_check(mins["recs"], gaps), "minutes_bad_lines": mins["bad"],
            "minutes_torn_tail": mins["torn_tail"], "events": evs["n"], "events_bad_lines": evs["bad"],
            "events_torn_tail": evs["torn_tail"], "alerts": al["n"], "alerts_bad_lines": al["bad"],
            "state_valid": st is not None, "status": st and st["status"], "exit_reason": st and st["exit_reason"],
            "chain_errors": st and st["chain"]["errors"], "radio_stats": st and st["radio"]["stats"],
            "disk": st and st["disk"], "untrusted_minutes": sum(1 for m in mins["recs"] if not m.get("trusted")),
            "untrusted_frontend": sum(1 for m in mins["recs"]
                                      if any(str(w).startswith("frontend_") for w in m.get("untrusted_because") or [])),
            "restored_minutes": st and st["chain"].get("restored_minutes"),
            "_mins": mins["recs"], "_evs": evs["recs"], "_alerts": al["recs"], "_gaps": gaps}


def strip(d: dict) -> dict:
    return {k: v for k, v in d.items() if not k.startswith("_")}


# ============================================================================================= throughput

def cmd_throughput(a):
    out = os.path.join(a.out, "throughput")
    files = corpus()[:a.n]
    r = Run(out, files, speed=0, file_gaps=True, settle=False).start()
    rc = r.wait()
    r.finish()
    instr = json.load(open(os.path.join(out, "instr.json")))
    sig_min = sum(os.path.getsize(f) / 4 / 62500 / 60 for f in files)
    cpu = instr["rusage"]["self_cpu_s"] + instr["rusage"]["children_cpu_s"]
    res = {"rc": rc, "files": len(files), "signal_min": sig_min, "wall_s": instr["wall_s"],
           "x_realtime": sig_min * 60 / instr["wall_s"], "cpu_s_per_signal_min": cpu / sig_min}
    wjson(os.path.join(a.out, "throughput.json"), res)
    print(json.dumps(res, indent=1))


# ============================================================================================= A. soak

def cmd_soak(a):
    out = os.path.join(a.out, "soak")
    files = corpus()
    if a.limit:
        files = files[:a.limit]
    r = Run(out, files, speed=a.speed, file_gaps=True, settle=False).start()
    last = [0.0]

    def tick(run):
        if time.time() - last[0] > 60:
            last[0] = time.time()
            s = run.samples[-1]
            print(f"[soak] t={s['t']:.0f}s minutes={s.get('minutes')} segs={s.get('segments')} "
                  f"rss={[p['rss_mb'] for p in s['procs']]} drops={s.get('ring_drops')}", flush=True)
    rc = r.wait(every=1.0, on_tick=tick)
    r.finish()
    analyse_soak(a.out, rc, files, a.speed, a.census)


def _trend(xs, ys):
    import numpy as np
    xs, ys = np.asarray(xs, float), np.asarray(ys, float)
    if xs.size < 3:
        return None
    return float(np.polyfit(xs, ys, 1)[0])


def analyse_soak(base, rc, files, speed, census):
    import numpy as np
    out = os.path.join(base, "soak")
    S = [json.loads(l) for l in open(os.path.join(out, "samples.jsonl"))]
    instr = json.load(open(os.path.join(out, "instr.json")))
    o = outputs_summary(os.path.join(out, "state"))
    mins = o["_mins"]
    sig_min = sum(os.path.getsize(f) / 4 / 62500 / 60 for f in files)
    # per-sample, per-role series; signal minute index from state.json (minutes committed)
    ser = {"main": [], "worker": []}
    for s in S:
        if s.get("minutes") is None:
            continue
        for p in s["procs"]:
            if p["role"] in ser:
                ser[p["role"]].append((s["minutes"], s["t"], p["rss_mb"], p["fds"], p["threads"], p["cpu_s"]))
    mem = {}
    for role, v in ser.items():
        if not v:
            continue
        v = np.array(v, float)
        m = v[:, 0]
        # steady level per 10 committed minutes: the minimum RSS in the bin (the worker saw-tooths ~0.5 -> 1.8 GB
        # within each commit; the minimum is the retained floor, the thing that grows with a leak)
        bins = (m // 10).astype(int)
        lv_min, lv_med, xs = [], [], []
        for b in np.unique(bins):
            k = bins == b
            if k.sum() >= 3:
                xs.append(b * 10)
                lv_min.append(v[k, 2].min())
                lv_med.append(np.median(v[k, 2]))
        lv_min, lv_med, xs = np.array(lv_min), np.array(lv_med), np.array(xs)
        warm = xs >= 60                                     # skip the first hour of signal (allocator warm-up)
        first = lv_min[warm][:6] if warm.sum() >= 12 else lv_min[:6]
        last_ = lv_min[warm][-6:] if warm.sum() >= 12 else lv_min[-6:]
        fmed = lv_med[warm][:6] if warm.sum() >= 12 else lv_med[:6]
        lmed = lv_med[warm][-6:] if warm.sum() >= 12 else lv_med[-6:]
        mem[role] = {"rss_floor_first_hour_mb": round(float(np.median(first)), 1),
                     "rss_floor_last_hour_mb": round(float(np.median(last_)), 1),
                     "rss_floor_growth_mb": round(float(np.median(last_) - np.median(first)), 1),
                     "rss_median_first_hour_mb": round(float(np.median(fmed)), 1),
                     "rss_median_last_hour_mb": round(float(np.median(lmed)), 1),
                     "rss_median_growth_mb": round(float(np.median(lmed) - np.median(fmed)), 1),
                     "rss_floor_slope_mb_per_24h_signal": round((_trend(xs[warm], lv_min[warm]) or 0) * 1440, 1),
                     "rss_peak_mb": round(float(v[:, 2].max()), 1),
                     "fds_min_max": [int(v[:, 3].min()), int(v[:, 3].max())],
                     "fds_first_last": [int(v[0, 3]), int(v[-1, 3])],
                     "threads_min_max": [int(v[:, 4].min()), int(v[:, 4].max())],
                     "cpu_s_last_sample": round(float(v[-1, 5]), 1)}
        np.savetxt(os.path.join(out, f"rss_{role}_per10min.csv"), np.c_[xs, lv_min, lv_med], delimiter=",",
                   header="minute,rss_floor_mb,rss_median_mb", fmt="%.1f")
    cpu = instr["rusage"]["self_cpu_s"] + instr["rusage"]["children_cpu_s"]
    # processing lag in WALL seconds: capture k's last sample left the radio at switch k (the next file starts); its
    # minutes are all written when the committed count reaches 10 (k + 1). With warm-up every minute of a 10-min
    # capture is committed at its segment end, so this is the whole chain's lag behind the input; it must not grow.
    sw_ = instr.get("switch_wall_s") or []
    mt = [(s_["t"], s_["minutes"]) for s_ in S if s_.get("minutes") is not None and s_.get("pid") == instr.get("pid")]
    lag_wall = []
    for k, tw in enumerate(sw_):
        need = 10 * (k + 1)
        done = next((t for t, n in mt if n >= need and t >= tw), None)
        if done is not None:
            lag_wall.append(done - tw)
    lag_wall = np.array(lag_wall)
    nq_ = max(1, lag_wall.size // 10)
    lagw = ({"median": round(float(np.median(lag_wall)), 2), "max": round(float(lag_wall.max()), 2),
             "first_10pct_median": round(float(np.median(lag_wall[:nq_])), 2),
             "last_10pct_median": round(float(np.median(lag_wall[-nq_:])), 2),
             "slope_s_per_100_captures": round((_trend(np.arange(lag_wall.size), lag_wall) or 0) * 100, 3),
             "n": int(lag_wall.size)} if lag_wall.size else None)
    delay = np.array([m["delay_s"] if m.get("delay_s") is not None else np.nan for m in mins], float)
    # minutes committed in stream: segment minutes 0-7 of a 10-min capture (8 and 9 are committed at the segment end,
    # when the mock clock has already jumped to the next capture's time, so their delay means nothing)
    full = np.array([m["seg_minute"] <= 7 and m.get("delay_s") is not None and m.get("delay_s") < 400
                     for m in mins])
    nq = max(1, len(delay) // 10)
    res = {"rc": rc, "speed": speed, "files": len(files), "signal_min": round(sig_min, 1),
           "signal_h": round(sig_min / 60, 2), "wall_s": round(instr["wall_s"], 1),
           "achieved_x": round(sig_min * 60 / instr["wall_s"], 2),
           "outputs": strip(o), "memory": mem,
           "cpu_s_per_signal_min": round(cpu / sig_min, 3),
           "cpu_main_s_per_signal_min": round(instr["rusage"]["self_cpu_s"] / sig_min, 3),
           "cpu_share_of_one_core": round(cpu / instr["wall_s"], 3),
           "commit_cpu_s": {"median": round(float(np.median([m["cpu_s"] for m in mins])), 3),
                            "max": round(float(max(m["cpu_s"] for m in mins)), 3)},
           "delay_stream_s": ({"in_stream_commits_median": round(float(np.median(delay[full])), 1),
                               "in_stream_commits_max": round(float(delay[full].max()), 1),
                               "first_10pct_median": round(float(np.median(delay[full][:nq])), 1),
                               "last_10pct_median": round(float(np.median(delay[full][-nq:])), 1)}
                              if full.any() else "none: with warm-up a 10-min capture commits at its segment end"),
           "segment_end_lag_wall_s": lagw,
           "ring_hw_s": round(instr["ring_hw_samples"] / 62500, 1), "file_switches": instr["file_switches"],
           "pace_late_max_s": round(instr["pace_late_max_s"], 3), "pace_late_p99_s": instr.get("pace_late_p99_s"),
           "heartbeat_age_max_s": max((s.get("hb_age") or 0) for s in S)}
    if census:
        res["vs_batch"] = per_capture_vs_batch(files, o["_mins"], o["_evs"], census)
    wjson(os.path.join(base, "soak.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "vs_batch"}, indent=1))
    if census:
        print(json.dumps(res["vs_batch"]["summary"], indent=1))


# ============================================================================================= E. batch equivalence

def per_capture_vs_batch(files, mins, evs, census) -> dict:
    """Per capture: live events (by time inside the capture) vs the batch census events; per minute: rates."""
    import csv
    import numpy as np
    rows = {r["id"]: r for r in csv.DictReader(open(os.path.join(census, "census.csv")))}
    et = np.array(sorted(e["t"] for e in evs))
    mt = {round(m["t"], 3): m for m in mins}
    caps, tot_b, tot_s = [], 0, 0
    mrel, crel, flips, nmin = [], [], 0, 0
    for f in files:
        cid = cap_id(f)
        if cid not in rows:
            continue
        U = sidecar_t0(f)
        L = os.path.getsize(f) / 4 / 62500
        s_t = et[(et >= U) & (et < U + L)]
        bev = np.genfromtxt(os.path.join(census, "census_sessions", f"{cid}.events.csv"), delimiter=",", names=True)
        b_t = (np.atleast_1d(bev["t_start_s"]) + U) if bev.size else np.zeros(0)
        nb, ns = int(rows[cid]["n_events"]), len(s_t)
        tot_b += nb
        tot_s += ns
        bm, _ = match_share(b_t, s_t)
        sm, _ = match_share(s_t, b_t)
        sess = json.load(open(os.path.join(census, "census_sessions", f"{cid}.session.json")))
        cw, cn = 0, 0
        for bmr in sess["minutes"]:
            m = mt.get(round(U + bmr["t0_s"], 3))
            if m is None:
                continue
            nmin += 1
            for key, acc in (("rate_per_min", mrel), ("conf_per_min", crel)):
                b, s = bmr[key], m[key]
                acc.append(abs(s - b) / b if b else (0.0 if s == 0 else 1.0))
            flips += int(bool(bmr["trusted"]) != bool(m["trusted"]))
            cn += 1
            cw += int(mrel[-1] <= 0.02)
        caps.append({"id": cid, "batch": nb, "stream": ns, "batch_matched_1ms": bm, "stream_matched_1ms": sm,
                     "minutes_rate_within_2pct": cw / cn if cn else None, "hum_frac": float(rows[cid]["hum_frac"]),
                     "gated_frac": float(rows[cid]["gated_frac"])})
    mrel, crel = np.array(mrel), np.array(crel)
    tot_rel = [abs(c["stream"] - c["batch"]) / c["batch"] for c in caps if c["batch"]]
    worst = sorted(caps, key=lambda c: -abs(c["stream"] - c["batch"]) / max(1, c["batch"]))[:10]
    summ = {"captures": len(caps), "events_batch": tot_b, "events_stream": tot_s,
            "events_total_rel_diff": round((tot_s - tot_b) / tot_b, 4),
            "batch_matched_1ms_pooled": round(float(np.average([c["batch_matched_1ms"] or 0 for c in caps],
                                                               weights=[c["batch"] for c in caps])), 4),
            "stream_matched_1ms_pooled": round(float(np.average([c["stream_matched_1ms"] or 0 for c in caps],
                                                                weights=[max(c["stream"], 1) for c in caps])), 4),
            "captures_identical_count": sum(c["batch"] == c["stream"] and (c["batch_matched_1ms"] in (None, 1.0))
                                            for c in caps),
            "captures_total_within_2pct": sum(r <= 0.02 for r in tot_rel),
            "captures_total_within_5pct": sum(r <= 0.05 for r in tot_rel),
            "minutes_compared": nmin,
            "minutes_rate_within_2pct": round(float((mrel <= 0.02).mean()), 4),
            "minutes_rate_within_10pct": round(float((mrel <= 0.10).mean()), 4),
            "minutes_conf_within_2pct": round(float((crel <= 0.02).mean()), 4),
            "minute_rate_reldiff_p95": round(float(np.percentile(mrel, 95)), 4),
            "trust_flips": flips,
            "hum_captures_events_rel_diff": round(
                (sum(c["stream"] for c in caps if c["hum_frac"] > 0) - sum(c["batch"] for c in caps if c["hum_frac"] > 0))
                / max(1, sum(c["batch"] for c in caps if c["hum_frac"] > 0)), 4),
            "clean_captures_events_rel_diff": round(
                (sum(c["stream"] for c in caps if c["hum_frac"] == 0) - sum(c["batch"] for c in caps if c["hum_frac"] == 0))
                / max(1, sum(c["batch"] for c in caps if c["hum_frac"] == 0)), 4),
            "worst10": [{k: (round(v, 4) if isinstance(v, float) else v) for k, v in c.items()} for c in worst]}
    return {"summary": summ, "per_capture": caps}


def cmd_equiv(a):
    d = json.load(open(os.path.join(a.out, "soak.json")))
    print(json.dumps(d.get("vs_batch", {}).get("summary"), indent=1))


# ============================================================================================= B. real time

def cmd_realtime(a):
    import numpy as np
    out = os.path.join(a.out, "realtime")
    files = [os.path.expanduser(f) for f in a.files]
    r = Run(out, files, speed=1.0, argv_extra=["--loop", "--duration", str(a.minutes * 60)]).start()
    rc = r.wait(every=1.0)
    r.finish()
    instr = json.load(open(os.path.join(out, "instr.json")))
    o = outputs_summary(os.path.join(out, "state"))
    mins = o["_mins"]
    delay = np.array([m["delay_s"] for m in mins if m["lookahead_min"] >= 1], float)
    cpu = instr["rusage"]["self_cpu_s"] + instr["rusage"]["children_cpu_s"]
    st = o["radio_stats"]
    S = r.samples
    res = {"rc": rc, "files": [os.path.basename(f) for f in files], "wall_s": round(instr["wall_s"], 1),
           "outputs": strip(o), "cpu_share_of_one_core": round(cpu / instr["wall_s"], 4),
           "cpu_main_share": round(instr["rusage"]["self_cpu_s"] / instr["wall_s"], 4),
           "commit_cpu_s_max": max(m["cpu_s"] for m in mins) if mins else None,
           "delay_s": {"median": float(np.median(delay)) if delay.size else None,
                       "max": float(delay.max()) if delay.size else None},
           "processing_latency_s_max": round(float(delay.max()) - 62.0, 1) if delay.size else None,
           "ring_hw_samples": instr["ring_hw_samples"], "ring_hw_s": round(instr["ring_hw_samples"] / 62500, 2),
           "ring_hw_frac": round(instr["ring_hw_samples"] / (120 * 62500), 4),
           "reader_late_max_s": round(instr["pace_late_max_s"], 4), "reader_late_p99_s": instr.get("pace_late_p99_s"),
           "overflows": st["overflows"] if st else None, "ring_drops": st["ring_drops"] if st else None,
           "rss_max_mb": {role: max((p["rss_mb"] for s in S for p in s["procs"] if p["role"] == role), default=None)
                          for role in ("main", "worker")},
           "hum_minutes": sum(1 for m in mins if (m["hum_frac"] or 0) > 0),
           "gated_frac_median": float(np.median([m["gated_frac"] for m in mins])) if mins else None}
    wjson(os.path.join(a.out, "realtime.json"), res)
    print(json.dumps(res, indent=1))


# ============================================================================================= C. faults

# fault runs: clean captures (no hum, nothing gated), so an event that differs from the no-fault baseline is the
# fault's doing; ``--set hum`` uses the fully hum-flagged 11-06 04-06Z storm captures instead (there a new segment
# changes the gate's own decisions, so the event comparison mixes fault and gate-history effects)
FAULT_SETS = {"clean": ["20261019/vlf_1500Z", "20261019/vlf_1530Z", "20261019/vlf_1600Z", "20261019/vlf_1630Z",
                        "20261019/vlf_1700Z", "20261019/vlf_1730Z"],
              "hum": ["20261106/vlf_0400Z", "20261106/vlf_0430Z", "20261106/vlf_0500Z", "20261106/vlf_0530Z",
                      "20261106/vlf_0600Z", "20261106/vlf_0630Z"]}
FAULT_FILES = FAULT_SETS["clean"][:3]
EXTRA = FAULT_SETS["clean"][3:]


def _vf(stem):
    return os.path.expanduser(os.path.join("~", "dlayer_run", "data", "vlf", stem + ".cs16"))


def fault_cases():
    F = lambda **k: k
    return {
        "baseline": {},
        "overflow_bursts": {"faults": [F(kind="overflow", at_s=300, n=3), F(kind="overflow", at_s=900, n=10),
                                       *[F(kind="overflow", at_s=1200 + 2 * i, n=1) for i in range(5)]]},
        "vanish_5s": {"faults": [F(kind="vanish", at_s=600, dur_s=5)]},
        "vanish_60s": {"faults": [F(kind="vanish", at_s=600, dur_s=60)]},
        "timeouts": {"faults": [F(kind="timeout", at_s=400, dur_s=2), F(kind="timeout", at_s=1000, dur_s=10)]},
        "short_reads": {"short_p": 0.3},
        "drop_dup": {"faults": [F(kind="drop", at_s=300, n=500), F(kind="drop", at_s=700, n=5000),
                                F(kind="dup", at_s=1100, n=3000), F(kind="dup", at_s=1500, n=800)]},
        "clock_2s": {"faults": [F(kind="clock", at_s=500, dur_s=2.0), F(kind="clock", at_s=1100, dur_s=-2.0)]},
        "clock_1h": {"faults": [F(kind="clock", at_s=500, dur_s=3600.0), F(kind="clock", at_s=1100, dur_s=-3600.0)]},
        "slow_disk_10x": {"slow_disk": [{"at_append": 20, "block_s": 30}]},
        "disk_low": {"argv_extra": ["--raw"], "free_gb": [[1, 50.0], [2, 3.0], [3, 0.3], [4, 50.0]],
                     "files": "4"},
        "enospc": {"enospc": {"from_append": 20, "n": 5}},
    }


def run_case(base, name, case, speed=10.0, files=None):
    out = os.path.join(base, "faults", name)
    if os.path.exists(os.path.join(out, "state")):
        subprocess.run(["rm", "-rf", os.path.join(out, "state")], check=False)
    files = [_vf(f) for f in ((FAULT_FILES + EXTRA)[:int(case["files"])] if case.get("files") else FAULT_FILES)]
    cfg = {k: v for k, v in case.items() if k not in ("faults", "argv_extra", "files")}
    extra = list(case.get("argv_extra", []))
    if case.get("faults"):
        extra += ["--mock-faults", json.dumps(case["faults"])]
    r = Run(out, files, speed=speed, argv_extra=extra, **cfg).start()
    rc = r.wait(every=1.0)
    r.finish()
    return r, rc, out


def fake_events(evs, base_evs, mins, tol=1e-3, base_mins=None):
    """Events of a faulted run not within tol of a baseline event, and how many lie within 10 s of a segment edge."""
    import numpy as np
    t = np.array([e["t"] for e in evs])
    bt = np.array([e["t"] for e in base_evs])
    share, ok = match_share(t, bt, tol)
    edges = []
    by = {}
    for m in mins:
        by.setdefault(m["segment"], []).append(m["t"])
    for s, ts in by.items():
        edges += [min(ts), max(ts) + 60.0]
    edges = np.array(sorted(edges))
    un = t[~ok] if t.size else t
    near = 0
    if un.size and edges.size:
        j = np.searchsorted(edges, un)
        d = np.minimum(np.abs(edges[np.clip(j - 1, 0, edges.size - 1)] - un), np.abs(edges[np.clip(j, 0, edges.size - 1)] - un))
        near = int((d <= 10).sum())
    # events whose time lies in no committed minute (inside a gap)
    mt = np.array(sorted(m["t"] for m in mins))
    in_gap = 0
    if t.size and mt.size:
        j = np.searchsorted(mt, t, side="right") - 1
        in_gap = int(((j < 0) | (t - mt[np.clip(j, 0, mt.size - 1)] >= 60.0)).sum())
    # NOTE: the 1 ms match is phase-sensitive: a segment that starts at another sample offset than the baseline's
    # (every gap does) finds a different set of near-threshold events (about a third match on clean captures even
    # with no fault, see the soak/throughput check), so the fair "fake events" tests are counts near the edges and
    # per-minute rates, below.
    near_n, near_b = 0, 0
    for e in edges:
        near_n += int(((t >= e - 10) & (t < e + 10)).sum())
        near_b += int(((bt >= e - 10) & (bt < e + 10)).sum())
    bmin = {}
    for m in base_mins or []:
        bmin[round(m["t"])] = m
    rel, nb, ns = [], 0, 0
    # rates of the faulted minutes against the baseline rate over the same span (baseline events counted directly)
    for m in mins:
        k = (bt >= m["t"]) & (bt < m["t"] + 60)
        b = int(k.sum())
        s = m["n_events"]
        nb += b
        ns += s
        rel.append(abs(s - b) / b if b else (0.0 if s == 0 else 1.0))
    rel = np.array(rel)
    return {"events": int(t.size), "matched_baseline_1ms": share, "unmatched": int((~ok).sum()) if t.size else 0,
            "unmatched_within_10s_of_segment_edge": near, "events_outside_any_minute": in_gap,
            "baseline_events_found": match_share(bt, t, tol)[0],
            "edge_windows": int(edges.size), "events_in_edge_windows": near_n,
            "baseline_events_in_edge_windows": near_b,
            "minutes_events_vs_baseline_same_span": [ns, nb],
            "minutes_rate_within_10pct_of_baseline": round(float((rel <= 0.10).mean()), 3) if rel.size else None,
            "minute_rate_reldiff_median": round(float(np.median(rel)), 3) if rel.size else None}


def case_report(name, r, rc, out, base_o=None):
    o = outputs_summary(os.path.join(out, "state"))
    instr = json.load(open(os.path.join(out, "instr.json"))) if os.path.exists(os.path.join(out, "instr.json")) else {}
    rec = {"case": name, "rc": rc, "outputs": strip(o), "ring_hw_s": round(instr.get("ring_hw_samples", 0) / 62500, 1),
           "disk_blocks": instr.get("disk_blocks"), "enospc_raised": instr.get("enospc_raised"),
           "short_injected": instr.get("short_injected"),
           "heartbeat_age_max_s": max((s.get("hb_age") or 0) for s in r.samples) if r.samples else None,
           "ring_fill_max_sampled": max((s.get("ring_fill") or 0) for s in r.samples) if r.samples else None,
           "worker_rss_max_mb": max((p["rss_mb"] for s in r.samples for p in s["procs"] if p["role"] == "worker"),
                                    default=None)}
    if base_o is not None:
        rec["vs_baseline"] = fake_events(o["_evs"], base_o["_evs"], o["_mins"], base_mins=base_o["_mins"])
    try:
        log = open(os.path.join(out, "daemon.log")).read()
        rec["log_tail"] = log.strip().splitlines()[-4:]
        rec["log_tracebacks"] = log.count("Traceback")
    except OSError:
        pass
    return rec, o


def wait_minutes(run, n):
    """Until THIS process (state.json pid) has committed n minutes (a shared state dir holds older processes' state)."""
    return lambda rr: rr.samples[-1].get("pid") == rr.p.pid and (rr.samples[-1].get("minutes") or 0) >= n


def cmd_faults(a):
    global FAULT_FILES, EXTRA
    FAULT_FILES, EXTRA = FAULT_SETS[a.set][:3], FAULT_SETS[a.set][3:]
    base = a.out
    res = {}
    resf = os.path.join(base, "faults.json" if a.set == "clean" else f"faults_{a.set}.json")
    if os.path.exists(resf):
        res = json.load(open(resf))
    cases = fault_cases()
    only = set(a.only.split(",")) if a.only else None

    def save():
        wjson(resf, res)

    def want(n):
        return (only is None or n in only) and (a.redo or n not in res)

    base_o = None
    r, rc, out = None, None, None
    if want("baseline") or True:
        bout = os.path.join(base, "faults", "baseline")
        if want("baseline") or not os.path.exists(os.path.join(bout, "state", "state.json")):
            r, rc, out = run_case(base, "baseline", {})
            rec, base_o = case_report("baseline", r, rc, out)
            res["baseline"] = rec
            save()
        else:
            base_o = outputs_summary(os.path.join(bout, "state"))
    for name, case in cases.items():
        if name == "baseline" or not want(name):
            continue
        print(f"[faults] {name}", flush=True)
        r, rc, out = run_case(base, name, case)
        rec, _ = case_report(name, r, rc, out, base_o if not case.get("files") else None)
        res[name] = rec
        save()

    # slow disk at 1x: the main loop blocked 30 s on a write; the ring must absorb it (120 s) with no drop
    name = "slow_disk_1x"
    if want(name):
        print(f"[faults] {name}", flush=True)
        out = os.path.join(base, "faults", name)
        r = Run(out, [_vf(FAULT_FILES[0])], speed=1.0, argv_extra=["--duration", "260", "--no-warmup"],
                slow_disk=[{"at_append": 1, "block_s": 30}]).start()
        rc = r.wait(every=0.5)
        r.finish()
        res[name], _ = case_report(name, r, rc, out)
        save()

    # SIGTERM in the middle of a minute
    name = "sigterm_mid_minute"
    if want(name):
        print(f"[faults] {name}", flush=True)
        out = os.path.join(base, "faults", name)
        r = Run(out, [_vf(f) for f in FAULT_FILES], speed=10.0).start()
        r.wait(max_s=1e9, on_tick=wait_minutes(r, 5))
        time.sleep(2.7)                                      # 27 s of stream into the next minute
        rc, dt_ = r.term(signal.SIGTERM)
        r.finish()
        rec, o = case_report(name, r, rc, out)
        rec["shutdown_s"] = dt_
        res[name] = rec
        save()

    # SIGKILL, then restart on the next capture; then a simulated torn last line (power cut mid-write) + restart
    name = "sigkill_restart"
    if want(name):
        print(f"[faults] {name}", flush=True)
        out = os.path.join(base, "faults", name)
        sd = os.path.join(out, "state")
        subprocess.run(["rm", "-rf", sd], check=False)
        r = Run(os.path.join(out, "run1"), [_vf(f) for f in FAULT_FILES[:2]], speed=10.0, state_dir=sd).start()
        r.wait(max_s=1e9, on_tick=wait_minutes(r, 6))
        time.sleep(1.3)
        kids = descendants(r.p.pid)
        rc1, _ = r.term(signal.SIGKILL)
        orphans = {}
        for w in (1, 5, 20):
            time.sleep(w - (sum(k for k in (1, 5, 20) if k < w)))
            orphans[f"after_{w}s"] = [k for k in kids if os.path.exists(f"/proc/{k}")]
        for k in kids:                                       # clean up whatever is left (by pid)
            try:
                os.kill(k, signal.SIGKILL)
            except OSError:
                pass
        r.finish()
        o1 = outputs_summary(sd)
        st1 = read_state(sd)
        r2 = Run(os.path.join(out, "run2"), [_vf(FAULT_FILES[2])], speed=10.0, state_dir=sd).start()
        r2.wait(max_s=1e9, on_tick=wait_minutes(r2, 3))
        rc2, _ = r2.term(signal.SIGTERM)
        r2.finish()
        o2 = outputs_summary(sd)
        # torn tail: what a kill or power cut in the middle of a write leaves
        torn = {}
        for nm in ("minutes", "events"):
            p = os.path.join(sd, f"{nm}.jsonl")
            with open(p, "rb") as f:
                lines = f.read().splitlines(keepends=True)
            with open(p, "ab") as f:
                f.write(lines[-1][:len(lines[-1]) // 2])
            torn[nm] = jsonl_check(p)["bad"]
        # ext4 after a power cut: a NUL-filled block inside the file, then a later complete line
        p = os.path.join(sd, "events.jsonl")
        with open(p, "rb") as f:
            ev_lines = f.read().splitlines(keepends=True)
        with open(p, "ab") as f:
            f.write(b"\0" * 4096 + b"\n" + ev_lines[0])
        torn["events_nul_block"] = jsonl_check(p)["bad"]
        r3 = Run(os.path.join(out, "run3"), [_vf(EXTRA[0])], speed=10.0, state_dir=sd).start()
        r3.wait(max_s=1e9, on_tick=wait_minutes(r3, 3))
        rc3, _ = r3.term(signal.SIGTERM)
        r3.finish()
        o3 = outputs_summary(sd)
        p_ok = True
        try:                                                 # the panel reads these files: does it survive?
            import importlib.util
            spec = importlib.util.spec_from_file_location("storm_panel", os.path.join(REPO, "tools", "storm_panel.py"))
            sp = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(sp)
            sp.load_live(sd, max_minutes=720)
        except Exception as e:
            p_ok = f"{type(e).__name__}: {e}"
        res[name] = {"case": name, "kill_rc": rc1, "state_after_kill": {"valid": st1 is not None,
                                                                         "status": st1 and st1["status"]},
                     "orphan_workers": orphans, "after_kill": strip(o1), "restart_rc": rc2, "after_restart": strip(o2),
                     "torn_bad_lines_injected": torn, "after_torn_restart_rc": rc3, "after_torn_restart": strip(o3),
                     "panel_load_live_after_torn": p_ok}
        save()

    # the radio lock: second instance (fake soapy = mock behind the SoapyRadio name; real driver poisoned)
    name = "second_instance"
    if want(name):
        print(f"[faults] {name}", flush=True)
        out = os.path.join(base, "faults", name)
        lock = os.path.join(out, "radio.lock")
        r1 = Run(os.path.join(out, "first"), [_vf(FAULT_FILES[0])], speed=1.0, fake_soapy=True,
                 argv_extra=["--lock-file", lock, "--duration", "40"]).start()
        r1.wait(max_s=8)
        st_first = read_state(r1.sd)
        t = time.time()
        r2 = Run(os.path.join(out, "second"), [_vf(FAULT_FILES[0])], speed=1.0, fake_soapy=True,
                 argv_extra=["--lock-file", lock, "--duration", "40"]).start()
        rc2 = r2.wait(max_s=60)
        dt2 = round(time.time() - t, 2)
        r2.finish()
        # a second MOCK instance on the same state dir (the mock takes no lock)
        r3 = Run(os.path.join(out, "mock_same_state"), [_vf(FAULT_FILES[0])], speed=1.0, state_dir=r1.sd,
                 argv_extra=["--duration", "10"]).start()
        rc3 = r3.wait(max_s=60)
        r3.finish()
        rc1 = r1.wait()
        r1.finish()
        res[name] = {"case": name, "first_status_at_8s": st_first and st_first["status"],
                     "first_rc": rc1, "second_rc": rc2, "second_exit_s": dt2,
                     "second_log": open(os.path.join(out, "second", "daemon.log")).read().strip().splitlines()[-2:],
                     "second_state_written": os.path.exists(os.path.join(out, "second", "state", "state.json")),
                     "mock_same_state_dir_rc": rc3,
                     "lock_file_after": open(lock).read() if os.path.exists(lock) else None}
        save()

    name = "dlayer_dummy"
    if want(name):
        print(f"[faults] {name}", flush=True)
        out = os.path.join(base, "faults", name)
        os.makedirs(out, exist_ok=True)
        dummy = os.path.join(out, "dlayer_run.py")
        with open(dummy, "w") as f:
            f.write("import time\ntime.sleep(45)\n")
        dp = subprocess.Popen([PY, dummy])
        time.sleep(1.0)
        t = time.time()
        r = Run(os.path.join(out, "watcher"), [_vf(FAULT_FILES[0])], speed=1.0, fake_soapy=True,
                argv_extra=["--lock-file", os.path.join(out, "radio.lock"), "--duration", "20"]).start()
        rc = r.wait(max_s=40)
        dt_ = round(time.time() - t, 2)
        if rc is None:
            rc, _ = r.term()
        r.finish()
        dp.kill()
        dp.wait()
        res[name] = {"case": name, "rc": rc, "exit_s": dt_,
                     "log": open(os.path.join(out, "watcher", "daemon.log")).read().strip().splitlines()[-2:],
                     "state_written": os.path.exists(os.path.join(out, "watcher", "state", "state.json"))}
        save()

    # a dlayer_run.py started AFTER the watcher (with interpreter flags the first guard missed): yield, exit 3
    name = "dlayer_late"
    if want(name):
        print(f"[faults] {name}", flush=True)
        out = os.path.join(base, "faults", name)
        os.makedirs(out, exist_ok=True)
        dummy = os.path.join(out, "dlayer_run.py")
        with open(dummy, "w") as f:
            f.write("import time\ntime.sleep(60)\n")
        r = Run(os.path.join(out, "watcher"), [_vf(FAULT_FILES[0])], speed=1.0, fake_soapy=True,
                argv_extra=["--lock-file", os.path.join(out, "radio.lock"), "--duration", "120"]).start()
        r.wait(max_s=8)
        st0 = read_state(r.sd)
        t = time.time()
        dp = subprocess.Popen([PY, "-X", "dev", "-u", dummy])
        rc = r.wait(max_s=60)
        dt_ = round(time.time() - t, 2)
        if rc is None:
            rc, _ = r.term()
        r.finish()
        dp.kill()
        dp.wait()
        st = read_state(r.sd)
        res[name] = {"case": name, "status_before": st0 and st0["status"], "rc": rc, "exit_s_after_dlayer_start": dt_,
                     "exit_reason": st and st["exit_reason"], "status": st and st["status"],
                     "lock_file_after": open(os.path.join(out, "radio.lock")).read()
                     if os.path.exists(os.path.join(out, "radio.lock")) else None}
        save()

    # the chain worker SIGKILLed (OOM killer) under full load: the daemon must notice, stop and EXIT (rc 1)
    name = "kill_worker"
    if want(name):
        print(f"[faults] {name}", flush=True)
        trials = []
        for k, (speed, after) in enumerate([(0, 25), (0, 40), (5, 30), (10, 45)]):
            out = os.path.join(base, "faults", name, f"t{k}")
            subprocess.run(["rm", "-rf", os.path.join(out, "state")], check=False)
            r = Run(out, [_vf(f) for f in FAULT_FILES], speed=speed).start()
            r.wait(max_s=after)
            wk = [p for p in descendants(r.p.pid) if (proc_info(p) or {}).get("role") == "worker"]
            t = time.time()
            for p in wk:
                os.kill(p, signal.SIGKILL)
            rc = r.wait(max_s=120)
            dt_ = round(time.time() - t, 2)
            if rc is None:
                rc, _ = r.term(signal.SIGKILL)
                rc = f"HUNG ({rc})"
            r.finish()
            st = read_state(r.sd)
            trials.append({"speed": speed, "kill_after_s": after, "workers_killed": len(wk), "rc": rc,
                           "exit_s": dt_, "status": st and st["status"], "exit_reason": st and st["exit_reason"],
                           "log_tracebacks": open(os.path.join(out, "daemon.log")).read().count("Traceback")})
        res[name] = {"case": name, "trials": trials,
                     "all_exited": all(isinstance(t["rc"], int) for t in trials)}
        save()

    # the daemon SIGKILLed while its feeder is mid-message (unpaced, queue full): no orphaned worker
    name = "kill_parent"
    if want(name):
        print(f"[faults] {name}", flush=True)
        trials = []
        for k, (speed, after) in enumerate([(0, 20), (0, 33), (0, 41), (5, 26), (5, 37), (10, 30)]):
            out = os.path.join(base, "faults", name, f"t{k}")
            subprocess.run(["rm", "-rf", os.path.join(out, "state")], check=False)
            r = Run(out, [_vf(f) for f in FAULT_FILES], speed=speed).start()
            r.wait(max_s=after)
            kids = descendants(r.p.pid)
            r.term(signal.SIGKILL)
            t = time.time()
            while any(os.path.exists(f"/proc/{p}") and (proc_info(p) or {}).get("role") != "tracker" and
                      (proc_info(p) or {}).get("rss_mb", 0) > 0 for p in kids) and time.time() - t < 30:
                time.sleep(0.2)
            left = [p for p in kids if os.path.exists(f"/proc/{p}") and (proc_info(p) or {}).get("rss_mb", 0) > 0]
            left_roles = [(proc_info(p) or {}).get("role") for p in left]
            for p in left:
                try:
                    os.kill(p, signal.SIGKILL)
                except OSError:
                    pass
            r.finish()
            trials.append({"speed": speed, "kill_after_s": after, "children": len(kids),
                           "gone_after_s": round(time.time() - t, 1), "left_after_30s": left_roles})
        res[name] = {"case": name, "trials": trials,
                     "orphaned_workers": sum(t["left_after_30s"].count("worker") for t in trials)}
        save()

    # a dead front end (all-zero samples) and a clipped one, between clean captures: never trusted, no crash
    name = "dead_frontend"
    if want(name):
        print(f"[faults] {name}", flush=True)
        import numpy as np
        out = os.path.join(base, "faults", name)
        os.makedirs(out, exist_ok=True)
        zf, cf = os.path.join(out, "zeros.cs16"), os.path.join(out, "clipped.cs16")
        np.zeros(2 * 62500 * 180, np.int16).tofile(zf)
        rng = np.random.default_rng(7)
        np.clip(rng.standard_normal(2 * 62500 * 180) * 12000, -32768, 32767).astype(np.int16).tofile(cf)
        subprocess.run(["rm", "-rf", os.path.join(out, "state")], check=False)
        files = [_vf(FAULT_FILES[0]), zf, _vf(FAULT_FILES[1]), cf, _vf(FAULT_FILES[2])]
        r = Run(out, files, speed=10.0).start()
        rc = r.wait(every=1.0)
        r.finish()
        rec, o = case_report(name, r, rc, out)
        mins = o["_mins"]
        t0 = mins[0]["t"] if mins else 0
        L = [os.path.getsize(f) / 4 / 62500 for f in files]
        edges = np.cumsum([0] + L) + t0 - 1.0                  # stream start = first minute start (align start)
        def span(i):
            return [m for m in mins if m["t"] >= edges[i] + 1 and m["t"] + 60 <= edges[i + 1] + 1]
        rec["zeros_minutes"] = [[m["trusted"], m["untrusted_because"]] for m in span(1)]
        rec["clipped_minutes"] = [[m["trusted"], m["untrusted_because"]] for m in span(3)]
        rec["clean_minutes_trusted"] = sum(m["trusted"] for i in (0, 2, 4) for m in span(i))
        rec["clean_minutes"] = sum(len(span(i)) for i in (0, 2, 4))
        for f in (zf, cf):
            os.remove(f)
        res[name] = rec
        save()

    # everything at once, ending with SIGTERM mid-minute
    name = "mixed"
    if want(name):
        print(f"[faults] {name}", flush=True)
        F = lambda **k: k
        faults = [F(kind="overflow", at_s=200, n=3), F(kind="timeout", at_s=500, dur_s=3),
                  F(kind="vanish", at_s=800, dur_s=5), F(kind="drop", at_s=1100, n=5000),
                  F(kind="clock", at_s=1400, dur_s=2.0), F(kind="dup", at_s=1700, n=3000),
                  F(kind="vanish", at_s=2000, dur_s=60), F(kind="overflow", at_s=2300, n=10),
                  F(kind="clock", at_s=2600, dur_s=-2.0), F(kind="timeout", at_s=2900, dur_s=10),
                  F(kind="stall", at_s=3200, dur_s=20), F(kind="drop", at_s=3300, n=300)]
        out = os.path.join(base, "faults", name)
        subprocess.run(["rm", "-rf", os.path.join(out, "state")], check=False)
        files = [_vf(f) for f in FAULT_FILES + EXTRA]
        r = Run(out, files, speed=10.0, argv_extra=["--raw", "--mock-faults", json.dumps(faults)], short_p=0.1,
                slow_disk=[{"at_append": 30, "block_s": 30}], free_gb=[[1, 50.0], [3, 3.0], [4, 50.0]]).start()
        r.wait(max_s=1e9, on_tick=lambda rr: rr.samples[-1]["t"] >= 290)    # ~48 min of stream at 10x
        time.sleep(3.1)
        rc, dt_ = r.term(signal.SIGTERM)
        r.finish()
        rec, o = case_report(name, r, rc, out)
        rec["shutdown_s"] = dt_
        rec["faults_injected"] = len(faults)
        res[name] = rec
        subprocess.run(["rm", "-rf", os.path.join(out, "state", "raw")], check=False)
        save()
    subprocess.run(["rm", "-rf", os.path.join(base, "faults", "disk_low", "state", "raw")], check=False)
    for k, v in res.items():
        print(k, json.dumps({kk: vv for kk, vv in v.items() if kk in ("rc", "outputs", "vs_baseline", "shutdown_s")},
                            default=str)[:600])


# ============================================================================================= D. alerts vs GLM

def cmd_refault(a):
    """Recompute the baseline comparison of every finished fault case (after a metric change)."""
    resf = os.path.join(a.out, "faults.json" if a.set == "clean" else f"faults_{a.set}.json")
    res = json.load(open(resf))
    fdir = os.path.join(a.out, "faults")
    base_o = outputs_summary(os.path.join(fdir, "baseline", "state"))
    cases = fault_cases()
    for name, rec in res.items():
        if name == "baseline" or name not in cases or cases[name].get("files"):
            continue
        o = outputs_summary(os.path.join(fdir, name, "state"))
        rec["vs_baseline"] = fake_events(o["_evs"], base_o["_evs"], o["_mins"], base_mins=base_o["_mins"])
    if "mixed" in res:
        o = outputs_summary(os.path.join(fdir, "mixed", "state"))
        res["mixed"]["vs_baseline_first3"] = fake_events(
            [e for e in o["_evs"] if e["t"] < base_o["_mins"][-1]["t"] + 60], base_o["_evs"],
            [m for m in o["_mins"] if m["t"] < base_o["_mins"][-1]["t"]], base_mins=base_o["_mins"])
    wjson(resf, res)
    for k, v in res.items():
        if "vs_baseline" in v:
            print(k, json.dumps(v["vs_baseline"]))


def cmd_alerts(a):
    import numpy as np
    sd = os.path.join(a.out, "soak", "state")
    o = outputs_summary(sd)
    mins, alerts = o["_mins"], o["_alerts"]
    files = corpus()
    caps = []
    gdir = os.path.expanduser(GLM_DIR)
    for f in files:
        cid = cap_id(f)
        U = sidecar_t0(f)
        g = os.path.join(gdir, f"glm_{cid[4:]}.npz")
        rate = {}
        per_min = None
        if os.path.exists(g):
            z = np.load(g)
            ft, fd = z["flash_t"], z["flash_dist_km"]          # times and distances only, never positions
            for km in (500, 1000, 2000):
                k = (fd <= km) & (ft >= U) & (ft < U + 600)
                rate[km] = float(k.sum()) / 10.0
            k = fd <= 1000
            per_min = np.histogram(ft[k], bins=np.arange(U, U + 601, 60))[0]
        caps.append({"id": cid, "U": U, "glm_per_min": rate, "glm1000_by_min": per_min})
    r1000 = np.array([c["glm_per_min"].get(1000, np.nan) for c in caps])
    p50, p75, p90 = np.nanpercentile(r1000, [50, 75, 90])
    by_t = sorted(caps, key=lambda c: c["U"])

    def cap_at(t):
        best = None
        for c in by_t:
            if c["U"] - 1 <= t < c["U"] + 600:
                best = c
        return best

    # hum boundaries: minute where hum_frac switches between 0 and > 0 (live rows, time order)
    ms = sorted(mins, key=lambda m: m["t"])
    hb = [ms[i]["t"] for i in range(1, len(ms)) if ((ms[i]["hum_frac"] or 0) > 0) != ((ms[i - 1]["hum_frac"] or 0) > 0)]
    hb = np.array(hb)
    rows = []
    for al in alerts:
        t = al["t"]
        c = cap_at(t)
        up = __import__("sferic_watch.live", fromlist=["LEVELS"]).LEVELS
        is_up = up.index(al["to"]) > up.index(al["from"])
        win = None
        if c is not None and c["glm1000_by_min"] is not None:
            m = int((t - c["U"]) // 60)
            win = float(c["glm1000_by_min"][max(0, m - 9):m + 1].mean())
        cid = c["id"] if c else None
        buzz_on = bool(cid and cid.split("_")[1] in ("20261103", "20261105", "20261106") and cid.endswith(("2200Z", "2230Z")))
        near_hum = bool(hb.size and np.abs(hb - t).min() <= 600)
        # GLM support: within-1000 km flash rate in the 10 min the statistic used at least the corpus 75th pct and
        # >= 3 per min
        sup = None if win is None else bool(win >= max(p75, 3.0))
        rows.append({"utc": al["utc"], "from": al["from"], "to": al["to"], "up": is_up, "capture": cid,
                     "stat_conf_per_min": al["stat_conf_per_min"], "baseline": al["baseline_conf_per_min"],
                     "baseline_source": al["baseline_source"],
                     "glm_flashes_per_min_1000km_last10": None if win is None else round(win, 1),
                     "glm_capture_rate_per_min": {k: round(v, 1) for k, v in (c["glm_per_min"] if c else {}).items()},
                     "glm_supported": sup, "on_buzz_switch_on_capture": buzz_on,
                     "within_10min_of_hum_boundary": near_hum})
    ups = [r for r in rows if r["up"]]
    covered_days = len(mins) / 1440
    span_days = (ms[-1]["t"] - ms[0]["t"]) / 86400 if ms else 0
    false_ups = [r for r in ups if r["glm_supported"] is False]
    # what the level was doing during the strongest GLM hours: captures with GLM >= p90 within 1000 km
    lvl_at = {}
    level_by_min = []
    from sferic_watch import live as lv
    A = lv.Alert()
    for m in ms:
        A.feed(m["t"], m)
        level_by_min.append((m["t"], A.level))
    lt = np.array([x[0] for x in level_by_min])
    lvv = np.array([x[1] for x in level_by_min])
    hi = [c for c in caps if c["glm_per_min"].get(1000, 0) >= max(p90, 3.0)]
    hi_alerted = 0
    for c in hi:
        k = (lt >= c["U"]) & (lt < c["U"] + 600)
        if k.any() and lvv[k].max() > 0:
            hi_alerted += 1
    res = {"alerts_total": len(alerts), "up_transitions": len(ups), "down_transitions": len(rows) - len(ups),
           "glm_1000km_corpus_per_min": {"p50": round(float(p50), 2), "p75": round(float(p75), 2),
                                         "p90": round(float(p90), 2)},
           "support_rule": "GLM flashes within 1000 km, mean per min over the 10 minutes ending at the alert minute,"
                           " >= max(corpus p75 capture rate, 3/min)",
           "ups_glm_supported": sum(1 for r in ups if r["glm_supported"]),
           "ups_not_supported": len(false_ups), "ups_no_glm": sum(1 for r in ups if r["glm_supported"] is None),
           "ups_on_buzz_switch_on": sum(1 for r in ups if r["on_buzz_switch_on_capture"]),
           "ups_within_10min_of_hum_boundary": sum(1 for r in ups if r["within_10min_of_hum_boundary"]),
           "hum_boundaries_seen": int(hb.size),
           "covered_signal_days": round(covered_days, 3), "calendar_span_days": round(span_days, 2),
           "false_ups_per_covered_day": round(len(false_ups) / covered_days, 3) if covered_days else None,
           "false_upper_bound_95_per_covered_day": round(3.0 / covered_days, 2) if not false_ups and covered_days else None,
           "high_glm_captures": len(hi), "high_glm_captures_with_level_above_quiet": hi_alerted,
           "replay_check_levels_match_daemon": None,
           "alerts": rows}
    wjson(os.path.join(a.out, "alerts.json"), res)
    print(json.dumps({k: v for k, v in res.items() if k != "alerts"}, indent=1))
    for r in rows:
        print(r)


# ============================================================================================= main

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("sub", choices=("throughput", "soak", "realtime", "faults", "refault", "alerts", "equiv", "child"))
    ap.add_argument("cfg", nargs="?", help="child: the run config (internal)")
    ap.add_argument("--out", help="output dir (outside the repo)")
    ap.add_argument("--speed", type=float, default=10.0)
    ap.add_argument("--limit", type=int, default=0, help="soak: only the first N captures")
    ap.add_argument("--n", type=int, default=3, help="throughput: captures")
    ap.add_argument("--census", default=None, help="batch census dir (census.csv + census_sessions/)")
    ap.add_argument("--minutes", type=float, default=30.0, help="realtime: wall minutes")
    ap.add_argument("--files", nargs="*", default=[os.path.join("~", "dlayer_run", "data", "vlf", "20261106", f)
                                                    for f in ("vlf_2200Z.cs16", "vlf_2230Z.cs16")])
    ap.add_argument("--only", default=None, help="faults: comma list of case names")
    ap.add_argument("--set", choices=tuple(FAULT_SETS), default="clean", help="faults: capture set")
    ap.add_argument("--redo", action="store_true", help="faults: rerun cases already in faults.json")
    a = ap.parse_args(argv)
    if a.sub == "child":
        return child(a.cfg)
    if not a.out:
        ap.error("--out is required")
    a.out = os.path.abspath(os.path.expanduser(a.out))
    if os.path.commonpath([a.out, REPO]) == REPO:
        ap.error("--out must be outside the repo")
    os.makedirs(a.out, exist_ok=True)
    return {"throughput": cmd_throughput, "soak": cmd_soak, "realtime": cmd_realtime, "faults": cmd_faults,
            "alerts": cmd_alerts, "equiv": cmd_equiv, "refault": cmd_refault}[a.sub](a)


if __name__ == "__main__":
    sys.exit(main() or 0)
