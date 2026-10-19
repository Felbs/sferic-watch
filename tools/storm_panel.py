"""storm_panel.py - sferic-watch's live storm scope (http://127.0.0.1:8647).

Reads a storm session directory (the VLF leg outputs of a sferic
capture chain) and serves the storm's story as it unfolds - ZERO radio
contention, this panel only reads files:

  * impulse rate per leg  -> the intensity curve + LIGHTNING JUMP alert
    (a sudden rate spike is a published severe-weather precursor,
    10-30 min ahead of radar confirmation)
  * power-line comb fraction -> the honesty dial (house vs sky)
  * tweek dispersion fits -> stroke distances (range-only: one loop
    antenna has no bearing - honest range, never a fake map) and the
    D-layer ceiling height when night tweeks land
  * VLF carrier traces (NAA/NLK/NML) -> the Early-VLF-event watch:
    lightning visibly denting the ionosphere over the transmitter path

Session dir layout (produced by the capture chain):
  vlf_HHMMSS.json   per-leg summary: impulse times, rate, carriers, tweeks
  naa_HHMMSS.npy    [n,4] float32: t_rel_s, carrier amplitudes

  python tools/storm_panel.py --session <dir> [--port 8647]
"""
import argparse
import json
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent


def powerline_frac(impulse_t):
    """Fraction of inter-impulse intervals on the 120 Hz mains comb.
    High = the house, not the sky (honesty dial, dlayer-diary lineage)."""
    if len(impulse_t) < 10:
        return None
    iei = np.diff(np.asarray(impulse_t)) * 1000.0
    ph = iei / (1000.0 / 120.0)
    on = np.abs(ph - np.round(ph)) * (1000.0 / 120.0) < 0.6
    return round(float(np.mean(on)), 2)


def load_session(session):
    legs = []
    for p in sorted(session.glob("vlf_*.json")):
        try:
            d = json.loads(p.read_text())
        except Exception:
            continue
        stamp = p.stem.split("_")[1]
        tw = d.get("tweeks") or {}
        legs.append({
            "stamp": stamp,
            "hhmm": f"{stamp[:2]}:{stamp[2:4]}",
            "secs": d.get("secs"),
            "rate": d.get("impulses_per_min"),
            "n": d.get("n_impulses"),
            "pl_frac": powerline_frac(d.get("impulse_t", [])),
            "n_tweeks": tw.get("n_tweeks_good"),
            "h_km": tw.get("h_km_median"),
            "d_km": (tw.get("d_km_list") or [])[:40],
            "carriers": d.get("carriers", {}),
        })
    return legs


def latest_carrier_trace(session, max_pts=600):
    npys = sorted(session.glob("naa_*.npy"))
    if not npys:
        return None
    cs = np.load(npys[-1])
    step = max(1, len(cs) // max_pts)
    cs = cs[::step]
    return {"stamp": npys[-1].stem.split("_")[1],
            "t": [round(float(x), 1) for x in cs[:, 0]],
            "series": {name: [round(float(v), 6) for v in cs[:, 1 + i]]
                       for i, name in enumerate(
                           ("NAA_24.0", "NLK_24.8", "NML_25.2"))}}


def jump_alert(legs):
    """Lightning-jump: latest sky-dominated rate vs the median of the
    prior three legs. >2x = JUMP (severity escalating)."""
    sky = [g for g in legs
           if g["rate"] and (g["pl_frac"] is None or g["pl_frac"] < 0.5)]
    if len(sky) < 2:
        return None
    cur = sky[-1]
    prior = [g["rate"] for g in sky[-4:-1]]
    if not prior:
        return None
    ref = sorted(prior)[len(prior) // 2]
    if ref > 0 and cur["rate"] > 2.0 * ref and cur["rate"] > 100:
        return {"jump": True,
                "msg": f"LIGHTNING JUMP at {cur['hhmm']}: "
                       f"{cur['rate']:.0f}/min vs {ref:.0f}/min median - "
                       "severity escalating (10-30 min lead is typical)"}
    return None


def make_handler(session):
    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, body, ctype="application/json"):
            self.send_response(code)
            self.send_header("Content-Type", ctype + "; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body if isinstance(body, bytes)
                             else body.encode())

        def do_GET(self):
            if self.path in ("/", "/index.html"):
                page = HERE / "storm_scope.html"
                self._send(200, page.read_bytes(), "text/html")
            elif self.path == "/storm.json":
                legs = load_session(session)
                self._send(200, json.dumps({
                    "t": time.time(),
                    "session": session.name,
                    "legs": legs,
                    "trace": latest_carrier_trace(session),
                    "alert": jump_alert(legs)}))
            else:
                self._send(404, "not found", "text/plain")
    return H


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--session", required=True,
                    help="storm session directory (vlf_*.json + naa_*.npy)")
    ap.add_argument("--port", type=int, default=8647)
    args = ap.parse_args()
    session = Path(args.session)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              make_handler(session))
    print(f"[storm_panel] http://127.0.0.1:{args.port}  session={session}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
