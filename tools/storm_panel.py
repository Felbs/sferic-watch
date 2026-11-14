"""storm_panel.py - sferic-watch's storm scope (http://127.0.0.1:8647).

Reads a folder of session files and serves them to tools/storm_scope.html. It never touches the radio; it only
reads files.

What it reads (engine v2, plan step 6): ``<id>.session.json`` files written by ``sferic_watch.report`` (schema in
sferic_watch/schema.json), for example from

    python -m sferic_watch.cli analyze FILE.cs16 --out DIR
    python tools/storm_panel.py --session DIR

Each minute of each capture becomes one point ("leg") on the intensity curve: gated events per minute, with the
residual 120 Hz comb share as the house-vs-sky dial (the SKY % tile is chance-corrected, ``sky_share``: pure sky
reads 100 %, not the 86 % that 1 - comb share gives). Labels carry MM-DD when the folder spans several UTC dates. A minute is drawn HOLLOW and its number greyed when the
report marks it untrusted: comb share above 0.25 (chance is 0.144) or more than 20 % of its time blanked by the
mains gate. The lightning-jump banner only compares trusted minutes, so a porch light switching on can no longer
look like a 40x storm jump.

Dropped on purpose: per-tweek range. The July tweek fit sits on its own cutoff bound (plan step 7: 29 of 77 fits
within 0.5 Hz of it), so no distance or ceiling is published until the tweek audit passes. The range card is
hidden.

Older session folders (``vlf_HHMMSS.json`` with ``impulses_per_min`` and ``impulse_t``) still load, one leg per
file, judged by the same comb-share rule; their tweek numbers are ignored for the same reason.

    python tools/storm_panel.py --session <dir> [--port 8647]
"""
import argparse
import json
import time
from datetime import datetime, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
COMB_TRUST_MAX = 0.25      # same rules as sferic_watch.report (kept here so the panel runs without the package)
GATED_TRUST_MAX = 0.20
COMB_CHANCE = 2 * 0.6 / (1000.0 / 120.0)   # 0.144: comb share of independent (pure sky) events


def sky_share(comb, chance=COMB_CHANCE):
    """Estimated share of events NOT locked to the mains comb, for the SKY % tile. A raw (1 - comb share) reads
    about 86 % for pure sky, because 14.4 % of independent intervals land on the comb by chance. A mix of a
    fraction b of buzz intervals (comb 1) and 1 - b independent ones (comb = chance) reads comb = b + (1 - b) chance,
    so b = (comb - chance) / (1 - chance); sky = 1 - b, clipped to [0, 1]. None when there is no comb share."""
    if comb is None:
        return None
    b = (float(comb) - chance) / (1.0 - chance)
    return round(min(1.0, max(0.0, 1.0 - b)), 3)


def powerline_frac(impulse_t):
    """Fraction of inter-impulse intervals within 0.6 ms of a multiple of 8.333 ms (chance 0.144 for independent
    impulses). For the older session files only; v2 sessions carry it as comb_share."""
    if len(impulse_t) < 10:
        return None
    iei = np.diff(np.asarray(impulse_t)) * 1000.0
    ph = iei / (1000.0 / 120.0)
    on = np.abs(ph - np.round(ph)) * (1000.0 / 120.0) < 0.6
    return round(float(np.mean(on)), 2)


def _when(utc, t0_s):
    """UTC datetime of a minute from the capture's start time plus its offset; None when there is no start time."""
    try:
        return datetime.fromisoformat(utc) + timedelta(seconds=float(t0_s))
    except (TypeError, ValueError):
        return None


def _hhmm(utc, t0_s, fallback):
    """HH:MM of a minute from the capture's start time plus its offset; the fallback label when there is none."""
    w = _when(utc, t0_s)
    return w.strftime("%H:%M") if w else fallback


def date_labels(legs):
    """When the legs span more than one UTC date, prefix every dated label with MM-DD, so minutes from different
    nights are told apart on the time axis (labels are HH:MM only within one date)."""
    days = {g["day"] for g in legs if g.get("day")}
    if len(days) > 1:
        for g in legs:
            if g.get("day"):
                g["hhmm"] = f"{g['day'][5:]} {g['hhmm']}"
    return legs


def legs_from_v2(doc):
    """Per-minute legs of one v2 session document."""
    cap = doc.get("capture", {})
    out = []
    for m in doc.get("minutes", []):
        hhmm = _hhmm(cap.get("utc_start"), m["t0_s"], f"{cap.get('id', '?')}+{int(m['t0_s'] // 60)}m")
        w = _when(cap.get("utc_start"), m["t0_s"])
        out.append({
            "stamp": hhmm.replace(":", "") + "00", "hhmm": hhmm, "day": w.strftime("%Y-%m-%d") if w else None,
            "capture": cap.get("id"), "sky": sky_share(m["comb_share"]),
            "utc_start": cap.get("utc_start"), "t0_s": m["t0_s"], "secs": m["dur_s"],
            "rate": m["rate_per_min"], "n": m["n_events"], "pl_frac": m["comb_share"],
            "gated_frac": m["gated_frac"], "hum_frac": m["hum_frac"],
            "v2_rate": m["v2_per_min"], "v2_chance": m["v2_chance_per_min"],
            "trusted": bool(m["trusted"]), "why": list(m["untrusted_because"]),
            "detector_version": doc.get("detector_version"),
            "n_tweeks": None, "h_km": None, "d_km": [], "carriers": {}})
    return out


def legs_from_legacy(p, d):
    """One leg from an older vlf_HHMMSS.json leg summary (no gate information: trust from the comb share only)."""
    stamp = p.stem.split("_")[1]
    pl = powerline_frac(d.get("impulse_t", []))
    why = ["comb_share"] if pl is not None and pl > COMB_TRUST_MAX else []
    return [{"stamp": stamp, "hhmm": f"{stamp[:2]}:{stamp[2:4]}", "day": None, "capture": p.stem, "sky": sky_share(pl), "utc_start": None, "t0_s": 0.0,
             "secs": d.get("secs"), "rate": d.get("impulses_per_min"), "n": d.get("n_impulses"), "pl_frac": pl,
             "gated_frac": None, "hum_frac": None, "v2_rate": None, "v2_chance": None,
             "trusted": not why, "why": why, "detector_version": "legacy",
             "n_tweeks": None, "h_km": None, "d_km": [], "carriers": d.get("carriers", {})}]


def load_session(session):
    """All legs in a session folder, oldest first: v2 ``*.session.json`` files when there are any, else the older
    ``vlf_*.json`` leg files."""
    session = Path(session)
    v2 = sorted(session.glob("*.session.json"))
    legs = []
    if v2:
        docs = []
        for p in v2:
            try:
                docs.append(json.loads(p.read_text()))
            except (OSError, ValueError):
                continue
        docs.sort(key=lambda d: (d.get("capture", {}).get("utc_start") or "", d.get("capture", {}).get("id", "")))
        for d in docs:
            legs += legs_from_v2(d)
        return date_labels(legs)
    for p in sorted(session.glob("vlf_*.json")):
        try:
            legs += legs_from_legacy(p, json.loads(p.read_text()))
        except (OSError, ValueError, IndexError):
            continue
    return legs


def latest_carrier_trace(session, max_pts=600):
    npys = sorted(Path(session).glob("naa_*.npy"))
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
    """Lightning jump: latest TRUSTED rate vs the median of the three trusted legs before it. More than 2x (and over
    100/min) = JUMP. Untrusted legs (comb share or blanking too high) never take part."""
    sky = [g for g in legs if g["rate"] and g.get("trusted", True)]
    if len(sky) < 2:
        return None
    cur = sky[-1]
    prior = [g["rate"] for g in sky[-4:-1]]
    ref = sorted(prior)[len(prior) // 2]
    if ref > 0 and cur["rate"] > 2.0 * ref and cur["rate"] > 100:
        return {"jump": True,
                "msg": f"LIGHTNING JUMP at {cur['hhmm']}: "
                       f"{cur['rate']:.0f}/min vs {ref:.0f}/min median - "
                       "severity escalating (10-30 min lead is typical)"}
    return None


# Added to the served page (storm_scope.html is left as it is): hollow points and a greyed rate tile follow the
# report's "trusted" flag instead of the page's own comb > 50 % rule, and the tweek range card is hidden.
TRUST_JS = """
<style>.sw-grey{opacity:.35}</style>
<script id="sw-trust">
"use strict";
(function(){
  const of = window.fetch;
  window.fetch = async function(u, o){
    const r = await of(u, o);
    if (String(u).indexOf("storm.json") >= 0) { try { window.__sw = await r.clone().json(); } catch(e){} }
    return r;
  };
  const draw = drawSeries;
  drawSeries = function(svg, xs, ys, opts){
    if (svg.id === "rate" && window.__sw)
      opts = {hollow: window.__sw.legs.map(g => g.trusted === false)};
    draw(svg, xs, ys, opts);
    if (svg.id !== "rate" || !window.__sw) return;
    const legs = window.__sw.legs, last = legs[legs.length-1];
    const sk = document.getElementById("t_sky");      // chance-corrected sky share, not 1 - comb share
    if (sk && last) sk.textContent = last.sky == null ? "\u2013" : Math.round(last.sky * 100);
    for (const id of ["t_rate", "t_sky"]) {
      const e = document.getElementById(id);
      const bad = last && last.trusted === false;
      e.classList.toggle("sw-grey", !!bad);
      e.title = bad ? "untrusted: " + last.why.join(", ") : "";
    }
  };
  const card = document.getElementById("range");
  if (card && card.closest(".card")) card.closest(".card").style.display = "none";
  const note = document.querySelector("#rate") && document.querySelector("#rate").parentNode.querySelector(".note");
  if (note) note.textContent = "one point per minute of gated events. Hollow (and a grey number above) = " +
    "untrusted: residual 120 Hz comb share over 0.25 (chance 0.144) or over 20% of the minute blanked by the " +
    "mains gate. The jump banner compares trusted minutes only. SKY % is corrected for the 14.4% chance comb " +
    "share of independent events (pure sky reads 100).";
  if (typeof poll === "function") poll();     // the page's first poll ran before this overlay was in place
})();
</script>
"""


def served_page():
    """storm_scope.html with the trust overlay inserted before </body>."""
    page = (HERE / "storm_scope.html").read_bytes()
    i = page.rfind(b"</body>")
    return page[:i] + TRUST_JS.encode() + page[i:] if i >= 0 else page + TRUST_JS.encode()


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
                self._send(200, served_page(), "text/html")
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
                    help="folder of <id>.session.json files (or older vlf_*.json + naa_*.npy)")
    ap.add_argument("--port", type=int, default=8647)
    args = ap.parse_args()
    session = Path(args.session)
    srv = ThreadingHTTPServer(("127.0.0.1", args.port),
                              make_handler(session))
    print(f"[storm_panel] http://127.0.0.1:{args.port}  session={session}")
    srv.serve_forever()


if __name__ == "__main__":
    main()
