"""Reading banked VLF captures (.cs16 + .json sidecar) without ever loading a whole file.

Capture format, as written by dlayer_run and the storm_live recorder:
  - interleaved int16 I,Q (4 bytes per complex sample), 62.5 kS/s, tuned to 15 kHz, so baseband 0 Hz = RF 15 kHz
    and RF 0 Hz sits at baseband -15 kHz
  - a JSON sidecar with the same stem (fs_hz, freq_hz, secs, utc, antenna, IFGR, edges_per_s, hum_share_120hz ...)
  - 600 s files are 150,000,000 bytes

What "DC handled" means here: the baseband mean (the LO leak at RF 15 kHz) is subtracted per slice. It is tiny on
these captures (about 0.7 LSB on 10-19 15Z against a 2,432 LSB rms), so it changes nothing in the 2.5-9 kHz band.
The big one is the receiver tone at RF ~0 Hz (-22.6 dBFS, 99.99% of the power); that is NOT removed here because it
is a band-selection job for the front end (plan step 3). read_slice returns full-scale-normalised complex64 so
``20*log10(rms)`` reads directly in dBFS.
"""
import json
import os
from datetime import datetime

import numpy as np

FS_DEFAULT = 62500.0
FC_DEFAULT = 15000.0
BYTES_PER_SAMPLE = 4          # int16 I + int16 Q
FULL_SCALE = 32768.0

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CORPUS_JSON = os.path.join(REPO, "bench", "corpus.json")


def sidecar_path(path):
    """The .json sidecar next to a .cs16 capture (same stem)."""
    return os.path.splitext(os.path.expanduser(path))[0] + ".json"


def read_sidecar(path):
    """Read the JSON sidecar of a capture; returns {} when there is none.

    Adds ``utc_start`` (timezone-aware datetime) parsed from the ``utc`` field when present. Sidecars carry no station
    position; nothing positional is added here either."""
    p = sidecar_path(path)
    if not os.path.exists(p):
        return {}
    with open(p) as f:
        d = json.load(f)
    if "utc" in d:
        try:
            d["utc_start"] = datetime.fromisoformat(d["utc"])
        except ValueError:
            pass
    return d


def capture_info(path):
    """Sample rate, centre, number of complete complex samples and duration of a capture (sidecar + file size)."""
    path = os.path.expanduser(path)
    side = read_sidecar(path)
    fs = float(side.get("fs_hz", FS_DEFAULT))
    fc = float(side.get("freq_hz", FC_DEFAULT))
    n = os.path.getsize(path) // BYTES_PER_SAMPLE
    return {"fs": fs, "fc": fc, "n_samples": int(n), "secs": n / fs, "sidecar": side}


def read_slice(path, t0, secs, dc="mean", fs=None, allow_short=False):
    """Read ``secs`` seconds starting ``t0`` seconds into a .cs16 capture as complex64, full scale = 1.0.

    Reads only that slice (np.fromfile with count/offset), so a 600 s, 150 MB file is never loaded whole.
    dc: "mean" subtracts the slice's complex mean (baseband DC = RF 15 kHz LO leak); "none" keeps the samples exactly
        as recorded (value/32768, bit-exact round trip back to int16 - the harness uses this to feed the OnAir binary
        the same bytes h2h fed it).
    fs: sample rate override; default from the sidecar, else 62.5 kS/s.
    allow_short: a slice that runs past the end of the file raises ValueError unless this is True (complete-capture
        rule: never score a silently truncated slice).
    """
    path = os.path.expanduser(path)
    if fs is None:
        fs = float(read_sidecar(path).get("fs_hz", FS_DEFAULT))
    s0 = int(round(t0 * fs))
    n = int(round(secs * fs))
    if s0 < 0 or n < 0:
        raise ValueError("t0 and secs must be >= 0")
    raw = np.fromfile(path, dtype=np.int16, count=2 * n, offset=BYTES_PER_SAMPLE * s0)
    if raw.size < 2 * n and not allow_short:
        raise ValueError(f"{path}: asked for {n} samples at {s0}, file has only {raw.size // 2}")
    raw = raw[: raw.size // 2 * 2]
    x = np.empty(raw.size // 2, dtype=np.complex64)
    x.real = raw[0::2]
    x.imag = raw[1::2]
    x /= np.float32(FULL_SCALE)
    if dc == "mean":
        if x.size:
            x -= x.mean(dtype=np.complex128).astype(np.complex64)
    elif dc != "none":
        raise ValueError("dc must be 'mean' or 'none'")
    return x


def iter_slices(path, secs, hop=None, t_start=0.0, t_stop=None, dc="mean"):
    """Yield (t0, x) over a capture in ``secs``-long slices every ``hop`` seconds (default hop = secs).

    Only complete slices are yielded."""
    info = capture_info(path)
    hop = secs if hop is None else hop
    t_stop = info["secs"] if t_stop is None else min(t_stop, info["secs"])
    t0 = t_start
    while t0 + secs <= t_stop + 1e-9:
        yield t0, read_slice(path, t0, secs, dc=dc, fs=info["fs"])
        t0 += hop


def to_cs16(x):
    """Inverse of read_slice(dc="none"): full-scale complex -> interleaved int16 (rounded, clipped)."""
    out = np.empty(2 * len(x), dtype=np.int16)
    out[0::2] = np.clip(np.round(x.real * FULL_SCALE), -32768, 32767)
    out[1::2] = np.clip(np.round(x.imag * FULL_SCALE), -32768, 32767)
    return out


def write_cs16(path, x, sidecar=None):
    """Write complex samples as a .cs16 (and an optional sidecar dict as .json). For tests and scratch files."""
    to_cs16(x).tofile(path)
    if sidecar is not None:
        with open(sidecar_path(path), "w") as f:
            json.dump(sidecar, f, indent=1)


def list_corpus(corpus_json=None, golden_only=False, role=None, must_exist=False):
    """Entries of the frozen regression corpus (bench/corpus.json) with ``path`` expanded to an absolute path.

    golden_only: drop entries with golden=false (e.g. tonight's live capture, kept for spot checks only).
    role: keep only entries whose ``roles`` list contains this label (e.g. "steady_hum", "clean_day").
    must_exist: drop entries whose file is not on this machine (the corpus is shared between machines)."""
    with open(corpus_json or CORPUS_JSON) as f:
        c = json.load(f)
    out = []
    for e in c["captures"]:
        e = dict(e)
        e["path"] = os.path.expanduser(e["path"])
        if golden_only and not e.get("golden", True):
            continue
        if role is not None and role not in e.get("roles", []):
            continue
        if must_exist and not os.path.exists(e["path"]):
            continue
        out.append(e)
    return out


def load_corpus(corpus_json=None):
    """The whole corpus.json as a dict (captures, injection pieces, bank, reference numbers)."""
    with open(corpus_json or CORPUS_JSON) as f:
        return json.load(f)
