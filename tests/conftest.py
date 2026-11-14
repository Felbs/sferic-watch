"""Shared pytest setup: repo root and bench/ on sys.path, and small SYNTHETIC captures (never the 150 MB banked files)."""
import json
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
for p in (ROOT, os.path.join(ROOT, "bench")):
    if p not in sys.path:
        sys.path.insert(0, p)

FS = 62500.0


@pytest.fixture
def fs():
    return FS


@pytest.fixture
def tiny_capture(tmp_path):
    """A 2 s synthetic .cs16 + sidecar written the way the recorder writes them: int16 I,Q interleaved.

    Content: a ramp in I and a constant -7 in Q, so every sample's value and position can be checked exactly.
    Returns (path, I, Q) with I/Q as the int16 arrays written."""
    n = int(2 * FS)
    i = (np.arange(n) % 2000 - 1000).astype(np.int16)
    q = np.full(n, -7, np.int16)
    raw = np.empty(2 * n, np.int16)
    raw[0::2], raw[1::2] = i, q
    p = tmp_path / "vlf_0000Z.cs16"
    raw.tofile(p)
    side = {"freq_hz": 15000.0, "fs_hz": FS, "secs": 2.0, "utc": "2026-11-05T04:30:00+00:00",
            "antenna": "Antenna C", "agc": False, "IFGR": 40.0}
    (tmp_path / "vlf_0000Z.json").write_text(json.dumps(side))
    return str(p), i, q


@pytest.fixture
def tiny_corpus(tmp_path, tiny_capture):
    """A corpus.json with one golden entry (the tiny capture) and one non-golden missing entry."""
    path, _, _ = tiny_capture
    c = {"captures": [{"id": "tiny", "path": path, "golden": True, "roles": ["clean_day"]},
                      {"id": "gone", "path": str(tmp_path / "nope.cs16"), "golden": False, "roles": ["steady_hum"]}]}
    p = tmp_path / "corpus.json"
    p.write_text(json.dumps(c))
    return str(p)
