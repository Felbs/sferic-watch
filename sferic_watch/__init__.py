"""sferic-watch engine v2: single-station lightning and ionosphere sensing from banked VLF IQ captures.

This file holds only the version and the tiny detector registry the bench harness scores.

A detector is any callable ``detector(x, fs) -> (times, peaks)``:
  x      complex64 IQ, full scale = 1.0 (what ``sferic_watch.io.read_slice`` returns), 62.5 kS/s centred on 15 kHz
  fs     sample rate in Hz
  times  1-D array of event onset times in seconds from x[0]
  peaks  1-D array of the same length (dB over the detector's own floor, or NaN if it has none)
It may also return a third item, a dict of extra numbers (for example the CPU the detector measured itself).

Register one with::

    import sferic_watch
    @sferic_watch.register("my_detector", ungated="my_detector_no_gate")
    def my_detector(x, fs): ...

``ungated`` names a twin with its mains gate switched off; the harness uses it to count false mains rejections.
"""

__version__ = "0.2.0.dev0"

_REGISTRY = {}


def register(name, ungated=None, note=""):
    """Decorator: add a detector to the registry under ``name``. Re-registering a name replaces it."""
    def deco(fn):
        _REGISTRY[name] = {"fn": fn, "ungated": ungated, "note": note or (fn.__doc__ or "").strip().split("\n")[0]}
        return fn
    return deco


def get_detector(name):
    """Return the registry entry (dict with fn, ungated, note) for ``name``; KeyError lists the known names."""
    if name not in _REGISTRY:
        raise KeyError(f"unknown detector {name!r}; registered: {sorted(_REGISTRY)}")
    return _REGISTRY[name]


def detector_names():
    """Sorted list of registered detector names."""
    return sorted(_REGISTRY)
