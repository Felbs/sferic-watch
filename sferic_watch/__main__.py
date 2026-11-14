"""``python -m sferic_watch ...`` runs the command line in :mod:`sferic_watch.cli` (same arguments).

The call sits behind ``if __name__ == "__main__"`` so that importing this
module does nothing. bench/harness.py ``load_plugins()`` imports every
submodule of the package, and without the guard that import ran the CLI on
the harness's own argv and exited with code 2.
"""
import sys

from sferic_watch.cli import main

if __name__ == "__main__":
    sys.exit(main())
