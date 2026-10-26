"""Picks the action-generator implementation used by the agent and the search.

`ag` is the compiled Cython module (action_generators_cy, built by scripts/build_cython.sh) when it
is available, otherwise the pure-Python reference action_generators. Both expose the same
functions with the same results. Set TRIPLEMIND_PURE_PYTHON=1 to force the Python version.
Diagnostics go to stderr only, and only when TRIPLEA_AGENT_LOG is 1: stdout carries the agent protocol.
"""

import os
import sys

if os.environ.get("TRIPLEMIND_PURE_PYTHON") == "1":
    import action_generators as ag
    IMPLEMENTATION = "python (TRIPLEMIND_PURE_PYTHON=1)"
else:
    try:
        import action_generators_cy as ag
        IMPLEMENTATION = "cython"
    except ImportError as e:
        import action_generators as ag
        IMPLEMENTATION = "python (Cython module not built)"
        if os.environ.get("TRIPLEA_AGENT_LOG") == "1":
            print(f"tripleMind: action_generators_cy unavailable ({e}); using the pure-Python "
                  f"generators. Build it with scripts/build_cython.sh.", file=sys.stderr, flush=True)
