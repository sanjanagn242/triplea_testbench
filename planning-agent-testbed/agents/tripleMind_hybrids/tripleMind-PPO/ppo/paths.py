"""Where the reused code lives, and the sys.path entries that make it importable.

Nothing is copied into this folder: the forward model, TripleMind's map graph and generators,
the behaviour-cloning preprocessing and model, the tripleMind-bc agent (whose phases this agent
inherits) and the testbench arena are imported from their own folders.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

PPO_PKG = Path(__file__).resolve().parent                  # tripleMind-PPO/ppo
AGENT_DIR = PPO_PKG.parent                                 # tripleMind-PPO
AGENTS = AGENT_DIR.parents[1]                              # planning-agent-testbed/agents
TESTBED = AGENTS.parent                                    # planning-agent-testbed
TRIPLEMIND = AGENTS / "tripleMind"
BC_DIR = AGENTS / "behaviour-cloning"
BC_AGENT_DIR = AGENT_DIR.parent / "tripleMind-bc"
FORWARD_MODEL = TESTBED / "forward-model" / "python"
TOOLS = TESTBED / "tools"
DEFAULT_XML = TESTBED / "games/capture-the-flag-diag/map/capture_the_flag/map/games/capture_the_flag.xml"
DEFAULT_RUNS = AGENT_DIR / "runs"
DEFAULT_CHECKPOINT = DEFAULT_RUNS / "ppo-selfplay" / "checkpoints" / "best.pt"

# Later entries end up first on sys.path.
for _p in (TOOLS, BC_DIR, BC_DIR / "data-processing", TRIPLEMIND, BC_AGENT_DIR, FORWARD_MODEL):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))


def load_module(name: str, path: Path):
    """Import a file under a unique module name (three agents here are all called agent.py)."""
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bc_agent_module():
    """tripleMind-bc/agent.py: TripleMindBC, the TripleMind phases with a pluggable formation
    chooser (its `bc` attribute)."""
    return load_module("tripleMind_bc_agent", BC_AGENT_DIR / "agent.py")


def triplemind_agent_module():
    """agents/tripleMind/agent.py: the original MCTS agent (used by the parity check)."""
    return load_module("tripleMind_mcts_agent", TRIPLEMIND / "agent.py")
