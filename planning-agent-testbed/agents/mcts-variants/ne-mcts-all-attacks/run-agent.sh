#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search: time per combat-move decision (s), fixed dice seeds per search (averaged per rollout)
# and seed.
export TRIPLEA_NE_SECONDS="${TRIPLEA_NE_SECONDS:-1.0}"
export TRIPLEA_NE_DICE_SEEDS="${TRIPLEA_NE_DICE_SEEDS:-1}"
export TRIPLEA_NE_SEED="${TRIPLEA_NE_SEED:-0}"
# Debugging: draw every combat-move search tree to planning-agent-testbed/logs/trees/ (edge_tree.py):
# SAVE_TREE 1 saves .dot and .html, TREE_PNG 1 also renders a .png (needs Graphviz `dot`),
# TREE_MAX_NODES caps the nodes drawn, TREE_RANKDIR lays the tree out top to bottom (TB) or left
# to right (LR).
export TRIPLEA_NE_SAVE_TREE="${TRIPLEA_NE_SAVE_TREE:-0}"
export TRIPLEA_NE_TREE_PNG="${TRIPLEA_NE_TREE_PNG:-1}"
export TRIPLEA_NE_TREE_MAX_NODES="${TRIPLEA_NE_TREE_MAX_NODES:-300}"
export TRIPLEA_NE_TREE_RANKDIR="${TRIPLEA_NE_TREE_RANKDIR:-TB}"
# The pure-Python TripleMind action generators (this folder has no Cython build).
export TRIPLEMIND_PURE_PYTHON=1
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
