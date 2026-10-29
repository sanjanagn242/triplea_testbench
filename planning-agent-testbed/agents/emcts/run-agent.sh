#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search, as in Baier and Cowling's EMCTS: seconds per combat move, bridge-burning phases, UCB1
# exploration constant (0 in the paper), seed, and rollout depth in the player's own turns (1: the
# paper's evaluation at the end of this turn, no rollout; n: n - 1 more rounds of TripleMind's
# rollout policy before the evaluation).
export EMCTS_SECONDS="${EMCTS_SECONDS:-1.0}"
export EMCTS_PHASES="${EMCTS_PHASES:-40}"
export EMCTS_EXP_CONST="${EMCTS_EXP_CONST:-0}"
export EMCTS_SEED="${EMCTS_SEED:-0}"
export EMCTS_ROLLOUT="${EMCTS_ROLLOUT:-1}"
# Search tree of each combat move in planning-agent-testbed/logs/trees/: SAVE_TREE 1 saves a .dot,
# TREE_PNG 1 also renders a .png (needs Graphviz `dot`), TREE_MAX_NODES caps the nodes drawn (the
# most visited, plus the bridge-burning path), TREE_MAX_DEPTH the levels drawn below that path,
# TREE_RANKDIR lays depth out left to right (LR) or
# top to bottom (TB).
export EMCTS_SAVE_TREE="${EMCTS_SAVE_TREE:-0}"
export EMCTS_TREE_PNG="${EMCTS_TREE_PNG:-1}"
export EMCTS_TREE_MAX_NODES="${EMCTS_TREE_MAX_NODES:-300}"
export EMCTS_TREE_MAX_DEPTH="${EMCTS_TREE_MAX_DEPTH:-3}"
export EMCTS_TREE_RANKDIR="${EMCTS_TREE_RANKDIR:-LR}"
# The pure-Python TripleMind action generators (this folder has no Cython build).
export TRIPLEMIND_PURE_PYTHON=1
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
