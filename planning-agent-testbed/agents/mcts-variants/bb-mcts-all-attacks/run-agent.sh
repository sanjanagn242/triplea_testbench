#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search: time per combat-move decision (s), UCT constant c of mean + c sqrt(ln n / n_j) (as
# TripleMind; the paper's weight is c = 4 on these [-1, 1] rewards), rollout epsilon
# (probability of the greedy action; paper: 0.5) and seed.
export TRIPLEA_BB_SECONDS="${TRIPLEA_BB_SECONDS:-1.0}"
export TRIPLEA_BB_EXP_CONST="${TRIPLEA_BB_EXP_CONST:-1.414}"
export TRIPLEA_BB_EPSILON="${TRIPLEA_BB_EPSILON:-0.5}"
# Estimated phases per turn (the budget is split between them; TripleMind's MCTS attacks 3-4
# territories a turn).
export TRIPLEA_BB_PHASES="${TRIPLEA_BB_PHASES:-4}"
export TRIPLEA_BB_SEED="${TRIPLEA_BB_SEED:-0}"
# Debugging: draw every combat-move search tree to planning-agent-testbed/logs/trees/ (edge_tree.py):
# SAVE_TREE 1 saves .dot and .html, TREE_PNG 1 also renders a .png (needs Graphviz `dot`),
# TREE_MAX_NODES caps the nodes drawn, TREE_RANKDIR lays the tree out top to bottom (TB) or left
# to right (LR).
export TRIPLEA_BB_SAVE_TREE="${TRIPLEA_BB_SAVE_TREE:-0}"
export TRIPLEA_BB_TREE_PNG="${TRIPLEA_BB_TREE_PNG:-1}"
export TRIPLEA_BB_TREE_MAX_NODES="${TRIPLEA_BB_TREE_MAX_NODES:-300}"
export TRIPLEA_BB_TREE_RANKDIR="${TRIPLEA_BB_TREE_RANKDIR:-TB}"
# The pure-Python TripleMind action generators (this folder has no Cython build).
export TRIPLEMIND_PURE_PYTHON=1
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
