#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search: MCTS over turn genomes. The root is the current state, its children genomes of my
# remaining turn, and every node's children genomes of the next player's turn. Seconds per
# searched phase (purchase, combat move and noncombat move are each searched), UCB1 exploration
# constant, seed, rollout length in rounds (after the last genome's turn, every player plays
# greedily until that many rounds have started after that turn's round; 0: stop at the end of
# that turn, 1: at the end of that round), and the % of expansions that add a child (the next
# turn's greedy genome) instead of a sibling (a mutation of the selected leaf).
export TBETS_SECONDS="${TBETS_SECONDS:-1.0}"
export TBETS_EXP_CONST="${TBETS_EXP_CONST:-0}"
export TBETS_SEED="${TBETS_SEED:-0}"
export TBETS_ROLLOUT="${TBETS_ROLLOUT:-2}"
export TBETS_GREEDY_PCT="${TBETS_GREEDY_PCT:-5}"
# Search tree of each searched phase in planning-agent-testbed/logs/trees/: SAVE_TREE 1 saves a .dot,
# TripleMind's style, with a collapsible .html of every genome. TREE_PNG 1 also renders a .png
# (needs Graphviz `dot`), TREE_MAX_NODES caps the genomes drawn in the .dot/.png (best mean first),
# TREE_RANKDIR stacks them (LR) or puts them side by side (TB).
export TBETS_SAVE_TREE="${TBETS_SAVE_TREE:-0}"
export TBETS_TREE_PNG="${TBETS_TREE_PNG:-0}"
export TBETS_TREE_MAX_NODES="${TBETS_TREE_MAX_NODES:-30}"
export TBETS_TREE_RANKDIR="${TBETS_TREE_RANKDIR:-LR}"
# The pure-Python TripleMind action generators (this folder has no Cython build).
export TRIPLEMIND_PURE_PYTHON=1
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
# Battles in the search's simulations: fast (the Fast AI's most likely outcome, no dice) or
# dice (the forward model's dice-rolling battles, default casualties, never retreat).
export TRIPLEA_ROLLOUT_BATTLE="${TRIPLEA_ROLLOUT_BATTLE:-fast}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
