#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG=1
# Search: time per phase (s), seed, UCT exploration constant, iteration cap besides the time.
export TRIPLEA_AGENT_SECONDS=600
export TRIPLEA_AGENT_SEED=0
export TRIPLEA_CLASSIC_MCTS_EXPLORATION=1.414
export TRIPLEA_CLASSIC_MCTS_MAX_ITERATIONS=100000
# Rounds a playout runs before it is stopped with no winner (scored as a loss).
export TRIPLEA_CLASSIC_MCTS_ROLLOUT=100
# Search trees in logs/trees/: 1 saves a .dot after every search, TREE_PNG 1 also renders a
# .png (needs Graphviz `dot`); TREE_MAX_NODES caps the nodes drawn.
export TRIPLEA_CLASSIC_MCTS_SAVE_TREE=1
export TRIPLEA_CLASSIC_MCTS_TREE_PNG=1
export TRIPLEA_CLASSIC_MCTS_TREE_MAX_NODES=500
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
