#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-1}"
# Search time per decision (s), rollout horizon in rounds (a playout still running then has no
# winner), and the search seed.
export TRIPLEA_MCTS_SECONDS="${TRIPLEA_MCTS_SECONDS:-2.0}"
export TRIPLEA_MCTS_ROLLOUT="${TRIPLEA_MCTS_ROLLOUT:-2}"
export TRIPLEA_MCTS_SEED="${TRIPLEA_MCTS_SEED:-0}"
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
