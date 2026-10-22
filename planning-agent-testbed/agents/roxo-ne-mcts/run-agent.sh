#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search, as in Roxo's NonExploringMcts: seconds and iteration cap per move decision (his code:
# 1000 ms and 500), playout length in move phases (40) and the seed of the Weak AI's shuffle.
export ROXO_SECONDS="${ROXO_SECONDS:-1.0}"
export ROXO_MAX_ITERATIONS="${ROXO_MAX_ITERATIONS:-500}"
export ROXO_PLAYOUT_PHASES="${ROXO_PLAYOUT_PHASES:-40}"
export ROXO_SEED="${ROXO_SEED:-0}"
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
