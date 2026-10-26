#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search, as in Roxo's OEP (thesis 4.4): seconds per move phase, population size, survival rate,
# mutation probability, playout length in rounds and the seed of the evolution and the Weak AI.
export ROXO_SECONDS="${ROXO_SECONDS:-1.0}"
export ROXO_OEP_POPULATION="${ROXO_OEP_POPULATION:-50}"
export ROXO_OEP_SURVIVAL="${ROXO_OEP_SURVIVAL:-0.5}"
export ROXO_OEP_MUTATION="${ROXO_OEP_MUTATION:-0.1}"
export ROXO_OEP_PLAYOUT_ROUNDS="${ROXO_OEP_PLAYOUT_ROUNDS:-10}"
export ROXO_SEED="${ROXO_SEED:-0}"
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
