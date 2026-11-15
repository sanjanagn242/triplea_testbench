#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search: time per searched decision (purchase, combat move, non-combat move) in seconds, split
# into burn phases; C of the UCT term 2C sqrt(2 ln n / n_j) (the paper's 1/sqrt 2), probability
# of the highest-rated action in rollouts, and seed.
export TRIPLEA_BB_SECONDS="${TRIPLEA_BB_SECONDS:-1.0}"
export TRIPLEA_BB_C="${TRIPLEA_BB_C:-0.7071067811865476}"
export TRIPLEA_BB_EPSILON="${TRIPLEA_BB_EPSILON:-0.5}"
export TRIPLEA_BB_SEED="${TRIPLEA_BB_SEED:-0}"
# The pure-Python TripleMind action generators (this folder has no Cython build).
export TRIPLEMIND_PURE_PYTHON=1
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
