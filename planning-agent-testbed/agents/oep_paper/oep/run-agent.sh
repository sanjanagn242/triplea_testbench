#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Evolution: time per searched decision (purchase, combat move, non-combat move) in seconds,
# population size, share kept each generation, probability that an offspring is mutated, actions
# per genome (the paper's 5, reduced to 3) and seed.
export TRIPLEA_OEP_SECONDS="${TRIPLEA_OEP_SECONDS:-1.0}"
export TRIPLEA_OEP_POPULATION="${TRIPLEA_OEP_POPULATION:-100}"
export TRIPLEA_OEP_SURVIVAL="${TRIPLEA_OEP_SURVIVAL:-0.5}"
export TRIPLEA_OEP_MUTATION="${TRIPLEA_OEP_MUTATION:-0.1}"
export TRIPLEA_OEP_GENOME_LENGTH="${TRIPLEA_OEP_GENOME_LENGTH:-3}"
export TRIPLEA_OEP_SEED="${TRIPLEA_OEP_SEED:-0}"
# The pure-Python TripleMind action generators (this folder has no Cython build).
export TRIPLEMIND_PURE_PYTHON=1
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
