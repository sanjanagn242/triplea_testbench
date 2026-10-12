#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-1}"
# Hard AI: seed of its randomness and simulation dice, battle simulations per estimate (scale;
# 1.0 as ProAi), and battle-phase samples in the simulated purchase turn.
export TRIPLEA_HARD_AI_SEED="${TRIPLEA_HARD_AI_SEED:-0}"
export TRIPLEA_HARD_AI_RUN_SCALE="${TRIPLEA_HARD_AI_RUN_SCALE:-1.0}"
export TRIPLEA_HARD_AI_BATTLE_SAMPLES="${TRIPLEA_HARD_AI_BATTLE_SAMPLES:-5}"
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
