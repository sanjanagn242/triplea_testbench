#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-1}"
# Evolution: time per searched decision (purchase, combat move, non-combat move) in seconds,
# population size, share kept each generation, probability that an offspring is mutated, actions
# per genome (the paper's 5, reduced to 3) and seed.
export TRIPLEA_OEP_SECONDS="${TRIPLEA_OEP_SECONDS:-1.0}"
export TRIPLEA_OEP_POPULATION="${TRIPLEA_OEP_POPULATION:-100}"
export TRIPLEA_OEP_SURVIVAL="${TRIPLEA_OEP_SURVIVAL:-0.5}"
export TRIPLEA_OEP_MUTATION="${TRIPLEA_OEP_MUTATION:-0.1}"
export TRIPLEA_OEP_GENOME_LENGTH="${TRIPLEA_OEP_GENOME_LENGTH:-100}"
# Evaluation: player turns played out after mine with TripleMind's deterministic policy (0: static
# evaluation, the paper's; 2 in a two-player game: the opponent's turn, then mine).
export TRIPLEA_OEP_ROLLOUT_LENGTH="${TRIPLEA_OEP_ROLLOUT_LENGTH:-0}"
export TRIPLEA_OEP_SEED="${TRIPLEA_OEP_SEED:-0}"
# Battles in the search's simulations: dice (the forward model's dice-rolling battles, default
# casualties, never retreat) or fast (the Fast AI's most likely outcome, no dice).
export TRIPLEA_BATTLE_RESOLUTION="${TRIPLEA_BATTLE_RESOLUTION:-dice}"
# TripleMind's action generators: 1 forces the pure-Python ones, 0 uses the Cython build.
export TRIPLEMIND_PURE_PYTHON="${TRIPLEMIND_PURE_PYTHON:-0}"
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"

# (Re)build the Cython action generators when the .pyx is newer than the compiled module. The
# lock serialises agents launched together; a failed build falls back to the Python generators.
(
  flock 9
  so="$(ls "$agent_dir"/action_generators_cy*.so 2>/dev/null | head -n 1 || true)"
  if [ -z "$so" ] || [ "$agent_dir/action_generators_cy.pyx" -nt "$so" ]; then
    bash "$agent_dir/scripts/build_cython.sh" || echo "oep: Cython build failed" >&2
  fi
) 9>"$agent_dir/.build.lock"

python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
