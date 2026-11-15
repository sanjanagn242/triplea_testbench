#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search: time per phase decision (purchase, combat move, non-combat move; s) and seed.
export TRIPLEA_GREEDY_SECONDS="${TRIPLEA_GREEDY_SECONDS:-1.0}"
export TRIPLEA_GREEDY_SEED="${TRIPLEA_GREEDY_SEED:-0}"
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
    bash "$agent_dir/scripts/build_cython.sh" || echo "greedy-action: Cython build failed" >&2
  fi
) 9>"$agent_dir/.build.lock"

python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
