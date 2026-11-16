#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search (thesis Sec. 3.10.1, 5.7): time per searched decision (purchase, combat move, non-combat
# move) in seconds, rounds a playout may advance (Sec. 5.5) and seed. C = 0 and the deterministic
# playout are the algorithm, not settings.
export TRIPLEA_ROXO_NE_SECONDS="${TRIPLEA_ROXO_NE_SECONDS:-1.0}"
export TRIPLEA_ROXO_NE_PLAYOUT_ROUNDS="${TRIPLEA_ROXO_NE_PLAYOUT_ROUNDS:-10}"
# Dice battles only: K fixed dice seeds per search, every playout averaged over them.
export TRIPLEA_ROXO_NE_DICE_SEEDS="${TRIPLEA_ROXO_NE_DICE_SEEDS:-1}"
export TRIPLEA_ROXO_NE_SEED="${TRIPLEA_ROXO_NE_SEED:-0}"
# Battles in the search's simulations: fast (the Fast AI's most likely outcome, no dice: the thesis's
# deterministic battles, Sec. 5.2.2) or dice (the forward model's dice-rolling battles, default
# casualties, never retreat).
export TRIPLEA_BATTLE_RESOLUTION="${TRIPLEA_BATTLE_RESOLUTION:-fast}"
# TripleMind's action generators (the forced air landing of the non-combat move): 1 forces the
# pure-Python ones, 0 uses the Cython build.
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
    bash "$agent_dir/scripts/build_cython.sh" || echo "roxo-paper-ne-mcts: Cython build failed" >&2
  fi
) 9>"$agent_dir/.build.lock"

python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
