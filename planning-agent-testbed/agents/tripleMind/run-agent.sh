#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG=0
# Game records in records/<faction>-game-<n>.jsonl: 1 on, 0 off.
export TRIPLEMIND_RECORD=0
# Search: time per combat-move decision (s), iteration budget (unused; the search is
# time-bounded), rollout horizon in rounds (a playout still running then has no winner),
# UCB1 exploration constant, and seed.
export TRIPLEA_MCTS_SECONDS="${TRIPLEA_MCTS_SECONDS:-1.0}"
export TRIPLEA_MCTS_ITERS="${TRIPLEA_MCTS_ITERS:-100000}"
export TRIPLEA_MCTS_ROLLOUT="${TRIPLEA_MCTS_ROLLOUT:-2}"
export TRIPLEA_MCTS_EXP_CONST="${TRIPLEA_MCTS_EXP_CONST:-1.414}"
export TRIPLEA_MCTS_SEED="${TRIPLEA_MCTS_SEED:-0}"
# Live map window: 1 shows it, 0 hides it. PURE_PYTHON 1 skips the Cython action generators.
export TRIPLEMIND_VIEW="${TRIPLEMIND_VIEW:-0}"
export TRIPLEMIND_PURE_PYTHON="${TRIPLEMIND_PURE_PYTHON:-0}"
# Map image before each combat-move search, in map_images/<game>/vs-<opponent>/: 1 on, 0 off.
export TRIPLEMIND_SAVE_MAP="${TRIPLEMIND_SAVE_MAP:-1}"
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).

export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"

# (Re)build the Cython action generators when the .pyx is newer than the compiled module. The
# lock serialises agents launched together; a failed build falls back to the Python generators.
(
  flock 9
  so="$(ls "$agent_dir"/action_generators_cy*.so 2>/dev/null | head -n 1 || true)"
  if [ -z "$so" ] || [ "$agent_dir/action_generators_cy.pyx" -nt "$so" ]; then
    bash "$agent_dir/scripts/build_cython.sh" || echo "tripleMind: Cython build failed" >&2
  fi
) 9>"$agent_dir/.build.lock"

python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
