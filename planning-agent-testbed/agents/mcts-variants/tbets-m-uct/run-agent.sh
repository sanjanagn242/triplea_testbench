#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG=0
# Game records in records/<faction>-game-<n>.jsonl: 1 on, 0 off.
export TRIPLEMIND_RECORD=0
# Search (one tree of atomic actions for purchase, combat move and non-combat move): time per decision (s),
# iteration budget (unused; the search is time-bounded), rollout horizon in rounds (1 ends the
# rollout when the round the leaf is in is over; a playout still running then has no winner),
# UCB1 exploration constant, and seed.
export TRIPLEA_MCTS_SECONDS="${TRIPLEA_MCTS_SECONDS:-1.0}"
export TRIPLEA_MCTS_ITERS="${TRIPLEA_MCTS_ITERS:-100000}"
export TRIPLEA_MCTS_ROLLOUT="${TRIPLEA_MCTS_ROLLOUT:-5}"
export TRIPLEA_MCTS_EXP_CONST="${TRIPLEA_MCTS_EXP_CONST:-1.414}"
# Epsilon-greedy combat-move and purchase rollouts: probability of a random move / unit, else
# the strongest attack / the most attack + defense per PU.
export TRIPLEA_MCTS_EPSILON="${TRIPLEA_MCTS_EPSILON:-0.5}"
# Combat-move rollout policy: epsilon_greedy (above) or evaluate (the attack whose resulting
# state evaluate_state scores highest for the mover).
export TRIPLEA_MCTS_COMBAT_ROLLOUT="${TRIPLEA_MCTS_COMBAT_ROLLOUT:-epsilon_greedy}"
# TBETS expansion: % of leaf expansions that add a child (the greedy action of the leaf's
# decision) instead of a sibling (a mutation: another action of its parent's decision), and
# whether every node selection passes through first gets one mutated child (1) or not (0).
export TBETS_GREEDY_PCT="${TBETS_GREEDY_PCT:-5}"
export TBETS_WIDEN="${TBETS_WIDEN:-1}"
export TRIPLEA_MCTS_SEED="${TRIPLEA_MCTS_SEED:-0}"
# Live map window: 1 shows it, 0 hides it. PURE_PYTHON 1 skips the Cython action generators.
export TRIPLEMIND_VIEW="${TRIPLEMIND_VIEW:-0}"
export TRIPLEMIND_PURE_PYTHON="${TRIPLEMIND_PURE_PYTHON:-0}"
# Search trees in trees/<game>/vs-<opponent>/, named by the root's phase (tree_* combat,
# tree_noncombat_* non-combat, tree_purchase_* purchase;
# .dot, .html and, with graphviz installed, .png): 1 on, 0 off.
export TRIPLEMIND_SAVE_TREE="${TRIPLEMIND_SAVE_TREE:-0}"
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).

export PYTHONPATH="$agent_dir/../../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"

# (Re)build the Cython action generators when the .pyx is newer than the compiled module. The
# lock serialises agents launched together; a failed build falls back to the Python generators.
(
  flock 9
  so="$(ls "$agent_dir"/action_generators_cy*.so 2>/dev/null | head -n 1 || true)"
  if [ -z "$so" ] || [ "$agent_dir/action_generators_cy.pyx" -nt "$so" ]; then
    bash "$agent_dir/scripts/build_cython.sh" || echo "tripleMind: Cython build failed" >&2
  fi
) 9>"$agent_dir/.build.lock"

# Rollout noise in rollout_stats.csv: every n-th rollout of a search is played a second time from
# the same state, outside the search (not backpropagated, not counted against the time budget).
# 0 plays none.
export TRIPLEA_ROLLOUT_NOISE_EVERY="${TRIPLEA_ROLLOUT_NOISE_EVERY:-10}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
