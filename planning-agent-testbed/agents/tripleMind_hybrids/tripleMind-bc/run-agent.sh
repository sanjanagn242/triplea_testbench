#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tripleMind_dir="$agent_dir/../../tripleMind"
bc_dir="$agent_dir/../../behaviour-cloning"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG=1
# Seed of the agent's randomness (and of the dice it uses to check its own moves).
export TRIPLEA_AGENT_SEED="${TRIPLEA_AGENT_SEED:-0}"
# Behaviour-cloning policy checkpoint (from behaviour-cloning/run_bc.py train) and torch threads.
export TRIPLEMIND_BC_CHECKPOINT="${TRIPLEMIND_BC_CHECKPOINT:-$bc_dir/runs/mlp-baseline/best.pt}"
export TRIPLEMIND_BC_THREADS="${TRIPLEMIND_BC_THREADS:-1}"
# Live map window: 1 shows it, 0 hides it. PURE_PYTHON 1 skips TripleMind's Cython generators.
export TRIPLEMIND_VIEW="${TRIPLEMIND_VIEW:-0}"
export TRIPLEMIND_PURE_PYTHON="${TRIPLEMIND_PURE_PYTHON:-0}"
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"

# (Re)build TripleMind's Cython action generators when the .pyx is newer than the compiled
# module, with the same lock as TripleMind; a failed build falls back to the Python generators.
(
  flock 9
  so="$(ls "$tripleMind_dir"/action_generators_cy*.so 2>/dev/null | head -n 1 || true)"
  if [ -z "$so" ] || [ "$tripleMind_dir/action_generators_cy.pyx" -nt "$so" ]; then
    bash "$tripleMind_dir/scripts/build_cython.sh" || echo "tripleMind-bc: Cython build failed" >&2
  fi
) 9>"$tripleMind_dir/.build.lock"

# The policy needs PyTorch: TRIPLEMIND_BC_PYTHON, else the behaviour-cloning venv, else
# TRIPLEA_TESTBENCH_PYTHON / python3 (which must then have torch, numpy>=2, networkx).
if [ -n "${TRIPLEMIND_BC_PYTHON:-}" ]; then
  python="$TRIPLEMIND_BC_PYTHON"
elif [ -x "$bc_dir/.venv/bin/python" ]; then
  python="$bc_dir/.venv/bin/python"
else
  python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
fi
exec "$python" -u "$agent_dir/agent.py"
