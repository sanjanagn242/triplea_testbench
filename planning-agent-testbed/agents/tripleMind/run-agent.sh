#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Agent logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt:
# 1 enables it, 0 disables it. Agents do not log unless this is 1.
export TRIPLEA_AGENT_LOG=0
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).

export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
export TRIPLEMIND_VIEW=0
export TRIPLEMIND_PURE_PYTHON=0

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
