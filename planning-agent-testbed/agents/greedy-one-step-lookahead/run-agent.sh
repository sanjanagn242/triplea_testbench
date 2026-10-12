#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Agent logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt:
# 1 enables it, 0 disables it. Agents do not log unless this is 1.
export TRIPLEA_AGENT_LOG=1
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
