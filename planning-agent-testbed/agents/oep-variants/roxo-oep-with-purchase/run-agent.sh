#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-0}"
# Search, as in Roxo's OEP (thesis 4.4): seconds per move phase, population size, survival rate,
# mutation probability, playout length in rounds and the seed of the evolution and the Weak AI.
export ROXO_SECONDS="${ROXO_SECONDS:-1.0}"
export ROXO_OEP_POPULATION="${ROXO_OEP_POPULATION:-50}"
export ROXO_OEP_SURVIVAL="${ROXO_OEP_SURVIVAL:-0.5}"
export ROXO_OEP_MUTATION="${ROXO_OEP_MUTATION:-0.1}"
export ROXO_OEP_PLAYOUT_ROUNDS="${ROXO_OEP_PLAYOUT_ROUNDS:-2}"
export ROXO_SEED="${ROXO_SEED:-0}"
# Genealogy of each search's genomes in planning-agent-testbed/logs/trees/ (the OEP counterpart of
# classical-mcts's tree dump): SAVE_TREE 1 saves a .dot after every move phase, TREE_PNG 1 also
# renders a .png (needs Graphviz `dot`), TREE_MAX_NODES caps the genomes drawn, TREE_RANKDIR lays
# generations out left to right (LR) or top to bottom (TB).
export ROXO_OEP_SAVE_TREE="${ROXO_OEP_SAVE_TREE:-0}"
export ROXO_OEP_TREE_PNG="${ROXO_OEP_TREE_PNG:-0}"
export ROXO_OEP_TREE_MAX_NODES="${ROXO_OEP_TREE_MAX_NODES:-300}"
export ROXO_OEP_TREE_RANKDIR="${ROXO_OEP_TREE_RANKDIR:-LR}"
# The pure-Python TripleMind action generators (purchase policy; this folder has no Cython build).
export TRIPLEMIND_PURE_PYTHON=1
# The testbench forward-model library (pip install -e planning-agent-testbed/forward-model/python
# also works; this keeps the repository copy importable without installation).
export PYTHONPATH="$agent_dir/../../../forward-model/python${PYTHONPATH:+:$PYTHONPATH}"
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
exec "$python" -u "$agent_dir/agent.py"
