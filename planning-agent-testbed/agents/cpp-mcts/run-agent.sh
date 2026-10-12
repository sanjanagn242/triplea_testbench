#!/usr/bin/env bash
# Builds (when sources changed) and starts the C++ MCTS agent. Game dynamics come from the
# testbench forward-model library; triplea-fm-config supplies its C++ binding's flags.
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-1}"
# Search time per decision (s), rollout horizon in rounds (a playout still running then has no
# winner), and the search seed.
export TRIPLEA_MCTS_SECONDS="${TRIPLEA_MCTS_SECONDS:-2.0}"
export TRIPLEA_MCTS_ROLLOUT="${TRIPLEA_MCTS_ROLLOUT:-2}"
export TRIPLEA_MCTS_SEED="${TRIPLEA_MCTS_SEED:-0}"
fm_config="$agent_dir/../../forward-model/cpp/triplea-fm-config"
build_dir="$agent_dir/build"
binary="$build_dir/cpp-mcts-agent"
compiler="${TRIPLEA_TESTBENCH_CXX:-g++}"

mkdir -p "$build_dir"
newest_source="$(ls -t "$agent_dir"/src/*.cpp "$agent_dir"/src/*.hpp \
  "$agent_dir"/../../forward-model/cpp/include/triplea_fm/*.hpp | head -n 1)"
if [[ ! -x "$binary" || "$newest_source" -nt "$binary" ]]; then
  # shellcheck disable=SC2046
  "$compiler" -std=c++20 -O2 -Wall -Wextra $("$fm_config" --cflags) \
    "$agent_dir/src/main.cpp" -o "$binary" $("$fm_config" --libs) >&2
fi
exec "$binary"
