#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
# Settings: change the values here; a value already set in the environment takes precedence.
# Logging to stderr and planning-agent-testbed/logs/agent-<faction>-game-<n>.txt: 1 on, 0 off.
export TRIPLEA_AGENT_LOG="${TRIPLEA_AGENT_LOG:-1}"
source_file="$agent_dir/agent.cpp"
build_dir="$agent_dir/build"
binary="$build_dir/simple-infantry-agent"
compiler="${TRIPLEA_TESTBENCH_CXX:-g++}"

mkdir -p "$build_dir"
if [[ ! -x "$binary" || "$source_file" -nt "$binary" ]]; then
  "$compiler" -std=c++20 -O2 -Wall -Wextra -pedantic "$source_file" -o "$binary"
fi
exec "$binary"
