#!/usr/bin/env bash
set -euo pipefail

agent_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
source_file="$agent_dir/agent.cpp"
build_dir="$agent_dir/build"
binary="$build_dir/simple-infantry-agent"
compiler="${TRIPLEA_TESTBENCH_CXX:-g++}"

mkdir -p "$build_dir"
if [[ ! -x "$binary" || "$source_file" -nt "$binary" ]]; then
  "$compiler" -std=c++20 -O2 -Wall -Wextra -pedantic "$source_file" -o "$binary"
fi
exec "$binary"
