#!/usr/bin/env bash
# Builds and runs the C++ binding tests.
set -euo pipefail
here="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
config="$here/../triplea-fm-config"
mkdir -p "$here/build"
# shellcheck disable=SC2046
"${TRIPLEA_TESTBENCH_CXX:-g++}" -std=c++20 -O2 -Wall -Wextra $("$config" --cflags) \
  "$here/binding_test.cpp" -o "$here/build/binding_test" $("$config" --libs)
"$here/build/binding_test" "$here/../../../games/capture-the-flag/map/capture_the_flag/map/games/capture_the_flag.xml"
