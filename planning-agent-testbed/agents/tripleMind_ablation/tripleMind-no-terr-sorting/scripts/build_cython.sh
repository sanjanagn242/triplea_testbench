#!/usr/bin/env bash
# Compiles action_generators_cy.pyx into an extension module next to it.
# Cython is only needed at build time, so it lives in a build-only venv (.venv-build) inside the
# agent folder; the compiled module runs on the same python3 the agent uses.
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
agent_dir="$(dirname -- "$script_dir")"  # agents/tripleMind_ablation/tripleMind-no-terr-sorting
python="${TRIPLEA_TESTBENCH_PYTHON:-python3}"
venv="$agent_dir/.venv-build"

if [ ! -x "$venv/bin/cythonize" ]; then
  echo "tripleMind: creating build venv with Cython in $venv" >&2
  "$python" -m venv "$venv"
  "$venv/bin/pip" install --quiet cython setuptools >&2
fi

# Hosts without the python3-devel package (e.g. PACE) lack Python.h. Point the compiler at a
# user-supplied header tree instead: TRIPLEA_PYTHON_INCLUDE, else ~/.local/include/python3.X.
include="$("$python" -c 'import sysconfig; print(sysconfig.get_paths()["include"])')"
if [ ! -f "$include/Python.h" ]; then
  pyver="$("$python" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
  fallback="${TRIPLEA_PYTHON_INCLUDE:-$HOME/.local/include/python$pyver}"
  if [ -f "$fallback/Python.h" ]; then
    export CFLAGS="-I$fallback${CFLAGS:+ $CFLAGS}"
  else
    echo "tripleMind: Python.h not found in $include or $fallback; install python3-devel or set TRIPLEA_PYTHON_INCLUDE" >&2
    exit 1
  fi
fi

cd "$agent_dir"
"$venv/bin/cythonize" -i -3 --quiet action_generators_cy.pyx >&2
rm -rf "$agent_dir/build"
echo "tripleMind: built $(ls "$agent_dir"/action_generators_cy*.so)" >&2
