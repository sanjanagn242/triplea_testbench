#!/usr/bin/env bash
# Runs one tournament per spec, in order; each is plotted and zipped in its own folder
# planning-agent-testbed/tournaments/<date>_<time>_<spec name>:
#   planning-agent-testbed/tools/run_tournament.sh [spec ...] [more tournament.py options]
# The spec defaults to tournaments/specs/2p_maps_1s.json; the options are passed to every tournament.
# A tournament that fails is reported at the end and the next spec still runs. The folders are
# git-ignored; commit the <folder>.zip next to each.
set -uo pipefail

repo="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo"

specs=()
while [[ $# -gt 0 && $1 != -* ]]; do
  specs+=("$1")
  shift
done
[[ ${#specs[@]} -gt 0 ]] || specs=("planning-agent-testbed/tournaments/specs/2p_maps_1s.json")
tournaments="planning-agent-testbed/tournaments"

failed=()
for spec in "${specs[@]}"; do
  name="$(date +%Y-%m-%d_%H-%M-%S)_$(basename "$spec" .json)"
  echo "=== $spec -> $tournaments/$name"
  if python3 planning-agent-testbed/tools/tournament.py --spec "$spec" --out "$tournaments/$name" --no-gui "$@" &&
     planning-agent-testbed/tournaments/scripts/generate_plots.sh "$tournaments/$name" &&
     (cd "$tournaments" && zip -qr "$name.zip" "$name"); then
    echo "Zipped to $tournaments/$name.zip"
  else
    echo "FAILED: $spec (not zipped)"
    failed+=("$spec")
  fi
done

if [[ ${#failed[@]} -gt 0 ]]; then
  echo "Failed specs: ${failed[*]}"
  exit 1
fi
