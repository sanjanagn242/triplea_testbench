#!/usr/bin/env bash
# Comparison plots of every agent in one tournament, written to <tournament>/plots/<map>/ and, with
# all maps combined, <tournament>/plots/all-maps/:
#   tournaments/scripts/generate_plots.sh planning-agent-testbed/tournaments/<date>_<time>
# Only some maps (repeatable; all-maps = every map combined):
#   tournaments/scripts/generate_plots.sh <tournament> --map minimap --map all-maps
set -euo pipefail

script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
tournament="${1:?usage: generate_plots.sh <tournament folder> [--map MAP ...] [--out DIR]}"
shift
python3 "$script_dir/plot_game_results.py" "$tournament" "$@"
python3 "$script_dir/plot_efficiency.py" "$tournament" "$@"
python3 "$script_dir/plot_rollout_stats.py" "$tournament" "$@"
# Only agents that write action_seq_stats.csv (tripleMind-with-opp-nodes so far); skip if none do.
python3 "$script_dir/plot_action_selection.py" "$tournament" "$@" || true
