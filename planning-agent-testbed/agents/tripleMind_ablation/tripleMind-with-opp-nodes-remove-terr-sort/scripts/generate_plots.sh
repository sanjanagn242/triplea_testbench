script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
agent_dir="$(dirname -- "$script_dir")"  # agents/tripleMind_hybrids/tripleMind-with-opp-model

# One set of plots per map and opponent folder (<metrics>/<game>/vs-<opponent>/). The metrics folder
# is this agent's metrics/ unless given, e.g. a tournament's:
#   scripts/generate_plots.sh planning-agent-testbed/tournaments/<date>_<time>/agents/<agent id>/metrics
metrics_root="${1:-$agent_dir/metrics}"
for folder in "$metrics_root"/*/vs-*/; do
  [ -f "$folder/efficiency.csv" ] || continue
  python3 "$script_dir/plot_efficiency.py" --folder "${folder%/}"
  [ -f "$folder/rollout_stats.csv" ] && python3 "$script_dir/plot_rollout_stats.py" --folder "${folder%/}"
done
