script_dir="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
agent_dir="$(dirname -- "$script_dir")"  # agents/tripleMind

python3 "$script_dir/plot_efficiency.py" --folder "$agent_dir/metrics"
