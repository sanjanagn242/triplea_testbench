"""Plots and summarises classical-mcts metrics/efficiency.csv (one row per search).

CSV columns: game, round, player, phase, num_iterations, avg_rollout_length, max_rollout_round,
max_rollout_rounds_ahead, max_depth, root_value, root_actions, time_taken.

Rows from older agent versions are read too, even when newer rows were appended under an older
header: 10 columns (no max_rollout_round, no max_rollout_rounds_ahead) and 11 columns (no
max_rollout_rounds_ahead). Their missing values are empty, and a metric with no values at all is
left out of the plots and the summary.

Usage (from the agent folder):
  python3 scripts/plot_efficiency.py                    # reads ./metrics/efficiency.csv
  python3 scripts/plot_efficiency.py --folder metrics --player Russians

Writes into the same folder:
  aggregate_game_summary.png  per-round mean and 25-75 percentile band, one line per phase, for
                              iterations, rollout length, furthest rollout round, rounds a
                              rollout advanced, tree depth and root actions
  agg_root_node_value.png     root node value per round, one line per phase
  efficiency_summary.txt      per-game and overall statistics by phase
"""

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

COLUMNS = ["game", "round", "player", "phase", "num_iterations", "avg_rollout_length",
           "max_rollout_round", "max_rollout_rounds_ahead", "max_depth", "root_value", "root_actions", "time_taken"]
PHASES = ["purchase", "combatMove", "nonCombatMove", "place"]  # turn order
# Fixed categorical order (one color and one marker per phase, never reassigned).
PHASE_STYLE = {
    "purchase": ("#2a78d6", "o"),
    "combatMove": ("#eb6834", "s"),
    "nonCombatMove": ("#1baf7a", "^"),
    "place": ("#eda100", "D"),
}
METRICS = {  # column -> (title, y label)
    "num_iterations": ("Iterations per search", "Iterations"),
    "avg_rollout_length": ("Average rollout length", "Actions per playout"),
    "max_rollout_round": ("Furthest round reached by a rollout", "Game round"),
    "max_rollout_rounds_ahead": ("Most rounds a rollout advanced", "Rounds past its start"),
    "max_depth": ("Maximum tree depth", "Depth"),
    "root_actions": ("Legal actions at the root", "Actions"),
    "root_value": ("Root node value", "Mean value"),
}
# Row layouts by field count: the current one and those written by earlier agent versions.
LAYOUTS = {
    len(COLUMNS): COLUMNS,
    len(COLUMNS) - 1: [c for c in COLUMNS if c != "max_rollout_rounds_ahead"],
    len(COLUMNS) - 2: [c for c in COLUMNS if c not in ("max_rollout_round", "max_rollout_rounds_ahead")],
}
NUMERIC = COLUMNS[4:]
TEXT, MUTED, GRID = "#1f1f1e", "#6b6a63", "#e4e3dc"

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, default=str(Path(__file__).resolve().parent.parent / "metrics"),
                    help="folder containing efficiency.csv (default: the agent's metrics folder)")
parser.add_argument("--player", type=str, default=None, help="only this faction's rows")
args = parser.parse_args()
FOLDER = Path(args.folder)



def load_rows(path):
    """Rows of efficiency.csv as a DataFrame with every column in COLUMNS. Each row is read by its
    own length (see LAYOUTS), whatever the header says, so files that mix formats load; missing
    values are NaN."""
    records, skipped = [], 0
    with open(path, newline="") as f:
        reader = csv.reader(f)
        next(reader, None)  # header
        for row in reader:
            if len(row) in LAYOUTS:
                records.append(dict(zip(LAYOUTS[len(row)], row)))
            elif row:
                skipped += 1
    if skipped:
        print(f"skipped {skipped} rows with an unexpected number of fields")
    frame = pd.DataFrame.from_records(records, columns=COLUMNS)
    frame[NUMERIC] = frame[NUMERIC].apply(pd.to_numeric, errors="coerce")
    frame[["game", "round"]] = frame[["game", "round"]].astype(int)
    return frame


df = load_rows(FOLDER / "efficiency.csv")
if args.player:
    df = df[df["player"] == args.player]
if df.empty:
    raise SystemExit("no rows to plot")
present = [c for c in NUMERIC if df[c].notna().any()]  # metrics with at least one value
phases = [p for p in PHASES if p in set(df["phase"])] + sorted(set(df["phase"]) - set(PHASES))


def style_axes(ax, title, ylabel):
    ax.set_title(title, fontsize=12, color=TEXT, loc="left")
    ax.set_ylabel(ylabel, fontsize=10, color=MUTED)
    ax.set_xlabel("Round", fontsize=10, color=MUTED)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    ax.xaxis.get_major_locator().set_params(integer=True)


def plot_metric(ax, column):
    """Mean across games per round (line) with the 25-75 percentile band, one series per phase."""
    for phase in phases:
        color, marker = PHASE_STYLE.get(phase, (MUTED, "x"))
        by_round = df[df["phase"] == phase].groupby("round")[column]
        mean, p25, p75 = by_round.mean(), by_round.quantile(0.25), by_round.quantile(0.75)
        mean, p25, p75 = mean.dropna(), p25.dropna(), p75.dropna()  # rounds without values
        if mean.empty:
            continue
        ax.fill_between(mean.index, p25.values, p75.values, color=color, alpha=0.15, linewidth=0)
        ax.plot(mean.index, mean.values, color=color, linewidth=2, marker=marker, markersize=6,
                markeredgecolor="white", markeredgewidth=1.5, label=phase)
    style_axes(ax, *METRICS[column])


def add_legend(fig, ax):
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(labels), frameon=False,
               fontsize=10, labelcolor=TEXT, bbox_to_anchor=(0.5, 1.0))


# -----------------------------
# Aggregate summary figure (small multiples, one metric per panel)
# -----------------------------
panels = [c for c in ["num_iterations", "avg_rollout_length", "max_rollout_round",
                      "max_rollout_rounds_ahead", "max_depth", "root_actions"] if c in present]
n_rows = (len(panels) + 1) // 2
fig, axes = plt.subplots(n_rows, 2, figsize=(13, 4.2 * n_rows), squeeze=False)
for ax, column in zip(axes.flat, panels):
    plot_metric(ax, column)
for ax in list(axes.flat)[len(panels):]:
    ax.set_visible(False)
add_legend(fig, axes.flat[0])
n_games = df["game"].nunique()
fig.suptitle(f"Search efficiency by round and phase (mean across {n_games} "
             f"{'game' if n_games == 1 else 'games'}, "
             "band = 25-75 percentile)", fontsize=13, color=TEXT, y=1 - 0.4 / (4.2 * n_rows))
fig.tight_layout(rect=(0, 0, 1, 1 - 0.65 / (4.2 * n_rows)))
fig.savefig(FOLDER / "aggregate_game_summary.png", dpi=150)
plt.close(fig)

# -----------------------------
# Root node value
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))
plot_metric(ax, "root_value")
ax.set_title("Root node value per round (across games)", fontsize=13, color=TEXT, loc="left")
ax.set_ylim(-0.02, 1.02)  # rewards are 0 (loss/unfinished) or 1 (win)
add_legend(fig, ax)
fig.tight_layout(rect=(0, 0, 1, 0.94))
fig.savefig(FOLDER / "agg_root_node_value.png", dpi=150)
plt.close(fig)

# -----------------------------
# Game-wise summary
# -----------------------------
STAT_COLUMNS = [c for c in ["num_iterations", "avg_rollout_length", "max_rollout_round",
                            "max_rollout_rounds_ahead", "max_depth", "root_value", "root_actions", "time_taken"] if c in present]


def phase_table(frame):
    """One line per phase: mean / median / min / max of each metric."""
    lines = []
    for phase in phases:
        rows = frame[frame["phase"] == phase]
        if rows.empty:
            continue
        lines.append(f"  {phase} ({len(rows)} searches)\n")
        for column in STAT_COLUMNS:
            s = rows[column].dropna()
            if s.empty:
                continue
            lines.append(f"    {column:<24} mean {s.mean():10.3f}  median {s.median():10.3f}  "
                         f"min {s.min():10.3f}  max {s.max():10.3f}\n")
    return lines


summary_lines = ["MCTS Efficiency Summary (Game-wise)\n", "=" * 40 + "\n\n"]
if args.player:
    summary_lines.append(f"Player: {args.player}\n\n")
for game_id, gdf in df.groupby("game"):
    summary_lines.append(f"Game {game_id}\n")
    summary_lines.append("-" * 20 + "\n")
    summary_lines.append(f"Rounds searched: {gdf['round'].nunique()} (last round {gdf['round'].max()})\n")
    summary_lines.append(f"Searches: {len(gdf)}, total iterations: {gdf['num_iterations'].sum()}\n")
    summary_lines.extend(phase_table(gdf))
    summary_lines.append("\n")

summary_lines.append("OVERALL (Across Games)\n")
summary_lines.append("=" * 30 + "\n")
summary_lines.append(f"Games: {df['game'].nunique()}, searches: {len(df)}\n")
summary_lines.extend(phase_table(df))

summary_path = FOLDER / "efficiency_summary.txt"
summary_path.write_text("".join(summary_lines))
print(f"Summary written to {summary_path}")
print(f"Plots written to {FOLDER / 'aggregate_game_summary.png'} and {FOLDER / 'agg_root_node_value.png'}")
