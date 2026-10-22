"""Compares how clearly each agent's search separated the action it chose from the alternatives, by
tree level, in one tournament (every agents/*/metrics/<map>/vs-*/action_seq_stats.csv, one row per
tree level of the chosen action sequence of a search: the chosen, most-visited child against its
siblings).

Deeper levels get fewer visits, so their choice can be close to a coin flip. These plots show at
which tree level (tl) the chosen child stops standing out. Levels with a single child (nothing to
choose between) are left out of the plots and counted in the summary.

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  action_selection_by_tl.png       by tree level, one color per agent (solid: the agent's own choice,
                                   dashed: the opponent's, as modelled in its tree); line = median,
                                   band = 25th-75th percentile over the searches:
                                     chosen visits / mean sibling visits (1 = no preference)
                                     margin over the runner-up, (visits - second) / visits (0 = tie)
                                     % of levels where the chosen child tied the runner-up
                                     visit z-score of the chosen child among its siblings
                                     avg value - runner-up's avg value
                                     visits of the chosen child (log scale)
  action_selection_by_tl.csv       the medians and tie / single-child shares per agent, chooser, tl
  action_selection_summary.txt     the same as tables, per agent

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_action_selection.py tournaments/<date>_<time> [--map MAP ...] [--out DIR]
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator

from common import TEXT, agent_colors, by_map, load, parse_args, save, style, tournament_agents

COLUMNS = ["player", "game", "round", "tl", "chooser", "num_children", "visit_count",
           "second_visit_count", "mean_visits", "visit_z", "avg_value", "second_avg_value"]
SIDES = {"self": "-", "opponent": "--"}

args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
df = load(args.tournament, "action_seq_stats.csv", COLUMNS)
if df.empty:
    raise SystemExit(f"no action_seq_stats.csv under {args.tournament / 'agents'}")
for c in COLUMNS[3:]:
    if c != "chooser":
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["side"] = np.where(df["chooser"] == df["player"], "self", "opponent")
df["single"] = df["num_children"] <= 1
df["ratio"] = df["visit_count"] / df["mean_visits"]
df["margin"] = (df["visit_count"] - df["second_visit_count"]) / df["visit_count"]
df["tied"] = 100.0 * (df["visit_count"] == df["second_visit_count"])
df["value_gap"] = df["avg_value"] - df["second_avg_value"]
colors = agent_colors(args.tournament)

PANELS = [  # column, y label, title, reference line (or None), log scale
    ("ratio", "Chosen visits / mean sibling visits", "Visits relative to siblings", 1.0, False),
    ("margin", "(visits - runner-up) / visits", "Margin over the runner-up", 0.0, False),
    ("tied", "% of levels", "Chosen child tied with the runner-up", None, False),
    ("visit_z", "z-score", "Visit z-score among siblings", 0.0, False),
    ("value_gap", "avg value - runner-up's", "Value gap to the runner-up", 0.0, False),
    ("visit_count", "Visits (log)", "Visits of the chosen child", None, True),
]


def by_tl(g, value):
    """Median and 25th-75th percentile of value per tree level; the mean for the tie share."""
    grouped = g.groupby("tl")[value]
    if value == "tied":
        mean = grouped.mean()
        return mean, mean, mean
    return grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)


def plot(m, path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    for ax, (value, ylabel, title, ref, log) in zip(axes.flat, PANELS):
        for (agent, side), g in m.groupby(["agent", "side"]):
            mid, lo, hi = by_tl(g, value)
            if mid.empty:
                continue
            ax.fill_between(mid.index, lo, hi, color=colors[agent], alpha=0.10, linewidth=0)
            ax.plot(mid.index, mid.values, color=colors[agent], linestyle=SIDES[side], linewidth=2,
                    marker="o", markersize=5)
        if ref is not None:
            ax.axhline(ref, color=TEXT, linewidth=1, linestyle=":")
        if log:
            ax.set_yscale("log")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Tree level (tl)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        style(ax)
    agents = sorted(m["agent"].unique())
    handles = [Line2D([], [], color=colors[a], linewidth=2, label=a) for a in agents]
    handles += [Line2D([], [], color=TEXT, linestyle=ls, linewidth=2,
                       label="agent's own choice" if side == "self" else "opponent's choice (modelled)")
                for side, ls in SIDES.items() if side in set(m["side"])]
    # visits fall with depth, so the last panel's upper right is free for the legend
    axes.flat[-1].legend(handles=handles, loc="upper right", frameon=False, fontsize=10,
                         labelcolor=TEXT, handlelength=4)
    fig.suptitle("How clearly the chosen action stood out from its siblings, by tree level "
                 "(line: median, band: 25th-75th percentile; dotted: no differentiation)",
                 fontsize=13)
    save(fig, path)


for map_name, m_all in by_map(df, args.maps):
    out = out_root / map_name
    m = m_all[~m_all["single"]]
    if m.empty:
        print(f"{map_name}: every level had a single child, nothing to compare")
        continue
    plot(m, out / "action_selection_by_tl.png")

    keys = ["agent", "side", "tl"]
    table = m.groupby(keys).agg(
        levels=("ratio", "size"), ratio=("ratio", "median"), margin=("margin", "median"),
        tied_pct=("tied", "mean"), visit_z=("visit_z", "median"),
        value_gap=("value_gap", "median"), visits=("visit_count", "median"),
        children=("num_children", "median"))
    single = m_all.groupby(keys)["single"].mean().mul(100).rename("single_child_pct")
    table = table.join(single, how="outer").reset_index()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "action_selection_by_tl.csv", index=False)

    lines = [f"Action selection by tree level: {map_name}\n", "=" * 60 + "\n\n",
             "One row per tree level of a search's chosen action sequence. Medians over the searches;\n"
             "ratio = chosen visits / mean sibling visits (1 = no preference), margin = (visits -\n"
             "runner-up) / visits (0 = tie), tied_pct = % of levels tied with the runner-up,\n"
             "single_child_pct = % of levels with one child (left out of the other columns).\n\n"]
    without = [a for a in tournament_agents(args.tournament) if a not in set(m_all["agent"])]
    if without:
        lines.append(f"No action_seq_stats.csv: {', '.join(without)}\n\n")
    for (agent, side), t in table.groupby(["agent", "side"]):
        name = f"{agent} ({'own choices' if side == 'self' else 'opponent choices'})"
        lines.append(f"{name}\n" + "-" * len(name) + "\n")
        lines.append(t.drop(columns=["agent", "side"]).round(3).to_string(index=False, na_rep="-")
                     + "\n\n")
    path = out / "action_selection_summary.txt"
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
