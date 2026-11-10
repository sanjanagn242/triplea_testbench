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
                                     std of the rewards through the chosen child
                                     value gap in standard errors of the two means (2: clear)
                                     % of levels where that gap is over 2 standard errors
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

from common import TEXT, agent_colors, agent_legend, by_map, load, parse_args, save, style, tournament_agents

COLUMNS = ["player", "game", "round", "tl", "chooser", "num_children", "visit_count",
           "second_visit_count", "mean_visits", "visit_z", "avg_value", "second_avg_value"]
# std of the rewards through the chosen / runner-up child; blank in files written before them
OPTIONAL = ["reward_std", "second_reward_std"]
SIDES = {"self": "-", "opponent": "--"}

args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
df = load(args.tournament, "action_seq_stats.csv", COLUMNS, OPTIONAL)
if df.empty:
    raise SystemExit(f"no action_seq_stats.csv under {args.tournament / 'agents'}")
for c in COLUMNS[3:] + OPTIONAL:
    if c != "chooser":
        df[c] = pd.to_numeric(df[c], errors="coerce")
df["side"] = np.where(df["chooser"] == df["player"], "self", "opponent")
df["single"] = df["num_children"] <= 1
df["ratio"] = df["visit_count"] / df["mean_visits"]
df["margin"] = (df["visit_count"] - df["second_visit_count"]) / df["visit_count"]
df["tied"] = 100.0 * (df["visit_count"] == df["second_visit_count"])
df["value_gap"] = df["avg_value"] - df["second_avg_value"]
# The value gap in standard errors of the difference of the two means (each reward_std /
# sqrt(visits); approximate, a child's rewards come from different leaves of its subtree). Blank
# when either child has under two visits or the file has no reward_std.
gap_se = np.sqrt(df["reward_std"] ** 2 / df["visit_count"]
                 + df["second_reward_std"] ** 2 / df["second_visit_count"])
df["gap_se"] = (df["value_gap"] / gap_se).where(gap_se > 0)
# % of levels whose gap is over two standard errors (clearly better), of those with gap_se
df["clear"] = (100.0 * (df["gap_se"] > 2)).where(df["gap_se"].notna())
colors = agent_colors(args.tournament)

PANELS = [  # column, y label, title, reference line (or None), log scale
    ("ratio", "Chosen visits / mean sibling visits", "Visits relative to siblings", 1.0, False),
    ("margin", "(visits - runner-up) / visits", "Margin over the runner-up", 0.0, False),
    ("tied", "% of levels", "Chosen child tied with the runner-up", None, False),
    ("visit_z", "z-score", "Visit z-score among siblings", 0.0, False),
    ("value_gap", "avg value - runner-up's", "Value gap to the runner-up", 0.0, False),
    ("visit_count", "Visits (log)", "Visits of the chosen child", None, True),
    ("reward_std", "std of the rewards", "Reward std through the chosen child", None, False),
    ("gap_se", "value gap / its standard error", "Value gap in standard errors (2: clear)", 2.0,
     False),
    ("clear", "% of levels", "Value gap over 2 standard errors", None, False),
]


def by_tl(g, value):
    """Median and 25th-75th percentile of value per tree level; the mean for the shares."""
    grouped = g.groupby("tl")[value]
    if value in ("tied", "clear"):
        mean = grouped.mean()
        return mean, mean, mean
    return grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)


def plot(m, path):
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    for ax, (value, ylabel, title, ref, log) in zip(axes.flat, PANELS):
        for (agent, side), g in m.groupby(["agent", "side"]):
            mid, lo, hi = by_tl(g, value)
            mid, lo, hi = mid.dropna(), lo.dropna(), hi.dropna()
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
    sides = [Line2D([], [], color=TEXT, linestyle=ls, linewidth=2,
                    label="agent's own choice" if side == "self" else "opponent's choice (modelled)")
             for side, ls in SIDES.items() if side in set(m["side"])]
    agent_legend(fig, agents, colors, sides)
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
        children=("num_children", "median"), reward_std=("reward_std", "median"),
        gap_se=("gap_se", "median"), clear_pct=("clear", "mean"))
    single = m_all.groupby(keys)["single"].mean().mul(100).rename("single_child_pct")
    table = table.join(single, how="outer").reset_index()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "action_selection_by_tl.csv", index=False)

    lines = [f"Action selection by tree level: {map_name}\n", "=" * 60 + "\n\n",
             "One row per tree level of a search's chosen action sequence. Medians over the searches;\n"
             "ratio = chosen visits / mean sibling visits (1 = no preference), margin = (visits -\n"
             "runner-up) / visits (0 = tie), tied_pct = % of levels tied with the runner-up,\n"
             "reward_std = std of the rewards through the chosen child, gap_se = value_gap in\n"
             "standard errors of the two means (reward_std / sqrt(visits) each; approximate),\n"
             "clear_pct = % of levels with gap_se over 2 (the chosen child clearly better; - when\n"
             "the agent logs no reward_std), single_child_pct = % of levels with one child (left out\n"
             "of the other columns).\n\n"]
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
