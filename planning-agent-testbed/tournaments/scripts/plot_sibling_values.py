"""Compares the sibling actions each agent scored at every level of its chosen action sequence, in one
tournament (every agents/*/metrics/<map>/vs-*/sibling_values.csv, one row per candidate per level:
its samples, mean value, std of the samples and its gap to the best one; written by
oep-greedy-action-combat so far).

action_selection_by_tl.png summarises the chosen action against the runner-up; these plots show the
whole sibling set: how far the alternatives (and ending the phase) fell behind the chosen action,
and whether the gap was larger than the dice noise of the estimates.

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  sibling_values_by_tl.png       by sequence level (tl), one color per agent; line = median,
                                 band = 25th-75th percentile over the levels:
                                   value gap to the runner-up (best - second best)
                                   gap of every non-chosen sibling to the best
                                   best - END (what attacking gained over stopping; 0 when END won)
                                   % of levels tied with the runner-up (solid) / where END won (dashed)
                                   % of levels with a clear winner: gap > 2 standard errors of the
                                     difference of the two means (deterministic: gap > 0)
                                   number of siblings (candidates incl. END)
  sibling_values_by_rank.png     median gap to the best by sibling rank (1 = chosen, 2 = runner-up, ...)
                                 and the value gap to the runner-up by game round
  sibling_values_by_tl.csv       the medians and shares per agent and tl
  sibling_values_summary.txt     the same as tables, per agent

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_sibling_values.py tournaments/<date>_<time> [--map MAP ...] [--out DIR]
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from common import (TEXT, agent_colors, agent_legend, by_map, load, mean_by_round, parse_args, save, stats, style,
                    tournament_agents)

COLUMNS = ["player", "game", "round", "tl", "action", "chosen", "samples", "mean_value",
           "std_value", "diff_to_best"]
LEVEL = ["agent", "map", "opponent", "player", "game", "round", "tl"]

args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
df = load(args.tournament, "sibling_values.csv", COLUMNS)
if df.empty:
    raise SystemExit(f"no sibling_values.csv under {args.tournament / 'agents'}")
for c in ["game", "round", "tl", "chosen", "samples", "mean_value", "std_value", "diff_to_best"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
colors = agent_colors(args.tournament)

# Rank within a level: the chosen action first (it is the best; END wins ties), then by value.
df = df.sort_values(LEVEL + ["chosen", "mean_value"], ascending=[True] * len(LEVEL) + [False, False])
df["rank"] = df.groupby(LEVEL).cumcount() + 1


def level_row(g):
    """One level's comparison: the chosen action against the runner-up and END."""
    best = g.iloc[0]
    others = g.iloc[1:]
    row = {"siblings": len(g), "end_won": best["action"] == "END"}
    end = g[g["action"] == "END"]
    row["end_gap"] = best["mean_value"] - end["mean_value"].iloc[0] if len(end) else np.nan
    if others.empty:
        row.update(gap=np.nan, tied=np.nan, clear=np.nan)
        return pd.Series(row)
    second = others.iloc[0]
    gap = best["mean_value"] - second["mean_value"]
    se = np.sqrt(best["std_value"] ** 2 / max(best["samples"], 1)
                 + second["std_value"] ** 2 / max(second["samples"], 1))
    row.update(gap=gap, tied=100.0 * (gap <= 1e-12),
               clear=100.0 * (gap > 2 * se if se > 0 else gap > 1e-12))
    return pd.Series(row)


levels = df.groupby(LEVEL, sort=False).apply(level_row).reset_index()
levels["end_won_pct"] = 100.0 * levels["end_won"].astype(float)
non_chosen = df[df["rank"] > 1]

PANELS = [  # frame, column, y label, title, reference line
    ("levels", "gap", "best - runner-up", "Value gap to the runner-up", 0.0),
    ("siblings", "diff_to_best", "best - sibling", "Gap of every non-chosen sibling to the best", 0.0),
    ("levels", "end_gap", "best - END", "Gain over ending the phase (END)", 0.0),
    ("levels", "tied", "% of levels", "Tied with the runner-up (solid) / END won (dashed)", None),
    ("levels", "clear", "% of levels", "Clear winner: gap > 2 SE of the difference", None),
    ("levels", "siblings", "candidates", "Siblings per level (incl. END)", None),
]


def by_tl(g, value):
    """Median and 25th-75th percentile of value per level; the mean for the % columns."""
    grouped = g.groupby("tl")[value]
    if value in ("tied", "clear", "end_won_pct"):
        mean = grouped.mean()
        return mean, mean, mean
    return grouped.median(), grouped.quantile(0.25), grouped.quantile(0.75)


def plot_by_tl(lv, sib, path):
    fig, axes = plt.subplots(2, 3, figsize=(18, 9.5))
    frames = {"levels": lv, "siblings": sib}
    for ax, (frame, value, ylabel, title, ref) in zip(axes.flat, PANELS):
        for agent, g in frames[frame].groupby("agent"):
            series = [(value, "-")] + ([("end_won_pct", "--")] if value == "tied" else [])
            for col, ls in series:
                mid, lo, hi = by_tl(g, col)
                if mid.dropna().empty:
                    continue
                ax.fill_between(mid.index, lo, hi, color=colors[agent], alpha=0.10, linewidth=0)
                ax.plot(mid.index, mid.values, color=colors[agent], linestyle=ls, linewidth=2,
                        marker="o", markersize=5)
        if ref is not None:
            ax.axhline(ref, color=TEXT, linewidth=1, linestyle=":")
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Sequence level (tl)", fontsize=11)
        ax.set_ylabel(ylabel, fontsize=11)
        ax.set_title(title, fontsize=12)
        style(ax)
    agent_legend(fig, sorted(lv["agent"].unique()), colors)
    fig.suptitle("How the chosen action compared with its siblings, by sequence level "
                 "(line: median, band: 25th-75th percentile; dotted: no difference)", fontsize=13)
    save(fig, path)


def plot_by_rank(sib_all, lv, path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5.5))
    for agent, g in sib_all.groupby("agent"):
        by_rank = g.groupby("rank")["diff_to_best"]
        mid, lo, hi = by_rank.median(), by_rank.quantile(0.25), by_rank.quantile(0.75)
        ax1.fill_between(mid.index, lo, hi, color=colors[agent], alpha=0.10, linewidth=0)
        ax1.plot(mid.index, mid.values, color=colors[agent], linewidth=2, marker="o", markersize=5)
    ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax1.set_xlabel("Sibling rank by value (1 = chosen)", fontsize=11)
    ax1.set_ylabel("best - sibling", fontsize=11)
    ax1.set_title("How fast the alternatives fall behind (median, 25th-75th percentile)", fontsize=12)
    style(ax1)
    mean_by_round(ax2, lv.dropna(subset=["gap"]), "gap", colors)
    ax2.axhline(0, color=TEXT, linewidth=1, linestyle=":")
    ax2.set_xlabel("Game round", fontsize=11)
    ax2.set_ylabel("best - runner-up", fontsize=11)
    ax2.set_title("Value gap to the runner-up by round (mean +/- sd, all levels)", fontsize=12)
    style(ax2)
    agent_legend(fig, sorted(set(sib_all["agent"]) | set(lv["agent"])), colors)
    save(fig, path)


for map_name, lv in by_map(levels, args.maps):
    out = out_root / map_name
    sib_all = df if map_name == "all-maps" else df[df["map"] == map_name]
    sib = sib_all[sib_all["rank"] > 1]
    plot_by_tl(lv, sib, out / "sibling_values_by_tl.png")
    plot_by_rank(sib_all, lv, out / "sibling_values_by_rank.png")

    table = lv.groupby(["agent", "tl"]).agg(
        levels=("gap", "size"), siblings=("siblings", "median"), gap=("gap", "median"),
        end_gap=("end_gap", "median"), tied_pct=("tied", "mean"), end_won_pct=("end_won_pct", "mean"),
        clear_pct=("clear", "mean"))
    spread = sib.groupby(["agent", "tl"])["diff_to_best"].median().rename("sibling_gap")
    table = table.join(spread, how="left").reset_index()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "sibling_values_by_tl.csv", index=False)

    lines = [f"Sibling values by sequence level: {map_name}\n", "=" * 60 + "\n\n",
             "One row per level of a chosen action sequence. Medians over the levels; gap = best -\n"
             "runner-up, sibling_gap = best - each non-chosen sibling, end_gap = best - END,\n"
             "tied_pct = % tied with the runner-up, end_won_pct = % where END was chosen,\n"
             "clear_pct = % where the gap exceeded 2 standard errors of the difference.\n\n"]
    without = [a for a in tournament_agents(args.tournament) if a not in set(lv["agent"])]
    if without:
        lines.append(f"No sibling_values.csv: {', '.join(without)}\n\n")
    for agent, t in table.groupby("agent"):
        g, s = lv[lv["agent"] == agent], sib[sib["agent"] == agent]
        lines.append(f"{agent}\n" + "-" * len(agent) + "\n")
        lines.append(f"levels {len(g)}, searches {g.groupby(LEVEL[:-1]).ngroups}\n")
        lines.append(f"gap to runner-up: {stats(g['gap'])}\n")
        lines.append(f"gap of non-chosen siblings: {stats(s['diff_to_best'])}\n")
        lines.append(f"gain over END: {stats(g['end_gap'])}\n\n")
        lines.append(t.drop(columns=["agent"]).round(4).to_string(index=False, na_rep="-") + "\n\n")
    path = out / "sibling_values_summary.txt"
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
