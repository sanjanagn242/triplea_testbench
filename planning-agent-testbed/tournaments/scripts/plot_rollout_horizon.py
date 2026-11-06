"""Compares the root-child return variance normalized by visit count against the rollout length the
search used, in one tournament (every agents/*/metrics/<map>/vs-*/rollout_horizon.csv, one row per
search: the rollout length in rounds and the metric after the search; written by hmcts-dh so far,
whose rollout length is set from this metric every rollout_freq rounds).

Rows with a blank metric (no root child had 2 visits) are left out.

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  rollout_horizon.png          one panel per root phase (purchase, combat move, non-combat move):
                               per rollout length, one box per agent of the per-search metric
                               (log scale)
  rollout_horizon_by_round.png rollout length by game round (mean +/- sd over the searches), per
                               phase: how the schedule moved
  rollout_horizon.csv          searches, median, mean, p25, p75 per agent, phase and rollout length
  rollout_horizon_summary.txt  the same as tables, per agent

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_rollout_horizon.py tournaments/<date>_<time> [--map MAP ...] [--out DIR]
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from common import (agent_colors, agent_legend, by_map, load, mean_by_round, parse_args, save, stats,
                    style, tournament_agents)

COLUMNS = ["player", "game", "round", "phase", "rollout_rounds", "root_return_var"]
PHASES = [("purchase", "Purchase"), ("combatMove", "Combat move"),
          ("nonCombatMove", "Non-combat move")]

args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
df = load(args.tournament, "rollout_horizon.csv", COLUMNS)
if df.empty:
    raise SystemExit(f"no rollout_horizon.csv under {args.tournament / 'agents'}")
for c in ["game", "round", "rollout_rounds", "root_return_var"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
colors = agent_colors(args.tournament)


def plot_metric(d, path):
    """Per phase: boxes of root_return_var per rollout length, side by side per agent."""
    measured = d.dropna(subset=["root_return_var"])
    phases = [(p, t) for p, t in PHASES if (measured["phase"] == p).any()]
    agents = sorted(measured["agent"].unique())
    lengths = sorted(measured["rollout_rounds"].dropna().astype(int).unique())
    width = 0.8 / len(agents)
    fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 5), sharey=True,
                             squeeze=False)
    for ax, (phase, title) in zip(axes[0], phases):
        g = measured[measured["phase"] == phase]
        for i, agent in enumerate(agents):
            ga = g[g["agent"] == agent]
            xs = [r for r in lengths if (ga["rollout_rounds"] == r).any()]
            if not xs:
                continue
            pos = [r - 0.4 + width * (i + 0.5) for r in xs]
            bp = ax.boxplot([ga.loc[ga["rollout_rounds"] == r, "root_return_var"] for r in xs],
                            positions=pos, widths=width * 0.8, patch_artist=True,
                            showfliers=False, manage_ticks=False,
                            medianprops={"color": "#0b0b0b", "linewidth": 2},
                            whiskerprops={"color": colors[agent]},
                            capprops={"color": colors[agent]})
            for patch in bp["boxes"]:
                patch.set_facecolor(colors[agent] + "55")
                patch.set_edgecolor(colors[agent])
                patch.set_linewidth(1.5)
        counts = g.groupby("rollout_rounds").size()
        ax.set_xticks(lengths, [f"{r}\nn={counts.get(r, 0)}" for r in lengths])
        ax.set_xlim(min(lengths) - 0.6, max(lengths) + 0.6)
        ax.set_yscale("log")
        ax.set_xlabel("Rollout length (rounds)", fontsize=11)
        ax.set_title(title, fontsize=12)
        style(ax)
    axes[0][0].set_ylabel("Root-child return variance / visits", fontsize=11)
    agent_legend(fig, agents, colors)
    fig.suptitle("Root-child return variance by rollout length (per search; n = searches)",
                 fontsize=13)
    save(fig, path)


def plot_by_round(d, path):
    """Per phase: the rollout length the searches used, by game round."""
    phases = [(p, t) for p, t in PHASES if (d["phase"] == p).any()]
    fig, axes = plt.subplots(1, len(phases), figsize=(6 * len(phases), 4.5), sharey=True,
                             squeeze=False)
    for ax, (phase, title) in zip(axes[0], phases):
        mean_by_round(ax, d[d["phase"] == phase], "rollout_rounds", colors)
        ax.yaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_xlabel("Game round", fontsize=11)
        ax.set_title(title, fontsize=12)
        style(ax)
    axes[0][0].set_ylabel("Rollout length (rounds)", fontsize=11)
    agent_legend(fig, sorted(d["agent"].unique()), colors)
    fig.suptitle("Rollout length by round (mean +/- sd over the searches)", fontsize=13)
    save(fig, path)


for map_name, d in by_map(df, args.maps):
    out = out_root / map_name
    plot_metric(d, out / "rollout_horizon.png")
    plot_by_round(d, out / "rollout_horizon_by_round.png")

    measured = d.dropna(subset=["root_return_var"])
    table = measured.groupby(["agent", "phase", "rollout_rounds"])["root_return_var"].agg(
        searches="size", median="median", mean="mean",
        p25=lambda s: s.quantile(0.25), p75=lambda s: s.quantile(0.75)).reset_index()
    out.mkdir(parents=True, exist_ok=True)
    table.to_csv(out / "rollout_horizon.csv", index=False)

    lines = [f"Root-child return variance / visits by rollout length: {map_name}\n",
             "=" * 60 + "\n\n",
             "One row per search; searches whose root had no child with 2 visits are left out.\n\n"]
    without = [a for a in tournament_agents(args.tournament) if a not in set(d["agent"])]
    if without:
        lines.append(f"No rollout_horizon.csv: {', '.join(without)}\n\n")
    for agent, t in table.groupby("agent"):
        g = measured[measured["agent"] == agent]
        lines.append(f"{agent}\n" + "-" * len(agent) + "\n")
        lines.append(f"searches {len(g)} (of {int((d['agent'] == agent).sum())})\n")
        lines.append(f"metric: {stats(g['root_return_var'])}\n")
        lines.append(f"rollout length: {stats(d.loc[d['agent'] == agent, 'rollout_rounds'])}\n\n")
        lines.append(t.drop(columns=["agent"]).to_string(index=False, float_format="{:.4g}".format)
                     + "\n\n")
    path = out / "rollout_horizon_summary.txt"
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
