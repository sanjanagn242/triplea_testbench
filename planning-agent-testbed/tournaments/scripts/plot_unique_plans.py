"""Compares how many distinct legal plans the EMCTS, TBETS and HMCTS agents evaluated per search, in
one tournament (one row per search): the unique_plans column of every
agents/{emcts,tbets}*/metrics/<map>/vs-*/efficiency.csv, and for heterogeneous-mcts,
tbets-m-uct and tbets-seg-hybrid (its tree) the num_action_sequences column of their efficiency.csv (combat move),
efficiency_purchase.csv and efficiency_noncombat.csv.

A plan is the genome as played: in emcts and emcts-puw every genome is legal, so one plan per
phenotype; in emcts-fh and emcts-segmented the decoded (repaired) genome, so stored genomes that
decode to the same plan count once; in tbets-complete-turn and tbets-segmented the decoded genomes
evaluated. heterogeneous-mcts and tbets-m-uct have no genomes: their count is the action sequences
(root-to-leaf paths) of their search tree. tbets-seg-hybrid stores its genomes in that tree, so
the same count is its distinct genomes.

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  unique_plans.png          one row per searched phase (combatMove for every agent; emcts-segmented,
                            the tbets agents, heterogeneous-mcts, tbets-m-uct and tbets-seg-hybrid also search purchase and
                            nonCombatMove): unique plans per search,
                            distribution per agent (box; dots = each game's mean), and by round
                            (line = mean, band = min to max across that round's searches)
  unique_plans_summary.txt  per agent and phase: games, searches, unique plans; and the mean, min
                            and max by round

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_unique_plans.py tournaments/<date>_<time> [--map MAP ...] [--out DIR]
"""
import contextlib
import io

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.ticker import MaxNLocator

from common import agent_colors, agent_legend, boxes_by_agent, by_map, load, parse_args, save, stats, style

COLUMNS = ["player", "game", "round", "unique_plans"]
COMBAT = "combatMove"
PHASES = [COMBAT, "purchase", "nonCombatMove"]  # row order of the plots
PLANNERS = ("emcts", "tbets")  # agents whose efficiency.csv has unique_plans
# Agents on heterogeneous-mcts's tree: one efficiency file per searched phase, num_action_sequences
# for the plans.
HMCTS = ("heterogeneous-mcts", "tbets-m-uct", "tbets-seg-hybrid")
HMCTS_FILES = {"efficiency.csv": COMBAT, "efficiency_purchase.csv": "purchase",
               "efficiency_noncombat.csv": "nonCombatMove"}
HMCTS_COLUMNS = ["player", "game", "round", "num_action_sequences"]

args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
# emcts-segmented and the tbets agents log one row per searched phase (their phase column); the
# others search only the combat move. Files without these columns (the other agents' and
# heterogeneous-mcts') are left out of these loads quietly.
with contextlib.redirect_stdout(io.StringIO()):
    phased = load(args.tournament, "efficiency.csv", COLUMNS + ["phase"])
    single = load(args.tournament, "efficiency.csv", COLUMNS)
    hmcts = [load(args.tournament, f, HMCTS_COLUMNS).assign(phase=phase)
             for f, phase in HMCTS_FILES.items()]
single = single[~single["agent"].isin(phased["agent"].unique())].assign(phase=COMBAT)
hmcts = pd.concat(hmcts, ignore_index=True).rename(columns={"num_action_sequences": "unique_plans"})
hmcts = hmcts[hmcts["agent"].str.startswith(HMCTS)]
df = pd.concat([phased, single], ignore_index=True)
df = df[df["agent"].str.startswith(PLANNERS) & ~df["agent"].str.startswith(HMCTS)]
df = pd.concat([df, hmcts], ignore_index=True)
if df.empty:
    raise SystemExit(f"no EMCTS/TBETS unique_plans or {'/'.join(HMCTS)} num_action_sequences under "
                     f"{args.tournament / 'agents'}")
for c in ["game", "round", "unique_plans"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df = df.dropna(subset=["unique_plans"])
colors = agent_colors(args.tournament)
Y_LABEL = ("Unique legal plans\n(heterogeneous-mcts, tbets-m-uct, tbets-seg-hybrid: action sequences)"
           if df["agent"].str.startswith(HMCTS).any() else "Unique legal plans")


def range_by_round(ax, d, value):
    """Mean of value per round, one line per agent, with a band from the round's min to its max."""
    for agent, g in d.groupby("agent"):
        by_round = g.groupby("round")[value]
        mean, lo, hi = by_round.mean(), by_round.min(), by_round.max()
        ax.fill_between(mean.index, lo, hi, color=colors[agent], alpha=0.12, linewidth=0)
        ax.plot(mean.index, mean.values, color=colors[agent], linewidth=2, label=agent)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


for map_name, m in by_map(df, args.maps):
    out = out_root / map_name
    phases = [p for p in PHASES if p in set(m["phase"])]
    if not phases:
        continue
    agents = sorted(m["agent"].unique())
    fig, axes = plt.subplots(len(phases), 2, figsize=(14, 5.5 * len(phases)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 3]})
    for (ax1, ax2), phase in zip(axes, phases):
        p = m[m["phase"] == phase]
        shown = sorted(p["agent"].unique())
        boxes_by_agent(ax1, {a: p.loc[p["agent"] == a, "unique_plans"] for a in shown}, colors,
                       {a: p[p["agent"] == a].groupby(["map", "game"])["unique_plans"].mean() for a in shown})
        ax1.set_ylabel(Y_LABEL, fontsize=12)
        ax1.set_title(f"{phase}: unique plans per search (dots: each game's mean)", fontsize=12)
        range_by_round(ax2, p, "unique_plans")
        ax2.set_xlabel("Round", fontsize=12)
        ax2.set_ylabel(Y_LABEL, fontsize=12)
        ax2.set_title(f"{phase}: mean by round (band: min to max across searches)", fontsize=12)
        for ax in (ax1, ax2):
            style(ax)
    agent_legend(fig, agents, colors)
    save(fig, out / "unique_plans.png")

    lines = [f"Unique legal plans per search: {map_name}\n", "=" * 60 + "\n\n"]
    for phase in phases:
        p = m[m["phase"] == phase]
        lines.append(f"{phase}\n" + "-" * len(phase) + "\n")
        for agent, a in p.groupby("agent"):
            games = len(a[["map", "game"]].drop_duplicates())
            lines.append(f"  {agent}\n    {games} games, {len(a)} searches\n"
                         f"    unique plans  {stats(a['unique_plans'])}\n")
        lines.append("\n")
        for agg, label in [("mean", "Mean"), ("min", "Min"), ("max", "Max")]:
            table = p.pivot_table(index="round", columns="agent", values="unique_plans", aggfunc=agg)
            lines.append(f"{label} unique plans by round ({phase})\n")
            lines.append(table.round(1).to_string(na_rep="-") + "\n\n")
    path = out / "unique_plans_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
