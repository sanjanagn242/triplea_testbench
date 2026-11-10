"""Rollout noise in the purchase, combat and non-combat searches of one tournament: how much of the
spread of a search's rollout rewards is the rollout's own noise (every agents/*/metrics/<map>/vs-*/
rollout_stats*.csv with the noise columns; files and rows without them are left out).

Every rollout is run from the leaf an iteration reached, once, so the spread of a search's rollout
rewards (the leaf spread) is the real difference between its leaves plus the rollout's noise. To
tell them apart the agents play every n-th rollout a second time from the same state
(TRIPLEA_ROLLOUT_NOISE_EVERY, see their helper.py): half the squared difference of such a pair's
rewards estimates the variance of a rollout's reward from one state, the noise (0 for
deterministic rollouts). Over the searches with at least one pair, pooled exactly (sums of the
pairs' half squared differences and of reward_m2):
  noise sd     sqrt(noise variance)
  leaf sd      sqrt(leaf variance), the spread of the rewards within a search
  noise share  noise variance / leaf variance: near 100%, the leaves' rewards differ mostly by
               noise and the search's values can hardly tell its moves apart
  signal sd    sqrt(max(leaf variance - noise variance, 0)), the real spread between the leaves
The phase of a search comes from its file (rollout_stats_purchase.csv, rollout_stats.csv,
rollout_stats_noncombat.csv) or, for an agent logging several phases to one file, its phase column.

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  rollout_noise.png              one row per phase: noise sd against leaf sd per agent (pooled), the
                                 noise share by decision round, and the noise sd per search (box;
                                 dots = each game's pooled noise sd)
  rollout_noise_by_search.csv    one row per search with pairs: phase, rollouts, noise_pairs,
                                 noise_sd, leaf_sd, noise_seconds
  rollout_noise_summary.txt      per phase and agent: searches, pairs, noise sd, leaf sd, noise
                                 share, signal sd, noise seconds per search; noise share by round

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_rollout_noise.py tournaments/<date>_<time> [--map MAP ...] [--out DIR]
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
from matplotlib.ticker import MaxNLocator

from common import (TEXT, agent_code, agent_colors, agent_legend, boxes_by_agent, by_map,
                    color_agent_ticks, parse_args, rollout_searches, save, stats, style,
                    tournament_agents)

# (phase, rollout file) in turn order
PHASES = [("purchase", "rollout_stats_purchase.csv"), ("combat", "rollout_stats.csv"),
          ("noncombat", "rollout_stats_noncombat.csv")]
NOISE = ["noise_pairs", "noise_var_sum", "noise_seconds"]
MIN_ROUND_SEARCHES = 5  # rounds with fewer searches of an agent are left out of its by-round share
SEARCH = ["agent", "map", "opponent", "phase", "player", "game", "round"]
BY_SEARCH = [*SEARCH, "rollouts", "noise_pairs", "noise_sd", "leaf_sd", "noise_seconds"]


def load_searches(tournament):
    """One row per search with at least one noise pair, of every rollout_stats*.csv (the rollout
    totals of common.rollout_searches, the NOISE columns, agent, map and opponent)."""
    frames = []
    for phase, filename in PHASES:
        for path in sorted((tournament / "agents").glob(f"*/metrics/*/vs-*/{filename}")):
            s, _ = rollout_searches(path)
            if s is None:
                continue
            if any(c not in s.columns for c in NOISE):
                print(f"skipping {path}: no noise columns (written before they existed)")
                continue
            s[NOISE] = s[NOISE].apply(pd.to_numeric, errors="coerce")
            s = s[s["noise_pairs"] > 0]  # rows of earlier runs are blank, small searches have none
            if s.empty:
                continue
            s["phase"] = s["phase"].fillna(phase)
            frames.append(s.assign(agent=path.parents[3].name, map=path.parent.parent.name,
                                   opponent=path.parent.name[len("vs-"):]))
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    df["noise_sd"] = np.sqrt(df["noise_var_sum"] / df["noise_pairs"])
    df["leaf_sd"] = np.sqrt(df["reward_m2"] / (df["rollouts"] - 1))
    return df


def pooled(g, keys):
    """Per keys, the noise and leaf variance of searches g pooled exactly (sums of the pairs' half
    squared differences and of reward_m2), with their sd, share and signal sd."""
    t = g.assign(dof=g["rollouts"] - 1).groupby(keys).agg(
        searches=("rollouts", "size"), pairs=("noise_pairs", "sum"),
        noise_var_sum=("noise_var_sum", "sum"), reward_m2=("reward_m2", "sum"), dof=("dof", "sum"),
        noise_s_per_search=("noise_seconds", "mean"))
    noise_var, leaf_var = t["noise_var_sum"] / t["pairs"], t["reward_m2"] / t["dof"]
    t["noise_sd"], t["leaf_sd"] = np.sqrt(noise_var), np.sqrt(leaf_var)
    t["noise_share_pct"] = (100 * noise_var / leaf_var).where(leaf_var > 0)
    t["signal_sd"] = np.sqrt((leaf_var - noise_var).clip(lower=0))
    t["pairs"] = t["pairs"].astype(int)
    return t[["searches", "pairs", "noise_sd", "leaf_sd", "noise_share_pct", "signal_sd",
              "noise_s_per_search"]]


args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
df = load_searches(args.tournament)
if df is None:
    raise SystemExit(f"no rollout noise in any rollout_stats*.csv under {args.tournament / 'agents'} "
                     f"(the agents write it since TRIPLEA_ROLLOUT_NOISE_EVERY was added)")
colors = agent_colors(args.tournament)


def noise_plot(m, path):
    """One row per phase with data: pooled noise sd against leaf sd per agent, the noise share by
    round, and the noise sd per search."""
    phases = [p for p, _ in PHASES if p in set(m["phase"])]
    fig, axes = plt.subplots(len(phases), 3, figsize=(20, 5.2 * len(phases)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 3, 2]})
    for (ax1, ax2, ax3), phase in zip(axes, phases):
        p = m[m["phase"] == phase]
        agents = sorted(p["agent"].unique())
        totals = pooled(p, "agent")
        by_round = pooled(p, ["agent", "round"])
        by_round = by_round[by_round["searches"] >= MIN_ROUND_SEARCHES]
        by_game = pooled(p, ["agent", "map", "game"])
        x = np.arange(len(agents))
        for xi, a in zip(x, agents):
            t = totals.loc[a]
            ax1.bar(xi - 0.2, t["leaf_sd"], width=0.4, color=colors[a] + "55", edgecolor=colors[a],
                    linewidth=1.5)
            ax1.bar(xi + 0.2, t["noise_sd"], width=0.4, color=colors[a])
            ax1.annotate(f"{t['noise_share_pct']:.0f}%", (xi + 0.2, t["noise_sd"]), xytext=(0, 3),
                         textcoords="offset points", ha="center", va="bottom", fontsize=9, color=TEXT)
        ax1.set_xticks(x, [agent_code(colors, a) for a in agents])
        ax1.set_ylabel("Reward sd", fontsize=12)
        ax1.set_title(f"{phase}: leaf sd (light) and noise sd (solid; label: noise share of the "
                      f"variance)", fontsize=11)
        ax1.set_ylim(0, 1.3 * totals["leaf_sd"].max())  # room for the legend above the bars
        ax1.legend(handles=[Patch(facecolor="#7d7c7855", edgecolor="#7d7c78", label="leaf sd"),
                            Patch(facecolor="#7d7c78", label="noise sd")],
                   frameon=False, fontsize=10, loc="upper center", ncol=2, labelcolor=TEXT)

        for a in agents:
            if a not in by_round.index.get_level_values("agent"):
                continue
            share = by_round.loc[a, "noise_share_pct"]
            ax2.plot(share.index, share.values, color=colors[a], linewidth=2, marker="o",
                     markersize=4)
        ax2.axhline(100, color=TEXT, linewidth=1, linestyle=":")
        ax2.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax2.set_xlabel("Decision round", fontsize=12)
        ax2.set_ylabel("Noise share of the leaf variance (%)", fontsize=12)
        ax2.set_title(f"{phase}: noise share by round (dotted: all of the spread is noise; rounds "
                      f"with {MIN_ROUND_SEARCHES}+ searches)", fontsize=12)

        boxes_by_agent(ax3, {a: p.loc[p["agent"] == a, "noise_sd"] for a in agents}, colors,
                       {a: by_game.loc[a, "noise_sd"] for a in agents})
        ax3.set_ylabel("Noise sd of a search", fontsize=12)
        ax3.set_title(f"{phase}: noise sd per search (dots: each game's)", fontsize=12)
        for ax in (ax1, ax2, ax3):
            style(ax)
        color_agent_ticks(ax1.get_xticklabels(), colors, agents)
    agent_legend(fig, sorted(m["agent"].unique()), colors)
    save(fig, path)


for map_name, m in by_map(df, args.maps):
    out = out_root / map_name
    noise_plot(m, out / "rollout_noise.png")
    out.mkdir(parents=True, exist_ok=True)
    m[BY_SEARCH].to_csv(out / "rollout_noise_by_search.csv", index=False)

    lines = [f"Rollout noise: {map_name}\n", "=" * 60 + "\n\n",
             "Noise sd = sd of the rewards of two rollouts from the same state (pooled over the\n"
             "searches' pairs; 0 = deterministic rollouts). Leaf sd = sd of the rewards of a search's\n"
             "rollouts (one per leaf), pooled within searches. Noise share = noise variance / leaf\n"
             "variance (near 100%: the leaves' rewards differ mostly by noise). Signal sd = the leaf\n"
             "spread left once the noise is taken out. Only searches with at least one pair.\n\n",
             "Agent codes (by-round tables and plots):\n"]
    lines += [f"  {agent_code(colors, a)}  {a}\n" for a in sorted(m["agent"].unique())] + ["\n"]
    without = [a for a in tournament_agents(args.tournament) if a not in set(m["agent"])]
    if without:
        lines.append(f"No rollout noise (no pairs, or written before the noise columns): "
                     f"{', '.join(without)}\n\n")
    for phase, _ in PHASES:
        p = m[m["phase"] == phase]
        if p.empty:
            continue
        lines.append(f"{phase}\n" + "=" * len(phase) + "\n")
        lines.append(pooled(p, "agent").round(4).to_string() + "\n\n")
        by_player = pooled(p, ["agent", "player"])
        for agent, a in p.groupby("agent"):
            lines.append(f"  {agent}\n    noise sd per search  {stats(a['noise_sd'])}\n")
            for player, t in by_player.loc[agent].iterrows():
                lines.append(f"    as {player}: searches {t['searches']:.0f}, noise sd "
                             f"{t['noise_sd']:.4f}, leaf sd {t['leaf_sd']:.4f}, noise share "
                             f"{t['noise_share_pct']:.0f}%\n")
        lines.append("\n")
        share = pooled(p, ["round", "agent"])
        share = share[share["searches"] >= MIN_ROUND_SEARCHES]["noise_share_pct"].unstack("agent")
        share.columns = [agent_code(colors, a) for a in share.columns]
        lines.append(f"{phase}: noise share (%) by round (- : under {MIN_ROUND_SEARCHES} searches)\n")
        lines.append(share.round(0).to_string(na_rep="-") + "\n\n")

    path = out / "rollout_noise_summary.txt"
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
