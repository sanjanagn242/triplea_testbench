"""Flatness of the rollouts in the purchase, combat and non-combat searches of one tournament: how much
the rollout rewards of a search's leaf nodes differ from each other.

Every rollout is run from the leaf an iteration of the search reached, so the spread of a search's
rollout rewards is the spread over its leaves. A row of rollout_stats*.csv has the totals of one
search's rollouts, its reward variance among them exact (files of earlier tournaments, one row per
rollout, are reduced to the same, see common.rollout_searches). A search whose leaves
all get (nearly) the same reward is flat: the rollouts can't tell the moves apart, often because they
stop before the moves' consequences show (rollout too short). A wide spread means the leaves differ,
either really or because the rollout policy is noisy; compare variants of one agent (e.g.
MCTS_ROLLOUT-2 vs MCTS_ROLLOUT-10, or a different epsilon) to tell which.

The phase of a search comes from the file it is in:
  rollout_stats_purchase.csv    purchase
  rollout_stats.csv             combat move
  rollout_stats_noncombat.csv   non-combat move
An agent appears only in the phases it logs rollouts for. An agent that logs several phases to
rollout_stats.csv (roxo-oep-with-purchase: combat and non-combat) names each search's phase in the
phase column; in its files of one row per rollout, they are split with the phase column of its
efficiency.csv, which has one row per search in the same order.

A search is one run of the search algorithm; its round is the decision round.

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  rollout_flatness.png          one row per phase: reward std over a search's leaves per agent (box;
                                dots = each game's mean), its mean by decision round, and the % of
                                flat searches (std below --flat-std)
  rollout_flatness_by_search.csv  one row per search: phase, n_leaves, avg_reward, var_reward,
                                std_reward, reward_range, avg_depth_reached, terminal_share
  rollout_flatness_summary.txt  per phase and agent: searches, leaves per search, reward std, flat %,
                                rollout depth reached and % terminal

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_rollout_flatness.py tournaments/<date>_<time> [--map MAP ...]
      [--out DIR] [--flat-std 0.001]
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from common import (SURFACE, TEXT, agent_code, agent_colors, agent_legend, boxes_by_agent, by_map,
                    color_agent_ticks, mean_by_round, parse_args, rollout_searches, rollout_spread, save,
                    stats, style)

# (phase, rollout file) in turn order
PHASES = [("purchase", "rollout_stats_purchase.csv"), ("combat", "rollout_stats.csv"),
          ("noncombat", "rollout_stats_noncombat.csv")]
# efficiency.csv phase values -> phase, for agents logging several phases to rollout_stats.csv
EFFICIENCY_PHASES = {"purchase": "purchase", "combatMove": "combat", "nonCombatMove": "noncombat"}
SEARCH = ["agent", "map", "opponent", "phase", "player", "game", "round"]
BY_SEARCH = [*SEARCH, "n_leaves", "avg_reward", "var_reward", "std_reward", "reward_range",
             "avg_depth_reached", "terminal_share"]


def efficiency_phases(path):
    """(player, game, round) -> phases of its searches in order, from the efficiency.csv next to
    path, when that file has a phase column naming more than one known phase; else None."""
    try:
        e = pd.read_csv(path.parent / "efficiency.csv", usecols=["player", "game", "round", "phase"])
    except (FileNotFoundError, ValueError, pd.errors.ParserError):
        return None
    if not e["phase"].isin(EFFICIENCY_PHASES).all() or e["phase"].nunique() < 2:
        return None
    e["phase"] = e["phase"].map(EFFICIENCY_PHASES)
    return {k: list(g["phase"]) for k, g in e.groupby(["player", "game", "round"], sort=False)}


def searches_of(path, phase):
    """One row per search of a rollout file (BY_SEARCH columns but agent / map / opponent)."""
    s, _ = rollout_searches(path)
    if s is None:
        return None
    s["n_leaves"] = s["rollouts"]
    s["avg_reward"] = s["reward_mean"]
    s["var_reward"], s["std_reward"], s["avg_depth_reached"], _ = rollout_spread(s)
    s["reward_range"] = s["reward_max"] - s["reward_min"]
    s["terminal_share"] = (s["terminal_wins"] + s["terminal_losses"]) / s["rollouts"]
    if s["phase"].notna().all():
        return s  # the agent named each search's phase
    s["phase"] = phase

    # one row per rollout: an agent logging several phases to one file is split by efficiency.csv
    phases = efficiency_phases(path) if phase == "combat" else None
    if phases:
        s["nth"] = s.groupby(["player", "game", "round"]).cumcount()
        size = s.groupby(["player", "game", "round"])["nth"].transform("size")
        keys = list(zip(s["player"], s["game"], s["round"]))
        aligned = [len(phases.get(k, ())) == n for k, n in zip(keys, size)]
        s["phase"] = [phases[k][i] if ok else None for k, i, ok in zip(keys, s["nth"], aligned)]
        dropped = s["phase"].isna().sum()
        if dropped:
            print(f"{path}: {dropped} searches left out, their rounds have a different number of "
                  f"searches than efficiency.csv rows")
        s = s.dropna(subset=["phase"])
    return s


def load_searches(tournament):
    frames = []
    for phase, filename in PHASES:
        for path in sorted((tournament / "agents").glob(f"*/metrics/*/vs-*/{filename}")):
            s = searches_of(path, phase)
            if s is None or s.empty:
                continue
            frames.append(s.assign(agent=path.parents[3].name, map=path.parent.parent.name,
                                   opponent=path.parent.name[len("vs-"):]))
    return pd.concat(frames, ignore_index=True)[BY_SEARCH] if frames else None


parser = parse_args(__doc__.splitlines()[0])
parser.add_argument("--flat-std", type=float, default=1e-3,
                    help="a search is flat when the reward std over its leaves is below this "
                         "(default 0.001)")
args = parser.parse_args()
out_root = args.out or args.tournament / "plots"
df = load_searches(args.tournament)
if df is None:
    raise SystemExit(f"no rollouts in any rollout_stats*.csv under {args.tournament / 'agents'}")
# A search with one leaf has no std (NaN): left out of the std plots and not counted as flat.
df["flat"] = df["std_reward"] < args.flat_std
colors = agent_colors(args.tournament)


def flatness_plot(m, path):
    """One row per phase with data: std over the leaves per agent, its mean by round, and % flat."""
    phases = [p for p, _ in PHASES if p in set(m["phase"])]
    fig, axes = plt.subplots(len(phases), 3, figsize=(20, 5.2 * len(phases)), squeeze=False,
                             gridspec_kw={"width_ratios": [2, 3, 1.4]})
    for (ax1, ax2, ax3), phase in zip(axes, phases):
        p = m[(m["phase"] == phase) & m["std_reward"].notna()]
        agents = sorted(p["agent"].unique())
        boxes_by_agent(ax1, {a: p.loc[p["agent"] == a, "std_reward"] for a in agents}, colors,
                       {a: p[p["agent"] == a].groupby(["map", "game"])["std_reward"].mean()
                        for a in agents})
        ax1.set_ylabel("Reward std over the leaves", fontsize=12)
        ax1.set_title(f"{phase}: reward std over a search's leaves (dots: each game's mean)",
                      fontsize=12)
        mean_by_round(ax2, p, "std_reward", colors)
        ax2.set_xlabel("Decision round", fontsize=12)
        ax2.set_ylabel("Mean reward std over the leaves", fontsize=12)
        ax2.set_title(f"{phase}: mean by round (band: +/- std across searches)", fontsize=12)
        flat = [100 * p.loc[p["agent"] == a, "flat"].mean() for a in agents]
        ax3.bar(range(len(agents)), flat, width=0.6, color=[colors[a] for a in agents],
                edgecolor=SURFACE, linewidth=2)
        for x, v in enumerate(flat):
            ax3.annotate(f"{v:.0f}%", (x, v), xytext=(0, 3), textcoords="offset points",
                         ha="center", va="bottom", fontsize=9, color=TEXT)
        ax3.set_xticks(range(len(agents)), [agent_code(colors, a) for a in agents])
        ax3.set_ylim(0, 105)
        ax3.set_ylabel("Flat searches (%)", fontsize=12)
        ax3.set_title(f"{phase}: std < {args.flat_std:g}", fontsize=12)
        for ax in (ax1, ax2, ax3):
            style(ax)
        color_agent_ticks(ax3.get_xticklabels(), colors, agents)  # after style() reset them
    agent_legend(fig, sorted(m["agent"].unique()), colors)
    save(fig, path)


for map_name, searches in by_map(df, args.maps):
    out = out_root / map_name
    flatness_plot(searches, out / "rollout_flatness.png")
    out.mkdir(parents=True, exist_ok=True)
    searches[BY_SEARCH].to_csv(out / "rollout_flatness_by_search.csv", index=False)

    lines = [f"Rollout flatness: {map_name}\n", "=" * 60 + "\n\n",
             "A search's leaves are its rollouts (one per iteration). Reward std = standard deviation\n"
             "of their rewards (the square root of the variance among the leaves). Low std with a\n"
             "low depth reached and few terminal rollouts: flat, the rollout is likely too short.\n"
             "High std: the leaves differ, really or through a noisy rollout policy. Flat = std\n"
             f"below {args.flat_std:g}. Compare agents with the same reward scale.\n\n"]
    for phase, _ in PHASES:
        p = searches[searches["phase"] == phase]
        if p.empty:
            continue
        lines.append(f"{phase}\n" + "=" * len(phase) + "\n")
        for agent, a in p.groupby("agent"):
            lines.append(f"  {agent}\n")
            lines.append(f"    searches {len(a)} in {len(a[['map', 'game']].drop_duplicates())} games, "
                         f"flat {100 * a['flat'].mean():.1f}%\n")
            lines.append(f"    leaves per search        {stats(a['n_leaves'])}\n")
            lines.append(f"    reward std over leaves   {stats(a['std_reward'])}\n")
            lines.append(f"    reward range             {stats(a['reward_range'])}\n")
            lines.append(f"    rollout depth reached    {stats(a['avg_depth_reached'])}\n")
            lines.append(f"    terminal rollouts        {100 * np.average(a['terminal_share'], weights=a['n_leaves']):.1f}%\n")
        table = p.pivot_table(index="round", columns="agent", values="std_reward", aggfunc="mean")
        table.columns = [agent_code(colors, c) for c in table.columns]
        lines.append("\n  Mean reward std over the leaves, by decision round\n")
        lines.append(table.round(4).to_string(na_rep="-") + "\n\n")
    lines.append("Agent codes: " + ", ".join(f"{agent_code(colors, a)} {a}"
                                             for a in sorted(searches["agent"].unique())) + "\n")
    path = out / "rollout_flatness_summary.txt"
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
