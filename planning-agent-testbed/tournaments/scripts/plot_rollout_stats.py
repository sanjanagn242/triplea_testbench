"""Compares the agents' rollouts in one tournament (every agents/*/metrics/<map>/vs-*/rollout_stats.csv,
one row per rollout: player, game, round, iteration, rollout_depth_reached, reward, is_terminal).

A turn is one search: the rollouts of one agent, as one player, in one game and round. A rollout that
reached depth n sampled n states, so the states sampled in a turn are the sum of rollout_depth_reached
over all its iterations.

The rollout files of a long tournament together outgrow memory (one row per rollout), so each file is
read on its own and reduced to one row per turn, plus counts of the rollout depths and rewards
(rounded to 4 decimals) for the summary's distributions; every plot and summary comes from those.

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  rollout_terminal.png             % of rollouts that stopped in a terminal state (win / loss): all
                                   rounds per agent, then one row per agent by decision round
  rollout_states_per_turn.png      states sampled per turn: distribution per agent (box; dots = each
                                   game's mean) and mean by round
  rollout_turn_stats.csv           one row per turn: avg_reward, var_reward, std_reward,
                                   avg_depth_reached, depth_reached_std, n_visits (rollouts)
  rollout_reward.png               every agent in one plot (one line each, all seats): mean reward per turn by round,
                                   with a band of the reward spread within a turn (narrow = clean
                                   reward signal, wide = noisy)
  rollout_stats_summary.txt        per agent: terminal %, depth, states and iterations per turn, reward
                                   spread within a turn (by player and round)

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_rollout_stats.py tournaments/<date>_<time> [--map MAP ...] [--out DIR]
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MaxNLocator

from common import (LOSS_COLOR, SURFACE, TEXT, WIN_COLOR, agent_code, agent_colors, agent_legend, boxes_by_agent,
                    color_agent_ticks, by_map, mean_by_round, parse_args, save, stats, stats_from_counts, style,
                    tournament_agents)

COLUMNS = ["player", "game", "round", "iteration", "rollout_depth_reached", "reward", "is_terminal"]
TURN = ["agent", "map", "opponent", "player", "game", "round"]
# rollout_turn_stats.csv: the turn and its rollouts' reward and depth
NODE_STATS = [*TURN, "avg_reward", "var_reward", "std_reward", "avg_depth_reached", "depth_reached_std", "n_visits"]


def load_turns(tournament):
    """One row per turn of every rollout_stats.csv (TURN, states, iterations, the NODE_STATS columns,
    wins and losses = rollouts ending in a terminal win / loss), and value counts (agent, map, value,
    count) of the rollout depths and of the rewards."""
    turns, depths, rewards = [], [], []
    for path in sorted((tournament / "agents").glob("*/metrics/*/vs-*/rollout_stats.csv")):
        r = pd.read_csv(path, usecols=lambda c: c in COLUMNS)
        missing = [c for c in COLUMNS if c not in r.columns]
        if missing:
            print(f"skipping {path}: no column(s) {', '.join(missing)}")
            continue
        if r.empty:
            continue
        for c in COLUMNS[1:]:
            r[c] = pd.to_numeric(r[c], errors="coerce")
        terminal = r["is_terminal"] == 1
        r["win"] = terminal & (r["reward"] > 0)
        r["loss"] = terminal & ~r["win"]
        t = r.groupby(TURN[3:]).agg(
            states=("rollout_depth_reached", "sum"),
            iterations=("iteration", "size"),
            avg_reward=("reward", "mean"),
            var_reward=("reward", "var"),
            std_reward=("reward", "std"),
            avg_depth_reached=("rollout_depth_reached", "mean"),
            depth_reached_std=("rollout_depth_reached", "std"),
            n_visits=("reward", "count"),
            wins=("win", "sum"),
            losses=("loss", "sum"),
        ).reset_index()
        keys = {"agent": path.parents[3].name, "map": path.parent.parent.name,
                "opponent": path.parent.name[len("vs-"):]}
        turns.append(t.assign(**keys))
        for column, counts in [(r["rollout_depth_reached"], depths), (r["reward"].round(4), rewards)]:
            vc = column.value_counts().rename_axis("value").reset_index(name="count")
            counts.append(vc.assign(agent=keys["agent"], map=keys["map"]))
    if not turns:
        return None, None, None
    return pd.concat(turns, ignore_index=True), pd.concat(depths), pd.concat(rewards)


args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
df, depth_counts, reward_counts = load_turns(args.tournament)
if df is None:
    raise SystemExit(f"no rollouts in any rollout_stats.csv under {args.tournament / 'agents'}")
df["reward_sd"] = df["std_reward"]
df["terminal"] = (df["wins"] + df["losses"]) / df["iterations"]
colors = agent_colors(args.tournament)


def outcome_pct(turns):
    """% of the turns' rollouts that ended in a terminal win, and in a terminal loss."""
    n = turns["iterations"].sum()
    return (100 * turns["wins"].sum() / n, 100 * turns["losses"].sum() / n) if n else (0, 0)


def counts_of(counts, map_name, agent):
    """value -> count of one agent's rollouts on a map (or all maps)."""
    c = counts[counts["agent"] == agent]
    if map_name in set(c["map"]):
        c = c[c["map"] == map_name]
    return c.groupby("value")["count"].sum()


def stacked(ax, x, win, loss, width):
    ax.bar(x, win, width=width, color=WIN_COLOR, label="Terminal win", edgecolor=SURFACE, linewidth=2)
    ax.bar(x, loss, width=width, bottom=win, color=LOSS_COLOR, label="Terminal loss",
           edgecolor=SURFACE, linewidth=2)
    for xi, top in zip(x, np.asarray(win) + np.asarray(loss)):
        ax.annotate(f"{top:.0f}%", (xi, top), xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=TEXT)


def terminal_plot(m, path):
    """Stacked rows, each with its own axes: all rounds per agent on top, then one row per agent by
    decision round."""
    agents = sorted(m["agent"].unique())
    fig, axes = plt.subplots(len(agents) + 1, 1, figsize=(16, 3.4 * (len(agents) + 1)), squeeze=False)
    axes = axes[:, 0]
    overall = [outcome_pct(m[m["agent"] == a]) for a in agents]
    stacked(axes[0], range(len(agents)), [w for w, _ in overall], [l for _, l in overall], 0.5)
    axes[0].set_xticks(range(len(agents)), [agent_code(colors, a) for a in agents])
    axes[0].set_title("All rounds", fontsize=13)
    axes[0].legend(frameon=False, fontsize=10, loc="upper left", labelcolor=TEXT)
    axes[0].set_ylim(0, max(10.0, max(w + l for w, l in overall) * 1.2))
    for ax, agent in zip(axes[1:], agents):
        a = m[m["agent"] == agent]
        rounds = np.arange(int(a["round"].min()), int(a["round"].max()) + 1)
        by_round = [outcome_pct(a[a["round"] == r]) for r in rounds]
        stacked(ax, rounds, [w for w, _ in by_round], [l for _, l in by_round], 0.7)
        ax.set_xticks(rounds)
        ax.set_xlim(rounds[0] - 0.6, rounds[-1] + 0.6)
        ax.set_ylim(0, max(10.0, max(w + l for w, l in by_round) * 1.2))
        ax.set_xlabel("Decision round", fontsize=12)
        ax.set_title(f"{agent_code(colors, agent)}  {agent}", fontsize=13, color=colors[agent])
    for ax in axes:
        ax.set_ylabel("Rollouts reaching\nterminal (%)", fontsize=12)
        style(ax)
        ax.tick_params(labelsize=11)
    # style() reset the tick colors; the codes are named in the per-agent row titles below
    color_agent_ticks(axes[0].get_xticklabels(), colors, agents)
    save(fig, path)


def states_plot(turns, path):
    agents = sorted(turns["agent"].unique())
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 3]})
    boxes_by_agent(ax1, {a: turns.loc[turns["agent"] == a, "states"] for a in agents}, colors,
                   {a: turns[turns["agent"] == a].groupby(["map", "game"])["states"].mean() for a in agents})
    ax1.set_ylabel("States sampled per turn", fontsize=12)
    ax1.set_title("States sampled per turn (dots: each game's mean)", fontsize=12)
    mean_by_round(ax2, turns, "states", colors)
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel("Mean states sampled per turn", fontsize=12)
    ax2.set_title("Mean states sampled per turn by round (band: +/- std across turns)", fontsize=12)
    for ax in (ax1, ax2):
        style(ax)
    agent_legend(fig, agents, colors)
    save(fig, path)


def reward_plot(node_stats, path):
    """One plot, one line per agent (its turns in every seat; rewards are from the searching player's
    point of view): per round, the mean of the turns' average rollout reward (line) and the mean
    spread of the rewards within a turn (band, +/- std_reward). A narrow band is a clean reward
    signal, a wide one a noisy one."""
    fig, ax = plt.subplots(figsize=(16, 7))
    for agent, g in node_stats.groupby("agent"):
        by_round = g.groupby("round")[["avg_reward", "std_reward"]].mean()
        x, mean, sd = by_round.index.values, by_round["avg_reward"].values, by_round["std_reward"].fillna(0).values
        ax.fill_between(x, mean - sd, mean + sd, color=colors[agent], alpha=0.12, linewidth=0)
        ax.plot(x, mean, color=colors[agent], linewidth=2)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_xlabel("Decision round", fontsize=12)
    ax.set_ylabel("Rollout reward", fontsize=12)
    ax.set_title("Rollout reward per turn by round (line: mean of the turns' average reward; "
                 "band: +/- mean std of the rewards within a turn)", fontsize=12)
    style(ax)
    agent_legend(fig, sorted(node_stats["agent"].unique()), colors)
    save(fig, path)


for map_name, turns in by_map(df, args.maps):
    out = out_root / map_name
    terminal_plot(turns, out / "rollout_terminal.png")
    states_plot(turns, out / "rollout_states_per_turn.png")
    out.mkdir(parents=True, exist_ok=True)
    turns[NODE_STATS].to_csv(out / "rollout_turn_stats.csv", index=False)
    reward_plot(turns, out / "rollout_reward.png")

    lines = [f"Rollout statistics: {map_name}\n", "=" * 60 + "\n\n",
             "A turn is one search (agent, map, opponent, player, game, round). States sampled = sum of\n"
             "rollout_depth_reached over the turn's rollouts. Reward sd = standard deviation of the\n"
             "rollout rewards within a turn (lower = cleaner reward signal; compare agents with the\n"
             "same reward scale).\n\n"]
    without = [a for a in tournament_agents(args.tournament) if a not in set(turns["agent"])]
    if without:
        lines.append(f"No rollouts (no search, or did not play this map): {', '.join(without)}\n\n")
    for agent, t in turns.groupby("agent"):
        win, loss = outcome_pct(t)
        lines.append(f"{agent}\n" + "-" * len(agent) + "\n")
        lines.append(f"  games {len(t[['map', 'game']].drop_duplicates())}, turns {len(t)}, "
                     f"rollouts {t['iterations'].sum()}\n")
        lines.append(f"  terminal {win + loss:.1f}% (win {win:.1f}%, loss {loss:.1f}%)\n")
        lines.append(f"  rollout depth reached     {stats_from_counts(counts_of(depth_counts, map_name, agent))}\n")
        lines.append(f"  states sampled per turn   {stats(t['states'])}\n")
        lines.append(f"  iterations per turn       {stats(t['iterations'])}\n")
        lines.append(f"  reward                    {stats_from_counts(counts_of(reward_counts, map_name, agent))}\n")
        lines.append(f"  reward sd within a turn   {stats(t['reward_sd'])}\n")
        for player, pt in t.groupby("player"):
            w, l = outcome_pct(pt)
            mean_reward = (pt["avg_reward"] * pt["n_visits"]).sum() / pt["n_visits"].sum()
            lines.append(f"    as {player}: turns {len(pt)}, terminal {w + l:.1f}%, mean states/turn "
                         f"{pt['states'].mean():.1f}, mean reward {mean_reward:.3f}, "
                         f"mean reward sd {pt['reward_sd'].mean():.3f}\n")
        lines.append("\n")

    for value, label in [("states", "Mean states sampled per turn"),
                         ("terminal", "Share of rollouts reaching terminal per turn (mean)"),
                         ("reward_sd", "Mean reward sd within a turn")]:
        table = turns.pivot_table(index="round", columns=["agent", "player"], values=value,
                                  aggfunc="mean")
        lines.append(f"{label}, by round\n")
        lines.append(table.round(3).to_string(na_rep="-") + "\n\n")

    path = out / "rollout_stats_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
