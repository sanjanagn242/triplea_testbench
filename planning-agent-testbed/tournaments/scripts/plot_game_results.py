"""Compares the agents' game outcomes in one tournament (every agents/*/metrics/<map>/vs-*/game_results.csv,
one row per game an agent played: game, player, rounds_played, won, turn_position, game_time).

A game the agent did not win is lost, or has no winner (it reached the round limit). That split comes
from the tournament's games.jsonl (winners of each game); without it both are counted as lost.

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  game_outcomes.png        win rate of every agent against every opponent (and all of them)
  game_seats.png           win rate of every agent by its place in the turn order (moves 1st, 2nd, ...)
  game_length.png          rounds played and game time per agent, won and lost games apart (lost
                           includes games without a winner)
  game_maps.png            (all-maps only) win rate of every agent on every map
  game_results_summary.txt per agent: games, wins, losses, no-winner games, win rate per opponent,
                           turn position (and map, in all-maps), rounds and time

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_game_results.py tournaments/<date>_<time> [--map MAP ...] [--out DIR]
"""
import json
import math

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

from common import (ALL_MAPS, SURFACE, TEXT, agent_code, agent_colors, agent_legend, by_map,
                    color_agent_ticks, load, parse_args, save, stats, style)

COLUMNS = ["game", "player", "rounds_played", "won", "turn_position", "game_time"]
# Reference sequential blue ramp (steps 100 -> 700) for the win-rate matrices.
WIN_RATE_CMAP = LinearSegmentedColormap.from_list(
    "win_rate", ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"])

args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
df = load(args.tournament, "game_results.csv", COLUMNS)
if df.empty:
    raise SystemExit(f"no game_results.csv under {args.tournament / 'agents'}")
for c in ["game", "rounds_played", "won", "turn_position", "game_time"]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
df["won"] = df["won"] == 1
colors = agent_colors(args.tournament)

# games.jsonl: (map, game number) -> winners, to tell a loss from a game without a winner.
winners = {}
games_jsonl = args.tournament / "games.jsonl"
if games_jsonl.exists():
    for line in games_jsonl.read_text().splitlines():
        if line.strip():
            g = json.loads(line)
            winners[(g.get("gameFolder"), g.get("game"))] = g.get("winners") or []
known = df.apply(lambda r: (r["map"], r["game"]) in winners, axis=1)
no_winner = df.apply(lambda r: not winners.get((r["map"], r["game"]), [None]), axis=1)
df["outcome"] = np.select([df["won"], known & no_winner], ["win", "no winner"], "loss")
OUTCOMES = ["win", "loss", "no winner"]


def wilson(wins, n, z=1.96):
    """95% Wilson score interval of a win rate (sensible for the small game counts of a tournament)."""
    if n == 0:
        return 0.0, 0.0
    p = wins / n
    centre = (p + z * z / (2 * n)) / (1 + z * z / n)
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / (1 + z * z / n)
    return max(0.0, centre - half), min(1.0, centre + half)


def ink_on(rgba):
    """Text color readable on a cell: white on dark fills, near-black on light ones (WCAG luminance)."""
    lin = [c / 12.92 if c <= 0.04045 else ((c + 0.055) / 1.055) ** 2.4 for c in rgba[:3]]
    luminance = 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]
    return "#ffffff" if (1.05 / (luminance + 0.05)) >= ((luminance + 0.05) / 0.0545) else "#0b0b0b"


def matrix(ax, table, counts, title, xlabel, column_label=None):
    """Win-rate matrix (agents x columns): cell color = win rate, text = rate and wins/games. Agents
    (rows, and opponent columns) are labelled by their code in their color; the figure legend names
    them."""
    column_label = column_label or (lambda c: agent_code(colors, c))
    im = ax.imshow(table.values.astype(float), cmap=WIN_RATE_CMAP, vmin=0, vmax=1, aspect="auto")
    for i, agent in enumerate(table.index):
        for j, col in enumerate(table.columns):
            rate = table.iloc[i, j]
            if pd.isna(rate):
                ax.text(j, i, "-", ha="center", va="center", fontsize=10, color=TEXT)
                continue
            wins, n = counts[(agent, col)]
            ax.text(j, i, f"{100 * rate:.0f}%\n{wins}/{n}", ha="center", va="center", fontsize=10,
                    color=ink_on(WIN_RATE_CMAP(rate)))
    ax.set_xticks(range(len(table.columns)), [column_label(c) for c in table.columns])
    ax.set_yticks(range(len(table.index)), [agent_code(colors, a) for a in table.index])
    agent_legend(ax.figure, sorted(set(table.index) | (set(table.columns) & set(colors))), colors)
    ax.set_xlabel(xlabel, fontsize=12)
    ax.set_title(title, fontsize=12)
    ax.tick_params(colors=TEXT, labelsize=10, length=0)
    color_agent_ticks(ax.get_xticklabels(), colors, table.columns)
    color_agent_ticks(ax.get_yticklabels(), colors, table.index)
    for side in ax.spines.values():
        side.set_visible(False)
    return im


def win_table(m, column):
    rates = m.pivot_table(index="agent", columns=column, values="won", aggfunc="mean")
    counts = {(a, c): (int(g["won"].sum()), len(g)) for (a, c), g in m.groupby(["agent", column])}
    return rates, counts


def outcomes_plot(m, path):
    """Win rate of every agent against every opponent, with all its games in the first column."""
    every = m.assign(opponent="all opponents")
    rates, counts = win_table(pd.concat([every, m]), "opponent")
    rates = rates[["all opponents"] + [c for c in rates.columns if c != "all opponents"]]
    fig, ax = plt.subplots(figsize=(4 + 1.8 * len(rates.columns), 1.8 + 0.8 * len(rates.index)))
    matrix(ax, rates, counts, "Win rate against each opponent", "Opponent")
    ax.axvline(0.5, color=SURFACE, linewidth=4)  # sets the overall column apart
    save(fig, path)


def ordinal(n):
    n = int(n)
    return f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"


def seats_plot(m, path):
    """Win rate by place in the turn order, whichever faction the agent played."""
    seat = m.dropna(subset=["turn_position"]).astype({"turn_position": int})
    if seat.empty:
        print(f"no turn_position recorded: {path.name} skipped")
        return
    rates, counts = win_table(seat, "turn_position")
    fig, ax = plt.subplots(figsize=(4 + 1.8 * len(rates.columns), 1.8 + 0.8 * len(rates.index)))
    matrix(ax, rates, counts, "Win rate by turn position", "Moves in the turn order",
           column_label=lambda c: f"moves {ordinal(c)}")
    save(fig, path)


def length_plot(m, path):
    agents = sorted(m["agent"].unique())
    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))
    for ax, value, label in [(axes[0], "rounds_played", "Rounds played"),
                             (axes[1], "game_time", "Game time (s)")]:
        for k, (won, offset, alpha) in enumerate([(True, -0.17, "88"), (False, 0.17, "33")]):
            data = [m.loc[(m["agent"] == a) & (m["won"] == won), value].dropna() for a in agents]
            pos = [i + 1 + offset for i, d in enumerate(data) if len(d)]
            data = [d for d in data if len(d)]
            if not data:
                continue
            bp = ax.boxplot(data, positions=pos, widths=0.3, patch_artist=True, showfliers=True,
                            medianprops={"color": "#0b0b0b", "linewidth": 2},
                            whiskerprops={"color": TEXT}, capprops={"color": TEXT},
                            flierprops={"markeredgecolor": TEXT, "markersize": 4})
            have = [a for a in agents if ((m["agent"] == a) & (m["won"] == won)).any()]
            for patch, a in zip(bp["boxes"], have):
                patch.set_facecolor(colors[a] + alpha)
                patch.set_edgecolor(colors[a])
                patch.set_linewidth(1.5)
                if won is False:
                    patch.set_hatch("//")
        ax.set_xticks([])
        ax.set_xlim(0.5, len(agents) + 0.5)
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(f"{label} (left box: won, hatched right box: lost)", fontsize=12)
        style(ax)
    agent_legend(fig, agents, colors)
    save(fig, path)


def maps_plot(m, path):
    rates, counts = win_table(m, "map")
    fig, ax = plt.subplots(figsize=(4 + 1.8 * len(rates.columns), 1.8 + 0.8 * len(rates.index)))
    matrix(ax, rates, counts, "Win rate on each map", "Map")
    save(fig, path)


for map_name, m in by_map(df, args.maps):
    out = out_root / map_name
    outcomes_plot(m, out / "game_outcomes.png")
    seats_plot(m, out / "game_seats.png")
    length_plot(m, out / "game_length.png")
    if map_name == ALL_MAPS:
        maps_plot(m, out / "game_maps.png")

    lines = [f"Game results: {map_name}\n", "=" * 60 + "\n\n"]
    if not winners:
        lines.append("No games.jsonl next to the agents folder: games without a winner are counted as\n"
                     "lost.\n\n")
    for agent, a in m.groupby("agent"):
        wins, n = int(a["won"].sum()), len(a)
        lo, hi = wilson(wins, n)
        counts = a["outcome"].value_counts()
        lines.append(f"{agent}\n" + "-" * len(agent) + "\n")
        lines.append(f"  games {n}, won {wins} ({100 * wins / n:.1f}%, 95% interval "
                     f"{100 * lo:.0f}-{100 * hi:.0f}%), "
                     + ", ".join(f"{o} {counts[o]}" for o in OUTCOMES[1:] if o in counts) + "\n")
        groups = [("opponent", "vs"), ("turn_position", "moving")] + (
            [("map", "on")] if map_name == ALL_MAPS else [])
        for column, name in groups:
            for key, g in a.groupby(column):
                key = ordinal(key) if column == "turn_position" else key
                lines.append(f"    {name} {key}: {int(g['won'].sum())}/{len(g)} won "
                             f"({100 * g['won'].mean():.1f}%)\n")
        lines.append(f"  rounds played   {stats(a['rounds_played'])}\n")
        lines.append(f"    in won games  {stats(a.loc[a['won'], 'rounds_played'])}\n")
        lines.append(f"  game time (s)   {stats(a['game_time'])}\n\n")

    path = out / "game_results_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
