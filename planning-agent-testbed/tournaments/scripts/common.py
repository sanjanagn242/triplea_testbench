"""Shared loading and styling for the tournament comparison plots.

A tournament keeps every agent's metrics under <tournament>/agents/<agent id>/metrics/<map>/vs-<opponent>/
(see ../README.md). load() reads one CSV name from all of them into a single DataFrame with three
extra columns, agent, map and opponent, so every plot can compare agents on the same map.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.ticker import MaxNLocator
import pandas as pd

# Reference categorical palette (fixed order); an agent keeps its color in every plot of a run.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948",
          "#14a6c0", "#8c5a2b", "#7d7c78", "#a3a800"]
WIN_COLOR, LOSS_COLOR = SERIES[0], SERIES[1]
TEXT = "#52514e"
GRID = "#e4e3df"
SURFACE = "#fcfcfb"
ALL_MAPS = "all-maps"  # output folder of the plots combining every map


def parse_args(description):
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("tournament", type=Path,
                        help="tournament folder, e.g. planning-agent-testbed/tournaments/<date>_<time>")
    parser.add_argument("--out", type=Path,
                        help="where to write plots and summaries (default <tournament>/plots)")
    parser.add_argument("--map", action="append", dest="maps", metavar="MAP",
                        help=f"plot only this map (its folder name, e.g. minimap); {ALL_MAPS} for "
                             f"all maps combined; repeatable (default: every map and {ALL_MAPS})")
    return parser


def load(tournament: Path, filename: str, columns, optional=()) -> pd.DataFrame:
    """Rows of every agents/<agent>/metrics/<map>/vs-<opponent>/<filename> with the given columns,
    plus agent, map and opponent. Files missing a column (written before it existed) are skipped
    with a note; a missing optional column is blank (NaN) instead."""
    frames = []
    columns = [*columns, *optional]
    for path in sorted((tournament / "agents").glob(f"*/metrics/*/vs-*/{filename}")):
        df = pd.read_csv(path)
        missing = [c for c in columns if c not in df.columns and c not in optional]
        if missing:
            print(f"skipping {path}: no column(s) {', '.join(missing)}")
            continue
        if df.empty:
            continue
        opponent_dir, map_dir, agent_dir = path.parent, path.parent.parent, path.parents[3]
        df = df.reindex(columns=list(columns))
        df["agent"] = agent_dir.name
        df["map"] = map_dir.name
        df["opponent"] = opponent_dir.name[len("vs-"):]
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=[*columns, "agent", "map", "opponent"])
    return pd.concat(frames, ignore_index=True)


# Columns of the rollout_stats*.csv files of earlier tournaments, one row per rollout; the agents
# now write one row per search (ROLLOUT_STATS_HEADER of their helper.py).
PER_ROLLOUT_COLUMNS = ["player", "game", "round", "iteration", "rollout_depth_reached", "reward",
                       "is_terminal"]
ROLLOUT_TOTALS = ["rollouts", "reward_sum", "reward_sq_sum", "reward_m2", "terminal_wins",
                  "terminal_losses", "depth_sum", "depth_sq_sum"]


def rollout_searches(path: Path):
    """(searches, depth counts) of one rollout_stats*.csv. searches has one row per search, in file
    order: player, game, round, phase (None in a file of one row per rollout), rollouts,
    reward_mean, reward_m2 (sum of the rewards' squared deviations from their mean), reward_min,
    reward_max, terminal_wins, terminal_losses, depth_sum and depth_sq_sum (of the rounds each
    rollout advanced). depth counts is depth -> rollouts over the file. A file of one row per
    rollout (earlier tournaments) is reduced to the same: a search is consecutive rows of one player
    and game until the iteration number starts over, and its round is the lowest of its rows.
    (None, None) for an empty file or one of neither layout."""
    columns = list(pd.read_csv(path, nrows=0).columns)
    if "rollouts" in columns:
        s = pd.read_csv(path)
        if s.empty:
            return None, None
        pairs = s["depth_counts"].astype(str).str.split(";").explode().str.split(":", expand=True)
        depth, count = pd.to_numeric(pairs[0]), pd.to_numeric(pairs[1])
        s["depth_sum"] = (depth * count).groupby(level=0).sum()
        s["depth_sq_sum"] = (depth * depth * count).groupby(level=0).sum()
        return s.drop(columns="depth_counts"), count.groupby(depth).sum()

    missing = [c for c in PER_ROLLOUT_COLUMNS if c not in columns]
    if missing:
        print(f"skipping {path}: no column(s) {', '.join(missing)}")
        return None, None
    r = pd.read_csv(path, usecols=PER_ROLLOUT_COLUMNS)
    if r.empty:
        return None, None
    for c in PER_ROLLOUT_COLUMNS[1:]:
        r[c] = pd.to_numeric(r[c], errors="coerce")
    new = ((r["iteration"].diff() <= 0) | (r["game"] != r["game"].shift())
           | (r["player"] != r["player"].shift()))
    terminal = r["is_terminal"] == 1
    r = r.assign(search=new.cumsum(), win=terminal & (r["reward"] > 0),
                 loss=terminal & (r["reward"] <= 0), depth_sq=r["rollout_depth_reached"] ** 2)
    s = r.groupby("search").agg(
        player=("player", "first"),
        game=("game", "first"),
        round=("round", "min"),
        rollouts=("reward", "count"),
        reward_mean=("reward", "mean"),
        reward_var=("reward", "var"),
        reward_min=("reward", "min"),
        reward_max=("reward", "max"),
        terminal_wins=("win", "sum"),
        terminal_losses=("loss", "sum"),
        depth_sum=("rollout_depth_reached", "sum"),
        depth_sq_sum=("depth_sq", "sum"),
    ).reset_index(drop=True)
    s["reward_m2"] = s.pop("reward_var").fillna(0) * (s["rollouts"] - 1)
    s["phase"] = None
    return s, r["rollout_depth_reached"].value_counts()


def combine_rollouts(df, keys):
    """Rows of rollout totals (rollout_searches() rows, or rows this returned) combined per keys:
    rollouts, reward_mean, reward_m2 (pooled exactly: the rows' m2 plus their means' spread),
    reward_min, reward_max, terminal wins and losses, depth_sum, depth_sq_sum, plus searches (rows
    combined)."""
    d = df.assign(reward_sum=df["reward_mean"] * df["rollouts"],
                  reward_sq_sum=df["reward_mean"] ** 2 * df["rollouts"])
    out = d.groupby(keys).agg(**{c: (c, "sum") for c in ROLLOUT_TOTALS},
                              reward_min=("reward_min", "min"), reward_max=("reward_max", "max"),
                              searches=("rollouts", "size")).reset_index()
    out["reward_mean"] = out["reward_sum"] / out["rollouts"]
    between = out["reward_sq_sum"] - out["reward_sum"] ** 2 / out["rollouts"]
    out["reward_m2"] = out["reward_m2"] + between.clip(lower=0)
    return out.drop(columns=["reward_sum", "reward_sq_sum"])


def rollout_spread(df):
    """(reward variance, reward std, mean depth, depth std) of rows of rollout totals; NaN variance
    and std for a row of one rollout."""
    n = df["rollouts"]
    var = df["reward_m2"] / (n - 1)
    depth_var = (df["depth_sq_sum"] - df["depth_sum"] ** 2 / n).clip(lower=0) / (n - 1)
    return var, np.sqrt(var), df["depth_sum"] / n, np.sqrt(depth_var)


def tournament_agents(tournament: Path):
    """Ids of every agent with a metrics folder in the tournament, sorted."""
    return sorted(p.parent.name for p in (tournament / "agents").glob("*/metrics") if p.is_dir())


def by_map(df, maps=None):
    """(folder name, rows) for every map, then ("all-maps", every row) combining the maps, so each
    script writes per-map plots to <out>/<map>/ and map-agnostic ones to <out>/all-maps/. maps
    (--map) keeps only the named ones; all-maps still combines every map of the tournament."""
    present = sorted(df["map"].unique())
    for name in maps or [*present, ALL_MAPS]:
        if name == ALL_MAPS:
            yield ALL_MAPS, df
        elif name in present:
            yield name, df[df["map"] == name]
        else:
            print(f"no data for map {name!r} (maps: {', '.join(present)}, or {ALL_MAPS})")


def agent_colors(tournament: Path):
    """Agent id -> color for every agent of the tournament (by sorted id), so an agent has the same
    color in every plot of the run, whichever agents a plot shows. Past twelve agents the colors
    repeat; the codes of agent_code() still tell them apart."""
    return {a: SERIES[i % len(SERIES)] for i, a in enumerate(tournament_agents(tournament))}


def style(ax, grid_axis="y"):
    ax.set_facecolor(SURFACE)
    ax.tick_params(colors=TEXT, labelsize=10)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    if grid_axis:
        getattr(ax, f"{grid_axis}axis").grid(True, color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    ax.title.set_color("#0b0b0b")
    ax.xaxis.label.set_color(TEXT)
    ax.yaxis.label.set_color(TEXT)


def agent_code(colors, agent):
    """Short id of an agent (A1, A2, ... in the order of agent_colors), the same in every plot of the
    run; for axes where the color can't identify the agent (win-rate matrices, win/loss bars).
    Anything not an agent of the run (e.g. "all opponents") is returned as is."""
    return f"A{list(colors).index(agent) + 1}" if agent in colors else str(agent)


def color_agent_ticks(labels, colors, agents):
    """Color tick labels (as set from agent_code) in their agent's color."""
    for label, a in zip(labels, agents):
        if a in colors:
            label.set_color(colors[a])
            label.set_fontweight("bold")


def agent_legend(fig, agents, colors, extra=()):
    """One legend for the whole figure, right of the plots: a color and code (agent_code) per agent,
    then any extra handles (e.g. line styles). Agent names are long, so the axes leave them out and
    the color identifies the agent; save() widens the image to fit the legend."""
    handles = [Line2D([], [], color=colors[a], linewidth=6, label=f"{agent_code(colors, a)}  {a}")
               for a in agents] + list(extra)
    fig.legend(handles=handles, loc="center left", bbox_to_anchor=(1.0, 0.5), frameon=False,
               fontsize=10, labelcolor=TEXT, handlelength=2.5)


def boxes_by_agent(ax, values_by_agent, colors, points_by_agent=None):
    """One box per agent (the per-decision distribution), colored by agent (see agent_legend), and
    optionally one dot per game (that game's mean) jittered over it."""
    agents = list(values_by_agent)
    bp = ax.boxplot([values_by_agent[a] for a in agents], widths=0.5, patch_artist=True,
                    showfliers=False, medianprops={"color": "#0b0b0b", "linewidth": 2},
                    whiskerprops={"color": TEXT}, capprops={"color": TEXT})
    for patch, a in zip(bp["boxes"], agents):
        patch.set_facecolor(colors[a] + "55")
        patch.set_edgecolor(colors[a])
        patch.set_linewidth(1.5)
    if points_by_agent:
        rng = np.random.default_rng(0)
        for i, a in enumerate(agents, 1):
            pts = list(points_by_agent.get(a, []))
            if pts:
                ax.scatter(i + rng.uniform(-0.15, 0.15, len(pts)), pts, s=36, color=colors[a],
                           edgecolor=SURFACE, linewidth=2, zorder=3)
    ax.set_xticks([])


def mean_by_round(ax, df, value, colors):
    """Mean of value per decision round, one line per agent (see agent_legend), with a band of
    +/- the standard deviation across that round's rows (turns)."""
    for agent, g in df.groupby("agent"):
        by_round = g.groupby("round")[value]
        series, sd = by_round.mean(), by_round.std().fillna(0)  # one turn in a round: no spread
        ax.fill_between(series.index, series - sd, series + sd, color=colors[agent], alpha=0.12,
                        linewidth=0)
        ax.plot(series.index, series.values, color=colors[agent], linewidth=2, label=agent)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))


def save(fig, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.patch.set_facecolor(SURFACE)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight", pad_inches=0.2)  # keeps a legend right of the plots
    plt.close(fig)
    print(f"Plot written to {path}")


def _stats_line(mean, median, p25, p75, lo, hi) -> str:
    f = lambda v: f"{v:.3g}" if abs(v) < 1000 else f"{v:.0f}"
    return f"mean {f(mean)}, median {f(median)}, p25-p75 {f(p25)} to {f(p75)}, min {f(lo)}, max {f(hi)}"


def stats(series) -> str:
    """mean / median / p25-p75 / min-max of a numeric series, for the summaries."""
    s = pd.to_numeric(series, errors="coerce").dropna()
    if s.empty:
        return "-"
    return _stats_line(s.mean(), s.median(), s.quantile(0.25), s.quantile(0.75), s.min(), s.max())


def stats_from_counts(counts) -> str:
    """stats() of the values a value -> count series stands for (e.g. summed value_counts() of
    files too big to hold together); quantiles are the lower of two values at a tie."""
    counts = counts[counts > 0].sort_index()
    if counts.empty:
        return "-"
    values, cum = counts.index.to_numpy(dtype=float), counts.cumsum().to_numpy()
    q = lambda p: values[np.searchsorted(cum, p * (cum[-1] - 1), side="right")]
    return _stats_line((values * counts.to_numpy()).sum() / cum[-1], q(0.5), q(0.25), q(0.75),
                       values[0], values[-1])
