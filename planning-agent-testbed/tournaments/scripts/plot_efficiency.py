"""Compares the agents' search efficiency in one tournament (every agents/*/metrics/<map>/vs-*/efficiency.csv,
one row per search, i.e. per combat move).

For each map, writes to <out>/<map>/, and for all maps combined to <out>/all-maps/:
  efficiency_iterations.png        iterations per search: distribution per agent (box; dots = each
                                   game's mean) and mean per round; searches beyond an agent's
                                   1.5 IQR whiskers are left out of both plots (not the summary)
  efficiency_action_sequences.png  action sequences visited (root-to-leaf paths of the search tree)
                                   per search: distribution per agent and mean per round
  efficiency_terr_attacked.png      territories attacked per combat move (the chosen action
                                   sequence, END and skips left out): distribution per agent and
                                   mean per round
  efficiency_tree_depth.png        size of the search tree as its depth (deepest node, in edges
                                   from the root) per search: distribution per agent and mean per
                                   round
  efficiency_summary.txt           per agent (and per opponent): games, searches, iterations, time,
                                   action sequences, root children, rollout depth, territories
                                   attacked, tree depth

The last two plots need the num_of_terr_attacked and max_tree_depth columns; efficiency.csv files
written before them only feed the other plots. Both are small integers, so no outliers are left out.

Agents without a search (e.g. tripleMind-bc, tripleMind-ppo leave num_iterations and
num_action_sequences blank) are listed in the summary but left out of the plots. A search agent that
leaves only some columns blank (e.g. oep_paper greedy: no MCTS iterations, rollouts or tree) is left out
of just those plots and summary lines.

Usage (from planning-agent-testbed/):
  python3 tournaments/scripts/plot_efficiency.py tournaments/<date>_<time> [--map MAP ...] [--out DIR]
"""
import matplotlib.pyplot as plt
import pandas as pd

from common import (ALL_MAPS, agent_colors, agent_legend, boxes_by_agent, by_map, load, mean_by_round,
                    parse_args, save, stats, style)

COLUMNS = ["player", "game", "round", "time_per_turn", "num_iterations", "root_value",
           "num_children_at_root", "max_rollout", "num_action_sequences"]

args = parse_args(__doc__.splitlines()[0]).parse_args()
out_root = args.out or args.tournament / "plots"
df = load(args.tournament, "efficiency.csv", COLUMNS)
if df.empty:
    raise SystemExit(f"no efficiency.csv under {args.tournament / 'agents'}")
for c in COLUMNS[2:]:
    df[c] = pd.to_numeric(df[c], errors="coerce")
# Newer columns, loaded apart so older efficiency.csv files still feed the plots above.
TREE_COLUMNS = ["player", "game", "round", "num_iterations", "num_action_sequences", "num_of_terr_attacked",
                "max_tree_depth"]
tree_df = load(args.tournament, "efficiency.csv", TREE_COLUMNS)
for c in TREE_COLUMNS[2:]:
    tree_df[c] = pd.to_numeric(tree_df[c], errors="coerce")
colors = agent_colors(args.tournament)


def searches(d):
    """Rows of agents that searched: an MCTS agent logs num_iterations, a greedy one only
    num_action_sequences."""
    return d[d["num_iterations"].notna() | d["num_action_sequences"].notna()]


def without_outliers(searched, value):
    """Rows within each agent's 1.5 IQR whisker range of value: the plots leave out the extreme
    searches (their min and max are in the summary), so a few long endgame searches don't stretch
    the axes."""
    def keep(g):
        q1, q3 = g[value].quantile([0.25, 0.75])
        return g[g[value].between(q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1))]
    return searched.groupby("agent", group_keys=False).apply(keep)


def compare(searched, value, label, title, path, drop_outliers=True):
    searched = searched.dropna(subset=[value])
    if searched.empty:
        return
    if drop_outliers:
        searched = without_outliers(searched, value)
    note = "outliers left out" if drop_outliers else "all searches"
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5), gridspec_kw={"width_ratios": [2, 3]})
    agents = sorted(searched["agent"].unique())
    boxes_by_agent(ax1, {a: searched.loc[searched["agent"] == a, value] for a in agents}, colors,
                   {a: searched[searched["agent"] == a].groupby(["map", "game"])[value].mean() for a in agents})
    ax1.set_ylabel(label, fontsize=12)
    ax1.set_title(f"{title} per search (dots: each game's mean; {note})", fontsize=12)
    mean_by_round(ax2, searched, value, colors)
    ax2.set_xlabel("Round", fontsize=12)
    ax2.set_ylabel(f"Mean {label.lower()}", fontsize=12)
    ax2.set_title(f"Mean {title.lower()} by round (band: +/- std across searches)", fontsize=12)
    for ax in (ax1, ax2):
        style(ax)
    agent_legend(fig, agents, colors)
    save(fig, path)


for map_name, m in by_map(df, args.maps):
    out = out_root / map_name
    searched = searches(m)
    if not searched.empty:
        compare(searched, "num_iterations", "Iterations", "Iterations",
                out / "efficiency_iterations.png")
        compare(searched, "num_action_sequences",
                "Action sequences", "Action sequences visited", out / "efficiency_action_sequences.png")
    tree = tree_df[tree_df["map"] == map_name] if map_name != ALL_MAPS else tree_df
    tree = searches(tree)
    if not tree.empty:
        compare(tree, "num_of_terr_attacked",
                "Territories attacked", "Territories attacked", out / "efficiency_terr_attacked.png",
                drop_outliers=False)
        compare(tree, "max_tree_depth", "Tree depth",
                "Search tree depth", out / "efficiency_tree_depth.png", drop_outliers=False)

    lines = [f"Search efficiency: {map_name}\n", "=" * 60 + "\n\n"]
    for agent, a in m.groupby("agent"):
        lines.append(f"{agent}\n" + "-" * len(agent) + "\n")
        for opponent, o in [("all opponents", a), *a.groupby("opponent")]:
            s = searches(o)
            games = len(o[["map", "game"]].drop_duplicates())
            lines.append(f"  vs {opponent}: {games} games, {len(o)} searches\n")
            lines.append(f"    time per turn (s)      {stats(o['time_per_turn'])}\n")
            if s.empty:
                lines.append("    no search (num_iterations and num_action_sequences blank)\n")
                continue
            lines.append(f"    iterations             {stats(s['num_iterations'])}\n")
            lines.append(f"    action sequences       {stats(s['num_action_sequences'])}\n")
            lines.append(f"    children at root       {stats(s['num_children_at_root'])}\n")
            lines.append(f"    max rollout            {stats(s['max_rollout'])}\n")
            lines.append(f"    root value             {stats(s['root_value'])}\n")
            t = tree[(tree["agent"] == agent) & ((tree["opponent"] == opponent) |
                                                 (opponent == "all opponents"))]
            if not t.empty:
                lines.append(f"    territories attacked   {stats(t['num_of_terr_attacked'])}\n")
                lines.append(f"    tree depth             {stats(t['max_tree_depth'])}\n")
        lines.append("\n")

    searched = searches(m)
    if not searched.empty:
        for value, label in [("num_iterations", "Mean iterations"),
                             ("num_action_sequences", "Mean action sequences")]:
            table = searched.pivot_table(index="round", columns="agent", values=value, aggfunc="mean")
            lines.append(f"{label} by round\n")
            lines.append(table.round(1).to_string(na_rep="-") + "\n\n")
    for value, label in [("num_of_terr_attacked", "Mean territories attacked"),
                         ("max_tree_depth", "Mean tree depth")]:
        if tree[value].notna().any():
            table = tree.pivot_table(index="round", columns="agent", values=value, aggfunc="mean")
            lines.append(f"{label} by round\n")
            lines.append(table.round(1).to_string(na_rep="-") + "\n\n")

    path = out / "efficiency_summary.txt"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(lines))
    print(f"Summary written to {path}")
