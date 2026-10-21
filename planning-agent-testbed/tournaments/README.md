# Capture The Flag (diag) tournament

This folder holds the configurations of one tournament on the diag Capture The Flag map. Each
`*.json` here is one matchup. `tools/tournament.py` runs all of them in order through the engine,
writes the win rates to a text file and leaves each agent's metrics in a folder named after its
opponent.

## Run it

From the repository root:

```bash
python3 planning-agent-testbed/tools/tournament.py \
  planning-agent-testbed/games/capture-the-flag-diag/config/tournament --games 10
```

This plays 10 games of every configuration. A game window opens for each configuration and closes
by itself when its games are done; leave it running until the script prints the report. Add
`--no-gui` to play without the game window.

### Options

| Option | Effect |
|---|---|
| `--games N` | Games per configuration. Written into every file here before starting. Without it each file keeps its own `games`. |
| `--first-game N` | Number of the tournament's first game (default 1). |
| `--clear-outputs` | Also empty the agents' records and logs before starting. |
| `--no-gui` / `--gui` | Hide or show the game window for every configuration, overriding each file's `showGame` (default: shown). |
| `--out DIR` | Write the report somewhere other than `planning-agent-testbed/tournaments/<timestamp>/`. |
| `--report-only FILE` | Rebuild the report from an existing `games.jsonl` without playing. |

## What happens when you start

1. **The configurations are renumbered, in these files.** Each file's `firstGameNumber` continues
   from the previous one, so game numbers never repeat. With `--games 10`:

   | File | Games |
   |---|---|
   | `1.tripleMind-hardAI.json` | 1–10 |
   | `2.hardAI-tripleMind.json` | 11–20 |
   | `3.no_reward-hardAI.json` | 21–30 |
   | `4.hardAI-no_reward.json` | 31–40 |
   | `5.no_reward-tripleMind.json` | 41–50 |
   | `6.tripleMind-no_reward.json` | 51–60 |

2. **The configurations are played** in name order. Numbers are compared as numbers, so `10.x`
   comes after `9.x`.

`clearAgentOutputs` in a file is ignored during a tournament, so a later configuration cannot wipe
what an earlier one wrote.

## Results

Each run gets its own folder, `planning-agent-testbed/tournaments/<date>_<time>/`:

- `results.txt`: the win rates, rewritten after every configuration, so it is useful even if you
  stop the run early. It has three parts:
  - **Per configuration:** wins and win rate of each faction, plus games with no winner.
  - **Head-to-head:** every pair of agents over all configurations and both seats. For example,
    `5.no_reward-tripleMind` and `6.tripleMind-no_reward` are combined into one
    "tripleMind-mcts vs tripleMind-mcts-no-reward-heuristic" result, with a per-seat breakdown.
  - **Per agent:** games, wins and win rate over the whole tournament.
- `games.jsonl`: one line per game (agents, winners, rounds, time), used to build the report.
- `configs.txt`: the configurations that were run.
- `agents/<agent id>/metrics/`: each agent's metrics from this tournament (see below).

A game that reaches `roundLimit` ends with no winner and counts as a game the agent did not win.

## Metrics

During a tournament every agent writes its metrics to the tournament folder, not to its own
`metrics/`: one folder per agent, then per map and per agent it played against:

```text
tournaments/<date>_<time>/agents/
  tripleMind-mcts/metrics/capture-the-flag-diag/
    vs-Hard-AI/                              # configurations 1 and 2
      game_results.csv
      efficiency.csv
      rollout_stats.csv
    vs-tripleMind-mcts-no-reward-heuristic/  # configurations 5 and 6
  tripleMind-mcts-no-reward-heuristic/metrics/capture-the-flag-diag/
    vs-Hard-AI/                              # configurations 3 and 4
    vs-tripleMind-mcts/                      # configurations 5 and 6
```

Agents without metrics (built-in players such as `Hard (AI)`, `does-nothing-python`) get no folder.
Both seats of a matchup share one folder. The `player` column (and `turn_position` in
`game_results.csv`) tells the sides apart. Game records (`records/<game>/vs-<opponent>/`), logs and
trees are still written in the agent's own folder. Games played outside a tournament still write
metrics to the agent's own `metrics/`.

To plot one agent's tournament metrics, pass the folder to its `scripts/generate_plots.sh`, for
example `agents/tripleMind/scripts/generate_plots.sh planning-agent-testbed/tournaments/<date>_<time>/agents/tripleMind-mcts/metrics`.
It makes a set of plots for each `vs-*` folder. Without an argument it plots the agent's own
`metrics/`.

### Comparing agents

`scripts/` compares every agent of one tournament. Each script writes one set of plots per map to
`tournaments/<date>_<time>/plots/<map>/` (where does an agent play better?) and one set with all maps
combined to `plots/all-maps/` (how does it play overall?):

```bash
planning-agent-testbed/tournaments/scripts/generate_plots.sh planning-agent-testbed/tournaments/<date>_<time>
```

By default it plots every map and `all-maps`. To plot only some, add `--map` (repeatable) with a
map folder name, or `all-maps` for every map combined:

```bash
planning-agent-testbed/tournaments/scripts/generate_plots.sh planning-agent-testbed/tournaments/<date>_<time> \
  --map capture-the-flag-diag --map all-maps
```

The same `--map` (and `--out DIR`) options work on each script on its own.

- `plot_game_results.py`: win rate against every opponent (and all of them) and in every seat, and rounds played and game time of
  won and lost games, plus `game_results_summary.txt`. `all-maps/` also has `game_maps.png`, the
  win rate of every agent on every map. Losses and games without a winner are told apart with the
  tournament's `games.jsonl`; without it a game without a winner counts as lost.
- `plot_efficiency.py`: iterations per search and action sequences visited (root-to-leaf paths of
  the search tree) per search, per agent (box; dots are each game's mean) and by round, plus
  `efficiency_summary.txt`.
- `plot_rollout_stats.py`: % of rollouts reaching a terminal state (win / loss) per agent and by
  round; states sampled per turn (the sum of `rollout_depth_reached` over a search's rollouts) per
  agent and by round; `rollout_reward.png`, every agent in one plot (one line each, both seats together): the mean reward
  per turn by round with a band of the reward spread within a turn (a wide band means a noisy
  reward); `rollout_turn_stats.csv`, one row per turn (`avg_reward`, `var_reward`, `std_reward`,
  `avg_depth_reached`, `depth_reached_std`, `n_visits`); plus `rollout_stats_summary.txt`.

Agents without a search (`tripleMind-bc`, `tripleMind-ppo`) appear in the summaries but not in the
search and rollout plots.

## Agent variants (same agent, different settings)

To compare settings of one agent, list it once per setting in a spec's `agents`, as the agent ID
followed by `KEY=VALUE` environment variables (see `specs/2p_maps.json`):

```json
"agents": [
  "tripleMind-mcts TRIPLEA_MCTS_ROLLOUT=2",
  "tripleMind-mcts TRIPLEA_MCTS_ROLLOUT=10 TRIPLEA_MCTS_SECONDS=2",
  {"agent": "tripleMind-mcts", "id": "tm-long-search", "env": {"TRIPLEA_MCTS_SECONDS": "5"}},
  "Hard (AI)"
]
```

```bash
python3 planning-agent-testbed/tools/tournament.py --spec planning-agent-testbed/tournaments/specs/2p_maps.json
# or override the spec's agents (one quoted argument per variant):
python3 planning-agent-testbed/tools/tournament.py --spec planning-agent-testbed/tournaments/specs/2p_maps.json \
  --agents "tripleMind-mcts TRIPLEA_MCTS_ROLLOUT=2" "tripleMind-mcts TRIPLEA_MCTS_ROLLOUT=10" "Hard (AI)"
```

Each variant is a separate agent. Its ID is the agent's ID plus its settings (a leading `TRIPLEA_`
dropped), e.g. `tripleMind-mcts-MCTS_ROLLOUT-10`, or the object form's `id`. That ID names its
metrics folder (`agents/tripleMind-mcts-MCTS_ROLLOUT-10/metrics/`), its opponents' `vs-` folders and
its lines in `results.txt` and the comparison plots. The tournament writes an `agent.json` for each
variant to `agents/generated-variants/<id>/` (ignored by git); its command sets the variables and
runs the agent's own `run-agent.sh`.

A setting only takes effect if the agent's run script reads it from the environment
(`export VAR="${VAR:-default}"`, as the tripleMind scripts do for the `TRIPLEA_MCTS_*` settings). The
tournament warns about a variable the script sets itself, such as `TRIPLEA_AGENT_LOG`. Game records
are still written to the base agent's own folder, shared by its variants.

## Adding or changing a matchup

- Copy one of the files and give it the next number, for example `7.bc-hardAI.json`.
- Set the agents by ID under `agents`. Agent IDs are the `id` in each agent's `agent.json`
  (`tripleMind-mcts`, `tripleMind-mcts-no-reward-heuristic`, `tripleMind-bc`, `tripleMind-ppo`,
  ...). Built-in players use their label, for example `"Hard (AI)"`.
- You do not need to set `firstGameNumber`; the tournament sets it.
- Keep the configurations in this folder (or another folder under `capture-the-flag-diag/config/`)
  so they use the diag map.
- To leave a configuration out, move it out of this folder. Every `*.json` here is run.
