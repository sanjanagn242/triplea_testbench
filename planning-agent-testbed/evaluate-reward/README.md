# Evaluate Reward

Builds a database of (state, round, reward) entries, where the reward comes from the environment
at the end of the game rather than from the agent. It is the true outcome for a state reached in
that round, which is what the agents' reward functions (for example `evaluate_state` in
`agents/tripleMind/mcts.py`) try to estimate.

So far this folder only builds the database. Comparing it with the agents' reward functions comes
later.

## How states are recorded without changing the agents

Every agent's `run-agent.sh` ends with `exec "${TRIPLEA_TESTBENCH_PYTHON:-python3}" -u agent.py`.
`collect_selfplay.py` plays the agent as a tournament variant that sets `TRIPLEA_TESTBENCH_PYTHON`
to [tap.py](tap.py). The tap starts the real interpreter and relays the protocol lines unchanged in
both directions. It keeps the observation from each `turn_request`, reduced to its abstract state
(see below). On `game_end` it labels them with the faction's reward and writes
`records/<faction>-game-<n>.json`. It writes this file before passing `game_end` on, because the
engine stops the agent as soon as it answers. Set `EVAL_REWARD_RAW=1` to record the full
observations instead, which are about 10 to 20 times larger.

| Outcome for the faction | Reward    |
|-------------------------|-----------|
| won                     | 1         |
| lost                    | -1        |
| no winner (round limit) | discarded |

By default only `combatMove` states are kept: the state before the agent's combat moves, which is
also the root of its search tree. `--phases all` keeps every decision phase.

A game that is stopped before `game_end` writes nothing.

## Running statistics per state

The database has one entry per distinct (faction, round, state). When the same state comes up
again in the same round, in this game or another one, its statistics are updated and no new record
is added.

Each observation is first reduced to an abstract state ([abstract_state.py](abstract_state.py)), and
two observations are the same state when their abstract states are equal. The abstract state keeps
only what changes during a game:

```json
{"round": 3, "step": "Combat Move", "active": "Russians",
 "players": {"Germans": {"PUs": 26},
             "Russians": {"PUs": 1, "reserve": {"infantry": 4, "armour": 1, "artillery": 2}}},
 "territories": {"Flag": {"owner": null, "units": {"Neutral": {"infantry": 3}}},
                 "TopBar": {"owner": "Russians", "units": {"Russians": {"infantry": 1}}}}}
```

- Units are counted per owner and type. Unit IDs, which are random in every game, are left out.
- A unit with hits or damage is counted as `<type>@<hits>h<damage>d`.
- Movement is left out, because states are recorded at the start of a phase.
- Static map and unit-type data (neighbours, production, attack, defense) is left out.
- Territories with no owner and no units are left out.

A Capture The Flag state goes from about 16 KB to 1.4 KB, and abstracting and hashing one takes
about 20 µs.

Each entry holds `count`, `wins`, `losses`, `reward_mean`, `reward_var` (a running mean and variance,
by Welford's method), `rounds_to_end_mean`, the `games` it came from, and the abstract `state`. With
`--raw` and records made with `EVAL_REWARD_RAW=1`, it also keeps the first raw observation seen of
it (`raw_state`).

The database is updated in place. Games that are already counted, by their `game_id`
(`<run>/<faction>-game-<n>`), are skipped, so merging a records folder again does not count it
twice.

## Usage

Edit [selfplay.json](selfplay.json) to choose which agents to collect samples from, then run:

```bash
python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py
python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py --games 100 --num-batches 4
python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py --config other-agents.json
```

```json
{
  "agents": [
    "tripleMind-mcts",
    {"agent": "tripleMind-mcts", "id": "tm-fast-evalrec", "env": {"TRIPLEA_MCTS_SECONDS": "0.3"}}
  ],
  "games": 10,
  "gameFolder": "capture-the-flag-diag",
  "map": {"name": "Capture The Flag", "gameName": "Capture The Flag", "aliases": ["capture_the_flag", "CTF"]},
  "roundLimit": 100,
  "phases": "combatMove",
  "env": {},
  "gui": false
}
```

- `agents`: each agent plays only against itself, in its own tournament, `games` games each.
  - A plain string is an agent ID.
  - An object adds settings for that agent only.
  - It plays as a variant with the ID `<agent>-evalrec`, so its metrics stay apart from normal
    tournaments. If the same agent is listed twice, give each entry its own `id`.
- `env`: settings for every agent.
- `gui`: `false` (the default) plays without a game window; `true` shows it.
- `--agents`, `--games`, `--round-limit`, `--phases` and `--env` override the config.

Every run merges its games into `database.json` in this folder (or the file given by `--database`),
so the statistics build up across runs. A run's own files go in `runs/<timestamp>/`:

- `selfplay.json`: a copy of the config used.
- `records/`: one file per agent, faction and won or lost game (`<agent id>-<faction>-game-<n>.json`),
  holding its states, result and final state.
- `tournament/<agent id>/`: that agent's tournament spec, `games.jsonl` and `results.txt`. The
  agent still writes its metric CSVs, because that can't be turned off without changing the agent.
  The runner deletes them (`agents/`) after the run unless you pass `--keep-metrics`.

To merge existing records, or to build a fresh database from them:

```bash
python3 planning-agent-testbed/evaluate-reward/build_database.py runs/<a>/records runs/<b>/records --out fresh.json
```

The variants' manifests in `agents/generated-variants/` name one run's records folder, so run one
collection at a time.

## Comparing an agent's reward function

[compare_reward.py](compare_reward.py) runs an agent's own reward function on every state in the
database and compares it with the environment's reward, round by round. The default is tripleMind's
`mcts.evaluate_state`.

```bash
python3 planning-agent-testbed/evaluate-reward/compare_reward.py
python3 planning-agent-testbed/evaluate-reward/compare_reward.py --agent-dir agents/tripleMind_hybrids/tripleMind-with-opp-nodes --min-count 2
```

The agent is not changed. The script imports the function from the agent's folder and calls it on
each state, rebuilt as the observation the agent would have been shown. The abstract state keeps
everything `evaluate_state` reads: territory owners and unit types and owners. Production and unit
costs come from the map XML. On the 93 states recorded in full, the rebuilt and full observations
gave identical values.

The two rewards are on different scales:

- The environment's reward is 1 or -1 per game, so a state's mean is in [-1, 1].
- `evaluate_state` gives `score - 0.5`, in [-0.5, 0.5], for a state that hasn't ended the game.

The script multiplies the agent's value by `--scale` (default 2) and clips it to [-1, 1]. Every
number is per sample, meaning a state counted once for each game it came up in, from the faction's
own point of view.

It writes these files to `comparisons/<agent folder>/`:

- `reward_vs_round.png`:
  - Top row: per faction, the mean actual reward and the mean scaled agent value by round.
  - Bottom row: the mean absolute error `|agent - outcome|`, and the share of samples where the
    agent's sign matches the winner (0 counts as half).
- `rounds.csv`: the numbers behind the plot, one row per faction and round.
- `states.csv`: one row per database state, with its count, wins, losses, actual mean and the
  agent's raw and scaled values.
