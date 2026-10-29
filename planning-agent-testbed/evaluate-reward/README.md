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
both directions. It keeps the observation from each `turn_request` and, on `game_end`, labels them
with the faction's reward and writes `records/<faction>-game-<n>.json`.

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
`--raw`, it also keeps the first raw observation seen of it (`raw_state`). The records folders keep
every raw observation either way.

The database is updated in place. Games that are already counted, by their `game_id`
(`<run>/<faction>-game-<n>`), are skipped, so merging a records folder again does not count it
twice.

## Usage

```bash
python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py --games 20
python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py --games 100 --num-batches 4 \
    --env TRIPLEA_MCTS_SECONDS=1.0
```

The default is `tripleMind-mcts` against itself on `capture-the-flag-diag`, with a round limit of
100. Every run merges its games into `database.json` in this folder (or the file given by
`--database`), so the statistics build up across runs. A run's own files go in `runs/<timestamp>/`:

- `evaluate-reward-selfplay.json`: the tournament spec.
- `records/`: one file per faction and won or lost game, holding its states, result and final state.
- `tournament/`: the tournament's `games.jsonl`, `results.txt` and agent metrics.

To merge existing records, or to build a fresh database from them:

```bash
python3 planning-agent-testbed/evaluate-reward/build_database.py runs/<a>/records runs/<b>/records --out fresh.json
```

The agent plays under the ID `<agent>-evalrec`, so its metrics stay apart from normal tournaments.
The variant's manifest in `agents/generated-variants/` names one run's records folder, so run one
collection at a time.
