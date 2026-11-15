# OEP paper: Greedy Action

Agent ID `greedy-action`. The Greedy Action baseline of Justesen, Mahlmann, Risi and Togelius,
*Playing Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search* (IEEE
Transactions on Games, 2018, Section V), applied to the TripleA purchase, combat move and
non-combat move. It is [greedy_action_combat](../greedy_action_combat/README.md) extended to the
other two phases.
The actions are those of every all-phase OEP agent (greedy_action, greedy_turn, vanilla_mcts,
ne_mcts, bb_mcts, oep), which differ only in how they select among them: `phases.py` holds them,
and it and the generators it uses (`action_generators.py`, its Cython build
`action_generators_cy.pyx` / `scripts/build_cython.sh`, `generators.py`,
`noncombat_generators.py`, `map_graph.py`, `helper.py`, `fast_battle.py`) are byte-identical
copies in each agent; [`../check-shared-files.sh`](../check-shared-files.sh) fails when they
drift. `run-agent.sh` rebuilds the `.so` when the `.pyx` is newer (or missing; a failed build
falls back to Python); `TRIPLEMIND_PURE_PYTHON=1` forces the Python generators. Both give the
same results.

## Algorithm

A one-ply search over the phase's actions, picking the one whose resulting state scores highest
under the heuristic, repeated until ending the phase scores best. It runs once per phase, so 3
times per turn. One action is:

| Phase | One action (`phases.py`) | Sorting heuristic (highest-rated first) |
|---|---|---|
| purchase | buy 1 unit of a candidate rule (land combat units for PUs) | attack + defence per PU; END last |
| combat move | attack one enemy territory with every unmoved unit that can reach it | attacks with strength > 1.5 × defence in TripleMind priority order, END, the other attacks |
| non-combat move | for the most important territory short of cover: a reinforcement set (strongest first) or its skip; END (with the positioning moves) once none is left | generation order |

```
plan = state at the phase's start (non-combat: after the forced air landing)
repeat:
    candidates = the actions offered after plan, in the sorting heuristic's order (END included
                 where offered); only END: play it, stop          (reached_end_of_turn = 1)
    score every candidate: value = evaluate_state(plan + candidate, then END [+ this turn's
        battles in the combat move]) in [-1, 1]
        combat move with dice: sampled round-robin until the level's time share is used (same
        dice seed for all candidates in a sweep); otherwise one sweep is exact
    best = the first candidate (sorting heuristic's order) with the highest mean value
    play best; if best is END: stop                                (reached_end_of_turn = 1)
out of time before END: the highest-rated actions until END
```

- **Heuristic**: TripleMind's `evaluate_state` unchanged (1 / -1 won / lost, else
  2 * (0.5 TUV share + 0.5 income share) - 1). It counts neither units in hand (just purchased)
  nor positions, so purchase and non-combat candidates almost always tie and the sorting
  heuristic decides. Scoring a whole turn instead is greedy turn's job.
- **Time budget**: `TRIPLEA_GREEDY_SECONDS` per phase decision (default 1.0). In the combat move
  with dice each level gets `remaining time / Phase.levels_left` (the attacks still offered);
  the other levels take one sweep each. If the budget runs out before END, the plan is completed
  with the highest-rated actions (`reached_end_of_turn = 0`).
- **Battle resolution**: `TRIPLEA_BATTLE_RESOLUTION` (`run-agent.sh`) picks how the search's battles are fought: `dice` (default: the forward model's dice, default casualties, no retreat) or `fast` (the Fast AI's most likely outcome, no dice; `fast_battle.py`). With `fast` one sample is exact, so the combat move scores each candidate once (no resampling sweeps). The real game always rolls dice.
- **Placement** is the engine's (`{}`), battles `fightAll`, as in TripleMind.

## Metrics

Under `metrics/<game>/vs-<opponent>/` (or `$TRIPLEA_TESTBENCH_OUTPUT_DIR/metrics/...` in a
tournament). TripleMind's columns come first, so `agents/tripleMind/scripts/generate_plots.sh
<metrics folder>` works unchanged; this agent's extra columns are appended, all but
`game_results.csv` ending with `phase` (`purchase`, `combatMove`, `nonCombatMove`): filter on it to compare with the
combat-only agents.

| File | Row | TripleMind columns, as used here | Extra columns |
|---|---|---|---|
| `game_results.csv` | per game | `won`, `rounds_played`, `game_time`, ... | `winners` |
| `efficiency.csv` | per phase decision | `time_per_turn` search time; `num_iterations` states visited (forward-model evaluations); `root_value` best value of the last level; `num_children_at_root` actions at level 1; `num_action_sequences` sum of actions over levels; `num_of_terr_attacked` sequence length (attacks, units bought or moves made); `max_tree_depth` levels searched; `max_rollout` 0 | `reached_end_of_turn`, `num_actions_per_level`, `best_value_per_level` (`;`-separated), `phase` |
| `rollout_stats.csv` | per evaluation | `iteration` evaluation index, `reward`, `is_terminal`; `rollout_depth_reached` 0 | `tl` (sequence level), `phase` |
| `action_seq_stats.csv` | per level | `num_children` actions at the level; `visit_count` samples of the chosen action; `avg_value` best value; `second_*` the sibling with the **second-best value**; `mean_avg_value`, `avg_value_std_dev`, `avg_value_z` over siblings | `value_gap` (best − second), `end_phase_value`, `states_visited`, `level_seconds`, `phase` |
| `sibling_values.csv` | per candidate per level | — | `action`, `chosen`, `samples`, `mean_value`, `std_value`, `diff_to_best`, `phase` |

## Running it

```bash
# Offline arena (from planning-agent-testbed/), against the library's HeuristicPolicy:
python3 tools/arena.py --xml games/capture-the-flag-diag/map/capture_the_flag/map/games/capture_the_flag.xml \
    --agent Russians=agents/oep_paper/greedy_action/run-agent.sh --games 1
```

In a tournament spec use `greedy-action TRIPLEA_GREEDY_SECONDS=1.0`.
