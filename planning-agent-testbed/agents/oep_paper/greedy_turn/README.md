# OEP paper: Greedy Turn

Agent ID `greedy-turn`. The Greedy Turn baseline of Justesen, Mahlmann, Risi and Togelius,
*Playing Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search* (IEEE
Transactions on Games, 2018, Section V), applied to the TripleA purchase, combat move and
non-combat move. It is [greedy_turn_combat](../greedy_turn_combat/README.md) extended to the
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

Paper: "a depth-first search among turns … the same heuristic as for Greedy Action rates all leaf
node states and selects the action sequence that leads to the highest rated state. A transposition
table keeps track of already visited game states … Because this search is usually not exhaustive,
choosing which actions to try first (i.e., action sorting) is critical."

It runs once per phase, so 3 times per turn. A node is a plan (the actions so far this phase); a
node where END is offered is also a leaf: ending the phase there (the non-combat move offers END
only once every territory short of cover is decided). The number of actions differs per phase,
so the depth is variable.

| Phase | One action (`phases.py`) | Action sorting (highest-rated first) | Leaf scored | Transposition key |
|---|---|---|---|---|
| purchase | buy 1 unit of a candidate rule (land combat units for PUs) | attack + defence per PU; END last | after END | units bought |
| combat move | attack one enemy territory with every unmoved unit that can reach it | attacks with strength > 1.5 × defence in TripleMind priority order, END, the other attacks | after this turn's battles (default casualties, no retreat) | where my moved units stand |
| non-combat move | for the most important territory short of cover: a reinforcement set (strongest first) or its skip; END (with the positioning moves) once none is left | generation order | after END | territories decided + where this phase's moved units stand |

```
DFS from the root plan (stack, children pushed in action-sorting order):
    node = pop
    if END is offered: leaf value = evaluate_state(node + END) in [-1, 1]
    for each action available after node's plan:
        child = node + action
        skip if child's state was already reached by another order (transposition table)
    push children
until the stack is empty (reached_end_of_turn = 1) or the time budget runs out (0)

combat move with dice battles, if the DFS finished: resample every leaf until the budget is used
(same dice seed for every leaf in a sweep)

value(node) = best mean leaf value in node's subtree
chosen sequence: from the root, the best sibling (END or a child) until END
    ties: the first in the action sorting wins
    a path stopping at a node the DFS did not score: the highest-rated actions until END
```

- **Heuristic**: TripleMind's `evaluate_state` unchanged (1 / -1 won / lost, else
  2 * (0.5 TUV share + 0.5 income share) - 1). It counts neither units in hand nor positions,
  so purchase and non-combat leaves almost always tie and the action sorting decides.
- **Time budget**: `TRIPLEA_GREEDY_SECONDS` per phase decision (default 1.0). The combat move
  with dice always uses all of it (DFS, then resampling). Purchase and non-combat roll no dice, so they
  stop early only when the DFS has scored every plan (the result is then exact); otherwise
  the DFS runs until the budget is used.
- **Battle resolution**: `TRIPLEA_BATTLE_RESOLUTION` (`run-agent.sh`) picks how the search's battles are fought: `dice` (default: the forward model's dice, default casualties, no retreat) or `fast` (the Fast AI's most likely outcome, no dice; `fast_battle.py`). With `fast` one sample is exact, so the combat move skips the resampling after the DFS. The real game always rolls dice.
- **Placement** is the engine's (`{}`), battles `fightAll`, as in TripleMind.

## Metrics

Under `metrics/<game>/vs-<opponent>/` (or `$TRIPLEA_TESTBENCH_OUTPUT_DIR/metrics/...` in a
tournament). Same files and columns as greedy_turn_combat, TripleMind's first (so
`agents/tripleMind/scripts/generate_plots.sh <metrics folder>` works), plus a last column `phase`
(`purchase`, `combatMove`, `nonCombatMove`) in all but `game_results.csv`: filter on it to compare
with the combat-only agents.

| File | Row | Columns |
|---|---|---|
| `game_results.csv` | per game | TripleMind's + `winners` |
| `efficiency.csv` | per phase decision | `time_per_turn`; `root_value` value of the chosen sequence; `num_children_at_root` actions at the root + END; `num_action_sequences` distinct sequences scored (leaves); `num_of_terr_attacked` sequence length (attacks, units bought or moves made); `max_tree_depth` deepest plan generated; `num_iterations`, `max_rollout` NaN; `reached_end_of_turn` 1 when the DFS searched every sequence; `num_actions_per_level` siblings at each level of the chosen sequence; `best_value_per_level` value of stopping at each level (its END leaf); `states_visited` evaluations incl. resamples; `transpositions` children skipped; `sweeps` samples per leaf; `phase` |
| `rollout_stats.csv` | per evaluation | `reward`, `is_terminal`, `tl` (actions in the plan + 1); `rollout_depth_reached` 0; `phase` |
| `action_seq_stats.csv` | per level of the chosen sequence | siblings = END (the node's leaf) and each child (valued by its best leaf); `visit_count` samples below the chosen sibling; `second_*` second-best by value; `value_gap`, `end_phase_value`, `states_visited` samples below the node; `level_seconds` NaN; `phase` |
| `sibling_values.csv` | per sibling per level | `action`, `chosen`, `samples`, `mean_value`, `std_value` (of the sibling's best leaf), `diff_to_best`, `phase` |

Levels where END is the only option (nothing left to do) are not logged, as in Greedy Action.

## Running it

```bash
# Offline arena (from planning-agent-testbed/), against the library's HeuristicPolicy:
python3 tools/arena.py --xml games/minimap/map/minimap/map/games/minimap.xml \
    --agent Russians=agents/oep_paper/greedy_turn/run-agent.sh --games 1
```

In a tournament spec use `greedy-turn TRIPLEA_GREEDY_SECONDS=1.0`.
