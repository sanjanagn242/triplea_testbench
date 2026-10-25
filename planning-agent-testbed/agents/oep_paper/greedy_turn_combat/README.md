# OEP paper: Greedy Turn (combat only)

Agent ID `oep-greedy-turn-combat`. The Greedy Turn baseline of Justesen, Mahlmann, Risi and
Togelius, *Playing Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search*
(IEEE Transactions on Games, 2018, Section V), applied to the TripleA combat move. Built from
[Greedy Action](../greedy_action_combat/README.md) (same actions, evaluation, phases and metric
files); the action generators, map graph and metric helpers are copies of TripleMind's
(`action_generators.py`, `generators.py`, `map_graph.py`, `helper.py`, pure Python).

UML (open in draw.io): [docs/greedy-turn-combat-uml.drawio](docs/greedy-turn-combat-uml.drawio):
1 class diagram, 2 sequence of one combat-move decision, 3 activity of the search.

## Algorithm

Paper: "a depth-first search among turns … the same heuristic as for Greedy Action rates all leaf
node states and selects the action sequence that leads to the highest rated state. A transposition
table keeps track of already visited game states … Because this search is usually not exhaustive,
choosing which actions to try first (i.e., action sorting) is critical."

One action is one attack, as in Greedy Action: **(all my unmoved units that can reach the target,
target territory)**. A node is a plan (the attacks so far), and every node is also a leaf: ending
the phase there.

```
DFS from the root plan (stack, children pushed in TripleMind territory priority order):
    node = pop
    leaf value = evaluate(node + END)            plan → end combat move → fight battles
                                                 (default casualties, no retreat)
                                                 → evaluate_state in [-1, 1]
    for each attack on an un-attacked target with reachable units:
        child = node + attack
        skip if where my moved units stand was already reached (transposition table)
    push children
until the stack is empty (reached_end_of_turn = 1) or the time budget runs out (0)

if the DFS finished: resample every leaf until the budget is used (same dice seed for every leaf
in a sweep; stop after two sweeps if no leaf's value varies)

value(node) = best mean leaf value in node's subtree
chosen sequence: from the root, go to the best child while it beats END there (END wins ties)
```

- **Time budget**: `TRIPLEA_GREEDY_SECONDS` per combat move (default 1.0), shared by the DFS and
  the resampling.
- **Action sorting**: TripleMind's `enemy_territories_by_priority` (victory cities, factory
  neighbours, production, ...). It decides which part of the tree is searched when the DFS cannot
  finish.
- **Evaluation point**: after this turn's battles, as in Greedy Action.
- **Other phases** are TripleMind's deterministic policy: first `purchase_actions` option,
  `noncombat_move`, engine placement (`{}`), `fightAll`.

## Metrics

Same files and columns as Greedy Action (TripleMind's columns first, so
`agents/tripleMind/scripts/generate_plots.sh` and `tournaments/scripts/generate_plots.sh` work),
with these meanings:

| File | Row | Columns |
|---|---|---|
| `game_results.csv` | per game | TripleMind's + `winners` |
| `efficiency.csv` | per combat move | `time_per_turn`; `root_value` value of the chosen sequence; `num_children_at_root` attacks at the root + END; `num_action_sequences` distinct full-turn sequences scored (leaves); `num_of_terr_attacked` sequence length; `max_tree_depth` deepest plan generated; `num_iterations`, `max_rollout` NaN; `reached_end_of_turn` 1 when the DFS searched every sequence; `num_actions_per_level` siblings at each level of the chosen sequence; `best_value_per_level` value of stopping at each level (its END leaf); `states_visited` forward-model evaluations incl. resamples; `transpositions` children skipped; `sweeps` samples per leaf |
| `rollout_stats.csv` | per evaluation | `reward`, `is_terminal`, `tl` (attacks in the plan + 1); `rollout_depth_reached` 0 |
| `action_seq_stats.csv` | per level of the chosen sequence | siblings = END (the node's leaf) and each child (valued by its best leaf); `visit_count` samples below the chosen sibling; `second_*` second-best by value; `value_gap`, `end_phase_value`, `states_visited` samples below the node; `level_seconds` NaN |
| `sibling_values.csv` | per sibling per level | `samples`, `mean_value`, `std_value` (of the sibling's best leaf), `diff_to_best` |

Levels where END is the only option (nothing left to attack) are not logged, as in Greedy Action.

## Running it

```bash
# Engine tournament: Greedy Turn, Greedy Action and TripleMind, every pair in both seats:
python3 tools/tournament.py --spec tournaments/specs/greedy_turn_vs_action_vs_tripleMind.json --no-gui
```
