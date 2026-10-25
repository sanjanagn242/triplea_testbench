# OEP paper: Non-exploring MCTS (combat only)

Agent ID `oep-ne-mcts-combat`. The Non-exploring MCTS of Justesen, Mahlmann, Risi and Togelius,
*Playing Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search* (IEEE
Transactions on Games, 2018, Sections V and VI), applied to the TripleA combat move. Same actions,
evaluation, other phases and metric files as [Greedy Action](../greedy_action_combat/README.md) and
[Greedy Turn](../greedy_turn_combat/README.md); the action generators, map graph and metric helpers
are copies of TripleMind's (pure Python).

Not to be confused with [roxo-ne-mcts](../../roxo-ne-mcts/README.md), Roxo's thesis version of
Non-Exploring MCTS (both players' moves in the tree, its own playout heuristic).

UML (open in draw.io): [docs/ne-mcts-combat-uml.drawio](docs/ne-mcts-combat-uml.drawio):
1 class diagram, 2 sequence of one combat-move decision, 3 activity of one iteration.

## Algorithm

Paper: Non-exploring MCTS uses "a nonexploring tree policy, i.e., C = 0, in combination with
deterministic rollouts. All children of a node are still visited at least once before any of them
are expanded further … since rollouts are deterministic … it is acceptable to value children based
only on one visit." Rollouts are depth-limited to one turn and rated with the heuristic; a
transposition table stores values and visits on edges (descent-path-only backpropagation); after
the time budget the tree is followed from the root along the best nodes.

One action is one attack: **(all my unmoved units that can reach the target, target territory)**,
or END.

```
root = plan with no attacks; table = {state key: node}      (key: where my moved units stand)
until the time budget is used:
    selection:  node = root; while every action of node has an edge:
                    edge = max mean value (C = 0, END wins ties); stop at an END edge
    expansion:  the node's next untried action ([END] + attacks in TripleMind priority order):
                END → END edge; attack → child plan, shared through the table if already reached
    rollout:    from the plan (unless END): attack the highest-priority target whose strength
                > 1.5 × defence, repeat until none; end combat move, fight the battles (default
                casualties, no retreat), evaluate_state in [-1, 1]
                averaged over the search's fixed dice seeds → deterministic, cached per node
    backup:     reward added to the edges of the descent path
chosen sequence: from the root, the best-mean edge at each level until END
```

- **Dice**: C = 0 needs deterministic values, so every evaluation of one search uses the same
  `TRIPLEA_NE_DICE_SEEDS` seeds (default 1; more seeds average out a single lucky or unlucky dice
  draw at a proportional cost). New seeds are drawn for every combat move.
- **Cache**: a node's rollout is deterministic, so it is computed once; later visits reuse the
  reward and only update the edge statistics. On small trees the search is exhausted early and
  the remaining time keeps re-walking the best path (millions of cheap iterations).
- **Time budget**: `TRIPLEA_NE_SECONDS` per combat move (default 1.0).
- **Other phases** are TripleMind's deterministic policy: first `purchase_actions` option,
  `noncombat_move`, engine placement (`{}`), `fightAll`.

## Metrics

Same files and column names as the greedy OEP agents (TripleMind's columns first, so the TripleMind
and tournament plot scripts work):

| File | Row | Columns |
|---|---|---|
| `game_results.csv` | per game | TripleMind's + `winners` |
| `efficiency.csv` | per combat move | `num_iterations` MCTS iterations (incl. cached ones); `root_value` mean reward over the root's edges; `num_children_at_root` root edges incl. END; `max_rollout` 0 (rollouts end with this turn); `num_action_sequences` plans at the tree's frontier; `num_of_terr_attacked`; `max_tree_depth` attacks in the deepest plan; `reached_end_of_turn` 1 when the chosen path ends with END (0: it stopped at an unexpanded node); `num_actions_per_level`, `best_value_per_level` (chosen edge's mean); `states_visited` forward-model evaluations (rollouts × dice seeds); `transpositions` edges into an existing node |
| `rollout_stats.csv` | per computed rollout | `reward`, `is_terminal`, `tl` (attacks in the plan + 1), `rollout_attacks` attacks the rollout policy added |
| `action_seq_stats.csv` | per level of the chosen sequence | real visits and mean values of the node's edges; `second_*` second-best by value; `value_gap`, `end_phase_value`, `states_visited` visits through the node; `level_seconds` NaN |
| `sibling_values.csv` | per edge per level | `samples` visits, `mean_value`, `std_value`, `diff_to_best` |

Levels where END is the only option are not logged, as in the greedy agents.

## Running it

```bash
python3 tools/tournament.py --spec tournaments/specs/oep_baselines_vs_tripleMind.json --no-gui
```
