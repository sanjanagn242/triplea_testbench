# OEP paper: Bridge Burning MCTS (combat only)

Agent ID `oep-bb-mcts-combat`. The BB-MCTS of Justesen, Mahlmann, Risi and Togelius, *Playing
Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search* (IEEE Transactions
on Games, 2018, Sections V and VI), applied to the TripleA combat move. Same actions, evaluation,
other phases and metric files as the other agents in [oep_paper](..): [Greedy
Action](../greedy_action_combat/README.md), [Greedy Turn](../greedy_turn_combat/README.md),
[Non-exploring MCTS](../ne_mcts_combat/README.md). The action generators, map graph and metric
helpers are copies of TripleMind's (pure Python).

UML (open in draw.io): [docs/bb-mcts-combat-uml.drawio](docs/bb-mcts-combat-uml.drawio):
1 class diagram, 2 sequence of one combat-move decision, 3 activity of the search.

## Algorithm

Paper: BB-MCTS is "the same as for Vanilla MCTS, but with the Bridge Burning strategy": it "splits
the time budget into a number of sequential phases equal to the number of actions in a turn.
During each phase, the search functions as an ordinary MCTS search. However, in the end of each
phase, all but the most promising node from the root are pruned and will never be added again …
treat the most promising child node as the root of the tree in the following phase." Vanilla MCTS
there: UCT `X + 2C sqrt(2 ln n / n_j)` with `C = 1/sqrt 2`, ε-greedy rollouts (ε = 0.5) limited
to one turn and rated with the heuristic, transposition table with values on edges and
descent-path-only backpropagation.

One action is one attack: **(all my unmoved units that can reach the target, target territory)**,
or END.

```
root = plan with no attacks; table = {state key: node}      (key: where my moved units stand)
repeat (one phase per committed action):
    actions at the root = [END] + attacks; no attacks → stop (END)
    phase time = remaining time / number of attacks at the root
    until the phase time is used, MCTS iterations below the root:
        selection:  while every action of the node has an edge: UCT-best edge
                    (mean reward mapped from [-1, 1] to [0, 1]); stop at an END edge
        expansion:  the node's next untried action (END first, then TripleMind priority order)
        rollout:    unless END: ε-greedy until END — with probability ε the greedy action (the
                    highest-priority attack with strength > 1.5 × defence, else END), otherwise
                    a random one of [END] + attacks; end combat move, fight the battles (fresh
                    dice, default casualties, no retreat), evaluate_state in [-1, 1]
        backup:     reward added to the edges of the descent path
    burn the bridges: commit the root's most visited edge (ties: best mean)
        END → stop; attack → its child becomes the root, everything else is never searched again
```

- **Phases**: a TripleA combat move has no fixed number of actions (the paper's had 5 action
  points). Each phase gets `remaining time / attacks at the current root`, an upper bound on the
  phases left, so a turn never runs out of phases. When END is committed early, the rest of the
  budget is unused.
- **Settings** (`run-agent.sh`): `TRIPLEA_BB_SECONDS` per combat move (1.0), `TRIPLEA_BB_EXP_CONST`
  C (0.7071 = 1/√2), `TRIPLEA_BB_EPSILON` probability of the greedy rollout action (0.5),
  `TRIPLEA_BB_SEED` (0).
- **Other phases** are TripleMind's deterministic policy: first `purchase_actions` option,
  `noncombat_move`, engine placement (`{}`), `fightAll`.

## Metrics

Same files and column names as the other OEP agents (TripleMind's columns first, so the TripleMind
and tournament plot scripts work):

| File | Row | Columns |
|---|---|---|
| `game_results.csv` | per game | TripleMind's + `winners` |
| `efficiency.csv` | per combat move | `num_iterations` MCTS iterations; `root_value` mean reward at the first root; `num_children_at_root` its edges incl. END; `max_rollout` 0 (rollouts end with this turn); `num_action_sequences` plans at the tree's frontier; `num_of_terr_attacked`; `max_tree_depth` attacks in the deepest plan; `reached_end_of_turn` 1 when END was committed (or nothing was left) before the time ran out; `num_actions_per_level` root edges per phase; `best_value_per_level` committed edge's mean per phase; `states_visited` rollouts; `transpositions` edges into an existing node; `iterations_per_level` iterations per phase |
| `rollout_stats.csv` | per rollout | `reward`, `is_terminal`, `tl` (attacks in the plan + 1), `rollout_attacks` attacks the rollout added |
| `action_seq_stats.csv` | per phase | root edges at the end of the phase: the committed (most visited) edge against `second_*`, the second-best by value; `value_gap`, `end_phase_value`, `states_visited` rollouts in the phase, `level_seconds` the phase's time |
| `sibling_values.csv` | per root edge per phase | `samples` visits, `mean_value`, `std_value`, `diff_to_best` (relative to the committed edge, so it can be negative when a less visited edge has a higher mean) |

Phases where END is the only option (nothing left to attack) are not searched or logged.

## Running it

```bash
python3 tools/tournament.py --spec tournaments/specs/oep_baselines_vs_tripleMind.json --no-gui
```
