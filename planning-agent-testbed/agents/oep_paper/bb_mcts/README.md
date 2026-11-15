# OEP paper: Bridge Burning MCTS (purchase, combat move, non-combat move)

Agent ID `bb-mcts`. The BB-MCTS of Justesen, Mahlmann, Risi and Togelius, *Playing Multiaction
Adversarial Games: Online Evolutionary Planning Versus Tree Search* (IEEE Transactions on Games,
2018, Sections V and VI): exactly [Vanilla MCTS](../vanilla_mcts/README.md) (C = 1/√2, ε = 0.5)
**plus the bridge burning step**. Unlike [bb_mcts_combat](../bb_mcts_combat/README.md) (combat move
only), this agent searches every phase request it gets: **purchase, combat move and non-combat
move**. Placement is
the engine's and battles are `fightAll`, as TripleMind. Action generators, map graph and metric
helpers are copies of TripleMind's (pure Python); `noncombat_generators.py` is all-phases-mcts's.

## Algorithm

One search per phase request, `TRIPLEA_BB_SECONDS` each. Each tree ply is one action, not
one turn. Paper: BB-MCTS "splits the time budget into a number of sequential phases … During each
phase, the search functions as an ordinary MCTS search. However, in the end of each phase, all but
the most promising node from the root are pruned and will never be added again … treat the most
promising child node as the root of the tree in the following phase."

```
root = state at the phase's start (non-combat: after the forced air landing); table = {key: node}
repeat (one burn phase per committed action), while time is left and root has more than END:
  burn phase time = remaining time / levels_left(root)    (upper bound on the actions still to choose)
  until the burn phase time is used, Vanilla MCTS iterations below root:
    selection:  while every action of the node has an edge:
                    edge = argmax X_j + 2C sqrt(2 ln n / n_j),  C = 1/sqrt 2
                    (n: the node's visits, n_j: the child edge's visits); stop at an END edge
    expansion:  the node's next untried action, in sorting-heuristic order; a child state already
                in the table is shared (transposition)
    rollout:    unless END: rest of the phase with the default policy, then END; then my remaining
                phases until my turn ends (searched phases: same default policy; battles: default
                casualties, no retreat; place: everything at the home factory), fresh dice;
                Δ = evaluate_state in [-1, 1]
                default policy, ε-greedy with ε = 0.5: with probability ε the highest-rated
                action by the sorting heuristic, otherwise a uniformly random one (END included)
    backup:     BackupNegamax (Algorithm 2) on the descent path only: each edge gets +Δ if its
                player is the max player (me), otherwise −Δ
  burn the bridges: commit root's most visited edge (ties: best mean); every other root child is
                    pruned for good. END → done; otherwise its child becomes the new root
out of time before END: from the last root, Vanilla MCTS's choice: the most visited edge (ties:
                 best mean) until END; past an unexpanded node, the highest-rated actions until END
```

A TripleA phase has no fixed number of actions (the paper's turns had 5 action points), so each
burn phase gets the remaining time divided by an upper bound on the actions still to choose, and a
search never runs out of burn phases before END. When END is committed early, the rest of the
budget is unused.

| Phase | `levels_left` (actions still to choose, upper bound) |
|---|---|
| purchase | the most units of any one offered rule the balance still pays for |
| combat move | attacks offered at the root (each target is attacked at most once) |
| non-combat move | my territories still short of cover and not yet decided |

Inside one phase every edge is mine, so the negamax backup always adds Δ; the sign rule is kept
as the paper writes it. Rewards are [-1, 1] (the testbed's `evaluate_state`), and C is applied as
given, unscaled. Rollouts stop at the end of my turn: the paper found short rollouts better than
long ones.

| Phase | One action | Sorting heuristic (highest-rated first) | Transposition key |
|---|---|---|---|
| purchase | buy 1 unit of a candidate rule (land combat units for PUs, as `purchase_actions`) | attack + defence per PU; END last | units bought |
| combat move | attack one enemy territory with every unmoved unit that can reach it (as the other OEP agents) | attacks with strength > 1.5 × defence in TripleMind priority order, END, the other attacks | where my moved units stand |
| non-combat move | for the most important territory short of cover: a reinforcement set (strongest first) or its skip; END when none is left, with the positioning moves | generation order: strongest set first, skip last | territories decided + where this phase's moved units stand |

The non-combat move first lands the air the engine would destroy (forced, not a tree ply). A
decision with only one action (nothing affordable, nothing to attack, no territory short of
cover) is answered without searching.

## Settings (`run-agent.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `TRIPLEA_BB_SECONDS` | 1.0 | search time per phase request, split into burn phases |
| `TRIPLEA_BB_C` | 0.7071 (1/√2) | C of the UCT term 2C √(2 ln n / n_j) |
| `TRIPLEA_BB_EPSILON` | 0.5 | rollout: probability of the highest-rated action |
| `TRIPLEA_BB_SEED` | 0 | search seed |

## Metrics

Same column names as the other OEP agents (TripleMind's columns first), plus a `phase` column:

| File | Row | Notes |
|---|---|---|
| `game_results.csv` | per game | TripleMind's + `winners` |
| `efficiency.csv` | per **combat-move** search | kept combat-only so it compares row for row with the combat-only OEP agents; `num_of_terr_attacked` is the actions chosen; `root_value`, `num_children_at_root` at the first root; `num_actions_per_level`, `best_value_per_level`, `iterations_per_level` per burn phase; `reached_end_of_turn` 1 when the sequence ends with an END edge of the tree |
| `efficiency_purchase.csv`, `efficiency_noncombat.csv` | per purchase / non-combat search | same columns |
| `rollout_stats.csv` | per rollout | `tl` actions in the node's state + 1, `rollout_actions` actions the default policy played, `phase` |
| `action_seq_stats.csv` | per burn phase | root edges at its end: the committed (most visited) edge against `second_*`, the second-best by value; `states_visited` rollouts in the burn phase, `level_seconds` its time; `phase` |
| `sibling_values.csv` | per root edge per burn phase | `diff_to_best` relative to the committed edge (negative when a less visited edge has a higher mean); `phase` |

## Running it

```bash
python3 tools/arena.py --xml games/capture-the-flag-diag/map/capture_the_flag/map/games/capture_the_flag.xml \
    --agent Russians=agents/oep_paper/bb_mcts/run-agent.sh --games 2
```

or add `"bb-mcts TRIPLEA_BB_SECONDS=1.0"` to a `tournaments/specs/*.json` agent list.
