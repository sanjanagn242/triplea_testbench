# OEP paper: Non-exploring MCTS (purchase, combat move, non-combat move)

Agent ID `ne-mcts`. The Non-exploring MCTS of Justesen, Mahlmann, Risi and Togelius, *Playing
Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search* (IEEE Transactions
on Games, 2018, Sections V and VI): exactly [Vanilla MCTS](../vanilla_mcts/README.md) with
**C = 0 and ε = 1**, so rollouts are fully deterministic and follow the sorting heuristic. Every
child of a node is visited at least once before any of them is expanded further. Unlike
[ne_mcts_combat](../ne_mcts_combat/README.md) (combat move only), this agent searches every phase
request it gets: **purchase, combat move and non-combat move**. Placement is the engine's and battles are `fightAll`, as TripleMind.
The actions are those of every all-phase OEP agent (greedy_action, greedy_turn, vanilla_mcts,
ne_mcts, bb_mcts, oep), which differ only in how they select among them: `phases.py` holds them,
and it and the generators it uses (`action_generators.py`, its Cython build
`action_generators_cy.pyx` / `scripts/build_cython.sh`, `generators.py`,
`noncombat_generators.py`, `map_graph.py`, `helper.py`, `fast_battle.py`) are byte-identical
copies in each agent; [`../check-shared-files.sh`](../check-shared-files.sh) fails when they
drift. `run-agent.sh` rebuilds the `.so` when the `.pyx` is newer (or missing; a failed build
falls back to the Python generators, which give the same results), and `TRIPLEMIND_PURE_PYTHON=1`
forces Python.

## Algorithm

One search per phase request, `TRIPLEA_NE_SECONDS` each. Each tree ply is one action, not
one turn.

```
root = state at the phase's start (non-combat: after the forced air landing); table = {key: node}
until the time budget is used:
    selection:  while every action of the node has an edge:
                    edge = argmax X_j + 2C sqrt(2 ln n / n_j) with C = 0, i.e. the best mean
                    (ties: sorting-heuristic order); stop at an END edge
    expansion:  the node's next untried action, in sorting-heuristic order; a child state already
                in the table is shared (transposition)
    rollout:    unless END: rest of the phase with the default policy, then END; then my remaining
                phases until my turn ends (searched phases: same default policy; battles: default
                casualties, no retreat; place: everything at the home factory), once per each of
                the search's K fixed dice seeds; Δ = the mean evaluate_state in [-1, 1]
                default policy, ε-greedy with ε = 1: always the highest-rated action by the
                sorting heuristic
                deterministic, so computed once per (node, END taken) and reused
    backup:     BackupNegamax (Algorithm 2) on the descent path only: each edge gets +Δ if its
                player is the max player (me), otherwise −Δ
chosen sequence: from the root, the most visited edge (ties: best mean) until END; if the path
                 stops at an unexpanded node, the highest-rated actions until END
```

Since the selection only descends from a node whose actions all have an edge, and expansion adds
one edge (one rollout) at a time in sorting-heuristic order, every child is valued by one visit
before any of them is expanded further; with deterministic rollouts one visit is enough. Battles
roll dice, so each search draws K dice seeds (`TRIPLEA_NE_DICE_SEEDS`, default 1; from
`TRIPLEA_NE_SEED`), every rollout is played once per seed and Δ is the mean of the K rewards: the
same plan always gets the same reward, and every node faces the same dice (common random numbers,
as greedy's sweeps). More seeds average out a single lucky or unlucky dice draw at K times the
cost per rollout. With `TRIPLEA_BATTLE_RESOLUTION=fast` there are no dice and K is 1. On small
trees the search is exhausted early and the remaining time re-walks the best path with cached
rewards.

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
| `TRIPLEA_NE_SECONDS` | 1.0 | search time per phase request |
| `TRIPLEA_NE_DICE_SEEDS` | 1 | dice battles: K fixed dice seeds per search, every rollout averaged over them |
| `TRIPLEA_NE_SEED` | 0 | search seed (draws each search's fixed dice) |
| `TRIPLEA_BATTLE_RESOLUTION` | dice | battles in the search's simulations: `dice` (the forward model's dice, default casualties, no retreat) or `fast` (the Fast AI's most likely outcome, no dice; `fast_battle.py`). The real game always rolls dice |

C = 0 and ε = 1 are fixed: they are what makes this NE-MCTS rather than Vanilla MCTS.

## Metrics

Same column names as the other OEP agents (TripleMind's columns first), plus a `phase` column:

| File | Row | Notes |
|---|---|---|
| `game_results.csv` | per game | TripleMind's + `winners` |
| `efficiency.csv` | per **combat-move** search | kept combat-only so it compares row for row with the combat-only OEP agents; `states_visited` counts computed (not cached) rollouts × K; `num_of_terr_attacked` is the actions chosen |
| `efficiency_purchase.csv`, `efficiency_noncombat.csv` | per purchase / non-combat search | same columns |
| `rollout_stats.csv` | per computed (not cached) rollout and dice seed | `tl` actions in the node's state + 1, `rollout_actions` actions the default policy played, `phase` |
| `action_seq_stats.csv` | per level of the chosen sequence with more than one edge | `states_visited` is the node's visits; `phase` |
| `sibling_values.csv` | per edge per such level | `phase` |

## Running it

```bash
python3 tools/arena.py --xml games/capture-the-flag-diag/map/capture_the_flag/map/games/capture_the_flag.xml \
    --agent Russians=agents/oep_paper/ne_mcts/run-agent.sh --games 2
```

or add `"ne-mcts TRIPLEA_NE_SECONDS=1.0"` to a `tournaments/specs/*.json` agent list.
