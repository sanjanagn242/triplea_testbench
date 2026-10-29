# Bridge Burning MCTS, all attacks (combat only)

Agent ID `bb-mcts-all-attacks`. A variant of [oep_paper's Bridge Burning
MCTS](../../oep_paper/bb_mcts_combat/README.md) that differs in two things: the tree's action
generation (`bb_mcts.py`: `BridgeBurningMCTS.tree_actions`) and how the budget is split into phases
(see Phases). See the original's README and UML for the rest of the algorithm, the settings and
the metric files, which are unchanged.

## Action generation

| | oep_paper's agent | This variant |
|---|---|---|
| Targets at a level | enemy land territories next to mine, not attacked yet, in TripleMind's priority order | every enemy territory not attacked yet that some unmoved unit of mine can reach, in map order (no priority, no adjacency requirement) |
| Actions per target | one: all reachable units | every attack set of TripleMind's `combat_actions` buildup for it: growing prefixes of the attackers (adjacent and strongest first) above 1.5x the defence and below 3.5x + 4, the full force if it beats the defence, or the cheapest unit for an undefended territory |
| END | first action of every node | unchanged |

A target whose attackers cannot reach any of these strengths has no action (it is simply not
attacked). Every level offers all the remaining targets' attack sets, so the tree, not a fixed
territory order, decides which targets are attacked and with which units; the plan is the path
through the tree, and the transposition table still merges plans that make the same attacks in a
different order.

The rollouts are unchanged: they still use oep_paper's actions (`candidate_actions`: one
all-units attack per target, TripleMind priority order).

## Phases

The second difference from oep_paper's agent. The paper split the budget into its fixed number of
actions per turn (5 action points). Here the number of phases per turn is a fixed estimate,
`TRIPLEA_BB_PHASES` (default 4, from the 3-4 territories TripleMind's MCTS attacks in a turn), to be
tuned:

- phase k gets `remaining time / phases left`, phases left = `TRIPLEA_BB_PHASES - (k - 1)`, but at
  most the number of targets left (a turn cannot have more phases than targets);
- a phase searches until its time is up *and* every root action has been tried once;
- a turn that needs more phases than estimated has no time left for them: each extra phase tries
  every root action once (usually already done by the earlier phases' search below it, so it costs
  nothing) and commits;
- when END is committed early, the rest of the budget is unused, as in oep_paper's agent.

`iterations_per_level` in `efficiency.csv` shows each phase's iterations.

## Files

A copy of `oep_paper/bb_mcts_combat` with:

| File | Change |
|---|---|
| `bb_mcts.py` | `tree_actions` builds the tree's actions; docstring |
| `action_generators.py` | TripleMind's, with `combat_actions` split into `target_attackers` and `attack_sets` (same results), as in `oep-variants/roxo-oep-with-purchase` |
| `bb_mcts.py` | phase split by `TRIPLEA_BB_PHASES` (see Phases) |
| `agent.py`, `agent.json`, `run-agent.sh` | ID, name, description; `TRIPLEA_BB_PHASES` |

The tree dump (`edge_tree.py`, `TRIPLEA_*_SAVE_TREE`) is the same as the original's: see its
[Drawing the search tree](../../oep_paper/bb_mcts_combat/README.md#drawing-the-search-tree).
