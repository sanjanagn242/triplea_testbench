# Non-exploring MCTS, all attacks (combat only)

Agent ID `ne-mcts-all-attacks`. A variant of [oep_paper's Non-exploring MCTS](../../oep_paper/ne_mcts_combat/README.md) that differs
**only in the tree's action generation** (`ne_mcts.py`: `NonExploringMCTS.tree_actions`). See the original's README
and UML for the algorithm, settings and metric files, which are unchanged.

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

The rollout policy is unchanged: from the reached plan it keeps attacking the highest-priority
target its units beat by 1.5x, as in oep_paper's agent.

## Files

A copy of `oep_paper/ne_mcts_combat` with:

| File | Change |
|---|---|
| `ne_mcts.py` | `tree_actions` builds the tree's actions; docstring |
| `action_generators.py` | TripleMind's, with `combat_actions` split into `target_attackers` and `attack_sets` (same results), as in `oep-variants/roxo-oep-with-purchase` |
| `agent.py`, `agent.json` | ID, name, description |

The tree dump (`edge_tree.py`, `TRIPLEA_*_SAVE_TREE`) is the same as the original's: see its
[Drawing the search tree](../../oep_paper/ne_mcts_combat/README.md#drawing-the-search-tree).
