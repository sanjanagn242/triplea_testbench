# Roxo NE-MCTS

A Python port of the Non-Exploring MCTS agent in M. Roxo's
[MCTSGrandStrategy](https://github.com/ManuelFelizardo/MCTSGrandStrategy) fork of TripleA
(`src/main/java/games/strategy/triplea/ai/mctsclean`), written for his MSc thesis *MCTS-based
Planning for Grand Strategy Games* (Instituto Superior Técnico). Where the thesis text and his code
disagree, this agent follows the code. Built from [template-python](../template-python/README.md);
the forward model is the testbench kernel (`triplea_fm`). The thesis's map and setting are
[games/minimap-roxo](../../games/minimap-roxo).

## Files

| File | Purpose | His code |
|---|---|---|
| `agent.json` | Manifest (ID `roxo-ne-mcts`, display name **Roxo NE-MCTS**) | |
| `run-agent.sh` | Launcher and settings (below); puts `forward-model/python` on `PYTHONPATH` | |
| `agent.py` | Protocol loop (template-python), one search per move phase, final routes, metrics | `AbstractNonExploringMctsAi`, `ForwardModel.calculateMoveRoutes` |
| `ne_mcts.py` | `NonExploringMcts` search and its `MCTSNode`s | `NonExploringMcts`, `MCTSNode` |
| `world_model.py` | `WorldModel` (one node's world and actions), `Game` (action generation, moves, Advance, playout, reward) | `WorldModel`, `NewAction*`, `ActionUtils`, `ModelTerritoryManager` |
| `weak_ai.py` | Playout heuristic | `WeakAi.calculateCombatMoveSimple`, `calculateNonCombatSimple` |
| `fast_battle.py` | Deterministic battles | `ActionUtils.simulateBattles`, `FastOddsEstimator` |
| `land_graph.py` | Shortest land routes for the heuristic's noncombat moves | `GameMap.getRoute`, `Utils.findNearest` |
| `docs/roxo-ne-mcts-uml.drawio` | Class diagram, sequence of one game, activity of one search | |

## How a turn is played

| Phase | Response |
|---|---|
| `purchase`, `place` | `{}`: the thesis setting has no production |
| `combatMove`, `nonCombatMove` | One search from the current state; the plan's units, routed from where they stand |
| `battle` | `{"fightAll": true}`: TripleA resolves the real battles with dice |

## The algorithm

**Search** (`NonExploringMcts`). Selection starts at the root and runs while the node is not
terminal, at most 100 levels. A node with an action it has not expanded yet expands it (copy the
world, apply the action, generate the child's actions). Otherwise selection goes to the child with
the best `Q / N`, with no exploration term. The playout's reward is added to every node on the
path. It is always the searching player's reward, also at the opponent's levels. A search stops
after 1 second or 500 iterations. The plan follows the best child from the root until an Advance
and keeps the non-skip actions.

**Actions** (`WorldModel.generateActions`). A node's actions concern one territory; decided
territories are *excluded* until the next phase.

- *Combat move*: the candidate territories are the enemy territories next to each territory the
  player owns, in that order. For the first one not excluded, the units that can reach it this move
  are collected. If their Weak AI strength is not more than 1.1x the territory's, it is excluded
  and the next one is tried. Otherwise the actions are the unit sets of `getUnitsUpToStrengthSet`
  (largest first), then skip. That function keeps a growing prefix of the units when its strength
  is above `threshold x defence` (threshold 1.1, +0.1 per kept prefix) and below
  `3.5 x defence + 4`. If the full set was not kept, it adds it, so an undefended territory gets
  one set: all units that can reach it.
- *Noncombat move*: for the first territory the player owns that is not excluded, the units that can
  reach it are grouped by the territory they stand in. Every non-empty combination of groups is an
  action, largest combinations first, then skip (2^k actions for k groups).
- When no territory is left: *Advance*. It ends the phase, resolves the battles and stops at the
  next move phase, whoever's it is. The tree therefore continues into our noncombat move and the
  opponent's turn.

Units used at one level can be offered again at a later level if they still have movement left.

**Playout.** From a node, up to 40 times: the acting player makes its heuristic moves for the
current move phase, then the game advances to the next move phase. The heuristic is Roxo's
simplified Weak AI:

- *Combat move*: first, tanks that have not moved blitz an enemy capital through an empty territory
  next to it if they are stronger than its defenders. Then it walks into undefended enemy
  territories with the cheapest unit. Then it attacks territories (capitals always considered) when
  it is more than 1.1x stronger, from the territories it owns, with just enough units. Enemy
  territories are shuffled, then sorted capitals, factories, infrastructure, production.
- *Noncombat move*: every stack that has not moved takes one step towards the nearest enemy
  capital (or the nearest enemy units or territory), if that territory is its own.

**Battles.** Every battle in the agent's simulations has the Fast AI estimate's most likely outcome.
The winning side keeps its most expensive units, `ceil(n x margin / 50)` of them, with *n* counting
the defenders' infrastructure. The attacker takes the territory unless it has no units left and
the defender has some, so it also takes it when both sides are wiped out.

**Reward.** A win scores `0.6 + 0.4 x (40 - phases) / 40` and a loss the negative, *phases* being the
move phases the playout played. A playout still running scores `(own - others) / (2.5 x all)`,
counting every unit on the map, factories and AA guns included.

## Where this differs from his code

These differences come from running on the testbench kernel instead of a clone of TripleA's
`GameData`:

- His heuristic moves are performed without validation (`MovePerformer`). Here the kernel checks
  every move and skips illegal ones, as `performMove` does for the search's own moves.
- He collects units in `HashSet`s, whose order (which sets the growing prefixes in
  `getUnitsUpToStrengthSet`) is not reproducible. Here units come in map order, then unit order.
- Routes are the kernel's shortest plausible routes (`toolkit.unit_routes`), not
  `GameMap.getRouteForUnit`. On a tie the two may pick different routes.
- He checks victory when the sequence reaches step 6, which is the start of a round on his map. The
  kernel checks it at the end of each round, which is the same moment.
- The Weak AI's shuffle uses Python's `random` seeded with `ROXO_SEED`, not Java's
  `Collections.shuffle`.
- Battle survivors are ordered by the map's production cost (`unit_cost`). His `getCost` falls back
  to TUV values for units without a production rule (factories and AA guns on this map), so they
  may rank differently.

## Settings (`run-agent.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `ROXO_SECONDS` | `1.0` | Search time per move phase (his `maxTime`, 1000 ms) |
| `ROXO_MAX_ITERATIONS` | `500` | Iteration cap per search (his `MaxIterations`); 0 for none |
| `ROXO_PLAYOUT_PHASES` | `40` | Playout length in move phases |
| `ROXO_SEED` | `0` | Seed of the heuristic's shuffle |
| `TRIPLEA_AGENT_LOG` | `0` | `1` logs each search to `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt` |

## Metrics

As for the TripleMind agents (using `agents/tripleMind/helper.py`), it writes three files to
`metrics/<game>/vs-<opponent>/`, or to `$TRIPLEA_TESTBENCH_OUTPUT_DIR/metrics/...` in a tournament:
`game_results.csv` (one row per game), `efficiency.csv` (one row per search, combat and noncombat
moves alike; `max_rollout` is in rounds) and `rollout_stats.csv` (one row per playout).

## Running it

Against the Hard AI in TripleA (from the repository root):

```bash
./gradlew :game-headed:run --args="triplea.testbench.config=$PWD/planning-agent-testbed/games/minimap-roxo/config/roxo-ne-mcts-vs-hard-ai.json triplea.testbench.autostart=true triplea.testbench.gui=false"
```

Offline, against the forward model's baseline policy (from `planning-agent-testbed/`):

```bash
python3 tools/arena.py --xml games/minimap-roxo/map/minimap/map/games/minimap.xml \
    --agent Russians=agents/roxo-ne-mcts/run-agent.sh --games 2 --max-rounds 35
```
