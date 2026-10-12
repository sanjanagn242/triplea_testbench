# Hard AI (Python)

A Python port of the engine's **Hard AI** (`games.strategy.triplea.ai.pro.ProAi`, shown as
"Hard (AI)" in TripleA) for the forward model's land/air V1 profile. It makes the same decisions as
the Java AI. The one change is where its simulations run: every battle estimate, the turn
simulation during purchase, and the check of every move and placement use the testbench forward
model (`triplea_fm`), never a copy of the engine.

Use it as a strong rule-based baseline: it is the closest testbench counterpart of the engine's
Hard AI, so agents can be compared against the Hard AI's logic under the forward model's rules.

## Files

| File | Java counterpart | Contents |
|---|---|---|
| `agent.json`, `run-agent.sh` | | Manifest and launcher (puts `forward-model/python` on `PYTHONPATH`) |
| `agent.py` | `ExternalAgentPlayer` protocol | Protocol loop: builds the state with `LiveBridge`, calls `HardAi`, encodes the answer |
| `hardai/ai.py` | `AbstractProAi`, `ProAi` | Phase flow, turn simulation in purchase, stored plans between phases |
| `hardai/combat_move.py` | `ProCombatMoveAi` | Attack selection, unit assignment, bombing raids, capital safety |
| `hardai/noncombat_move.py` | `ProNonCombatMoveAi` | Defence, best-value moves, air basing, AA guns |
| `hardai/purchase.py` | `ProPurchaseAi`, `ProPurchaseUtils` | Purchases with planned placement, then placement |
| `hardai/battle.py` | `AbstractProAi.selectCasualties`, `ProRetreatAi`, `ProSimulateTurnUtils.simulateBattles` | Battle-time policies and the simulated battle phase |
| `hardai/territory_manager.py` | `ProTerritoryManager` | Attack/defend options for this player, its allies and enemies |
| `hardai/odds.py` | `ProBattleUtils`, `ProOddsCalculator` | Strength estimates; battle odds through `triplea_fm.battlesim` |
| `hardai/values.py` | `ProTerritoryValueUtils` | Territory values |
| `hardai/sorting.py` | `ProSortMoveOptionsUtils` | Order in which units are assigned |
| `hardai/moves.py` | `ProMoveUtils` | Routes for planned moves, merged and checked by the kernel |
| `hardai/data.py` | `ProData`, `ProTerritory`, `ProPurchaseOption(Map)`, `ProBattleResult`, ... | Shared analysis data |

## How a turn is played

| Request | What happens |
|---|---|
| `game_start` | `fm.load_definition(xml, strict=False)`. If the map is outside the V1 profile, every refusal is logged and the agent answers with empty actions (battles: `fightAll`) |
| `purchase` | As `AbstractProAi.purchase`: on a copy of the state the forward model plays the rest of the turn. The AI plans its combat move, the kernel applies it and fights the battles, and the AI plans its noncombat move, which the kernel applies. Purchases and their placement are then decided on that simulated end-of-turn board (`ProPurchaseAi.purchase`). The combat moves and the placement plan are stored |
| `combatMove` | The stored combat moves are replayed if the kernel still accepts all of them. Otherwise the combat move is planned again on the live state |
| `battle` | `{"fightAll": true}`. In protocol v1 the engine picks this faction's casualties and retreats |
| `nonCombatMove` | Planned on the live state. Units planned for placement count as defenders that cannot move, as in ProAi |
| `place` | Places the bought units where planned, then places any units left (`placeRemainingUnits`) |

The simulated battle phase is played by the kernel with this AI's own casualty selection
(`selectCasualties`) and retreat decision (`ProRetreatAi`). Defenders use the library default
order of loss. The battles are fought several times (`TRIPLEA_HARD_AI_BATTLE_SAMPLES`, default 5).
The run closest to the average outcome is kept: average units left on each side and territories
owned. ProAi does the same per battle with its odds calculator
(`AggregateResults.getBattleResultsClosestToAverage`).

## Where the forward model is used

| Simulation | Forward-model function |
|---|---|
| Battle odds (win %, TUV swing, typical survivors, rounds) | `triplea_fm.battlesim.simulate_battle`, which fights the battle with the kernel's battle program. The run count is ProAi's `max(16, 100 − smaller army)`. Results are cached per (territory, attacking units, defending units) during a phase |
| Reachable territories for this player, its allies and enemies (including next-turn enemy attacks) | `triplea_fm.toolkit.unit_routes` |
| Rest-of-turn simulation in purchase | `fm.apply_in_place` (moves, battle decisions, end of phase) on clones |
| Checking every move and placement before it is sent | `fm.apply_in_place` / `fm.validate` on a clone, in order. Rejected merged moves are retried unit by unit |
| Air landing, casualty order, retreat destinations | `triplea_fm.rules.can_land_here`, `toolkit.default_casualty_order`, `rules.retreat_destinations` |

`battlesim`, `toolkit.unit_routes` and `toolkit.default_casualty_order` were added to the library
for this agent. They are covered by the forward-model tests and documented in the library README.

## Scope and deviations from the Java AI

**Omitted, because the V1 profile refuses maps that use them:** fleets, transports, amphibious
assaults, bombardment, carriers, scrambling, factory purchase and repair, bids, technology and
politics.

**Behavioural differences:**

* **Battle odds** come from the forward model's battle rules instead of the engine's
  `BattleCalculator`. The calculator player behaves the same way: default order of loss, no
  retreat, or retreat when only air is left for strafing estimates.
* **Purchase randomness** (`ProPurchaseUtils.randomizePurchaseOption`) uses a seeded generator
  instead of `Math.random`. Collections are iterated in a fixed order instead of `HashMap` order.
  A game is therefore reproducible for a given `TRIPLEA_HARD_AI_SEED`.
* **Unit values (TUV)** are the cheapest production cost of each unit type over all frontiers
  (`GameDefinition.unit_cost`). `TuvCostsCalculator` takes the player's own frontier first.
* **Strength estimates** do not count the hit points of infrastructure (AA guns). The engine
  counts them although such units can never be casualties in V1 battles.
* **Capital defensive re-run in noncombat moves.** `ProNonCombatMoveAi` re-runs its best-value
  moves with a capital defence range when the capital lacks local land superiority. It runs them
  on a copy of the territory manager, but the moves it executes come from the map taken before
  the copy. The only visible effect is that AA guns are not moved. This port reproduces that
  behaviour.
* Win thresholds are ProAi's dice values (90 % / 65 %). Low Luck is outside the V1 profile.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TRIPLEA_HARD_AI_SEED` | 0 | Seed of the purchase randomness and of the AI's simulation dice |
| `TRIPLEA_HARD_AI_RUN_SCALE` | 1.0 | Scales battle simulations per estimate (1.0 = ProAi's run count) |
| `TRIPLEA_HARD_AI_BATTLE_SAMPLES` | 5 | Battle-phase samples in the purchase simulation |
| `TRIPLEA_AGENT_LOG` | 0 (in `run-agent.sh`) | 1 logs each phase's plan, timing and any rejected action |
| `TRIPLEA_TESTBENCH_PYTHON` | python3 | Interpreter used by `run-agent.sh` |

No Python packages beyond the forward-model library are needed.

## Running

* **In the testbench GUI:** assign `hard-ai-python` to a faction in a game profile, for example
  `"agents": {"Russians": "hard-ai-python"}`, then use **Planning Agent Testbench → Run Simulations**.
* **Offline arena** (the forward model stands in for the engine):
  ```bash
  python3 planning-agent-testbed/tools/arena.py --agent Russians=planning-agent-testbed/agents/hard-ai-python/run-agent.sh --games 2
  python3 planning-agent-testbed/tools/arena.py --xml planning-agent-testbed/games/minimap/map/minimap/map/games/minimap.xml \
    --agent Russians=planning-agent-testbed/agents/hard-ai-python/run-agent.sh \
    --agent Italians=planning-agent-testbed/agents/python-mcts/run-agent.sh --games 2
  ```
  Both should report `rejected=0`. On Capture the Flag a whole game takes a few seconds. On
  Minimap a purchase, which includes the turn simulation, takes about 0.5 s.
