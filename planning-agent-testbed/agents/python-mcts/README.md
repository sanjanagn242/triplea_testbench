# Python MCTS agent

A testbench agent that plays each phase with open-loop Monte Carlo tree search (MCTS), simulating
the game with the testbench forward-model library (`planning-agent-testbed/forward-model`). The
agent contains only its search. Rules, legal-action generation, the observation-to-state bridge
and the baseline rollout policy all come from the library.

This README is also a guide for writing a similar agent. It covers the protocol loop, how the
search state is built, how actions are generated, how the tree is searched, how the final actions
are chosen, and every library function the agent calls. [The C++ agent](../cpp-mcts/README.md) runs
the same algorithm against the library's C++ binding.

Contents:

1. [Architecture](#1-architecture)
2. [Protocol loop (`agent.py`)](#2-protocol-loop-agentpy)
3. [From observation to search state](#3-from-observation-to-search-state)
4. [Action generation](#4-action-generation)
5. [The search (`mcts.py`)](#5-the-search-mctspy)
6. [Action selection](#6-action-selection)
7. [Rollout policy and evaluation](#7-rollout-policy-and-evaluation)
8. [Function reference](#8-function-reference)
9. [Configuration and running](#9-configuration-and-running)
10. [Replicating this agent](#10-replicating-this-agent)
11. [Limitations](#11-limitations)

## 1. Architecture

```
TripleA engine ──turn_request (JSON line)──▶ agent.py ──LiveBridge.build_state──▶ SimulationState (root)
                                                │                                     │
                                                │                     mcts.PhaseSearch.search(root)
                                                │                       │  toolkit.candidates   (action generation)
                                                │                       │  HeuristicPolicy.prior (ordering)
                                                │                       │  kernel.apply_in_place / validate (simulation)
                                                │                       │  HeuristicPolicy.act + evaluate (rollout, value)
                                                ◀───────── list of actions for the rest of the phase
TripleA engine ◀──response (JSON line)── agent.encode (LiveBridge.*_to_json)
```

| Layer | Owner | Code |
|---|---|---|
| Protocol, logging, time budget per phase | This agent | `agent.py` |
| Tree search: expansion, widening, UCT, backup, final choice | This agent | `mcts.py` |
| Game rules, dice, validation, transitions | Library kernel | `triplea_fm.kernel` |
| Legal candidate actions | Library ActionToolkit | `triplea_fm.toolkit` |
| Rollout policy, priors, state evaluation | Library baselines | `triplea_fm.baselines` |
| Observation → `SimulationState`; actions → protocol JSON | Library bridge | `triplea_fm.testbench` |

The search only uses the library's public API. To change the agent's strategy, replace the
policy, priors or evaluation. The rules never need to change.

## 2. Protocol loop (`agent.py`)

The testbench starts `run-agent.sh`, which puts `forward-model/python` on `PYTHONPATH` and runs
`agent.py`. The agent reads one JSON request per line from stdin and writes one JSON response per
line to stdout. Diagnostics go to stderr and to `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt`.
Stdout carries protocol lines only.

| Request | What the agent does | Response |
|---|---|---|
| `game_start` | `fm.load_definition(gameXml, strict=False)`. If the support report is not `supported`, it logs every refusal and answers later requests with empty actions. Otherwise it builds `LiveBridge(definition, player)` and one `PhaseSearch(definition, Precomp(definition), rollout_rounds, seed)` that is reused for the whole game | `{"ready": true, "protocolVersion": 1}` |
| `turn_request`, phase `battle` | No search. In protocol v1 the engine's built-in AI chooses this agent's casualties and retreats | `{"fightAll": true}` |
| `turn_request`, phase `purchase` / `combatMove` / `nonCombatMove` / `place` | Build the root state (§3), check it with `fm.create_state`, set the time budget, run `PhaseSearch.search(root)`, encode the result | `{"purchases": [...]}`, `{"moves": [...]}` or `{"placements": [...]}` |
| any other | Nothing to decide | `{}` |

**Time budget.** Each phase gets `TRIPLEA_MCTS_SECONDS × share`, where `PHASE_TIME_SHARE` is
`combatMove` 1.5, `nonCombatMove` 0.8, `purchase` 0.6 and `place` 0.3. Combat moves decide the
game, so they get the most time.

**Encoding (`Agent.encode`).**

| Phase | Encoding |
|---|---|
| Purchase | All `PurchaseAction`s in the sequence are merged into one quantity map, because a purchase does not end the phase and the search may buy in several steps. Then `LiveBridge.purchase_to_json` |
| Moves | Every `MoveAction` in order, via `LiveBridge.move_to_json`. Each move carries the full `route`, so blitzes and multi-step flights are executed as searched, and an `intent` (`RAID` makes the Java adapter answer "bomb" when the engine asks) |
| Placement | Every `PlaceAction` via `LiveBridge.place_to_json` |

`EndPhaseAction` is never sent; the engine ends the phase after applying the list.

**Error handling.** Any exception inside a request is logged and answered with
`{"error": ...}`. The loop never exits on an error, so one bad decision does not end the game.

## 3. From observation to search state

`LiveBridge.build_state(observation, phase)` returns a complete `SimulationState` positioned at
this player's decision for `phase`:

* territory owners, units (with movement already used), unit reserves (held purchases) and
  treasuries come from the observation;
* the step is the player's step of that phase kind, and the decision is `PURCHASE`, `MOVE` or
  `PLACE`;
* for placement it computes the placement capacities (`rules.placement_snapshot`);
* the observation does not contain some history, so the bridge rebuilds it from memory kept across
  requests and documents each assumption in its module docstring. Territories conquered this turn
  are those the player owns now but did not own at its first request of the turn. A unit's
  original owner is the first owner the bridge saw. Nothing counts as fought this turn.

`fm.create_state(definition, state)` runs the kernel's invariant checks. If the state is
inconsistent, the agent logs it and answers `{}`, so the game continues without a search.

The search never changes this root state. Every iteration works on `root.clone()`.

## 4. Action generation

Tree edges come from `triplea_fm.toolkit.candidates(d, s)`, which dispatches on
`s.decision.kind`. Every generator passes its proposals through `kernel.validate_batch`, so **every
candidate is legal in the state it was generated from**. The generators are deliberately not
complete. They propose a compact, useful subset:

| Decision | Generator | Candidates |
|---|---|---|
| `PURCHASE` | `purchase_candidates(d, s, max_types=2)` | For each affordable frontier rule: spend everything on it, with the remainder spent on the cheapest rule. For each pair of rules: bundles in ratios 1:1, 2:1 and 1:2, again topped up with the cheapest. Plus `EndPhaseAction` (buy nothing) |
| `MOVE` | `move_candidates(d, s)` | Units of the actor are grouped by (territory, type, movement left). For each group and each destination returned by `reachable_routes`: move the whole group, and move one unit of it. For strategic bombers in combat move, also a `RAID` variant. For mixed stacks, the whole stack to each adjacent territory. Plus `EndPhaseAction` |
| `PLACE` | `place_candidates(d, s)` | For each eligible factory territory: place as many held units as its remaining capacity allows. Plus `EndPhaseAction` |
| `BATTLE_SELECTION` | `battle_candidates` | Every pending battle the actor may fight now |
| `CASUALTIES` | `casualty_candidates` | `default_casualties` (lowest power, then cost, then air with least movement), cheapest-first, most-expensive-first |
| `RETREAT` | `retreat_candidates` | `CONTINUE` plus every legal retreat destination |

**Routes.** `reachable_routes(d, s, start, max_steps, group_air, group_blitz)` runs a BFS and
returns one shortest plausible route per destination. Intermediate territories must be passable
for that kind of group:

* air may not cross neutral land;
* land may not enter water;
* land may pass friendly territory without enemy combat units, and may pass enemy territory in
  combat move only if the group can blitz and the territory is blitzable;
* impassable territories are never crossed.

The kernel then validates the full move. This covers landing feasibility for air, the end-of-route
rules and stopping in neutral territory.

**Ordering and limiting.** `PhaseSearch._expand` sorts the candidates by
`HeuristicPolicy.prior(s, a)` (highest first) and keeps the first `max_children` (24). These
become the node's `untried` list. The prior is a hand-made score:

| Action | Prior |
|---|---|
| `EndPhase` | 0.5 |
| Combat move into enemy territory with no defenders | 1.0 + value/10 for land units; 0.05 for air only |
| Combat move into defended enemy territory | power/(defense+1) + value/20 − 0.5, where power includes own units already there |
| `RAID` | 0.3 |
| Any other move | 0.4 + 0.1 × (distance to nearest target from start − from destination) |
| Purchase | 0.5 + 0.01 × number of units bought |

Territory value is `production + 12 × victory city + 15 if a capital` (`Precomp.value`).
Distances are BFS distances over passable territories (`Precomp.dist`).

## 5. The search (`mcts.py`)

`PhaseSearch` searches **the rest of the agent's current phase**. The tree contains only the
agent's own actions within that phase. Everything after the phase, including battles, other
players' turns and dice, is handled by the rollout.

### 5.1 Phase boundary

```python
_same_phase(root, s) = s.decision is not None and s.terminal is None
                       and s.decision.actor == root.decision.actor
                       and s.decision.step_id == root.decision.step_id
                       and s.decision.kind == root.decision.kind
                       and s.round == root.round
```

A path in the tree ends when the phase ends. That happens after `EndPhaseAction`, or when the
kernel moves to the next step automatically because nothing is left to decide.

### 5.2 Open loop

Moves in this game can be followed by dice: fly-over AA fire during a move, and battles after the
combat move. The agent does not create chance nodes. Each iteration draws a fresh game RNG,
`fm.SeededRng(self.rnd.getrandbits(62))`, and replays the tree's action sequence from the root.
Statistics of a node therefore average over dice outcomes (open-loop MCTS).

Because outcomes differ between iterations, an edge that was legal when it was created can be
illegal in another sample. For example, the unit it moves may have been shot down by AA. This is
handled by checking with the kernel at selection time (§5.4).

### 5.3 One iteration

```
s   = root_state.clone()                    # never touch the root
rng = SeededRng(fresh seed)                 # game dice for this iteration
node, path = root, [root]
while same_phase(root_state, s):
    if node.untried is None: expand(s, node)                 # candidates + prior ordering
    allowed = max(1, int(widening * sqrt(node.n + 1)))       # progressive widening
    if node.untried and len(node.children) < allowed:
        a = node.untried.pop(0)                              # best remaining prior
        if apply_in_place(d, s, a, rng) == REJECTED: break   # stale under this sample: drop it
        node = new child(a); path.append(node); break        # expand one node per iteration
    child = select(node, s)                                  # UCT over children valid in s
    if child is None: break
    apply_in_place(d, s, child.action, rng); node = child; path.append(node)
value = rollout(s, rng, me, root_state.round)                # heuristic play, then evaluate
for n in path: n.n += 1; n.w += value                        # backup (single perspective)
```

The loop repeats until `time_budget` expires or `max_iterations` is reached. `expand` runs the
first time a node is reached and uses the state of the sample that reached it.

### 5.4 Tree policy: progressive widening and UCT

* **Widening.** A node with `n` visits may have at most `max(1, ⌊widening·√(n+1)⌋)` children
  (`widening` = 2.0). New children are taken in prior order. A node first explores its 1–2 most
  promising actions and considers more as it gets visits. This keeps the search useful when there
  are many legal moves.
* **UCT.** Among children whose action is still valid in the current sample
  (`fm.validate(d, s, child.action).ok`):
  `score = w/n + c·√(ln(N+1)/n)` with `c` = `exploration` = 0.7, where `N` is the parent's visit
  count. Values are in [0, 1], so the constant is small.

### 5.5 Backup

The value from the rollout is the root player's evaluation in [0, 1]. It is added to every node on
the path. There is only one point of view: the tree contains only the agent's own decisions, so
there are no opponent nodes to negate or switch.

## 6. Action selection

`_best_sequence(root_state, root)` turns the tree into the action list sent to the engine:

1. Clone the root and create a fixed RNG (`SeededRng(12345)`), so the replay is reproducible.
2. At each node, choose the child with the most visits (ties broken by total value) among
   children that were visited and are valid in the replay state.
3. Apply it, append it, and descend.
4. If the tree has no usable child, the remaining actions come from `HeuristicPolicy.act`
   (greedy completion).
5. Stop at `EndPhaseAction`, when the phase ends, or if an action is rejected.

Choosing by visit count rather than mean value is the standard robust choice. It ignores children
that were tried only a few times with lucky dice.

## 7. Rollout policy and evaluation

**Rollout (`_rollout`).** From the leaf state, every player, the agent included, is played by
`HeuristicPolicy.act(s, rnd)` until the game ends, `start_round + rollout_rounds` rounds have
passed (default 2), or 5000 steps. If the policy proposes an action the kernel rejects, the
rollout applies `EndPhaseAction` instead. If that is also rejected, the rollout stops. Rollouts
use `fm.apply_in_place`, which mutates the iteration's clone and is about 5× faster than
`transition`.

**`HeuristicPolicy` (library baseline).** It plays every decision kind for any player:

| Decision | Rule |
|---|---|
| Purchase | Ends the phase if units are already held. Otherwise alternates the best "offensive" unit ((attack + 0.5·movement)/cost) and the best "defensive" unit (defense/cost) until the money runs out; infrastructure and sea units excluded |
| Combat move | For each enemy or neutral land neighbour of each stack of movable land units: an empty target is taken with the cheapest single unit (score = value + 1). A defended target is attacked with the whole stack only if the attack power of all units adjacent to it is at least `1.2 × defense + 1` (score = value + 2·(power − defense)). The best-scoring valid move is chosen, one move per call |
| Noncombat move | Air units that cannot land where they are fly to the nearest legal landing spot. Land units step toward the nearest enemy target through own territory, as a group of the same type. Land units do not leave the capital while it has 2 or fewer defenders |
| Battle selection | Lowest pending battle id |
| Casualties | `toolkit.default_casualties` |
| Retreat | Continue if attack power ≥ 0.6 × defense, otherwise retreat to the first legal destination |
| Placement | All held units at the factory with unlimited capacity, otherwise at the highest-value factory with room |

**`evaluate(d, pc, s, player)`.** The value is 1 or 0 at a terminal state (win or loss).
Otherwise each player's score is:

```
1 + Σ owned territories (3·production + 30·victory city) + 25 if it holds its own capital
  + Σ cost of its non-infrastructure units + its treasury
```

The evaluation is the player's share of the total score, between 0 and 1. It changes smoothly
with material, territory and economy, which suits short rollouts.

## 8. Function reference

**Forward-model kernel (`import triplea_fm as fm`).**

| Function | Used for | Notes |
|---|---|---|
| `fm.load_definition(xml, strict=False)` | `game_start` | Returns `(definition, SupportReport)`. With `strict=False` it does not raise; check `report.supported` and log `report.features` |
| `fm.create_state(d, state)` | Checking each root state | Raises `fm.StateError` on invariant violations |
| `state.clone()` | One copy per iteration and per final replay | Deep, independent copy |
| `fm.validate(d, s, a)` | UCT filtering, final selection | Pure; uses no randomness |
| `fm.apply_in_place(d, s, a, rng)` | Tree descent, expansion, rollouts, final replay | Returns `"DECISION"`, `"TERMINAL"`, `"REJECTED"` or `"FAILED"`. A rejected action leaves `s` and `rng` unchanged |
| `fm.SeededRng(seed)` | Game dice per iteration | The agent's own randomness (`random.Random`) is kept separate |
| `fm.EndPhaseAction()`, `fm.PurchaseAction(...)` | Fallbacks and merging purchases | Action dataclasses: `kind` is `"purchase"`, `"move"`, `"place"`, `"end_phase"`, ... |
| `fm.REJECTED` | Status comparison | |

State fields the search reads: `s.decision.actor`, `.kind`, `.step_id`; `s.round`; `s.terminal`.

**ActionToolkit (`from triplea_fm import toolkit`).** `candidates(d, s)` and, through it,
`purchase_candidates`, `move_candidates`, `reachable_routes`, `place_candidates`,
`battle_candidates`, `casualty_candidates`, `retreat_candidates`, `default_casualties` (see §4).

**Baselines (`from triplea_fm.baselines import ...`).**

| Name | Used for |
|---|---|
| `Precomp(d)` | Static tables, built once per game: all-pairs BFS distances `dist`, unit cost `cost`, territory `value` |
| `HeuristicPolicy(d, pc)` | `.prior(s, a)` orders expansions; `.act(s, rnd)` drives rollouts and greedy completion |
| `evaluate(d, pc, s, player)` | Leaf value in [0, 1] |

**Bridge (`from triplea_fm.testbench import LiveBridge`).** `LiveBridge(d, player)`,
`.build_state(observation, phase)`, `.purchase_to_json(a)`, `.move_to_json(a)`,
`.place_to_json(a)`.

**This agent.**

| Function | Purpose |
|---|---|
| `Agent.handle`, `.game_start`, `.turn`, `.encode`, `.log` | Protocol loop (§2) |
| `PhaseSearch(d, pc, time_budget, max_iterations, rollout_rounds, max_children, exploration, widening, seed)` | Search object, reused across decisions |
| `PhaseSearch.search(root_state) -> list[action]` | Runs MCTS for the rest of the phase |
| `PhaseSearch._same_phase`, `_expand`, `_select`, `_rollout`, `_best_sequence` | §5.1, §4, §5.4, §7, §6 |
| `Node(action)` with `children`, `untried`, `n`, `w` | Tree node; `untried` is `None` until expanded |

## 9. Configuration and running

| Variable | Default | Meaning |
|---|---|---|
| `TRIPLEA_MCTS_SECONDS` | 2.0 | Base search time per decision, scaled per phase (combat move ×1.5, noncombat ×0.8, purchase ×0.6, place ×0.3) |
| `TRIPLEA_MCTS_ROLLOUT` | 2 | Rollout horizon in rounds |
| `TRIPLEA_MCTS_SEED` | 0 | Search seed |
| `TRIPLEA_TESTBENCH_PYTHON` | python3 | Interpreter used by `run-agent.sh` |

The other search parameters (`max_children` = 24, `exploration` = 0.7, `widening` = 2.0,
`max_iterations` = 100000) are `PhaseSearch` constructor arguments.

* **In the testbench GUI:** assign `python-mcts` to a faction in a game profile (for example
  `"agents": {"Russians": "python-mcts"}`), then use **Planning Agent Testbench → Run Simulations**.
* **Headless, real engine** (agent vs. Easy AI):
  ```bash
  TRIPLEA_LIVE_AGENT_TEST=1 TRIPLEA_MCTS_SECONDS=0.5 ./gradlew --offline :game-headed:test --tests '*ExternalAgentLiveGameTest*' --rerun -i
  ```
  For another map set `TRIPLEA_LIVE_XML`, e.g.
  `planning-agent-testbed/games/minimap/map/minimap/map/games/minimap.xml`. Without `--rerun`, Gradle
  skips the test when only environment variables changed.
* **Offline arena** (the forward model stands in for the engine; heuristic opponents):
  ```bash
  python3 planning-agent-testbed/tools/arena.py --games 4 --seconds 0.5
  python3 planning-agent-testbed/tools/arena.py --xml planning-agent-testbed/games/minimap/map/minimap/map/games/minimap.xml \
    --agent Russians=planning-agent-testbed/agents/python-mcts/run-agent.sh --games 4 --seconds 0.5
  ```

## 10. Replicating this agent

To write a new search agent (a different MCTS variant, beam search, rolling-horizon evolution, an
RL policy, ...) on the same library:

1. **Copy the shell.** `agent.json` (id and launch command), `run-agent.sh` (sets `PYTHONPATH` to
   `forward-model/python`) and the protocol loop in `agent.py`. Add the id to `agents/agents.json`.
2. **Load once per game.** `fm.load_definition(xml, strict=False)`. Refuse gracefully if
   unsupported. Build `LiveBridge` and any static tables (`Precomp`) once.
3. **Build a root per request.** `bridge.build_state(obs, phase)`, then `fm.create_state` to check it.
4. **Simulate on clones only.** `s = root.clone()` per iteration. Use `apply_in_place` for speed,
   or `transition` when you need the previous state, the events or the chance outcomes.
5. **Keep two RNGs.** Game dice must come from an `fm.SeededRng` passed to the kernel. Your
   agent's own random choices use its own generator. Never reuse one for the other.
6. **Generate actions with the toolkit, or your own generator.** Toolkit candidates are always
   legal. If you write your own, filter them with `fm.validate_batch`.
7. **Handle open-loop staleness.** When the same action sequence is replayed under different dice,
   check each stored action with `fm.validate` before applying it.
8. **Define the search horizon explicitly.** The phase boundary test in §5.1 is what makes the
   output a complete phase plan. A different scope (one action, a whole turn) needs a different test.
9. **Return what the protocol expects.** Merge purchases, keep moves in order with full routes,
   and drop `EndPhaseAction`. Use the bridge's `*_to_json` functions.
10. **Test offline first.** Use `tools/arena.py` (head-to-head with `--agent X=... --agent Y=...`),
    then the headless real-engine test. Both report rejected actions. A sound agent should have 0.

Places to improve the strategy without touching anything else: the prior (`HeuristicPolicy.prior`),
the rollout policy (`HeuristicPolicy.act`), the evaluation (`evaluate`), the candidate set, and
the parameters `max_children`, `widening`, `exploration` and `rollout_rounds`.

## 11. Limitations

* Only the agent's own phase is in the tree. Opponent replies are modelled only by the heuristic
  rollout policy.
* Candidate generation is incomplete: one route per destination, whole groups or single units,
  and at most 24 children per node after prior ordering.
* Nothing is reused between decisions; each phase starts a new tree.
* Battle decisions (casualties, retreats) are not searched. In protocol v1 the engine's built-in AI
  makes them for this agent, while the search models them with `default_casualties` and the
  heuristic retreat rule.
* The root state contains bridge assumptions about history that the observation does not provide
  (§3).
