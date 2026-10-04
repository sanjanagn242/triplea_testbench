# C++ MCTS agent

A testbench agent whose search, rollout policy, priors and evaluation are written in C++. It
simulates the game with the testbench forward-model library (`planning-agent-testbed/forward-model`)
through the library's C++ binding, `#include <triplea_fm/forward_model.hpp>`. It uses the same
rules, dice and legal-action generator as every other agent, and contains no game rules and no
binding code of its own.

This README is also a guide for writing a similar C++ agent. It covers the protocol loop, the
binding's data model and action encoding, how actions are generated, how the tree is searched,
how the final actions are chosen, and every function used. The algorithm is the same as
[the Python agent](../python-mcts/README.md). Here it is described in terms of the C++ code.

Contents:

1. [Architecture](#1-architecture)
2. [Protocol loop (`src/main.cpp`)](#2-protocol-loop-srcmaincpp)
3. [The binding: handles, `Static`, `View`, action encoding](#3-the-binding-handles-static-view-action-encoding)
4. [Action generation](#4-action-generation)
5. [The search (`src/mcts.hpp`)](#5-the-search-srcmctshpp)
6. [Action selection](#6-action-selection)
7. [Rollout policy, priors and evaluation (`src/policy.hpp`)](#7-rollout-policy-priors-and-evaluation-srcpolicyhpp)
8. [Function reference](#8-function-reference)
9. [Build, configuration and running](#9-build-configuration-and-running)
10. [Replicating this agent](#10-replicating-this-agent)
11. [Performance and limitations](#11-performance-and-limitations)

## 1. Architecture

```
TripleA engine ──turn_request (JSON line)──▶ main.cpp Agent
                                               │  model.parse_request(line)          (library)
                                               │  model.state_from_request() → state handle (library bridge)
                                               │  PhaseSearch::search(state)           (src/mcts.hpp)
                                               │     model.candidates / clone / apply / validate / view
                                               │     Policy::prior, act, evaluate      (src/policy.hpp)
                                               │  model.encode_response(phase, actions) (library bridge)
TripleA engine ◀──response (JSON line)─────────┘
```

| Layer | Owner | Code |
|---|---|---|
| Protocol, logging, time budget per phase | This agent | `src/main.cpp` |
| Tree search: expansion, widening, UCT, backup, final choice | This agent | `src/mcts.hpp` (`agent::PhaseSearch`) |
| Rollout policy, move priors, state evaluation | This agent | `src/policy.hpp` (`agent::Policy`) |
| Game rules, dice, validation, transitions, cloning | Library kernel | via `fm::ForwardModel` |
| Legal candidate actions | Library ActionToolkit | `fm::ForwardModel::candidates` |
| Observation → state; actions → protocol JSON | Library bridge | `state_from_request`, `encode_response` |

Every action the agent creates itself (in `Policy`) is checked with `model.validate` before use.
The agent never decides legality on its own.

## 2. Protocol loop (`src/main.cpp`)

`run-agent.sh` builds the binary if needed (§9) and runs it. The binary reads one JSON request
per line from stdin and writes one JSON line to stdout. The log goes to stderr and to
`planning-agent-testbed/logs/agent-<faction>-game-<n>.txt`.

The agent never parses JSON itself. `model.parse_request(line)` returns
`[type, phase, playerName, gameNumber]` and keeps the full request inside the library for the
next call.

| Request | What the agent does | Response |
|---|---|---|
| `game_start` | `model.start_game()` loads the request's game XML (≥ 0 means supported). `model.messages()` returns any refusal reasons, which are logged. If supported: `model.static_info()` once, and one `PhaseSearch(model, static, seed)` reused for the game | `{"ready": true, "protocolVersion": 1}` |
| `turn_request`, phase `battle` | No search. In protocol v1 the engine's built-in AI chooses this agent's casualties and retreats | `{"fightAll": true}` |
| `turn_request`, phase `purchase` / `combatMove` / `nonCombatMove` / `place` | `model.state_from_request()` builds and checks the root state (−1 plus `messages()` on failure). Set the time budget, run `search(state)`, `model.free(state)`, then `model.encode_response(phase, actions)` | Purchases (merged), moves (ordered, full routes and intent) or placements |
| any other | Nothing to decide | `{}` |

**Time budget.** Each phase gets `TRIPLEA_MCTS_SECONDS × share`, where `kPhaseShare` is
`combatMove` 1.5, `nonCombatMove` 0.8, `purchase` 0.6 and `place` 0.3.

**Encoding.** `encode_response` merges all purchase actions into one purchase, sends moves in
order with their full `route` and `intent` (`RAID` makes the Java adapter answer "bomb"), sends
placements, and drops `END_PHASE`.

**Errors.** An exception inside a request is logged and answered with an error line. The loop
keeps running.

## 3. The binding: handles, `Static`, `View`, action encoding

**Handles.** States and RNGs live inside the library and are referred to by `int` handles:

| Call | Meaning |
|---|---|
| `int s = model.clone(root)` | Independent copy of a state |
| `model.free(s)` | Release a state handle (every `clone` and `state_from_request` must be freed) |
| `int r = model.rng_new(seed)` / `model.rng_free(r)` | Game dice generator (counter-based, seeded) |
| `fm::Status model.apply(s, action, r)` | Apply one action in place, followed by automatic and chance effects up to the next decision. Returns `DECISION`, `TERMINAL`, `REJECTED` or `FAILED`. A rejected action leaves `s` and `r` unchanged |
| `bool model.validate(s, action)` | Pure legality check; uses no randomness |
| `std::vector<fm::Action> model.candidates(s)` | Legal candidate actions for the current decision (§4) |
| `fm::View model.view(s)` | Snapshot of the state as plain fields |

**`fm::Static`** (from `model.static_info()`, read once per game):

| Field | Meaning |
|---|---|
| `territories`, `players`, `alliances` | Counts |
| `production[t]`, `victory[t]`, `capital_of[t]` (player or `NULL_PLAYER`), `water[t]`, `impassable[t]` | Territory data |
| `neighbors[t]`, `dist[a][b]` | Adjacency and all-pairs BFS distance over passable territories |
| `types[k]` (`fm::UnitType`) | `domain` (0 land, 1 air, 2 sea), `movement`, `attack`, `defense`, `cost`, flags `can_blitz`, `infrastructure`, `can_move_in_combat`, `artillery`, `supportable`, `strategic_bomber`, `factory`; helpers `air()`, `land()` |
| `rules[r]` (`fm::Rule`) | `cost`, `other_resources`, `result_type`, `result_qty` |
| `frontier[p]` | Production rule ids player `p` may buy |
| `alliance_of[p]`, `allied(a, b)` | Alliances (`allied` is false for `NULL_PLAYER`) |

**`fm::View`** (from `model.view(s)`):

| Field | Meaning |
|---|---|
| `round`, `step_id`, `step` (`STEP_PURCHASE`, `STEP_COMBAT`, `STEP_BATTLE`, `STEP_NONCOMBAT`, `STEP_PLACE`, `STEP_OTHER`) | Position in the turn |
| `decision` (`DEC_PURCHASE`, `DEC_MOVE`, `DEC_BATTLE_SELECTION`, `DEC_CASUALTIES`, `DEC_RETREAT`, `DEC_PLACE`, `DEC_NONE`), `actor` | Who must decide what |
| `terminal`, `winner` (alliance id or −1) | Game end |
| `owner[t]`, `conquered[t]`, `balance[p]` | Ownership, conquered this turn, treasury |
| `units` (`fm::Unit`: `id`, `type`, `owner`, `loc` (`HELD` = −2 for purchased, unplaced), `move_left`, `in_combat`), `unit(id)` | Units |
| `caps` (`fm::Cap`: `territory`, `cap` (−1 unlimited), `used`) | Placement capacities in the place phase |
| `battle_id`, `batch_id`, `hits`, `defenders_hit`, `eligible`, `retreat_destinations` | Current battle decision |
| `battles` (`fm::BattleInfo`: `id`, `site`, `attacker`, `raid`, `active`) | Pending battles |

**Action encoding** (`fm::Action` = `std::vector<int>`, codes in `fm::ActionCode`):

| Action | Encoding |
|---|---|
| End phase | `{END_PHASE}` |
| Purchase | `{PURCHASE, rule, count, rule, count, ...}` |
| Move | `{MOVE, intent (0 normal, 1 raid), n, unit_1 … unit_n, t_0, t_1, …}`; the route includes the start territory |
| Select battle | `{SELECT_BATTLE, battle_id}` |
| Casualties | `{CASUALTIES, batch_id, unit, ...}` |
| Retreat | `{RETREAT, choice (0 continue, 1 retreat), destination or −1}` |
| Place | `{PLACE, territory, unit, ...}` |

## 4. Action generation

Tree edges come from `model.candidates(s)`. This is the library's ActionToolkit (the same
generator the Python agent uses), returned in the encoding above. Every candidate has been checked
with the kernel's batch validation, so **every candidate is legal in the state it was generated
from**. The set is deliberately not complete:

| Decision | Candidates |
|---|---|
| Purchase | For each affordable rule: spend everything on it, remainder on the cheapest rule. For each pair of rules: bundles in ratios 1:1, 2:1, 1:2, topped up with the cheapest. Plus `END_PHASE` |
| Move | Units grouped by (territory, type, movement left). For each group and each reachable destination (one shortest plausible BFS route each): the whole group and one unit. `RAID` variants for strategic bombers in combat move. Whole mixed stacks to each neighbour. Plus `END_PHASE` |
| Place | Per eligible factory: as many held units as fit. Plus `END_PHASE` |
| Battle selection / casualties / retreat | Every pending battle / three casualty orders / continue plus every legal retreat destination |

During route BFS, air may not cross neutral land. Land may not enter water, may pass friendly
territory without enemy combat units, and may pass blitzable enemy territory in combat move if the
group can blitz. Impassable territories are never crossed. The kernel validates the rest
(landing for air, stopping rules).

**Ordering and limiting (`PhaseSearch::expand`).** Candidates are scored with
`Policy::prior(view, action)`, stable-sorted highest first, and truncated to
`config.max_children` (24). The result becomes the node's `untried` list. Priors are listed in §7.

## 5. The search (`src/mcts.hpp`)

`agent::PhaseSearch` searches **the rest of the agent's current phase**. Tree edges are only the
agent's own actions in that phase. Battles, other players' turns and all dice after the phase are
simulated in the rollout.

### 5.1 Phase boundary

```cpp
static bool same_phase(const View& root, const View& v) {
  return !v.terminal && v.decision != fm::DEC_NONE && v.actor == root.actor &&
         v.step_id == root.step_id && v.decision == root.decision && v.round == root.round;
}
```

### 5.2 Open loop

There are no chance nodes. Each iteration creates a new game RNG,
`model.rng_new(rnd_() >> 2)`, and replays the tree's action sequence from a clone of the root, so
node statistics average over dice outcomes. `rnd_` is a `std::mt19937_64` used only for search
seeds. Game dice always come from the library RNG handle. Because outcomes differ between
iterations (for example fly-over AA can destroy a unit), a stored edge can be illegal in a given
sample. Selection therefore skips children that `model.validate` rejects.

### 5.3 One iteration

```cpp
int s = model.clone(root_state);
int rng = model.rng_new(fresh_seed);
Node* node = &root; path = {&root}; View v = model.view(s);
while (same_phase(root_view, v)) {
  if (!node->expanded) expand(s, v, *node);                          // candidates + prior order
  size_t allowed = max(1, size_t(widening * sqrt(node->n + 1.0)));  // progressive widening
  if (!node->untried.empty() && node->children.size() < allowed) {
    Action a = pop_front(node->untried);
    if (model.apply(s, a, rng) >= fm::REJECTED) break;              // stale in this sample: drop
    node = add_child(node, a); path.push_back(node); v = model.view(s);
    break;                                                           // one expansion per iteration
  }
  Node* child = select(*node, s);                                    // UCT over valid children
  if (!child) break;
  model.apply(s, child->action, rng); node = child; path.push_back(node); v = model.view(s);
}
double value = rollout(s, rng, me, root_view.round, v);
for (Node* n : path) { n->n += 1; n->w += value; }                   // single-perspective backup
model.free(s); model.rng_free(rng);
```

The loop runs until `config.time_budget` seconds pass (`steady_clock`) or `config.max_iterations`
is reached.

### 5.4 Tree policy

* **Progressive widening.** At most `max(1, ⌊widening·√(n+1)⌋)` children per node
  (`widening` = 2.0). New children are added in prior order.
* **UCT.** Over children whose action is valid in the current sample:
  `w/n + exploration·√(ln(N+1)/n)` with `exploration` = 0.7 (values are in [0, 1]).

### 5.5 Backup

The rollout returns the root player's value in [0, 1]. It is added to every node on the path.
There is no perspective switch, because the tree holds only this player's decisions.

`agent::Node` holds `action`, `children` (`unique_ptr<Node>`), `untried`, `expanded`, `n` and
`w`. The whole tree is freed when `search` returns.

## 6. Action selection

`best_sequence(root_state, root_view, root)`:

1. Clone the root and use a fixed RNG (`rng_new(12345)`) so the replay is reproducible.
2. At each node, take the visited child with the most visits (ties broken by value) whose action
   `model.validate` accepts in the replay state.
3. Apply it and append it to the result.
4. Without a usable child, continue with `Policy::act` (greedy completion).
5. Stop at `END_PHASE`, when the phase ends, or if an action is rejected. Free the clone and RNG.

## 7. Rollout policy, priors and evaluation (`src/policy.hpp`)

`agent::Policy` is the agent's own C++ heuristic, a port of the library baseline. It works only
from `View` and `Static`. Territory value `value_[t] = production + 12·victory + 15 if a capital`
is computed in the constructor.

**Rollout (`PhaseSearch::rollout`).** Every player, the agent included, is played by
`Policy::act(state, view)` until the game ends, no decision is pending, the round exceeds
`start_round + rollout_rounds` (default 2), or `max_rollout_steps` (5000) is reached. If the
policy's action is `REJECTED` or `FAILED`, the rollout applies `END_PHASE`. If that also fails,
the rollout stops. The leaf is scored with `Policy::evaluate(view, me)`.

**`Policy::act` by decision:**

| Decision | Function | Rule |
|---|---|---|
| Purchase | `purchase` | End the phase if units are already held. Otherwise alternate the best offensive unit ((attack + 0.5·movement)/cost) and the best defensive unit (defense/cost) until the money runs out; infrastructure and sea units excluded |
| Combat move | `combat_move` | For each enemy or neutral land neighbour of each stack of movable land units: empty target → cheapest single unit (score value + 1). Defended target → whole stack, only if the attack power of all units adjacent to the target ≥ `1.2·defense + 1` (score value + 2·(power − defense)). Highest-scoring valid move, one per call |
| Noncombat move | `noncombat_move`, `land_air` | Air that cannot land where it is: BFS (no neutral land, no impassable) to the nearest legal landing spot. Land: move the same-type group to an own neighbour closer to the nearest enemy target. Land units do not leave the capital while it has 2 or fewer defenders |
| Battle selection | `select_battle` | Lowest pending battle id |
| Casualties | `casualties` | Lowest power first (artillery counts its support when attacking), then cost, then air with least movement left |
| Retreat | `retreat` | Continue if attack power ≥ 0.6 × defense, otherwise retreat to the first destination |
| Placement | `place` | Unlimited-capacity factory first, otherwise the highest-value factory with room |

Helpers: `attack_power` (artillery support +1 per supported unit), `defense_power`,
`enemy_combat`, `units_at`, `target_distance` (BFS distance to the nearest enemy territory with
value), `can_land`, `neutral_land`, `enemy_territory`.

**`Policy::prior(view, action)`**, used to order and limit expansions:

| Action | Prior |
|---|---|
| `END_PHASE` | 0.5 |
| Purchase | 0.5 + 0.01 × number of units |
| Combat move, `RAID` | 0.3 |
| Combat move into undefended enemy territory | 1.0 + value/10 with land units; 0.05 air only |
| Combat move into defended enemy territory | power/(defense+1) + value/20 − 0.5 |
| Other moves | 0.4 + 0.1 × (target distance at start − at destination) |

**`Policy::evaluate(view, player)`.** At a terminal state the value is 1 if the player's alliance
won, otherwise 0. Otherwise each player's score is
`1 + Σ owned (3·production + 30·victory) + 25 for holding its own capital + Σ unit cost (non-infrastructure) + treasury`,
and the value is the player's share of the total.

## 8. Function reference

**`fm::ForwardModel` (library binding).**

| Method | Used in | Purpose |
|---|---|---|
| `ForwardModel(python_dir = "")` | `Agent` member | Starts the embedded interpreter and imports the library. One per process |
| `parse_request(line)` | `handle` | `[type, phase, playerName, gameNumber]`; stores the request |
| `start_game()` | `game_start` | Loads the stored request's XML. ≥ 0 if supported |
| `messages()` | `game_start`, `turn` | Refusal or error messages from the last call |
| `static_info()` | `game_start` | `fm::Static` |
| `state_from_request()` | `turn` | Root state handle from the stored `turn_request`, or −1 |
| `encode_response(phase, actions)` | `turn` | Testbench JSON for the phase |
| `clone`, `free`, `rng_new`, `rng_free` | search, rollout, final replay | Handle management |
| `view(s)` | everywhere | `fm::View` snapshot |
| `candidates(s)` | `expand` | Legal candidates (ActionToolkit) |
| `validate(s, a)` | `select`, `best_sequence`, `Policy` | Legality check |
| `apply(s, a, rng)` | descent, expansion, rollout, replay | In-place transition with status |
| `load_xml`, `initial_state`, `snapshot_json` | not used by the agent | For offline tools and tests |

**This agent.**

| Symbol | File | Purpose |
|---|---|---|
| `Agent::handle`, `game_start`, `turn`, `log` | `main.cpp` | Protocol loop (§2) |
| `agent::SearchConfig` | `mcts.hpp` | `time_budget`, `max_iterations` (100000), `rollout_rounds` (2), `max_children` (24), `exploration` (0.7), `widening` (2.0), `max_rollout_steps` (5000) |
| `agent::PhaseSearch(model, static, seed)`, `search(state)`, `iterations` | `mcts.hpp` | MCTS for the rest of the phase |
| `same_phase`, `expand`, `select`, `rollout`, `best_sequence` | `mcts.hpp` | §5.1, §4, §5.4, §7, §6 |
| `agent::Policy(model, static)`: `act`, `prior`, `evaluate` and the per-decision functions | `policy.hpp` | §7 |

## 9. Build, configuration and running

`run-agent.sh` compiles `src/main.cpp` into `build/cpp-mcts-agent` whenever the agent sources or
the library header are newer than the binary:

```bash
g++ -std=c++20 -O2 $(forward-model/cpp/triplea-fm-config --cflags) src/main.cpp -o build/cpp-mcts-agent \
    $(forward-model/cpp/triplea-fm-config --libs)
```

Requirements: `g++` with C++20 and the Python 3.10+ development headers (`python3-dev`).
Overrides: `TRIPLEA_TESTBENCH_CXX`, `TRIPLEA_TESTBENCH_PYTHON_CONFIG`, `TRIPLEA_FM_PYTHONPATH`.

| Variable | Default | Meaning |
|---|---|---|
| `TRIPLEA_MCTS_SECONDS` | 2.0 | Base search time per decision, scaled per phase |
| `TRIPLEA_MCTS_ROLLOUT` | 2 | Rollout horizon in rounds |
| `TRIPLEA_MCTS_SEED` | 0 | Search seed |

* **Testbench GUI:** assign `cpp-mcts` to a faction, e.g. `"agents": {"Russians": "cpp-mcts"}`.
* **Real engine, headless:**
  ```bash
  TRIPLEA_LIVE_AGENT_TEST=1 TRIPLEA_LIVE_AGENT=cpp-mcts TRIPLEA_MCTS_SECONDS=0.5 \
    ./gradlew --offline :game-headed:test --tests '*ExternalAgentLiveGameTest*' --rerun -i
  ```
  Add `TRIPLEA_LIVE_FACTION` and `TRIPLEA_LIVE_XML` to choose the faction and map.
* **Offline arena** (also head-to-head with the Python agent):
  ```bash
  python3 planning-agent-testbed/tools/arena.py \
    --agent Russians=planning-agent-testbed/agents/cpp-mcts/run-agent.sh \
    --agent Italians=planning-agent-testbed/agents/python-mcts/run-agent.sh --games 3 --seconds 0.5
  ```
  Add `--xml <game.xml>` for another map, e.g. Minimap.

## 10. Replicating this agent

To write another C++ agent on the library:

1. **Copy the shell.** `agent.json`, `run-agent.sh` (the build line in §9) and the protocol loop in
   `main.cpp`. Add the id to `agents/agents.json`.
2. **Create one `fm::ForwardModel` per process.** Use `parse_request` for every line,
   `start_game` for `game_start`, `state_from_request` for each decision and `encode_response` for
   the answer. This way you never parse or write the testbench JSON yourself.
3. **Read `static_info()` once** and build your own tables from it (values, distances, unit
   classes).
4. **Simulate on clones only.** One `clone` and one `rng_new` per iteration. **Free every handle**
   (`free`, `rng_free`), or memory grows with every iteration.
5. **Keep game dice in library RNG handles.** Use your own generator only for search choices.
6. **Generate candidates with `model.candidates`,** or build your own `fm::Action`s and check them
   with `model.validate`.
7. **Handle open-loop staleness** by validating stored actions before replaying them.
8. **Define the search horizon** with a boundary test such as `same_phase`.
9. **Test offline first** with `tools/arena.py`, then the headless real-engine test. Both report
   rejected actions. A sound agent should have 0.

Places to change strategy without touching anything else: `Policy::prior`, `Policy::act`,
`Policy::evaluate`, and the `SearchConfig` parameters.

## 11. Performance and limitations

* Every library call runs the Python kernel, so this agent runs about as many iterations as the
  Python agent (a few hundred MCTS iterations per second on CTF). The binding's interface is
  designed so that a later native kernel or C ABI (spec §14) can replace the implementation
  without changing this agent.
* Only the agent's own phase is in the tree. Opponents are modelled only by the rollout policy.
* Candidates are incomplete (one route per destination, whole groups or single units, ≤ 24 per
  node). Nothing is reused between decisions.
* Battle decisions are not searched. In protocol v1 the engine's built-in AI makes them for this
  agent.
* The root state includes the library bridge's assumptions about history that the observation
  does not provide (see the Python agent README, §3).
