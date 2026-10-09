# Planning Agent Developer Guide

This guide describes how to build an agent for the planning testbed. The engine owns the game and
remains authoritative: an agent receives a rules document and state, chooses an action, and TripleA
validates that action through its existing delegates.

## Agent folder and dependencies

Keep each implementation isolated in `planning-agent-testbed/agents/<agent-id>/`. Put its entry
point and its own `requirements.txt` or equivalent dependency manifest in that folder. Add an
`agent.json` manifest in the same folder and list its ID in `planning-agent-testbed/agents/agents.json`.
The shared engine player reads the manifest command and starts the external process; no new Java
class is needed for each policy or language. Python agents need Python 3.10 or newer; Simple
Infantry is C++20 and requires `g++` with C++20 support.

The testbed-level `planning-agent-testbed/requirements.txt` is the place to aggregate Python
dependencies needed across the experiment. Install it from the repository root with:

```bash
python3 -m pip install -r planning-agent-testbed/requirements.txt
```

An individual Python agent can be installed by its own manifest, when it has one:

```bash
python3 -m pip install -r planning-agent-testbed/agents/your-agent/requirements.txt
```

Simple Infantry compiles itself on launch using `g++`. If that name is not on the engine process's
`PATH`, set `TRIPLEA_TESTBENCH_CXX` to the compiler path, for example `/usr/bin/g++`.

## Process and message protocol

The current bridge uses newline-delimited JSON over the process's standard input and output. It
starts one persistent process per agent-controlled faction for each game. Read exactly one JSON
request line and write exactly one JSON response line for each request. Do not write diagnostics or
pretty-printed JSON to stdout; log diagnostics to stderr or the agent's own log file. The Java side
uses the response line as the action object directly, without an outer `response` envelope.

Logging is opt-in for every agent. Each agent's `run-agent.sh` exports `TRIPLEA_AGENT_LOG`
(`1` enables logging, `0` disables it; an existing value in the environment takes precedence). An
agent must write nothing to stderr or its log file, and must not create the log file, unless
`TRIPLEA_AGENT_LOG` is `1`. The only exceptions are crash tracebacks and build output from
the launcher (for example compiler errors).

Every request includes `schemaVersion` (currently `1`), `requestId`, `playerName`, and
`gameNumber`.

### Game initialization

The first request is a `game_start` message with:

- `gameXml`: the XML rules document visible to the faction.
- `initialState`: the initial state observation.
- `playerName`: the faction controlled by this process.
- `gameNumber`: the batch number, starting at 1.
- `schemaVersion` and `requestId`: protocol metadata.

Return an acknowledgement such as:

```json
{"ready": true, "protocolVersion": 1}
```

Parse and retain rules and initial state at this point. The agent may maintain its own internal
representation for the rest of the game.

### Turn requests

At the faction's decision points, the engine sends `type: "turn_request"` with `phase`, `state`,
`nonCombat`, and the same `playerName`, `gameNumber`, and protocol metadata. Requests are sent only
for that faction's turn. Current phases include `purchase`, `tech`, `combatMove`, `battle`,
`nonCombatMove`, and `place`.

The current `state` object has `observationMode`, `observationPlayer`, `gameName`, `mapName`,
`round`, `step`, `activePlayer`, `diceSides`, `gameProperties`, `players`, and `territories`.
Player entries contain resource amounts, technologies, and units in reserve. Territory entries
contain ownership, water/impassable flags, neighbors, production/objective properties, and units.
Unit entries contain stable IDs for the game, type/owner, hits/damage, movement left/used, combat
stats, and air/sea/infrastructure flags. Unowned territories are serialized with `owner: null`.

Current actions are:

| Phase | Action object | Adapter behavior |
| --- | --- | --- |
| `purchase` | `{"purchases": [{"rule": "buyInfantry", "count": 4}]}` | Resolves production rule names from the current player's frontier and submits the requested counts. |
| `combatMove` | `{"moves": [{"from": "A", "to": "B", "unitIds": ["..."]}]}` | Resolves unit IDs and submits each route through the move delegate. |
| `battle` | `{"fightAll": true}` | Asks TripleA to resolve currently listed battles. |
| `tech` | `{}` | The first agent takes no technology action. |
| `nonCombatMove` | `{"moves": [...]}` or `{}` | Submits any returned moves; the first agent currently returns no non-combat moves. |

The earlier `purchaseCount`/`rule`/`unitType` shape remains accepted for compatibility, but new
agents should use `purchases[]` so the action describes production rules generically.

Optional fields, ignored by older agents:

| Field | Where | Adapter behavior |
| --- | --- | --- |
| `route` | each entry of `moves[]` | Full path from start to end (for example `["A", "B", "C"]`), so one move can cross several territories, e.g. an armour blitz. Without it the adapter uses the single step `from` → `to`. |
| `intent` | each entry of `moves[]` | `"RAID"` makes the adapter answer "bomb" when the engine asks whether bombers moving to that territory should bomb; `"NORMAL"` answers "attack". If omitted, the built-in AI decides. |
| `placements` | response to `place` | `[{"territory": "X", "unitIds": ["..."]}]`, applied in order. Units left unplaced are destroyed at the end of the step, as in the engine. |

The engine now also sends a `place` turn request. If the response has no `placements` field
(for example the older `placeAt` shape), the adapter keeps the built-in `WeakAi` placement.
Rejected moves and placements are logged as warnings by the engine.

## Forward model

The testbench provides a forward-model library in `planning-agent-testbed/forward-model`, so agents
never implement game rules themselves. It is independent of the engine and covers the V1 land/air
rules. It provides `validate` and `transition` on hypothetical states, explicit seeded or replayed
dice, candidate generation, a baseline policy, and an observation-to-state bridge.

* Python agents: `import triplea_fm` (put `forward-model/python` on `PYTHONPATH` or `pip install -e` it).
* C++ agents: `#include <triplea_fm/forward_model.hpp>` and build with
  `forward-model/cpp/triplea-fm-config --cflags/--libs`.

Both bindings reach the same kernel. See [the library README](../forward-model/README.md).
`agents/python-mcts` and `agents/cpp-mcts` are complete examples. `tools/arena.py` plays full games
offline with the library standing in for the engine.

If the policy encounters an error, it may return `{"error": "description"}`. The engine records
that as an agent communication/protocol error. An empty object is a valid no-action response.

## Rules, observations, and partial visibility

Use `gameXml` to derive map-specific facts such as production rules, unit costs, movement, victory
conditions, and special properties. Do not hard-code one map's production-rule names when those
facts can be discovered from XML. Use the request's state as the current engine snapshot and retain
any additional policy memory inside the process.

The current XML provider sends the common XML unchanged to each player, and the current state
provider marks observations `FULL`. `PlayerGameXmlProvider` and `StateObservationProvider` are the
engine seams for future per-player rule and state filters. Do not assume those filters already
exist.

The engine checks actions through TripleA's delegates. A returned action is a proposal, not proof
that it is legal. Use territory names and unit IDs exactly as sent in the observation, and expect an
invalid proposal to be rejected or logged by the engine.

## Registering and running an agent

1. Add the agent code and runtime dependencies under `agents/<agent-id>/`. For a Python agent,
   copy `agents/template-python/` and follow its README.
2. Add `agents/<agent-id>/agent.json` with its stable ID, display name, language, entry point, and
   launch command. Optionally list `outputDirectories` (paths relative to the agent folder, which is
   also the process working directory) so a run started with `clearAgentOutputs` empties them first;
   paths outside the agent folder are ignored. The registry discovers manifests at startup and creates a selectable entry that
   uses the shared `ExternalAgentPlayer`; do not add a Java player class or player label.
3. Add the ID to `agents/agents.json` for the human-readable inventory.
4. Assign the stable ID to faction names in a game's JSON profile under
   `planning-agent-testbed/games/<game>/config/`. The UI shows the manifest's display name.
5. Start the headed client, choose **Planning Agent Testbench → Run Simulations**, and start the
   run. Agent logs are written to
   `planning-agent-testbed/logs/agent-<faction>-game-<number>.txt`; the batch clears prior generated
   logs before game 1.

The current C++ example is in `planning-agent-testbed/agents/simple-infantry/`. See
[TESTBENCH_FLOW.md](TESTBENCH_FLOW.md) for the complete engine sequence.
