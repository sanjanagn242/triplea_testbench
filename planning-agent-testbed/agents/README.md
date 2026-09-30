# Planning agents

Agent implementations live in one folder per agent. `agents.json` is the inventory of agents
implemented for this testbed. To make an agent selectable in the headed client, register its player
type in `TestbenchAgentRegistry` and connect that type to an engine-side adapter. The adapter starts
one process per assigned faction at game initialization. It sends one `game_start` JSON message with
that player's game XML and initial state, then sends `turn_request` messages with the current state
at that faction's decision phases. The agent returns one JSON action per request, all over
newline-delimited standard input/output. TripleA delegates validate and apply actions returned by
the agent.

## Simple Infantry (Python)

`simple-infantry/agent.py` is the first external agent. It reads the player's XML at game start,
buys as many infantry units as it can afford, advances groups toward enemy capitals, and asks the
engine to resolve its available battles. Placement currently falls back to TripleA's built-in
`WeakAi` and does not request an agent action. The Python agent uses only the standard library and
retains its own state in memory for the duration of the game.

The engine launches `python3` by default. If the GUI launcher has a restricted `PATH`, set the
executable explicitly before starting the game:

```bash
TRIPLEA_TESTBENCH_PYTHON=/usr/bin/python3 ./gradlew :game-headed:run
```

The equivalent Java system property is `-Dtriplea.testbench.python=/path/to/python3`.

At the beginning of a simulation batch, the testbench refreshes its generated logs. Each faction's
agent writes its startup, received phase requests, round, game, returned actions, and protocol errors
to `planning-agent-testbed/logs/agent-<faction>-game-<number>.txt`. The first line is the simulation
number provided by the engine. The engine console log also includes the detailed cause if
communication with the process fails.

The agent computes candidate adjacent territory moves from the observation's territory connections,
unit owners, and movement remaining. The game engine remains authoritative and rejects any illegal
proposal. The bot is intentionally basic, so it does not optimize battle odds, transports,
technology, or special map rules. The current XML provider sends the common XML unchanged;
`PlayerGameXmlProvider` is the per-player seam for a visibility filter. State is currently marked
`FULL` and contains the same map state for each faction.

To select it in a testbench JSON configuration, use the exact label:

```json
"Russians": "Simple Infantry (Python)"
```

Use `capture-the-flag-simple-vs-easy.json` for one Python agent against an Easy AI, or
`capture-the-flag-four-simple-agents.json` to run four copies against one another.
