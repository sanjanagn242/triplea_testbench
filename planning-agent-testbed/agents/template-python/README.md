# Python agent template

Starting point for a new Python agent. The protocol loop, logging and launcher are already in
place; you only write the decision for each phase. Unchanged, the template is a valid agent that
buys, moves and places nothing and lets TripleA resolve its battles.

## Files

| File | Purpose |
|---|---|
| `agent.json` | Manifest the engine reads to discover and launch the agent (ID, display name, command) |
| `run-agent.sh` | Launcher. Sets `TRIPLEA_AGENT_LOG`, puts `forward-model/python` on `PYTHONPATH` and runs `agent.py` |
| `agent.py` | Protocol loop plus one method per phase for you to fill in |

## Making a new agent

1. Copy the folder: `cp -r agents/template-python agents/<your-agent-id>`.
2. In the new `agent.json`, set `id` to `<your-agent-id>` (it must match the folder name and be
   unique) and change `name` (shown in the faction dropdown) and `description`.
3. Add `<your-agent-id>` to `agents/agents.json`.
4. Fill in the phase methods in `agent.py` (below). Put per-game setup in `Agent.game_start()`.
5. If the agent needs Python packages, add a `requirements.txt` to its folder.
6. Assign the ID to a faction in a game profile under `planning-agent-testbed/games/<game>/config/`
   and start a run from **Planning Agent Testbench → Run Simulations**.

The engine lists every folder with an `agent.json`, so the template itself also appears in the
dropdown as **Python Template**. Leave its ID as `template-python`; two manifests with the same ID
or name conflict.

## Phase methods

`Agent.turn()` sends each `turn_request` to the method for its phase, passing the current
observation (`request["state"]`). Phases without a method get `{}`.

| Phase | Method | Template returns | Fill in with |
|---|---|---|---|
| `purchase` | `purchase` | `{}` | `{"purchases": [{"rule": "buyInfantry", "count": 4}]}` |
| `combatMove` | `combat_move` | `{}` | `{"moves": [{"from": "A", "to": "B", "unitIds": ["..."]}]}` |
| `battle` | `battle` | `{"fightAll": true}` | Usually left as is |
| `nonCombatMove` | `non_combat_move` | `{}` | `{"moves": [...]}` |
| `place` | `place` | `{}` | `{"placements": [{"territory": "X", "unitIds": ["..."]}]}` |

See [AGENT_DEVELOPER_GUIDE.md](../../docs/AGENT_DEVELOPER_GUIDE.md) for the observation format and
the optional move fields (`route`, `intent`).

## Rules

- Write exactly one single-line JSON response per request on stdout, and nothing else.
- Send diagnostics to `self.log(...)`. It writes to stderr and to
  `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt`, but only when `TRIPLEA_AGENT_LOG` is
  `1`. `run-agent.sh` sets it; change it there to turn logging on or off. Any logging you add must
  respect the same flag.
- `main()` catches exceptions and replies with `{"error": ...}`, so a bug in a phase method does
  not break the protocol loop.

## Using the forward model

`run-agent.sh` makes the testbench forward-model library importable, so an agent can simulate the
game:

```python
import triplea_fm as fm
from triplea_fm.testbench import LiveBridge

definition, report = fm.load_definition(request["gameXml"], strict=False)  # in game_start
bridge = LiveBridge(definition, self.player)
state = bridge.build_state(request["state"], phase)                        # in a phase method
```

[python-mcts](../python-mcts/README.md) is a complete agent built this way.

## Running it standalone

```bash
printf '%s\n' '{"type":"game_start","playerName":"Test"}' \
              '{"type":"turn_request","phase":"battle","state":{}}' | bash run-agent.sh
```
