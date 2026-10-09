# Python Does Nothing

A Python agent that answers every turn request, in every phase, with an empty action (`{}`). It
buys, moves and places nothing. Use it as a passive opponent or as a baseline when testing other
agents.

| Request | Response |
|---|---|
| `game_start` | `{"ready": true, "protocolVersion": 1}` |
| `turn_request` (any phase) | `{}` |

Logging follows the testbed rule: the agent writes to stderr and to
`planning-agent-testbed/logs/agent-<faction>-game-<n>.txt` only when `TRIPLEA_AGENT_LOG` is `1`,
which `run-agent.sh` sets.

To write a new Python agent, start from [template-python](../template-python/README.md) instead.
