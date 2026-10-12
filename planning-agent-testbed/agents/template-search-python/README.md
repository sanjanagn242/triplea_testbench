# Python search-agent template

Starting point for an agent that searches with the forward model (`triplea_fm`). The agent is a
class: subclass `SearchAgent` and implement one method, `search()`. The base class does the rest:
the protocol, logging, loading the map, building the forward-model state from each observation,
time budgets, checking your actions with the kernel and encoding the answer.

Unchanged, the template is a working agent: its example `search()` is a greedy one-step lookahead
(see `agent.py`). Use [template-python](../template-python/README.md) instead if you want to write
the per-phase JSON answers yourself without the forward model.

## Files

| File | Edit? | Purpose |
|---|---|---|
| `agent.py` | yes | `MyAgent(SearchAgent)`: your `search()` and optional hooks; `MyAgent().run()` |
| `search_agent.py` | normally no | The `SearchAgent` base class (protocol, state, checking, encoding, helpers) |
| `agent.json` | yes | Manifest: change `id`, `name`, `description` |
| `run-agent.sh` | rarely | Launcher: sets `TRIPLEA_AGENT_LOG`, puts `forward-model/python` on `PYTHONPATH` |

## Making a new agent

1. Copy the folder: `cp -r agents/template-search-python agents/<your-agent-id>`.
2. In the new `agent.json`, set `id` to `<your-agent-id>` (it must match the folder name and be
   unique) and change `name` and `description`.
3. Add `<your-agent-id>` to `agents/agents.json`.
4. Implement `search()` in `agent.py` (rename the class if you like).
5. Test offline with `tools/arena.py` (below), then assign the id to a faction in a game profile
   and run it from **Planning Agent Testbench → Run Simulations**.

## What you implement

```python
class MyAgent(SearchAgent):
    def search(self, root, phase, deadline):
        ...
        return [action, action, ...]
```

| Argument | Meaning |
|---|---|
| `root` | `triplea_fm.SimulationState` at this player's decision for `phase`. Do not modify it; work on `root.clone()` |
| `phase` | `"purchase"`, `"combatMove"`, `"nonCombatMove"` or `"place"` |
| `deadline` | `time.time()` value to finish by (`time_budget(phase)` seconds after the request arrived) |
| return | The actions for the rest of the phase, in order: `fm.PurchaseAction`, `fm.MoveAction`, `fm.PlaceAction`. `[]` ends the phase without acting; anything after an `fm.EndPhaseAction` is ignored |

What the base class does with the result:

* replays the actions in order on a clone with the kernel and drops (and logs) any it rejects,
  including actions that no longer fit because dice changed the state (e.g. fly-over AA);
* merges purchases into one `{"purchases": [...]}` answer, sends moves with full routes and
  intents (`RAID` for bombing) and placements, translating unit ids back to the engine's.

Optional hooks:

| Method | Default | Override to |
|---|---|---|
| `setup()` | nothing | Build per-game tables once (after `self.d`, `self.me` are set) |
| `battle(observation)` | `{"fightAll": true}` | Protocol v1 offers nothing else; the engine picks casualties and retreats |
| `time_budget(phase)` | `seconds × PHASE_TIME_SHARE[phase]` | Change the time per phase |

## Helpers available in `search()`

| Helper | Does |
|---|---|
| `self.legal_actions(s)` | Toolkit candidates for `s.decision`: a compact, always-legal subset |
| `self.legal_actions(s, exhaustive=True)` | Every legal action (`triplea_fm.actionspace`); can be thousands of moves |
| `self.simulate(s, a, rng=None)` | New state after `a`, or `None` if rejected (`s` unchanged) |
| `self.apply(s, a, rng)` | Apply `a` to `s` in place (fast, for clones); `False` if rejected |
| `self.same_phase(root, s)` | `True` while `s` is still the phase that started at `root` |
| `self.new_rng()` | Fresh game dice (`fm.SeededRng`) seeded from the agent's own generator |
| `self.rollout(s, rng, rounds=2)` | Play all players with the library's `HeuristicPolicy`, in place |
| `self.evaluate(s)` | This player's value of `s` in [0, 1] (`baselines.evaluate`) |
| `self.log(message)` | Diagnostics (only when `TRIPLEA_AGENT_LOG` is 1); never `print` to stdout |

Attributes: `self.d` (`GameDefinition`), `self.me` (player index), `self.player_name`,
`self.rnd` (`random.Random` for the agent's own choices), `self.precomp` (distances, unit costs,
territory values), `self.policy` (`HeuristicPolicy`), `self.bridge` (`LiveBridge`). Keep game
dice (`new_rng()`) and the agent's own randomness (`self.rnd`) separate. The whole `triplea_fm`
library can also be used directly; see the [library README](../../forward-model/python/README.md).

If the map is outside the forward model's profile (or its XML cannot be read), `search()` is never called: the agent logs the
reasons and answers every request with an empty action.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TRIPLEA_AGENT_SECONDS` | 2.0 | Base search time; ×0.6 purchase, ×1.5 combat move, ×0.8 noncombat move, ×0.3 place |
| `TRIPLEA_AGENT_SEED` | 0 | Seed of `self.rnd` and of `new_rng()` |
| `TRIPLEA_AGENT_LOG` | 0 (in `run-agent.sh`) | 1 logs every phase: time, accepted/rejected actions, the answer |
| `TRIPLEA_TESTBENCH_PYTHON` | python3 | Interpreter used by `run-agent.sh` |

## Testing

```bash
# offline games, the forward model standing in for the engine (--seconds sets TRIPLEA_AGENT_SECONDS)
python3 planning-agent-testbed/tools/arena.py \
  --agent Russians=planning-agent-testbed/agents/template-search-python/run-agent.sh --games 2 --seconds 0.5

# the protocol by hand
printf '%s\n' '{"type":"game_start","playerName":"Test","gameXml":"<game/>"}' \
              '{"type":"turn_request","phase":"battle","state":{}}' | bash run-agent.sh
```

The arena reports `rejected=`; the base class already filters illegal actions, so it should be 0.
