# TripleMind BC

TripleMind ([agents/tripleMind](../../tripleMind)) with one change: the attack formation for each
target is chosen by a **behaviour-cloning policy** trained on TripleMind's own MCTS decisions,
not by MCTS. Everything else is TripleMind's: the map graph, the action generators, the target
heuristic and the other phases.

| Phase | TripleMind (MCTS) | TripleMind BC |
|---|---|---|
| purchase | first purchase from `purchase_actions` | same |
| combatMove: which target | `combat_actions` heuristic (most important first) | same |
| combatMove: which formation | MCTS over the generator's formations (attack sets + skip) | BC policy over the same formations |
| nonCombatMove | `noncombat_move` | same |
| place | `{}` (engine places all purchased units) | same |
| battle | fight all | same |

Combat move, step by step (as MCTS walks down its tree): generate the formations for the next
target from the current state, let the policy pick one (possibly the skip), apply its moves with
the forward model, mark its units as committed, and repeat until the generator has no target left.
The policy sees the features it was trained on: the generator's candidates are canonicalised and
featurised by the same functions as the training data
([attack_formation_dataset.py](../../behaviour-cloning/data-processing/attack_formation_dataset.py)).

The structure follows [template-search-python](../../template-search-python/README.md): the agent
is a `SearchAgent` subclass.

## Files

| File | Purpose |
|---|---|
| `agent.py` | `TripleMindBC(SearchAgent)`: TripleMind's phases, `combat_move()` with the BC policy |
| `bc_policy.py` | `BCPolicy`: loads the checkpoint, builds the training features, picks a formation |
| `search_agent.py` | The `SearchAgent` base class, copied from template-search-python (only the log folder differs: this agent is one level deeper) |
| `agent.json` | Manifest (`id` `tripleMind-bc`; also listed in `agents/agents.json`) |
| `run-agent.sh` | Launcher: settings, `PYTHONPATH`, TripleMind's Cython build, interpreter with PyTorch |
| `docs/tripleMind-bc-uml.drawio` | UML: class diagram; sequence diagrams of start-up and every phase, and of one combat move with the BC policy; the policy's neural-network architecture |
| `../docs/tripleMind-hybrids-uml.drawio` | UML of all TripleMind variants (MCTS, BC, PPO): family class diagram, components and data flow, phase comparison, combat move in each variant |

Code it uses from elsewhere (not copied): TripleMind's `map_graph.py` and `generators.py` /
action generators, and from `behaviour-cloning/` the preprocessing feature functions and the
`bc_baseline` model.

## Requirements

1. A trained checkpoint. By default `behaviour-cloning/runs/mlp-baseline/best.pt`:

   ```bash
   cd planning-agent-testbed/agents/behaviour-cloning
   .venv/bin/python run_bc.py train --data-dir data --out runs/mlp-baseline
   ```

2. Python with PyTorch, NumPy 2 and networkx. `run-agent.sh` uses `behaviour-cloning/.venv` when
   it exists (see the [behaviour-cloning README](../../behaviour-cloning/README.md) for creating
   it); set `TRIPLEMIND_BC_PYTHON` to use another interpreter.

3. The map the policy was trained on. The checkpoint stores its feature names (territories, unit
   types, features), and the agent compares them with the game's at the first combat move. The
   current policy was trained on the two-player Capture the Flag map
   (`games/capture-the-flag-diag`); on another map, or with more than one opponent, the agent logs
   that the policy is disabled and plays the generator's first (strongest) formation instead.

## Configuration

| Variable | Default | Meaning |
|---|---|---|
| `TRIPLEMIND_BC_CHECKPOINT` | `behaviour-cloning/runs/mlp-baseline/best.pt` | Policy checkpoint |
| `TRIPLEMIND_BC_THREADS` | 1 | PyTorch threads |
| `TRIPLEMIND_BC_PYTHON` | (unset) | Interpreter; else `behaviour-cloning/.venv/bin/python`, else `TRIPLEA_TESTBENCH_PYTHON` / `python3` |
| `TRIPLEMIND_VIEW` | 0 (in `run-agent.sh`) | 1 shows TripleMind's live map window |
| `TRIPLEMIND_PURE_PYTHON` | 0 | 1 uses TripleMind's Python generators instead of the Cython build |
| `TRIPLEA_AGENT_SEED` | 0 | Seed of the agent's randomness |
| `TRIPLEA_AGENT_LOG` | 0 (in `run-agent.sh`) | 1 logs to stderr and `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt` |

With logging on, each combat decision is logged with the policy's choice and its probability:

```
22:58:59   RussianStartLeft: attack 1 of 4 (p=0.68) combat on RussianStartLeft: 2 units, strength=2
22:59:02   ItalianStart: skip 0 of 1 (p=1.00) combat: skip ItalianStart
```

## Testing

From `planning-agent-testbed/`:

```bash
# BC (Russians) against TripleMind MCTS (Germans) on the map the policy was trained on
python3 tools/arena.py --xml games/capture-the-flag-diag/map/capture_the_flag/map/games/capture_the_flag.xml \
  --agent Russians=agents/tripleMind_hybrids/tripleMind-bc/run-agent.sh \
  --agent Germans=agents/tripleMind/run-agent.sh --games 2

# the protocol by hand
printf '%s\n' '{"type":"game_start","playerName":"Test","gameXml":"<game/>"}' \
              '{"type":"turn_request","phase":"battle","state":{}}' \
  | bash agents/tripleMind_hybrids/tripleMind-bc/run-agent.sh
```

In the testbench, assign `tripleMind-bc` to a faction in a game profile and run it from
**Planning Agent Testbench → Run Simulations**; the testbench finds `agent.json` in this nested
folder (it searches `agents/` three levels deep).
