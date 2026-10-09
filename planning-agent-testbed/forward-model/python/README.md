# triplea_fm — forward-model kernel and Python API (profile `forward-model-v1`)

This is the kernel of the testbench forward-model library; see [../README.md](../README.md) for
the library layout and the C++ binding.

A standalone, algorithm-independent simulator of TripleA game dynamics, built from
[forward-model-implementation-spec.md](../../forward-model-implementation-spec.md) and the rule
traces in [forward-model-rule-model.md](../../forward-model-rule-model.md). Agents use it for
search (MCTS), planning, RL environments and evaluation. It has no dependency on a running
TripleA game. Pure Python 3.10+, no third-party packages.

## Quick start

```python
import sys; sys.path.insert(0, "planning-agent-testbed/forward-model/python")
import triplea_fm as fm
from triplea_fm import toolkit

xml = open("planning-agent-testbed/games/capture-the-flag/map/capture_the_flag/map/games/capture_the_flag.xml").read()
definition, report = fm.load_definition(xml)          # raises UnsupportedDefinition if refused
state = fm.create_initial_state(definition)            # Russians' purchase decision, round 1
rng = fm.SeededRng(seed=42)

while not fm.is_terminal(state):
    action = toolkit.candidates(definition, state)[0]  # your agent chooses here
    result = fm.transition(definition, state, action, rng)
    state, rng = result.next_state, result.next_rng    # inputs are never mutated
print(fm.outcome(state).winners)
```

## Public API (spec §12)

| Function | Contract |
|---|---|
| `load_definition(xml, overrides=None, strict=True)` | XML → immutable `GameDefinition` + `SupportReport`. Unsupported/uncertified features are refused with reasons. |
| `create_initial_state(d)` | Applies the setup recipe once and advances to the first decision. |
| `create_state(d, snapshot)` | Installs a complete hypothetical state after invariant checks (never merges with setup). |
| `validate(d, s, a)` / `validate_batch(d, s, actions)` | Pure legality checks; consume no randomness. |
| `transition(d, s, a, rng)` | Applies one action, then automatic and chance effects up to the next decision or terminal state. Returns `TransitionResult(status, next_state, validation, next_rng, events, chance_outcomes, error)`. Rejected/failed calls return the input state and RNG unchanged. |
| `apply_in_place(d, s, a, rng)` | Same semantics, mutating `s` and `rng` (for disposable rollout states; ~5x faster, no undo). |
| `clone`, `is_terminal`, `outcome`, `serialize_state`, `deserialize_state` | As named; snapshots are canonical JSON. |

Decisions (`state.decision.kind`): `PURCHASE`, `MOVE`, `BATTLE_SELECTION`, `CASUALTIES`, `RETREAT`,
`PLACE`. `decision.actor` may be any player, including the null player (`fm.NULL_PLAYER`) when
neutral defenders choose casualties. Actions: `PurchaseAction`, `MoveAction(units, route, intent)`
(`intent="RAID"` for strategic bombing), `SelectBattleAction`, `SelectCasualtiesAction`,
`RetreatAction`, `PlaceAction`, `EndPhaseAction`.

Randomness is explicit: `SeededRng` (counter-based SplitMix64, cloning = copying) or `TapeRng`
(strict replay of recorded draws). Agent randomness must use a separate generator.

## Agent-side helpers (not part of the kernel)

* `triplea_fm.toolkit` — the ActionToolkit (spec §23.3): candidate generators for every decision
  kind, all filtered through `validate_batch` so every candidate is legal, plus
  `default_casualties` / `default_casualty_order` and `unit_routes` (route candidates for any
  player's units, also in a hypothetical next-turn move phase).
* `triplea_fm.actionspace` — `legal_actions(d, s, distinct_units=False)`, the exhaustive legal
  action set of the pending decision (a generator; every action passes `validate`, and nothing
  legal is missed). Interchangeable units are collapsed by default, so each legal action appears
  once up to renaming unit ids; `distinct_units=True` enumerates every unit-id subset. From an
  agent's observation: `LiveBridge.legal_actions(observation, phase)`; C++:
  `model.legal_actions(state, distinct_units, limit)`. Unlike `toolkit.candidates`, sets can be
  large (thousands of moves late in a Capture the Flag game).
* `triplea_fm.battlesim` — `simulate_battle`, the counterpart of TripleA's odds calculator: fights
  a hypothetical battle (units given as `(type, owner)` pairs) many times under the kernel's
  battle rules and reports win/draw rates, survivors per run and the run closest to the average.
* `triplea_fm.testbench` — `LiveBridge` builds a `SimulationState` from a live testbench
  observation (protocol v1) and converts actions back to the testbench JSON format;
  `observation_from_state` does the reverse for offline arenas. Its belief assumptions are listed
  in the module docstring.
* `triplea_fm.baselines` — `HeuristicPolicy`, a baseline that plays every decision for any
  player (the arena's reference opponent, usable for rollouts), and `evaluate`, a generic
  state score.
* `triplea_fm.flat` — the plain-data binding API used by the C++ binding and future bindings.

## Supported scope

V1 land/air rules, traced from the pinned engine: phase sequencing with zero-bid skips;
purchases with frontier and capital gates; land and air movement with neutral, blitz, newly
captured territory and enemy blocking rules; air landing checks and removal of air that cannot
land; fly-over AA; land/air battles with defensive AA, artillery support, simultaneous fire,
casualty decisions (automatic where the engine auto-selects), attacker retreat and stalemates;
strategic bombing raids (treasury damage, WW2V2 per-bomber cap); capture, infrastructure capture,
liberation and capital treasury transfer; placement with original/captured factory capacities;
income with the capital gate; Total Victory checks at end of round.

Map-independent settings beyond CTF's defaults, traced from the same engine revision:

| Property | Effect in the model |
|---|---|
| `Always on AA` | Fly-over AA also fires during noncombat moves (`AaInMoveUtil.getTerritoriesWhereAaWillFire`), so noncombat moves can draw dice |
| `Territory Turn Limit` | The engine's `PU_CAP`: a territory's total bombing loss in one turn is capped at its production, on top of the WW2V2 per-bomber cap (`StrategicBombingRaidBattle`) |
| `neutralCharge` ≠ 0 | Admitted only as inert, when every null-owned land territory is impassable (the charge then can never apply). Otherwise refused; moves into injected passable neutral land are rejected |

Capture the Flag and Minimap load as supported (Minimap's sea units and sea zones are declared but
never placed or purchasable, so they are inert like CTF's). Star Wars is refused with explicit
reasons (naval units, transports, triggers, politics and many uncatalogued properties).

## Status against the spec

This is a working, tested implementation, **not a conformance-certified one**:

* **No offline TripleA oracle or differential suite yet** (spec §17–18, Stage 3). Rules were
  traced from source and are covered by unit tests, but not yet replayed against the engine.
  The opt-in live test (`ExternalAgentLiveGameTest`) checks that the agent plays legally in the
  real engine, which is weaker evidence.
* **ADR-KERNEL/ADR-LANGUAGE**: chosen provisionally as one Python kernel. C++ reaches the same
  kernel through the embedded-interpreter binding in `../cpp`; Java bindings and the binary
  service are not built yet.
* **Normalization** reimplements the parser defaults V1 needs in Python instead of reusing
  TripleA's Java `GameParser`. Unknown properties, attachments and unit options are refused
  rather than guessed.
* **ADR-RNG**: counter-based SplitMix64 (`splitmix64-ctr` v1). Dice are assigned to units
  strongest-first, then by unit id. Exact engine dice ordering is an oracle question.
* Actions carry no `ActionHeader` (digest/decision id); the actor is implied by the decision.
* Serialization is canonical JSON only; there is no binary format.
* Unresolved gates (spec §22) are refused, not guessed: simultaneous victory returns
  `FAILED/UNCERTIFIED_CONFIGURATION`; multi-hit units, non-PU resources, nonzero bids, technology
  and construction placement are refused.

## Tests

```bash
cd planning-agent-testbed/forward-model/python
python3 -m unittest discover tests
```

Tests cover normalization baselines (CTF and Minimap), refusals, atomicity of rejected actions, determinism and
serialization round trips, neutral/blitz/air/AA movement rules, battles (simultaneous fire,
casualty and retreat decisions, capture), capital capture, raids, placement capacities, income,
victory timing, Minimap's noncombat AA, per-turn raid cap and inert neutral charge, the battle
simulator (including dice-for-dice agreement with a kernel battle), the exhaustive action set
(soundness and completeness against the toolkit over random games), and toolkit
soundness over random games.
