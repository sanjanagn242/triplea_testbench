"""Plain-data binding API of the forward-model library (spec §14).

This is the single entry point that non-Python bindings use. The C++ binding
(forward-model/cpp/include/triplea_fm/forward_model.hpp) embeds the Python interpreter and calls
these functions; future bindings (Java/JNI, a binary service) should call the same functions. All
values crossing the boundary are ints, strs, or (nested) lists of ints. States and RNGs stay here
and are referred to by integer handles, one interpreter per process.

Action encoding (list of ints):
  [0]                                   EndPhase
  [1, rule, count, rule, count, ...]    Purchase
  [2, intent, n, u1..un, t0, t1, ...]   Move (intent 0 NORMAL, 1 RAID; route includes the start)
  [3, battle_id]                        SelectBattle
  [4, batch_id, u1, ...]                SelectCasualties
  [5, choice, destination]              Retreat (choice 0 CONTINUE, 1 RETREAT; destination -1 if none)
  [6, territory, u1, ...]               Place

Status codes returned by apply(): 0 DECISION, 1 TERMINAL, 2 REJECTED, 3 FAILED.
static_info() and view() document the flat integer layouts.

Functions:
  definitions  load_xml(xml) / start_game() (testbench game_start), messages(), static_info()
  states       initial_state(), state_from_request(), clone(h), free(h), view(h), snapshot_json(h)
  dynamics     rng_new(seed), rng_free(r), validate(h, a), apply(h, a, r), candidates(h)
  protocol     parse_request(line), encode_response(phase, actions)
"""

from __future__ import annotations

import json

from . import kernel, rules, toolkit
from .actions import (CONTINUE, NORMAL, RAID, RETREAT, EndPhaseAction, MoveAction, PlaceAction,
                      PurchaseAction, RetreatAction, SelectBattleAction, SelectCasualtiesAction)
from .definition import BATTLE, COMBAT_MOVE, NONCOMBAT_MOVE, PLACE, PURCHASE, load_definition
from .rng import SeededRng
from .testbench import LiveBridge

_STEP_CODE = {PURCHASE: 1, COMBAT_MOVE: 2, BATTLE: 3, NONCOMBAT_MOVE: 4, PLACE: 5}
_DEC_CODE = {"PURCHASE": 0, "MOVE": 1, "BATTLE_SELECTION": 2, "CASUALTIES": 3, "RETREAT": 4, "PLACE": 5}

_d = None
_live = None
_messages = []
_request = None
_states = {}
_rngs = {}
_next = [1]


def _handle(table, obj):
    h = _next[0]
    _next[0] += 1
    table[h] = obj
    return h


# ---- protocol helpers ----

def parse_request(line: str):
    """Returns [type, phase, playerName, gameNumber] and remembers the request."""
    global _request
    _request = json.loads(line)
    return [str(_request.get("type", "")), str(_request.get("phase", "")),
            str(_request.get("playerName", "")), int(_request.get("gameNumber", 1))]


def start_game() -> int:
    """Load the remembered game_start request. Returns the controlled player id, or -1 if the
    forward model refuses the map (reasons in messages())."""
    global _d, _live, _messages
    _messages = []
    d, report = load_definition(_request["gameXml"], strict=False)
    if not report.supported:
        _messages = [f"{f.code} {f.feature_id}={f.effective_value}: {f.explanation}" for f in report.features]
        _d = None
        return -1
    _d = d
    _live = LiveBridge(d, _request["playerName"])
    _messages = [f"definition {d.definition_digest[:12]}"]
    return d.player_index[_request["playerName"]]


def load_xml(xml: str) -> int:
    """Load a game definition directly (no testbench request). Returns 1 if supported, else 0
    (reasons in messages())."""
    global _d, _live, _messages
    d, report = load_definition(xml, strict=False)
    _messages = [f"{f.code} {f.feature_id}={f.effective_value}: {f.explanation}" for f in report.features]
    _d = d if report.supported else None
    _live = None
    return 1 if report.supported else 0


def initial_state() -> int:
    """Handle of the setup state at its first decision."""
    return _handle(_states, kernel.create_initial_state(_d))


def snapshot_json(h: int) -> str:
    """Canonical JSON snapshot of a state (for logging, checkpoints or other processes)."""
    return json.dumps(_states[h].to_json(), sort_keys=True)


def messages():
    return list(_messages)


def state_from_request() -> int:
    """Handle of the state built from the remembered turn_request, or -1 (reason in messages())."""
    global _messages
    try:
        s = _live.build_state(_request["state"], _request["phase"])
        kernel.create_state(_d, s)
    except Exception as e:  # incomplete or unsupported observation
        _messages = [repr(e)]
        return -1
    return _handle(_states, s)


def encode_response(phase: str, actions) -> str:
    """Testbench JSON for the chosen action sequence of a phase."""
    acts = [decode(a) for a in actions]
    if phase == "purchase":
        merged = {}
        for a in acts:
            if a.kind == "purchase":
                for r, c in a.quantities:
                    merged[r] = merged.get(r, 0) + c
        return json.dumps(_live.purchase_to_json(PurchaseAction(tuple(sorted(merged.items())))))
    if phase in ("combatMove", "nonCombatMove"):
        return json.dumps({"moves": [_live.move_to_json(a) for a in acts if a.kind == "move"]})
    if phase == "place":
        return json.dumps({"placements": [_live.place_to_json(a) for a in acts if a.kind == "place"]})
    return "{}"


# ---- handles ----

def clone(h: int) -> int:
    return _handle(_states, _states[h].clone())


def free(h: int) -> None:
    _states.pop(h, None)


def rng_new(seed: int) -> int:
    return _handle(_rngs, SeededRng(seed))


def rng_free(r: int) -> None:
    _rngs.pop(r, None)


# ---- actions ----

def decode(a):
    k = a[0]
    if k == 0:
        return EndPhaseAction()
    if k == 1:
        return PurchaseAction(tuple((a[i], a[i + 1]) for i in range(1, len(a), 2)))
    if k == 2:
        n = a[2]
        return MoveAction(tuple(a[3:3 + n]), tuple(a[3 + n:]), RAID if a[1] == 1 else NORMAL)
    if k == 3:
        return SelectBattleAction(a[1])
    if k == 4:
        return SelectCasualtiesAction(a[1], tuple(a[2:]))
    if k == 5:
        return RetreatAction(RETREAT if a[1] == 1 else CONTINUE, None if a[2] < 0 else a[2])
    if k == 6:
        return PlaceAction(tuple(a[2:]), a[1])
    raise ValueError(f"bad action code {a}")


def encode(a):
    k = a.kind
    if k == "end_phase":
        return [0]
    if k == "purchase":
        return [1] + [x for rc in a.quantities for x in rc]
    if k == "move":
        return [2, 1 if a.intent == RAID else 0, len(a.units)] + list(a.units) + list(a.route)
    if k == "select_battle":
        return [3, a.battle_id]
    if k == "casualties":
        return [4, a.hit_batch_id] + list(a.killed)
    if k == "retreat":
        return [5, 1 if a.choice == RETREAT else 0, -1 if a.destination is None else a.destination]
    if k == "place":
        return [6, a.territory] + list(a.units)
    raise ValueError(k)


def candidates(h: int):
    s = _states[h]
    return [encode(a) for a in toolkit.candidates(_d, s)] if s.decision is not None else []


def validate(h: int, action) -> int:
    return 1 if kernel.validate(_d, _states[h], decode(action)).ok else 0


def apply(h: int, action, r: int) -> int:
    try:
        st = kernel.apply_in_place(_d, _states[h], decode(action), _rngs[r])
    except Exception:
        return 3
    return {kernel.DECISION: 0, kernel.TERMINAL: 1, kernel.REJECTED: 2}[st]


# ---- flat layouts ----

def static_info():
    """[nT, nP, nUT, nRules, nAlliances, treasury,
        per territory: production, victory, capital_of, water, impassable, nNeighbors, neighbors...,
        per player: alliance,
        per unit type: domain, movement, attack, defense, can_blitz, infrastructure,
                       can_move_in_combat, artillery, supportable, strategic_bomber, factory, cost,
        per rule: treasury cost, other-resource cost flag, result type (-1 if none/multi), result qty,
        per player: nFrontier, rule ids...]"""
    d = _d
    out = [len(d.territory_names), len(d.player_names), len(d.unit_types), len(d.production_rules),
           len(d.alliance_names), d.treasury_resource]
    for t in range(len(d.territory_names)):
        out += [d.production[t], d.victory_value[t], d.capital_of[t], int(d.is_water[t]),
                int(d.impassable[t]), len(d.neighbors[t])] + list(d.neighbors[t])
    out += list(d.alliance_of)
    for ut in d.unit_types:
        out += [ut.domain, ut.movement, ut.attack, ut.defense, int(ut.can_blitz), int(ut.is_infrastructure),
                int(ut.can_move_in_combat), int(ut.is_artillery), int(ut.is_artillery_supportable),
                int(ut.is_strategic_bomber), int(ut.is_factory), d.unit_cost(ut.id)]
    for rule in d.production_rules:
        tre = sum(a for r, a in rule.costs if r == d.treasury_resource)
        other = int(any(r != d.treasury_resource for r, _ in rule.costs))
        if len(rule.results) == 1:
            out += [tre, other, rule.results[0][0], rule.results[0][1]]
        else:
            out += [tre, other, -1, 0]
    for p in range(len(d.player_names)):
        out += [len(d.frontier[p])] + list(d.frontier[p])
    return out


def view(h: int):
    """[round, step_id, step_code, decision_code(-1 none), actor, terminal, winner_alliance(-1),
        nT, owners..., conquered flags...,
        nP, balances (treasury)...,
        nU, per unit: id, type, owner, loc(-2 held), movement_left, in_combat,
        nCaps, per producer: territory, cap(-1 unlimited), used,
        decision battle id(-1), hit batch id(-1), hits, side(1 = defenders are hit), nEligible, ids...,
        nDestinations, territories...,
        nBattles, per battle: id, site, attacker, kind(0 normal, 1 raid), active]"""
    d = _d
    s = _states[h]
    dec = s.decision
    step = d.steps[s.step]
    out = [s.round, s.step, _STEP_CODE.get(step.kind, 0), -1 if dec is None else _DEC_CODE[dec.kind],
           -99 if dec is None else dec.actor, int(s.terminal is not None),
           s.terminal.winners[0] if s.terminal is not None and s.terminal.winners else -1]
    nt = len(d.territory_names)
    out.append(nt)
    out += s.owner
    out += [1 if t in s.conquered else 0 for t in range(nt)]
    out.append(len(d.player_names))
    out += [b[d.treasury_resource] for b in s.balances]
    out.append(len(s.units))
    for uid in sorted(s.units):
        u = s.units[uid]
        out += [uid, u.type, u.owner, u.loc, rules.movement_left(d, u), int(u.in_combat)]
    caps = s.placement_cap or {}
    out.append(len(caps))
    for t in sorted(caps):
        out += [t, caps[t], s.placement_used.get(t, 0)]
    if dec is not None and dec.battle_id is not None:
        out.append(dec.battle_id)
        hb = dec.hit_batch
        if hb is not None:
            out += [hb.id, hb.hits, 1 if hb.side == "DEFENDER" else 0, len(hb.eligible)] + list(hb.eligible)
        else:
            out += [-1, 0, 0, 0]
        if dec.kind == "RETREAT":
            dests = rules.retreat_destinations(d, s, s.battles[dec.battle_id])
            out += [len(dests)] + dests
        else:
            out.append(0)
    else:
        out += [-1, -1, 0, 0, 0, 0]
    out.append(len(s.battles))
    for bid in sorted(s.battles):
        b = s.battles[bid]
        out += [bid, b.site, b.attacker, 1 if b.kind == "RAID" else 0, 1 if b.status == "ACTIVE" else 0]
    return out
