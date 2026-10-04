"""Agent-side ActionToolkit (spec §23.3).

Optional candidate generators and default decision helpers that agents may use, replace or ignore.
This module is agent code: it uses only the kernel's public API and read-only state access, and
the kernel never imports it.

Soundness: every generator filters its candidates through kernel.validate_batch, so every returned
action is valid in the supplied state. Completeness is documented per generator; none of them
ranks, scores or prunes by strategy except where a function name says "default_".
"""

from __future__ import annotations

from collections import deque
from itertools import combinations_with_replacement
from typing import Dict, List, Optional, Sequence, Tuple

from . import kernel
from .actions import (CONTINUE, NORMAL, RAID, RETREAT, EndPhaseAction, MoveAction, PlaceAction,
                      PurchaseAction, RetreatAction, SelectBattleAction, SelectCasualtiesAction)
from .definition import COMBAT_MOVE, NULL_PLAYER, GameDefinition
from .state import (BATTLE_SELECTION_D, CASUALTIES_D, MOVE_D, PLACE_D, PURCHASE_D, RETREAT_D,
                    SimulationState)
from . import rules


def _valid(d, s, actions):
    return [a for a, v in zip(actions, kernel.validate_batch(d, s, actions)) if v.ok]


# ---------------------------------------------------------------------------------------------
# Purchases
# ---------------------------------------------------------------------------------------------

def purchase_candidates(d: GameDefinition, s: SimulationState, max_types: int = 2,
                        include_empty: bool = True) -> List[PurchaseAction]:
    """Spend-everything bundles over 1..max_types frontier rules in several ratios.

    Not complete (the full space is every affordable vector); it covers 'all-in' bundles for each
    rule and mixed bundles for every pair of rules at ratios 1:1, 2:1 and 1:2.
    """
    p = s.decision.actor
    r = d.treasury_resource
    budget = s.balances[p][r]
    rules_ = [rid for rid in d.frontier[p]
              if all(res == r for res, _ in d.production_rules[rid].costs)]
    cost = {rid: sum(a for _, a in d.production_rules[rid].costs) for rid in rules_}
    rules_ = [rid for rid in rules_ if 0 < cost[rid] <= budget]
    seen, out = set(), []

    def add(counts: Dict[int, int]):
        key = tuple(sorted((k, v) for k, v in counts.items() if v > 0))
        if key and key not in seen:
            seen.add(key)
            out.append(PurchaseAction(key))

    cheapest = min(rules_, key=lambda x: cost[x]) if rules_ else None
    for rid in rules_:
        n = budget // cost[rid]
        counts = {rid: n}
        rest = budget - n * cost[rid]
        if cheapest is not None and rest >= cost[cheapest]:
            counts[cheapest] = counts.get(cheapest, 0) + rest // cost[cheapest]
        add(counts)
    if max_types >= 2:
        for a, b in combinations_with_replacement(rules_, 2):
            if a == b:
                continue
            for wa, wb in ((1, 1), (2, 1), (1, 2)):
                unit = wa * cost[a] + wb * cost[b]
                k = budget // unit
                if k == 0:
                    continue
                counts = {a: wa * k, b: wb * k}
                rest = budget - k * unit
                if cheapest is not None and rest >= cost[cheapest]:
                    counts[cheapest] = counts.get(cheapest, 0) + rest // cost[cheapest]
                add(counts)
    if include_empty:
        out.append(EndPhaseAction())
    return _valid(d, s, out)


# ---------------------------------------------------------------------------------------------
# Moves
# ---------------------------------------------------------------------------------------------

def _passable_middle(d, s, p, t, combat, group_air, group_blitz) -> bool:
    if d.impassable[t]:
        return False
    if group_air:
        return not rules.is_neutral_land(d, s, t)
    if d.is_water[t]:
        return False
    owner = s.owner[t]
    friendly = owner != NULL_PLAYER and d.is_allied(p, owner)
    if friendly and not (t in s.conquered and t not in s.blitzed):
        return not rules.enemy_combat_units(d, s, p, t)
    if not combat:
        return False
    return group_blitz and rules.is_blitzable(d, s, p, t)


def reachable_routes(d: GameDefinition, s: SimulationState, start: int, max_steps: int,
                     group_air: bool, group_blitz: bool) -> Dict[int, Tuple[int, ...]]:
    """One shortest plausible route per destination within max_steps (BFS over territories a group
    of this kind could pass through). Routes are candidates; callers must still validate."""
    p = s.decision.actor
    combat = d.steps[s.step].kind == COMBAT_MOVE
    parent = {start: None}
    depth = {start: 0}
    q = deque([start])
    while q:
        t = q.popleft()
        if depth[t] >= max_steps:
            continue
        if t != start and not _passable_middle(d, s, p, t, combat, group_air, group_blitz):
            continue
        for n in d.neighbors[t]:
            if n in parent:
                continue
            parent[n] = t
            depth[n] = depth[t] + 1
            q.append(n)
    routes = {}
    for t in parent:
        if t == start:
            continue
        path = [t]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        routes[t] = tuple(reversed(path))
    return routes


def move_candidates(d: GameDefinition, s: SimulationState, singles: bool = True,
                    include_end: bool = True) -> List[object]:
    """Candidate moves for the current MOVE decision.

    For each territory and each group of the actor's units sharing (type, movement left), proposes
    moving the whole group (and, if singles=True, one unit of it) to every destination reachable
    by one shortest plausible route; also whole-stack moves of all units that can reach each
    adjacent destination, and RAID variants for strategic bombers. Complete for single-step moves of
    whole groups; multi-step alternatives beyond one route per destination are not enumerated.
    """
    p = s.decision.actor
    combat = d.steps[s.step].kind == COMBAT_MOVE
    out: List[object] = []
    by_terr: Dict[int, List] = {}
    for u in s.units.values():
        if u.owner != p or u.loc < 0:
            continue
        ut = d.unit_types[u.type]
        if rules.movement_left(d, u) <= 0 or (combat and not ut.can_move_in_combat):
            continue
        by_terr.setdefault(u.loc, []).append(u)
    for t, units in sorted(by_terr.items()):
        groups: Dict[Tuple[int, int], List] = {}
        for u in sorted(units, key=lambda x: x.id):
            groups.setdefault((u.type, rules.movement_left(d, u)), []).append(u)
        for (type_id, ml), us in sorted(groups.items()):
            ut = d.unit_types[type_id]
            routes = reachable_routes(d, s, t, ml, ut.is_air, ut.can_blitz)
            ids = tuple(u.id for u in us)
            for dest, route in routes.items():
                out.append(MoveAction(ids, route))
                if singles and len(ids) > 1:
                    out.append(MoveAction(ids[:1], route))
                if combat and ut.is_strategic_bomber:
                    out.append(MoveAction(ids, route, RAID))
        # whole stack to adjacent territories
        if len(groups) > 1:
            ids = tuple(u.id for u in sorted(units, key=lambda x: x.id))
            for n in d.neighbors[t]:
                out.append(MoveAction(ids, (t, n)))
    if include_end:
        out.append(EndPhaseAction())
    return _valid(d, s, out)


# ---------------------------------------------------------------------------------------------
# Placement
# ---------------------------------------------------------------------------------------------

def place_candidates(d: GameDefinition, s: SimulationState) -> List[object]:
    """For each eligible producer: place as many held units as capacity allows (all held types,
    in id order), plus EndPhase. Not complete over splits between producers."""
    p = s.decision.actor
    held = [u.id for u in s.held_units(p) if not d.unit_types[u.type].is_factory]
    out: List[object] = []
    for t, cap in sorted((s.placement_cap or {}).items()):
        room = len(held) if cap < 0 else cap - s.placement_used.get(t, 0)
        if room > 0 and held:
            out.append(PlaceAction(tuple(held[:room]), t))
    out.append(EndPhaseAction())
    return _valid(d, s, out)


# ---------------------------------------------------------------------------------------------
# Battles
# ---------------------------------------------------------------------------------------------

def battle_candidates(d: GameDefinition, s: SimulationState) -> List[object]:
    """Every pending battle the actor may select now (complete)."""
    return _valid(d, s, [SelectBattleAction(b) for b in sorted(s.battles)])


def unit_power(d: GameDefinition, unit_type_id: int, attacking: bool) -> int:
    ut = d.unit_types[unit_type_id]
    return ut.attack if attacking else ut.defense


def default_casualties(d: GameDefinition, s: SimulationState) -> SelectCasualtiesAction:
    """Default order of loss in the spirit of CasualtyOrderOfLosses: lowest (power, cost) first,
    keeping artillery support where possible; air with less movement left dies first."""
    dec = s.decision
    hb = dec.hit_batch
    attacking = hb.side == "ATTACKER"
    units = [s.units[u] for u in hb.eligible]
    n_inf = sum(1 for u in units if d.unit_types[u.type].is_artillery_supportable)

    def key(u):
        ut = d.unit_types[u.type]
        power = ut.attack if attacking else ut.defense
        if attacking and ut.is_artillery and n_inf > 0:
            power += 1  # artillery's support value
        ml = rules.movement_left(d, u) if ut.is_air else 0
        return (power, d.unit_cost(u.type), ml, u.id)

    order = sorted(units, key=key)
    need = min(hb.hits, len(units))
    return SelectCasualtiesAction(hb.id, tuple(u.id for u in order[:need]))


def casualty_candidates(d: GameDefinition, s: SimulationState, limit: int = 4) -> List[object]:
    """The default selection plus cheapest-cost-first and one-of-each-type variations."""
    dec = s.decision
    hb = dec.hit_batch
    need = min(hb.hits, len(hb.eligible))
    units = [s.units[u] for u in hb.eligible]
    cands = [default_casualties(d, s)]
    by_cost = sorted(units, key=lambda u: (d.unit_cost(u.type), u.id))
    cands.append(SelectCasualtiesAction(hb.id, tuple(u.id for u in by_cost[:need])))
    by_cost_desc_power = sorted(units, key=lambda u: (-d.unit_cost(u.type), u.id))
    cands.append(SelectCasualtiesAction(hb.id, tuple(u.id for u in by_cost_desc_power[:need])))
    seen, out = set(), []
    for c in cands:
        k = tuple(sorted(c.killed))
        if k not in seen:
            seen.add(k)
            out.append(c)
    return _valid(d, s, out[:limit])


def retreat_candidates(d: GameDefinition, s: SimulationState) -> List[object]:
    """CONTINUE plus every legal retreat destination (complete)."""
    b = s.battles[s.decision.battle_id]
    out = [RetreatAction(CONTINUE)] + [RetreatAction(RETREAT, t) for t in rules.retreat_destinations(d, s, b)]
    return _valid(d, s, out)


def candidates(d: GameDefinition, s: SimulationState) -> List[object]:
    """Dispatch to the generator for the current decision kind."""
    k = s.decision.kind
    if k == PURCHASE_D:
        return purchase_candidates(d, s)
    if k == MOVE_D:
        return move_candidates(d, s)
    if k == PLACE_D:
        return place_candidates(d, s)
    if k == BATTLE_SELECTION_D:
        return battle_candidates(d, s)
    if k == CASUALTIES_D:
        return casualty_candidates(d, s)
    if k == RETREAT_D:
        return retreat_candidates(d, s)
    return []
