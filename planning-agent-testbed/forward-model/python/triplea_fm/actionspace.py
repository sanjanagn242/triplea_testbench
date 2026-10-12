"""Exhaustive legal action generation for the pending decision of a state.

legal_actions(d, s) yields every action the kernel accepts in `s`: each one is checked with
kernel.validate, and the enumeration covers the whole space the validator can accept, so
nothing legal is missed. Unlike toolkit.candidates (a compact, useful subset), this is the full
action set, which can be very large. It is a generator, so callers can stop early.

What is enumerated, per decision kind:
  PURCHASE          every nonzero affordable count vector over the frontier rules, then EndPhase
  MOVE              every nonempty set of the actor's units in one territory, along every route
                    (paths of adjacent territories, no territory twice) within their movement, as
                    NORMAL and, for all-bomber groups, RAID; then EndPhase
  PLACE             every nonempty set of held units, at every producer with room; then EndPhase
  BATTLE_SELECTION  every pending battle that may be fought now
  CASUALTIES        every set of exactly the required number of eligible units
  RETREAT           CONTINUE and every legal retreat territory

Interchangeable units. By default (distinct_units=False) units that the rules cannot tell apart
(same territory, type, owner, original owner, movement used, damage, combat flag and battle role)
form one class, and a set of units is generated once per count of each class, using the lowest
unit ids of the class. Every legal action is then equivalent to exactly one generated action: it
leads to the same state up to renaming unit ids. With distinct_units=True every subset of unit ids
is generated instead (exponentially more actions).

Purchases are generated in canonical form (rules sorted, zero counts omitted), and only
purchases with a positive count of some rule; buying nothing is EndPhase.

The state can be a hypothetical one or one built from an agent's observation
(testbench.LiveBridge.legal_actions). This is an agent-side helper; the kernel does not import it.
"""

from __future__ import annotations

from itertools import combinations, product
from typing import Dict, Iterator, List, Sequence, Tuple

from . import kernel, rules
from .actions import (CONTINUE, NORMAL, RAID, RETREAT, EndPhaseAction, MoveAction, PlaceAction,
                      PurchaseAction, RetreatAction, SelectBattleAction, SelectCasualtiesAction)
from .definition import COMBAT_MOVE, GameDefinition
from .state import (BATTLE_SELECTION_D, CASUALTIES_D, MOVE_D, PLACE_D, PURCHASE_D, RETREAT_D,
                    SimulationState)


def legal_actions(d: GameDefinition, s: SimulationState, distinct_units: bool = False) -> Iterator[object]:
    """Every legal action for s.decision (see the module docstring). Nothing if the game is over."""
    dec = s.decision
    if dec is None or s.terminal is not None:
        return
    gen = {PURCHASE_D: _purchases, MOVE_D: _moves, PLACE_D: _placements,
           BATTLE_SELECTION_D: _battle_selections, CASUALTIES_D: _casualties,
           RETREAT_D: _retreats}[dec.kind]
    for a in gen(d, s, distinct_units):
        if kernel.validate(d, s, a).ok:
            yield a
    if dec.kind in (PURCHASE_D, MOVE_D, PLACE_D) and kernel.validate(d, s, EndPhaseAction()).ok:
        yield EndPhaseAction()


def count_legal_actions(d: GameDefinition, s: SimulationState, distinct_units: bool = False) -> int:
    return sum(1 for _ in legal_actions(d, s, distinct_units))


# ---------------------------------------------------------------------------------------------
# Unit classes and selections
# ---------------------------------------------------------------------------------------------

def _battle_roles(s: SimulationState) -> Dict[int, Tuple]:
    roles: Dict[int, List] = {}
    for b in s.battles.values():
        for uid in b.attackers:
            roles.setdefault(uid, []).append((b.id, "A", b.origins.get(uid)))
        for uid in b.defenders:
            roles.setdefault(uid, []).append((b.id, "D", None))
    return {uid: tuple(sorted(r, key=repr)) for uid, r in roles.items()}


def _classes(s: SimulationState, units) -> List[List[int]]:
    """Units the rules cannot tell apart, grouped; each class sorted by id."""
    roles = _battle_roles(s)
    groups: Dict[tuple, List[int]] = {}
    for u in sorted(units, key=lambda x: x.id):
        key = (u.loc, u.type, u.owner, u.orig_owner, u.moved, u.hits, u.in_combat, roles.get(u.id, ()))
        groups.setdefault(key, []).append(u.id)
    return list(groups.values())


def _selections(classes: Sequence[Sequence[int]], distinct_units: bool, min_total: int = 1,
                max_total: int = None) -> Iterator[Tuple[int, ...]]:
    """Unit id sets with min_total..max_total units: one per count vector over the classes, or
    every id subset if distinct_units."""
    if distinct_units:
        ids = sorted(u for c in classes for u in c)
        hi = len(ids) if max_total is None else min(max_total, len(ids))
        for k in range(max(min_total, 0), hi + 1):
            for combo in combinations(ids, k):
                if combo:
                    yield combo
        return
    for counts in _count_vectors([len(c) for c in classes], min_total, max_total):
        yield tuple(sorted(u for c, k in zip(classes, counts) for u in c[:k]))


def _count_vectors(sizes: Sequence[int], min_total: int, max_total=None) -> Iterator[Tuple[int, ...]]:
    cap = sum(sizes) if max_total is None else max_total
    out = [0] * len(sizes)

    def rec(i, total):
        if i == len(sizes):
            if total >= max(min_total, 1):
                yield tuple(out)
            return
        for k in range(0, min(sizes[i], cap - total) + 1):
            out[i] = k
            yield from rec(i + 1, total + k)
        out[i] = 0

    yield from rec(0, 0)


# ---------------------------------------------------------------------------------------------
# Generators (unvalidated; legal_actions filters them)
# ---------------------------------------------------------------------------------------------

def _purchases(d, s, distinct_units) -> Iterator[PurchaseAction]:
    p = s.decision.actor
    if not rules.has_enough_capitals(d, s, p):
        return
    rule_ids = sorted(set(d.frontier[p]))
    budget = list(s.balances[p])
    costs = []
    for rid in rule_ids:
        c = [0] * len(budget)
        for r, a in d.production_rules[rid].costs:
            c[r] += a
        if not any(c):
            raise ValueError(f"production rule {d.production_rules[rid].name} is free: "
                             "the purchase action set is infinite")
        costs.append(c)
    counts = [0] * len(rule_ids)

    def rec(i, left):
        if i == len(rule_ids):
            if any(counts):
                yield PurchaseAction(tuple((rid, k) for rid, k in zip(rule_ids, counts) if k))
            return
        most = min((left[r] // c for r, c in enumerate(costs[i]) if c > 0))
        for k in range(most + 1):
            counts[i] = k
            yield from rec(i + 1, [left[r] - k * costs[i][r] for r in range(len(left))])
        counts[i] = 0

    yield from rec(0, budget)


def _walks(d, start: int, max_len: int, any_land: bool) -> List[Tuple[int, ...]]:
    """Every route from start with 1..max_len steps through adjacent territories, visiting no
    territory twice (the engine's Route rejects loops). Impassable territories never appear;
    water never appears for land units."""
    out = []
    path = [start]

    def rec():
        if len(path) > 1:
            out.append(tuple(path))
        if len(path) - 1 == max_len:
            return
        for n in d.neighbors[path[-1]]:
            if d.impassable[n] or (any_land and d.is_water[n]) or n in path:
                continue
            path.append(n)
            rec()
            path.pop()

    rec()
    return out


def _moves(d, s, distinct_units) -> Iterator[MoveAction]:
    p = s.decision.actor
    combat = d.steps[s.step].kind == COMBAT_MOVE
    movable = [u for u in s.units.values()
               if u.owner == p and u.loc >= 0 and rules.movement_left(d, u) > 0
               and (not combat or d.unit_types[u.type].can_move_in_combat)]
    by_loc: Dict[int, list] = {}
    for u in movable:
        by_loc.setdefault(u.loc, []).append(u)
    walk_cache: Dict[tuple, List[Tuple[int, ...]]] = {}
    for start in sorted(by_loc):
        for ids in _selections(_classes(s, by_loc[start]), distinct_units):
            units = [s.units[u] for u in ids]
            types = [d.unit_types[u.type] for u in units]
            max_len = min(rules.movement_left(d, u) for u in units)
            any_land = any(t.is_land for t in types)
            raid = combat and all(t.is_strategic_bomber for t in types)
            key = (start, max_len, any_land)
            if key not in walk_cache:
                walk_cache[key] = _walks(d, start, max_len, any_land)
            for route in walk_cache[key]:
                yield MoveAction(ids, route, NORMAL)
                if raid:
                    yield MoveAction(ids, route, RAID)


def _placements(d, s, distinct_units) -> Iterator[PlaceAction]:
    p = s.decision.actor
    held = [u for u in s.held_units(p)
            if not d.unit_types[u.type].is_factory and d.unit_types[u.type].domain != 2]
    if not held:
        return
    classes = _classes(s, held)
    for t, cap in sorted((s.placement_cap or {}).items()):
        room = len(held) if cap < 0 else cap - s.placement_used.get(t, 0)
        if room <= 0:
            continue
        for ids in _selections(classes, distinct_units, 1, room):
            yield PlaceAction(ids, t)


def _battle_selections(d, s, distinct_units) -> Iterator[SelectBattleAction]:
    for bid in sorted(s.battles):
        yield SelectBattleAction(bid)


def _casualties(d, s, distinct_units) -> Iterator[SelectCasualtiesAction]:
    hb = s.decision.hit_batch
    need = min(hb.hits, len(hb.eligible))
    eligible = [s.units[u] for u in hb.eligible]
    for ids in _selections(_classes(s, eligible), distinct_units, need, need):
        yield SelectCasualtiesAction(hb.id, ids)


def _retreats(d, s, distinct_units) -> Iterator[RetreatAction]:
    yield RetreatAction(CONTINUE)
    for t in rules.retreat_destinations(d, s, s.battles[s.decision.battle_id]):
        yield RetreatAction(RETREAT, t)
