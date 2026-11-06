"""Playout heuristic: Roxo's simplified Weak AI (WeakAi.moveSimple in his MCTSGrandStrategy fork:
calculateCombatMoveSimple and calculateNonCombatSimple), ported to the forward model.

Compared with TripleA's WeakAi it attacks at 1.1x instead of 1.37x, adds a tank blitz on enemy
capitals through an empty territory next to them, only moves units that have not moved, only
attacks from and moves into territories the player owns, and has no transport, sea or bomber
moves. As in his code, the enemy territories are shuffled before they are sorted, so ties are
broken at random. His code performs the moves without checking them; here the kernel rejects
illegal ones.
"""

from __future__ import annotations

import random
from typing import Iterable, List

from triplea_fm import NULL_PLAYER, MoveAction
from triplea_fm import rules
from triplea_fm.definition import AIR, LAND, SEA

NO_RULE_COST = 2**31 - 1  # AiUtils.getCost without a production rule: Integer.MAX_VALUE


def strength(d, units: Iterable, attacking: bool) -> float:
    """AiUtils.strength for land and air units (sea=false)."""
    total = 0.0
    artillery = supportable = 0
    for u in units:
        ut = d.unit_types[u.type]
        if ut.is_infrastructure or ut.domain not in (LAND, AIR):
            continue
        total += 2 + 1.5 * ut.hit_points
        if attacking:
            total += ut.attack
            if ut.attack == 0:
                total -= 1.2
        else:
            total += ut.defense
        artillery += ut.is_artillery
        supportable += ut.is_artillery_supportable
    if attacking:
        total += min(artillery, supportable)
    return total


def potential_attackers_strength(d, s, t: int) -> float:
    """Utils.getStrengthOfPotentialAttackers: attack strength next to t of the enemies of t's owner."""
    owner = s.owner[t]
    total = 0.0
    for x in d.neighbors[t]:
        if d.is_water[x]:
            continue
        enemies = [u for u in s.units_in(x) if owner != NULL_PLAYER and d.is_at_war(owner, u.owner)]
        total += strength(d, enemies, True)
    return total


def cost(d, u) -> int:
    """AiUtils.getCost: the player's production rule cost, Integer.MAX_VALUE without one."""
    for rid in d.frontier[u.owner]:
        rule = d.production_rules[rid]
        if any(t == u.type for t, _ in rule.results):
            return sum(a for r, a in rule.costs if r == d.treasury_resource)
    return NO_RULE_COST


def _is_capital(d, t: int) -> bool:
    return d.capital_of[t] != NULL_PLAYER


def _has_land_units_of(d, s, t: int, player: int) -> bool:
    return any(u.owner == player and d.unit_types[u.type].is_land for u in s.units_in(t))


def _has_enemy_land_units(d, s, t: int, player: int) -> bool:
    return any(d.is_at_war(player, u.owner) and d.unit_types[u.type].is_land for u in s.units_in(t))


def _owned_with_land_units(d, s, t: int, player: int) -> List[int]:
    """Neighbours of t owned by player and holding its land units."""
    return [x for x in d.neighbors[t] if s.owner[x] == player and _has_land_units_of(d, s, x, player)]


def _units_up_to_strength(d, max_strength: float, units: List) -> List:
    """Utils.getUnitsUpToStrength."""
    if strength(d, units, True) < max_strength:
        return list(units)
    out = []
    for u in units:
        out.append(u)
        if strength(d, out, True) > max_strength:
            return out
    return out


def combat_moves(d, s, player: int, rnd: random.Random) -> List[MoveAction]:
    """calculateCombatMoveSimple."""
    moves: List[MoveAction] = []
    moved = set()
    enemy_owned = [t for t in range(len(d.territory_names))
                   if not d.is_water[t] and not d.impassable[t] and rules.is_enemy_territory(d, s, player, t)]
    rnd.shuffle(enemy_owned)
    enemy_owned.sort(key=lambda t: (
        not _is_capital(d, t),                                                       # capitals first
        not any(d.unit_types[u.type].can_produce for u in s.units_in(t)),           # then factories
        not any(d.unit_types[u.type].is_infrastructure for u in s.units_in(t)),     # then infrastructure
        -d.production[t]))                                                           # then PU value

    # tanks blitz an enemy capital through an empty territory next to it
    taken = False
    for first in range(len(d.territory_names)):
        if s.owner[first] == player or not _is_capital(d, first):
            continue
        defence = strength(d, s.units_in(first), False) if s.units_in(first) else 0.0
        for t in d.neighbors[first]:
            if rules.is_blitzable(d, s, player, t) and not s.units_in(t):
                for origin in _owned_with_land_units(d, s, t, player):
                    units = [u for u in s.units_in(origin) if d.unit_types[u.type].can_blitz and u.moved == 0]
                    if units and strength(d, units, True) > defence:
                        moves.append(MoveAction(tuple(u.id for u in units), (origin, t, first)))
                        moved.update(u.id for u in units)
                        taken = True
                        break
                if taken:
                    break

    # walk into undefended enemy territories with the cheapest unit
    for enemy in enemy_owned:
        if strength(d, s.units_in(enemy), False) != 0:
            continue
        taken = False
        for origin in _owned_with_land_units(d, s, enemy, player):
            if taken:
                break
            for u in sorted(s.units_in(origin), key=lambda u: cost(d, u)):
                ut = d.unit_types[u.type]
                if (u.owner == player and ut.is_land and not ut.is_infrastructure and ut.movement > 0
                        and u.moved == 0 and not ut.is_aa and ut.can_move_in_combat and u.id not in moved):
                    moves.append(MoveAction((u.id,), (origin, enemy)))
                    moved.add(u.id)
                    taken = True
                    break

    # attack what it is 1.1x stronger than (capitals always considered)
    def attackable(u):
        ut = d.unit_types[u.type]
        return (u.owner == player and not ut.is_strategic_bomber and u.id not in moved and not ut.is_aa
                and ut.movement > 0 and u.moved == 0 and not ut.is_infrastructure
                and ut.can_move_in_combat and ut.domain != SEA)

    for enemy in enemy_owned:
        enemy_strength = strength(d, s.units_in(enemy), False)
        if not (enemy_strength > 0 or _is_capital(d, enemy)):
            continue
        origins = _owned_with_land_units(d, s, enemy, player)
        dont_move_from = set()
        ours = 0.0
        for origin in origins:
            if _is_capital(d, origin) and \
                    potential_attackers_strength(d, s, origin) > strength(d, s.units_in(origin), False):
                dont_move_from.add(origin)
                continue
            ours += strength(d, [u for u in s.units_in(origin) if attackable(u)], True)
        if ours <= 1.1 * enemy_strength:
            continue
        needed = 2.5 * enemy_strength + 4
        for origin in origins:
            if origin in dont_move_from:
                continue
            units = [u for u in s.units_in(origin) if attackable(u)]
            fronts = sum(1 for x in d.neighbors[origin] if _has_enemy_land_units(d, s, x, player))
            if fronts > 1:
                units = _units_up_to_strength(d, needed, units)
            needed -= strength(d, units, True)
            if units:
                moved.update(u.id for u in units)
                moves.append(MoveAction(tuple(u.id for u in units), (origin, enemy)))
    return moves


def _first_capital(d, s, player: int):
    """TerritoryAttachment.getFirstOwnedCapitalOrFirstUnownedCapital."""
    capitals = [t for t in range(len(d.territory_names)) if d.capital_of[t] == player]
    owned = [t for t in capitals if s.owner[t] == player]
    return owned[0] if owned else (capitals[0] if capitals else None)


def noncombat_moves(d, s, player: int, graph) -> List[MoveAction]:
    """calculateNonCombatSimple: every stack that has not moved takes one step towards the nearest
    enemy capital, or else the nearest enemy land units or enemy territory, if that step is into a
    territory the player owns. A capital with stronger enemies next to it keeps its units (compared
    with units still to place: none here)."""
    moves: List[MoveAction] = []
    to_place = strength(d, s.held_units(player), False)
    for t in range(len(d.territory_names)):
        if d.is_water[t]:
            continue
        if _is_capital(d, t) and potential_attackers_strength(d, s, t) > to_place:
            continue
        units = [u for u in s.units_in(t)
                 if u.owner == player and not d.unit_types[u.type].is_aa and d.unit_types[u.type].movement > 0
                 and u.moved == 0 and not d.unit_types[u.type].is_infrastructure
                 and d.unit_types[u.type].is_land]
        if not units:
            continue
        goal, best = None, None
        for other in range(len(d.player_names)):
            capital = _first_capital(d, s, other)
            if capital is None or d.is_allied(player, s.owner[capital]) or capital not in graph.land:
                continue
            dist = graph.dist[capital][t]
            if 0 < dist < (best if best is not None else float("inf")) and dist < 10**9:
                goal, best = capital, dist
        if goal is None:
            goal = graph.nearest(t, [x for x in graph.land if _has_enemy_land_units(d, s, x, player)])
            if goal is None:
                goal = graph.nearest(t, [x for x in graph.land if rules.is_enemy_territory(d, s, player, x)])
        step = graph.first_step(t, goal) if goal is not None else None
        if step is None or s.owner[step] != player:
            continue
        moves.append(MoveAction(tuple(u.id for u in units), (t, step)))
    return moves
