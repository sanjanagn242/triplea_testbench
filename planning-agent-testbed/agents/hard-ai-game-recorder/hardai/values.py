"""Territory values (ProTerritoryValueUtils), land territories only (the V1 profile has no fleets)."""

from __future__ import annotations

import math
from collections import deque
from typing import Collection, Dict, List, Set

from .data import ProData
from .odds import estimate_strength

MIN_FACTORY_CHECK_DISTANCE = 9


def find_territory_attack_value(pd: ProData, player: int, t: int) -> float:
    is_enemy_factory = 1 if pd.has_infra_factory_enemy_land(player, t) else 0
    value = 3.0 * pd.d.production[t] * (is_enemy_factory + 1)
    if pd.is_neutral_land(t):
        strength = estimate_strength(pd, t, pd.units_in(t), [], False)
        value += -(strength / 8) * pd.min_cost_per_hit_point
    return value


def find_territory_values(pd: ProData, player: int, cant_be_held: Collection[int],
                          to_attack: Collection[int], to_check: Collection[int]) -> Dict[int, float]:
    cant_be_held, to_attack = set(cant_be_held), set(to_attack)
    max_land_mass = find_max_land_mass_size(pd)
    capitals_and_factories = _enemy_capitals_and_factories_value(pd, player, max_land_mass, cant_be_held,
                                                                 to_attack)
    out = {}
    for t in sorted(to_check):
        if pd.is_land(t):
            out[t] = _land_value(pd, t, player, max_land_mass, capitals_and_factories, cant_be_held, to_attack)
        else:
            out[t] = 0.0  # water value only matters for fleets, outside the V1 profile
    return out


def find_max_land_mass_size(pd: ProData) -> int:
    visited: Set[int] = set()
    best = 1
    for t in range(pd.n):
        if pd.d.is_water[t] or t in visited:
            continue
        size = 0
        q = deque([t])
        visited.add(t)
        while q:
            x = q.popleft()
            size += 1
            for n in pd.d.neighbors[x]:
                if n not in visited and pd.can_potentially_move_land(n):
                    visited.add(n)
                    q.append(n)
        best = max(best, size)
    return best


def _enemy_capitals_and_factories_value(pd: ProData, player, max_land_mass, cant_be_held, to_attack):
    potential = set(pd.potential_enemy_players(player))
    targets = {t for t in range(pd.n)
               if pd.has_infra_factory(t) and (pd.s.owner[t] in potential or t in cant_be_held)}
    n_potential_enemy = sum(1 for t in range(pd.n) if pd.s.owner[t] in potential)
    if len(targets) * 2 >= n_potential_enemy:
        targets.clear()
    targets.update(pd.live_enemy_capitals(player))
    targets -= to_attack
    land = pd.can_potentially_move_land
    out = {}
    for t in sorted(targets):
        factory_production = pd.d.production[t] if pd.has_infra_factory(t) else 0
        player_production = pd.player_production(pd.s.owner[t]) if pd.d.capital_of[t] != -1 else 0.0
        is_neutral = 1 if pd.is_neutral_land(t) else 0
        land_mass = 1 + len(pd.neighbors_within(t, 6, land))
        out[t] = (math.sqrt(factory_production + math.sqrt(player_production)) * 32 / (1 + 3.0 * is_neutral)
                  * land_mass / max_land_mass)
    return out


def _nearby_capitals_and_factories(pd: ProData, start: int, targets: Collection[int]) -> List[int]:
    """findNearbyEnemyCapitalsAndFactories: targets within distance 9, or the nearest ring found."""
    targets = set(targets)
    found = []
    seen = {start: 0}
    q = deque([start])
    current = -1
    while q:
        t = q.popleft()
        dist = seen[t]
        if t in targets:
            found.append(t)
        if dist != current:
            current = dist
            if not (current <= MIN_FACTORY_CHECK_DISTANCE or not found):
                break
        for n in pd.d.neighbors[t]:
            if n not in seen:
                seen[n] = dist + 1
                q.append(n)
    return found


def _land_value(pd: ProData, t, player, max_land_mass, capitals_and_factories, cant_be_held, to_attack) -> float:
    if t in cant_be_held:
        return 0.0
    land = pd.can_potentially_move_land
    values = []
    for target in _nearby_capitals_and_factories(pd, t, capitals_and_factories):
        dist = pd.distance(t, target, land)
        if dist > 0:
            values.append(capitals_and_factories[target] / 2 ** dist)
    values.sort(reverse=True)
    capital_or_factory_value = sum(v / 2 ** i for i, v in enumerate(values))

    nearby_enemy_value = 0.0
    for x in sorted(pd.neighbors_within(t, 2, land)):
        if not (pd.is_enemy_territory(player, x) or x in cant_be_held) or x in to_attack:
            continue
        dist = pd.distance(t, x, land)
        if dist <= 0:
            continue
        value = float(pd.d.production[x])
        if pd.is_neutral_land(x):
            value = find_territory_attack_value(pd, player, x) / 3
        elif _allied_land_without_enemy_neighbors(pd, player, x):
            value *= 0.1
        if value > 0:
            nearby_enemy_value += value / 2 ** dist
    land_mass = 1 + len(pd.neighbors_within(t, 6, land))
    value = nearby_enemy_value * land_mass / max_land_mass + capital_or_factory_value
    if pd.has_infra_factory(t):
        value *= 1.1
    return value


def _allied_land_without_enemy_neighbors(pd: ProData, player, t) -> bool:
    return (pd.is_land(t) and pd.is_allied_territory(player, t)
            and not any(pd.is_enemy_not_neutral_land(player, n) for n in pd.d.neighbors[t]))
