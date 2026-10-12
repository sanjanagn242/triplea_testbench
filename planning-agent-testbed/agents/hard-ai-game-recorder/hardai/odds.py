"""Battle estimates (ProBattleUtils) and the odds calculator (ProOddsCalculator).

Strength estimates are the Java AI's closed-form heuristics. Every actual battle simulation goes
through the forward model's battle simulator (triplea_fm.battlesim), which fights the battle with
the kernel's own battle rules; results are cached per (territory, attackers, defenders).
"""

from __future__ import annotations

import math
from typing import Dict, List, Tuple

import triplea_fm as fm
from triplea_fm import battlesim

from .data import BattleResult, ProData

SHORT_RANGE = 2
MEDIUM_RANGE = 3


def _fighting(pd: ProData, units) -> List:
    """Matches.unitCanBeInBattle for land/air battles: infrastructure never fights."""
    return [u for u in units if not pd.ut(u).is_infrastructure]


def estimate_power(pd: ProData, t: int, my_units, enemy_units, attacking: bool) -> float:
    d = pd.d
    units = _fighting(pd, my_units)
    total = 0
    if attacking:
        support = sum(1 for u in units if pd.ut(u).is_artillery)
        for u in units:
            ut = pd.ut(u)
            strength = ut.attack
            if ut.is_artillery_supportable and support > 0:
                strength += 1
                support -= 1
            total += min(strength, d.dice_sides) * ut.attack_rolls
    else:
        total = sum(min(pd.ut(u).defense, d.dice_sides) * pd.ut(u).defense_rolls for u in units)
    return total * 6.0 / d.dice_sides


def estimate_strength(pd: ProData, t: int, my_units, enemy_units, attacking: bool) -> float:
    hit_points = sum(pd.ut(u).hit_points - u.hits for u in _fighting(pd, my_units))
    return 2.0 * hit_points + estimate_power(pd, t, my_units, enemy_units, attacking)


def estimate_strength_difference(pd: ProData, t: int, attackers, defenders) -> float:
    """0 = absolute defender strength, 50 = even, 100+ = absolute attacker strength."""
    if all(pd.ut(u).is_infrastructure for u in attackers) or \
            estimate_power(pd, t, attackers, defenders, True) <= 0:
        return 0.0
    if all(pd.ut(u).is_infrastructure for u in defenders) or \
            estimate_power(pd, t, defenders, attackers, False) <= 0:
        return 99_999.0
    a = estimate_strength(pd, t, attackers, defenders, True)
    b = estimate_strength(pd, t, defenders, attackers, False)
    return (a - b) / b ** 0.85 * 50 + 50


def check_for_overwhelming_win(pd: ProData, t: int, attackers, defenders) -> bool:
    if not defenders and attackers:
        return True
    if estimate_power(pd, t, defenders, attackers, False) == 0 and attackers:
        return True
    attack_power = estimate_power(pd, t, attackers, defenders, True) * pd.d.dice_sides / 6.0
    hit_points = sum(pd.ut(u).hit_points - u.hits for u in _fighting(pd, defenders))
    return attack_power / pd.d.dice_sides >= hit_points


def territory_has_local_land_superiority(pd: ProData, t, distance: int, player: int,
                                         purchase_territories=None) -> bool:
    if t is None:
        return True
    for i in range(2, distance + 1):
        land = lambda x: pd.can_move_land(player, x, False)  # noqa: E731
        near_enemy = pd.neighbors_within(t, i, land) | {t}
        enemy = [u for x in sorted(near_enemy) for u in pd.enemy_units(player, x)
                 if not pd.is_passive_neutral(u.owner)]
        near_allied = pd.neighbors_within(t, i - 1, land) | {t}
        allied = [u for x in sorted(near_allied) for u in pd.allied_units(player, x)]
        for ppt in (purchase_territories or {}).values():
            for place in ppt.can_place:
                if place.t in near_allied:
                    allied.extend(place.place_units)
        if estimate_strength_difference(pd, t, enemy, allied) > 50:
            return False
    return True


def territory_has_local_land_superiority_after_moves(pd: ProData, t: int, distance: int, player: int,
                                                     move_map) -> bool:
    land = lambda x: pd.can_move_land(player, x, False)  # noqa: E731
    near_enemy = pd.neighbors_within(t, distance, land) | {t}
    enemy = [u for x in sorted(near_enemy) for u in pd.enemy_units(player, x)
             if not pd.is_passive_neutral(u.owner)]
    near_allied = pd.neighbors_within(t, distance - 1, land) | {t}
    allied = {}
    for x in sorted(near_allied):
        if x in move_map:
            allied.update(dict.fromkeys(move_map[x].max_defenders()))
    for x, pt in move_map.items():
        if x not in near_allied:
            for u in pt.units + pt.temp_units:
                allied.pop(u, None)
    return estimate_strength_difference(pd, t, enemy, list(allied)) <= 50


class OddsCalculator:
    """ProOddsCalculator on top of triplea_fm.battlesim."""

    def __init__(self, seed: int = 0, run_scale: float = 1.0):
        self.rng = fm.SeededRng(seed, stream=7)
        self.run_scale = run_scale
        self.cache: Dict[tuple, Tuple[float, float, float, float, Tuple[int, ...], Tuple[int, ...], float]] = {}
        self.calls = 0
        self.simulated = 0

    def reset_cache(self):
        self.cache.clear()
        self.calls = 0
        self.simulated = 0

    # ---- public API (ProOddsCalculator) ----
    def estimate_attack(self, pd: ProData, t: int, attackers, defenders) -> BattleResult:
        r = self._no_attackers_or_defenders(pd, t, attackers, defenders)
        if r is not None:
            return r
        if estimate_strength_difference(pd, t, attackers, defenders) < 45:
            return BattleResult(0, -999, False, (), list(defenders), 1)
        return self.call_battle_calc(pd, t, attackers, defenders)

    def estimate_defend(self, pd: ProData, t: int, attackers, defenders) -> BattleResult:
        r = self._no_attackers_or_defenders(pd, t, attackers, defenders)
        if r is not None:
            return r
        diff = estimate_strength_difference(pd, t, attackers, defenders)
        if diff > 55:
            only_air = pd.is_land(t) and attackers and all(pd.ut(u).is_air for u in attackers)
            return BattleResult(100 + diff, 999 + diff, not only_air, list(attackers), (), 1)
        return self.call_battle_calc(pd, t, attackers, defenders)

    def calculate(self, pd: ProData, t: int, attackers, defenders) -> BattleResult:
        r = self._no_attackers_or_defenders(pd, t, attackers, defenders)
        if r is not None:
            return r
        return self.call_battle_calc(pd, t, attackers, defenders)

    def call_battle_calc(self, pd: ProData, t: int, attackers, defenders,
                         retreat_when_only_air: bool = False) -> BattleResult:
        attackers, defenders = list(attackers), list(defenders)
        if not attackers or not defenders:
            return BattleResult()
        self.calls += 1
        att = sorted(attackers, key=lambda u: (u.type, u.owner, u.id))
        dfd = sorted(defenders, key=lambda u: (u.type, u.owner, u.id))
        key = (t, tuple((u.type, u.owner) for u in att), tuple((u.type, u.owner) for u in dfd),
               retreat_when_only_air)
        cached = self.cache.get(key)
        if cached is None:
            cached = self._simulate(pd, t, att, dfd, retreat_when_only_air)
            self.cache[key] = cached
        win, att_left_tuv_mean, def_left_tuv_mean, rounds, att_idx, def_idx, att_left_count = cached
        main_att = _fighting(pd, attackers)
        main_def = _fighting(pd, defenders)
        tuv_swing = pd.tuv(main_def) - pd.tuv(main_att) + (att_left_tuv_mean - def_left_tuv_mean)
        if pd.is_neutral_land(t):
            tuv_swing = att_left_tuv_mean - pd.tuv(main_att)
        avg_att = [att[i] for i in att_idx]
        avg_def = [dfd[i] for i in def_idx]
        if pd.d.is_water[t]:
            has_land = bool(avg_att)
        else:
            has_land = any(pd.ut(u).is_land for u in avg_att)
        return BattleResult(win * 100, tuv_swing, has_land, avg_att, avg_def, rounds)

    # ---- internals ----
    def _no_attackers_or_defenders(self, pd: ProData, t, attackers, defenders):
        no_defenders = not any(not pd.ut(u).is_infrastructure for u in defenders)
        only_air = pd.is_land(t) and attackers and all(pd.ut(u).is_air for u in attackers)
        if not attackers or (no_defenders and only_air):
            return BattleResult()
        if no_defenders:
            return BattleResult(100, 0.1 + pd.tuv(_fighting(pd, defenders)), True, list(attackers), (), 0)
        return None

    def _simulate(self, pd: ProData, t, att, dfd, retreat_when_only_air):
        runs = max(16, 100 - min(len(att), len(dfd)))
        runs = max(1, int(math.ceil(runs * self.run_scale)))
        result = battlesim.simulate_battle(pd.d, pd.s, t, [(u.type, u.owner) for u in att],
                                           [(u.type, u.owner) for u in dfd], self.rng, runs,
                                           retreat_when_only_air)
        self.simulated += runs
        val = pd.unit_value
        att_tuv = sum(sum(val[att[i].type] for i in r.attackers_left) for r in result.runs) / runs
        def_tuv = sum(sum(val[dfd[i].type] for i in r.defenders_left) for r in result.runs) / runs
        closest = result.closest_to_average()
        return (result.attacker_win_rate, att_tuv, def_tuv, result.average_rounds,
                closest.attackers_left, closest.defenders_left, result.average_attackers_left)
