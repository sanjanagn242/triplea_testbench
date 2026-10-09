"""Combat move planning (ProCombatMoveAi), land and air units.

Produces the attack map: for each target territory the units that attack it (`units`) and the
strategic bombers that raid it (`bombers`). Turning the map into forward-model moves is done by
moves.py. Amphibious assaults, bombardment and naval battles are outside the V1 profile.
"""

from __future__ import annotations

import math
from typing import Dict, List

from .data import ProData, ProTerritory, TempUnit
from .odds import (MEDIUM_RANGE, check_for_overwhelming_win, estimate_strength_difference,
                   territory_has_local_land_superiority)
from .sorting import sort_unit_move_options, sort_unit_needed_options_then_attack
from .territory_manager import TerritoryManager
from .values import find_territory_attack_value, find_territory_values

MIN_BOMBING_SCORE = 4  # avoid bombing low production factories with AA


class CombatMoveAi:
    def __init__(self, ai):
        self.ai = ai
        self.calc = ai.calc
        self.pd: ProData = None
        self.player = None
        self.tm: TerritoryManager = None
        self.is_defensive = False

    def log(self, msg):
        self.ai.log(msg)

    def do_combat_move(self, pd: ProData) -> Dict[int, ProTerritory]:
        self.pd = pd
        self.player = player = pd.player
        self.tm = tm = TerritoryManager(self.calc, pd)
        self.is_defensive = not territory_has_local_land_superiority(pd, pd.my_capital, MEDIUM_RANGE, player)
        self.log(f"combat move: defensive stance={self.is_defensive}")

        tm.populate_attack_options()
        tm.populate_enemy_defense_options()
        attack_options = tm.remove_territories_that_cant_be_conquered()
        cleared = [p.t for p in attack_options]
        tm.populate_enemy_attack_options(cleared, cleared)
        self._determine_territories_that_can_be_held(attack_options, cleared)
        self._prioritize_attack_options(attack_options)
        self._remove_territories_that_arent_worth_attacking(attack_options)

        self._determine_territories_to_attack(attack_options)

        cleared = [p.t for p in attack_options]
        tm.populate_enemy_attack_options(cleared, cleared)
        self._determine_territories_that_can_be_held(attack_options, cleared)
        self._remove_territories_that_arent_worth_attacking(attack_options)

        already_moved = self._move_one_defender_to_land_territories_bordering_enemy(attack_options)
        self._determine_units_to_attack_with(attack_options, already_moved)

        if pd.my_capital is not None and pd.s.owner[pd.my_capital] == player:
            self._remove_attacks_until_capital_can_be_held(attack_options)

        self.ai.stored_strafing = tm.strafing_territories()
        names = pd.d.territory_names
        for patd in attack_options:
            self.log(f"  attack {names[patd.t]} value={patd.value:.2f} units={len(patd.units)}"
                     f" bombers={len(patd.bombers)} strafing={patd.strafing}")
        return tm.attack_options.territory_map

    # ---------------------------------------------------------------------------------------
    def _prioritize_attack_options(self, attack_options: List[ProTerritory]):
        pd, player, d = self.pd, self.player, self.pd.d
        is_ffa = 1 if pd.is_ffa(player) else 0
        for patd in list(attack_options):
            t = patd.t
            is_land = 1 if pd.is_land(t) else 0
            is_neutral = 1 if pd.is_neutral_land(t) else 0
            is_can_hold = 1 if patd.can_hold else 0
            defending = [u for u in patd.max_enemy_defenders(player) if not pd.ut(u).is_infrastructure]
            is_empty_land = 1 if pd.is_land(t) and not defending else 0
            adjacent_to_capital = pd.my_capital is not None and pd.my_capital in d.neighbors[t]
            is_not_neutral_adjacent_to_capital = 1 if adjacent_to_capital and \
                pd.is_enemy_not_passive_neutral_land(player, t) else 0
            is_factory = 1 if pd.has_infra_factory(t) else 0
            production, is_capital = d.production[t], 1 if d.capital_of[t] != -1 else 0
            tuv_swing = patd.max_battle_result.tuv_swing
            if is_ffa and tuv_swing > 0:
                tuv_swing *= 0.5
            territory_value = ((1 + is_land + is_can_hold * (1 + 2.0 * is_ffa * is_land)) * (1 + is_empty_land)
                               * (1 + is_factory) * production)
            attack_value = ((tuv_swing + territory_value) * (1 + 4.0 * is_capital)
                            * (1 + 2.0 * is_not_neutral_adjacent_to_capital) * (1 - 0.9 * is_neutral))
            if attack_value <= 0 and pd.is_neutral_land(t):
                attack_value = self._negative_neutral_value(patd, attack_value)
            patd.value = attack_value
            near_capital = pd.my_capital is not None and pd.distance(pd.my_capital, t) <= 3
            if attack_value <= 0 or (self.is_defensive and attack_value <= 8 and near_capital):
                attack_options.remove(patd)
        attack_options.sort(key=lambda p: -p.value)

    def _negative_neutral_value(self, patd: ProTerritory, attack_value: float) -> float:
        pd, player, t = self.pd, self.player, patd.t
        nearby = pd.neighbors(t, lambda x: pd.can_move_land(player, x, True))
        nearby_enemy = [x for x in nearby if pd.is_enemy_territory(player, x)]
        nearby_owned = [x for x in nearby if pd.owned_units(player, x)]
        through = lambda x: pd.is_allied_territory(player, x) or (  # noqa: E731
            pd.is_enemy_territory(player, x) and not pd.is_passive_neutral(pd.s.owner[x]))
        nearby_enemy_value = 0.0
        for enemy_t in nearby_enemy:
            all_have_route = True
            for allied_t in nearby_owned:
                dist = pd.distance(allied_t, enemy_t, through, ignore_end=True)
                if dist < 0 or dist > 2:
                    all_have_route = False
                    break
            if not all_have_route:
                value = find_territory_attack_value(pd, player, enemy_t)
                if value > 0:
                    nearby_enemy_value += value
        if nearby_enemy_value > 0:
            return nearby_enemy_value * 0.001 / (1 - attack_value)
        diff = estimate_strength_difference(pd, t, list(patd.max_units), patd.max_enemy_defenders(player))
        if diff > 500:
            return diff * 0.00001 / (1 - attack_value)
        return attack_value

    def _determine_territories_to_attack(self, prioritized: List[ProTerritory]):
        pd, player = self.pd, self.player
        num_to_attack = min(1, len(prioritized))
        while True:
            to_try = prioritized[:num_to_attack]
            self._try_to_attack_territories(to_try, [])
            successful = True
            for patd in to_try:
                if patd.battle_result is None:
                    patd.estimate_battle_result(self.calc, player)
                estimate = estimate_strength_difference(pd, patd.t, patd.units, patd.max_enemy_defenders(player))
                result = patd.battle_result
                if not patd.strafing and estimate < patd.strength_estimate and (
                        result.win_percentage < pd.min_win_percentage or not result.has_land_unit_remaining):
                    successful = False
            if successful:
                for patd in to_try:
                    patd.can_attack = True
                    estimate = estimate_strength_difference(pd, patd.t, patd.units,
                                                            patd.max_enemy_defenders(player))
                    if estimate < patd.strength_estimate:
                        patd.strength_estimate = estimate
                num_to_attack += 1
                if num_to_attack > len(prioritized):
                    break
            else:
                del prioritized[num_to_attack - 1]
                if num_to_attack > len(prioritized):
                    num_to_attack -= 1
            if not prioritized:
                break

    def _determine_territories_that_can_be_held(self, prioritized: List[ProTerritory], cleared: List[int]):
        pd, player, calc = self.pd, self.player, self.calc
        enemy_attack = self.tm.enemy_attack_options
        attack_map = self.tm.attack_options.territory_map
        to_check = set()
        for patd in prioritized:
            to_check.add(patd.t)
            to_check.update(u.loc for u in patd.max_units if not pd.ut(u).is_air)
        values = find_territory_values(pd, player, [], cleared, to_check)
        for patd in prioritized:
            t = patd.t
            if patd.strafing:
                patd.can_hold = False
                continue
            enemy_max = enemy_attack.get_max(t)
            if enemy_max is not None:
                patd.max_enemy_units = list(enemy_max.max_units)
            is_factory = 1 if pd.has_infra_factory(t) else 0
            non_air = [u for u in patd.max_units if not pd.ut(u).is_air]
            average = sum(values[u.loc] for u in non_air) / len(non_air) * 0.75 if non_air else math.nan
            territory_value = values[t] * (1 + 4.0 * is_factory)
            if pd.is_land(t) and territory_value < average:
                attack_map[t].can_hold = False
                continue
            if enemy_max is not None:
                result = calc.estimate_attack(pd, t, list(patd.max_units), patd.max_enemy_defenders(player))
                remaining = [u for u in result.average_attackers_remaining if not pd.ut(u).is_air]
                result2 = calc.calculate(pd, t, patd.max_enemy_units, remaining)
                patd.can_hold = ((not result2.has_land_unit_remaining and pd.is_land(t))
                                 or result2.tuv_swing < 0 or result2.win_percentage < pd.min_win_percentage)
            else:
                attack_map[t].can_hold = True

    def _remove_territories_that_arent_worth_attacking(self, prioritized: List[ProTerritory]):
        pd, player = self.pd, self.player
        enemy_attack = self.tm.enemy_attack_options
        for patd in list(prioritized):
            t = patd.t
            is_neutral = pd.is_neutral_land(t)
            diff = estimate_strength_difference(pd, t, list(patd.max_units), patd.max_enemy_defenders(player))
            if not patd.can_hold and enemy_attack.get_max(t) is not None and pd.is_land(t) \
                    and is_neutral and diff <= 500:
                prioritized.remove(patd)
                continue
            if is_neutral and diff <= 500:
                targets = {p.t for p in prioritized}
                attack_from = {u.loc for u in patd.max_units} & set(pd.d.neighbors[t])
                for x in sorted(attack_from):
                    enemy_neighbors = set(pd.neighbors(x, lambda n: pd.is_enemy_not_neutral_land(player, n)))
                    if not enemy_neighbors <= targets:
                        prioritized.remove(patd)
                        break

    def _move_one_defender_to_land_territories_bordering_enemy(self, prioritized: List[ProTerritory]) -> List:
        pd, player, d = self.pd, self.player, self.pd.d
        unit_move_map = self.tm.attack_options.unit_move_map
        targets = {p.t for p in prioritized}
        already_moved = []
        for t in pd.my_unit_territories:
            if not pd.is_land(t):
                continue
            # unitCantBeMovedAndIsAlliedDefenderAndNotInfra
            has_allied_land = any(
                not pd.ut(u).is_infrastructure and d.is_allied(player, u.owner)
                and (u.owner != player or pd.movement_left(u) <= 0) for u in pd.units_in(t))
            enemy_neighbors = [n for n in d.neighbors[t] if n not in targets
                               and pd.is_enemy_not_neutral_land(player, n)
                               and any(pd.ut(u).is_land and not pd.ut(u).is_infrastructure
                                       and pd.ut(u).movement > 0 for u in pd.enemy_units(player, n))]
            if has_allied_land or not enemy_neighbors:
                continue
            min_unit, min_cost = None, math.inf
            for u in pd.owned_units(player, t):
                if not pd.ut(u).is_infrastructure and pd.value_of(u) < min_cost:
                    min_unit, min_cost = u, pd.value_of(u)
            if min_unit is not None:
                unit_move_map.pop(min_unit, None)
                already_moved.append(min_unit)
        return already_moved

    def _units_to_add(self, unit, already_moved) -> List:
        """ProTransportUtils.getUnitsToAdd: the unit itself unless it is already used."""
        used = set(already_moved)
        for p in self.tm.attack_options.territory_map.values():
            used.update(p.all_defenders())
        return [] if unit in used else [unit]

    def _determine_units_to_attack_with(self, prioritized: List[ProTerritory], already_moved: List):
        pd, player, calc = self.pd, self.player, self.calc
        attack_map = self.tm.attack_options.territory_map
        enemy_attack = self.tm.enemy_attack_options
        unit_attack_map = self.tm.attack_options.unit_move_map
        while True:
            sorted_options = self._try_to_attack_territories(prioritized, already_moved)
            for p in attack_map.values():
                p.bombers.clear()
            attacked_with = set()
            for p in attack_map.values():
                attacked_with.update(p.units)
            self._determine_territories_that_can_be_bombed(attack_map, sorted_options, attacked_with)

            sorted_options = sort_unit_needed_options_then_attack(pd, player, sorted_options, attack_map, calc)
            added = []
            # Air units in any territory with no AA (don't move planes to empty territories)
            for unit in list(sorted_options):
                if not pd.ut(unit).is_air:
                    continue
                min_t, min_win = None, math.inf
                for t in sorted_options[unit]:
                    patd = attack_map[t]
                    if not pd.has_infra_factory_enemy_land(player, t) and not self._can_air_safely_land(unit, t):
                        continue
                    if patd.battle_result is None:
                        patd.estimate_battle_result(calc, player)
                    result = patd.battle_result
                    if result.win_percentage < min_win or (not result.has_land_unit_remaining and min_t is None):
                        defenders = patd.max_enemy_defenders(player)
                        overwhelming = check_for_overwhelming_win(pd, t, patd.units, defenders)
                        has_aa = any(pd.ut(u).is_aa for u in defenders)
                        if not has_aa and not overwhelming:
                            min_win, min_t = result.win_percentage, t
                if min_t is not None:
                    attack_map[min_t].set_battle_result(None)
                    attack_map[min_t].units.append(unit)
                    added.append(unit)
            for u in added:
                sorted_options.pop(u, None)

            sorted_options = sort_unit_needed_options_then_attack(pd, player, sorted_options, attack_map, calc)
            # Territories that can be held and still need units
            for unit in list(sorted_options):
                if unit in added:
                    continue
                min_t = None
                for t in sorted_options[unit]:
                    patd = attack_map[t]
                    if patd.can_hold:
                        if patd.battle_result is None:
                            patd.estimate_battle_result(calc, player)
                        result = patd.battle_result
                        overwhelming = check_for_overwhelming_win(pd, t, patd.units, patd.max_enemy_defenders(player))
                        if not overwhelming and result.battle_rounds > 2:
                            min_t = t
                            break
                if min_t is not None:
                    attack_map[min_t].set_battle_result(None)
                    to_add = self._units_to_add(unit, already_moved)
                    attack_map[min_t].units.extend(to_add)
                    added.extend(to_add)
            for u in added:
                sorted_options.pop(u, None)

            # Determine if all attacks are worth it
            used = [u for p in prioritized for u in p.units]
            to_remove = None
            for patd in prioritized:
                t = patd.t
                if patd.battle_result is None:
                    patd.estimate_battle_result(calc, player)
                result = patd.battle_result
                can_hold, enemy_counter_tuv_swing = True, 0.0
                if enemy_attack.get_max(t) is not None:
                    remaining = [u for u in result.average_attackers_remaining if not pd.ut(u).is_air]
                    result2 = calc.calculate(pd, t, patd.max_enemy_units, remaining)
                    if patd.can_hold and result2.tuv_swing > 0:
                        unused = [u for u in patd.max_units if u not in set(used)] + remaining
                        result3 = calc.calculate(pd, t, patd.max_enemy_units, unused)
                        if result3.tuv_swing < result2.tuv_swing:
                            result2, remaining = result3, unused
                    can_hold = ((not result2.has_land_unit_remaining and pd.is_land(t))
                                or result2.tuv_swing < 0 or result2.win_percentage < pd.min_win_percentage)
                    if result2.tuv_swing > 0:
                        enemy_counter_tuv_swing = result2.tuv_swing
                is_neutral = pd.is_neutral_land(t)
                is_land = 1 if pd.is_land(t) else 0
                is_can_hold = 1 if can_hold else 0
                is_factory = 1 if pd.has_infra_factory(t) else 0
                is_ffa = 1 if pd.is_ffa(player) else 0
                production = pd.d.production[t]
                capital_value = pd.player_production(pd.s.owner[t]) if pd.d.capital_of[t] != -1 else 0.0
                territory_value = ((1 + is_land + is_factory + is_can_hold * (1 + 2.0 * is_ffa + 1.5 * is_factory))
                                   * production + capital_value)
                tuv_swing = result.tuv_swing
                if is_ffa and tuv_swing > 0:
                    tuv_swing *= 0.5
                attack_value = (1 + tuv_swing + territory_value * result.win_percentage / 100
                                - enemy_counter_tuv_swing * 2 / 3)
                all_can_attack_other = True
                if is_neutral and attack_value < 0:
                    for u in patd.units:
                        if not any(p2 is not patd and p2.t in unit_attack_map.get(u, ()) for p2 in prioritized):
                            all_can_attack_other = False
                            break
                if not patd.strafing and (
                        result.win_percentage < pd.min_win_percentage or not result.has_land_unit_remaining
                        or (is_neutral and not can_hold)
                        or (attack_value < 0 and (not is_neutral or all_can_attack_other
                                                  or result.battle_rounds >= 4))):
                    to_remove = patd
            if to_remove is None:
                break
            prioritized.remove(to_remove)

    def _determine_territories_that_can_be_bombed(self, attack_map, sorted_options, attacked_with):
        for bomber, targets in self.tm.attack_options.bomber_move_map.items():
            if bomber in attacked_with:
                return  # already attacked bombers cannot move
            self._best_bombing_attack(attack_map, sorted_options, targets, bomber)

    def _best_bombing_attack(self, attack_map, sorted_options, targets, bomber):
        pd, player = self.pd, self.player
        best_t, best_score = None, MIN_BOMBING_SCORE
        for t in targets:
            target_units = [u for u in pd.enemy_units(player, t) if pd.ut(u).can_produce and pd.ut(u).can_be_damaged]
            if not target_units or not self._can_air_safely_land(bomber, t):
                continue
            no_aa = 0 if any(pd.ut(u).is_aa for u in pd.units_in(t)) else 1
            remaining_damage = pd.d.production[t] - 3 * len(attack_map[t].bombers)
            score = (1 + 9 * no_aa) * remaining_damage
            if score >= best_score:
                best_score, best_t = score, t
        if best_t is not None:
            attack_map[best_t].bombers.append(bomber)
            sorted_options.pop(bomber, None)

    def _try_to_attack_territories(self, prioritized: List[ProTerritory], already_moved: List):
        pd, player, calc = self.pd, self.player, self.calc
        d = pd.d
        attack_map = self.tm.attack_options.territory_map
        unit_attack_map = self.tm.attack_options.unit_move_map
        for p in attack_map.values():
            p.units.clear()
            p.set_battle_result(None)
        targets = [p.t for p in prioritized]
        options = {}
        for u, ts in unit_attack_map.items():
            can = {t: None for t in targets if t in ts}
            if can:
                options[u] = can
        sorted_options = sort_unit_move_options(pd, options)
        added = []

        # Enough land units in territories to have at least a chance of winning
        for unit in list(sorted_options):
            ut = pd.ut(unit)
            expensive_land = ut.is_land and pd.value_of(unit) > 2 * pd.min_cost_per_hit_point
            if ut.is_air or expensive_land or unit in added:
                continue
            estimates = {}
            for t in sorted_options[unit]:
                patd = attack_map[t]
                if d.is_water[t] and not patd.can_hold:
                    continue
                defenders = patd.max_enemy_defenders(player)
                estimate = estimate_strength_difference(pd, t, patd.units, defenders)
                if any(pd.ut(u).is_aa for u in defenders):
                    estimate -= 10
                estimates[estimate] = t
            if estimates and min(estimates) < 40:
                t = estimates[min(estimates)]
                to_add = self._units_to_add(unit, already_moved)
                attack_map[t].units.extend(to_add)
                added.extend(to_add)
        for u in added:
            sorted_options.pop(u, None)

        sorted_options = sort_unit_needed_options_then_attack(pd, player, sorted_options, attack_map, calc)
        # Non-air units in territories that can be held
        for unit in list(sorted_options):
            if pd.ut(unit).is_air or unit in added:
                continue
            min_t, min_win = None, pd.win_percentage
            for t in sorted_options[unit]:
                patd = attack_map[t]
                if not patd.currently_wins and patd.can_hold:
                    if patd.battle_result is None:
                        patd.estimate_battle_result(calc, player)
                    result = patd.battle_result
                    if result.win_percentage < min_win or (not result.has_land_unit_remaining and min_t is None):
                        min_win, min_t = result.win_percentage, t
            if min_t is not None:
                attack_map[min_t].set_battle_result(None)
                to_add = self._units_to_add(unit, already_moved)
                attack_map[min_t].units.extend(to_add)
                added.extend(to_add)
        for u in added:
            sorted_options.pop(u, None)

        sorted_options = sort_unit_needed_options_then_attack(pd, player, sorted_options, attack_map, calc)
        # Air units in territories that can't be held (don't move planes to empty territories)
        enemy_capitals = set(pd.live_enemy_capitals(player))
        allied_capitals = set(pd.live_allied_capitals(player))
        for unit in list(sorted_options):
            if not pd.ut(unit).is_air:
                continue
            min_t, min_win = None, pd.win_percentage
            for t in sorted_options[unit]:
                patd = attack_map[t]
                if patd.currently_wins or patd.can_hold:
                    continue
                adjacent_to_allied_capital = any(n in allied_capitals for n in d.neighbors[t])
                if t not in enemy_capitals and not adjacent_to_allied_capital and self._uses_over_half_range(unit, t):
                    continue
                if patd.battle_result is None:
                    patd.estimate_battle_result(calc, player)
                result = patd.battle_result
                if result.win_percentage < min_win or (not result.has_land_unit_remaining and min_t is None):
                    defenders = patd.max_enemy_defenders(player)
                    no_defenders = not any(not pd.ut(u).is_infrastructure for u in defenders)
                    overwhelming = check_for_overwhelming_win(pd, t, patd.units, defenders)
                    has_aa = any(pd.ut(u).is_aa for u in defenders)
                    if not no_defenders and not overwhelming and (not has_aa or result.win_percentage < min_win):
                        min_win, min_t = result.win_percentage, t
                        if patd.strafing:
                            break
            if min_t is not None:
                attack_map[min_t].set_battle_result(None)
                attack_map[min_t].units.append(unit)
                added.append(unit)
        for u in added:
            sorted_options.pop(u, None)

        sorted_options = sort_unit_needed_options_then_attack(pd, player, sorted_options, attack_map, calc)
        # Remaining units in any territory that needs it (don't move planes to empty territories)
        for unit in list(sorted_options):
            if unit in added:
                continue
            is_air = pd.ut(unit).is_air
            min_t, min_win = None, pd.win_percentage
            for t in sorted_options[unit]:
                patd = attack_map[t]
                if patd.currently_wins:
                    continue
                adjacent_to_allied_factory = any(pd.has_infra_factory_allied_land(player, n) for n in d.neighbors[t])
                if is_air and not adjacent_to_allied_factory and self._uses_over_half_range(unit, t) and (
                        patd.value < pd.value_of(unit) or (pd.is_land(t) and not patd.can_hold)):
                    continue
                if patd.battle_result is None:
                    patd.estimate_battle_result(calc, player)
                result = patd.battle_result
                if result.win_percentage < min_win or (not result.has_land_unit_remaining and min_t is None):
                    defenders = patd.max_enemy_defenders(player)
                    no_defenders = not any(not pd.ut(u).is_infrastructure for u in defenders)
                    overwhelming = check_for_overwhelming_win(pd, t, patd.units, defenders)
                    has_aa = any(pd.ut(u).is_aa for u in defenders)
                    if not is_air or (not no_defenders and not overwhelming
                                      and (not has_aa or result.win_percentage < min_win)):
                        min_win, min_t = result.win_percentage, t
            if min_t is not None:
                attack_map[min_t].set_battle_result(None)
                to_add = self._units_to_add(unit, already_moved)
                attack_map[min_t].units.extend(to_add)
                added.extend(to_add)
        for u in added:
            sorted_options.pop(u, None)
        return sorted_options

    def _uses_over_half_range(self, unit, t) -> bool:
        pd, player = self.pd, self.player
        fly = lambda x: pd.can_move_air_no_aa(player, x)  # noqa: E731
        distance = pd.distance(unit.loc, t, fly, ignore_end=True)
        return distance > pd.movement_left(unit) // 2

    def _can_air_safely_land(self, unit, t) -> bool:
        pd = self.pd
        adjacent_to_allied_factory = any(pd.has_infra_factory_allied_land(self.player, n) for n in pd.d.neighbors[t])
        return adjacent_to_allied_factory or not self._uses_over_half_range(unit, t)

    def _remove_attacks_until_capital_can_be_held(self, prioritized: List[ProTerritory]):
        pd, player, calc, d = self.pd, self.player, self.calc, self.pd.d
        attack_map = self.tm.attack_options.territory_map
        capital = pd.my_capital
        place_units = []
        if pd.has_factory_not_conquered_owned_land(player, capital):
            place_units = find_max_purchase_defenders(pd, player, capital,
                                                      pd.purchase_options.land_options(),
                                                      pd.s.balances[player][d.treasury_resource])
        while prioritized:
            targets = [p.t for p in prioritized]
            self.tm.populate_enemy_attack_options(targets, [capital])
            enemy_max = self.tm.enemy_attack_options.get_max(capital)
            if enemy_max is None:
                break
            defenders = pd.allied_units(player, capital) + place_units
            for n in pd.neighbors(capital, pd.is_land):
                defenders.extend(u for u in pd.owned_units(player, n)
                                 if pd.ut(u).is_land and pd.movement_left(u) > 0)
            attacking = {u for p in attack_map.values() for u in p.units}
            defenders = [u for u in defenders if u not in attacking]
            result = calc.estimate_defend(pd, capital, list(enemy_max.max_units), defenders)
            if not result.has_land_unit_remaining:
                break
            near = set(pd.neighbors(capital, pd.is_land)) | {capital}
            best_t, best_ratio = None, 0.0
            for t, patd in attack_map.items():
                near_units = sum(1 for u in patd.units if u.loc in near)
                if patd.value == 0:
                    ratio = math.inf if near_units else 0.0
                else:
                    ratio = near_units / patd.value
                if ratio > best_ratio:
                    best_t, best_ratio = t, ratio
            if best_t is None:
                break
            patd = attack_map[best_t]
            if patd in prioritized:
                prioritized.remove(patd)
            patd.units.clear()
            patd.set_battle_result(None)
            self.log(f"  removing attack on {d.territory_names[best_t]} to hold the capital")


def unit_production(pd: ProData, player: int, t: int) -> int:
    """ProPurchaseUtils.getUnitProduction: unlimited at an original factory of the player."""
    factories = [u for u in pd.units_in(t) if u.owner == player and pd.ut(u).can_produce]
    if pd.d.original_factory[t] and any(u.orig_owner == player for u in factories):
        return 10 ** 9
    return max(0, pd.d.production[t])


def find_max_purchase_defenders(pd: ProData, player: int, t: int, land_options, pus: int) -> List[TempUnit]:
    """ProPurchaseUtils.findMaxPurchaseDefenders."""
    best, best_eff = None, 0.0
    for ppo in land_options:
        if ppo.defense_efficiency > best_eff and ppo.cost <= pus:
            best, best_eff = ppo, ppo.defense_efficiency
    units = []
    if best is not None:
        remaining_production = unit_production(pd, player, t)
        spent = 0
        while best.cost <= pus - spent and remaining_production >= best.quantity:
            spent += best.cost
            remaining_production -= best.quantity
            units.extend(best.create_temp_units())
    return units
