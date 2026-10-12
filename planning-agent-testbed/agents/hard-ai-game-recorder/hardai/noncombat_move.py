"""Noncombat move planning (ProNonCombatMoveAi), land and air units plus AA guns.

Produces a move map: for each territory the owned units that should end the phase there. moves.py
turns it into forward-model moves. Transports, fleets and carriers are outside the V1 profile.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

from .combat_move import find_max_purchase_defenders
from .data import OrderedSet, ProData, ProTerritory, PurchaseTerritory
from .odds import estimate_strength_difference, territory_has_local_land_superiority_after_moves
from .sorting import sort_unit_move_options
from .territory_manager import TerritoryManager
from .values import find_territory_values


def can_land_on_carrier(ut) -> bool:
    return any(k == "carrierCost" and int(v) > 0 for k, v in ut.deferred)


class NonCombatMoveAi:
    def __init__(self, ai):
        self.ai = ai
        self.calc = ai.calc
        self.pd: ProData = None
        self.player = None
        self.tm: TerritoryManager = None

    def log(self, msg):
        self.ai.log(msg)

    def do_noncombat_move(self, pd: ProData,
                          purchase_territories: Optional[Dict[int, PurchaseTerritory]]) -> Dict[int, ProTerritory]:
        self.pd = pd
        self.player = player = pd.player
        self.tm = tm = TerritoryManager(self.calc, pd)
        tm.populate_defense_options([])
        self._find_units_that_cant_move(purchase_territories)
        infra_move_map = self._find_infra_units_that_can_move()
        moved_one = self._move_one_defender_to_land_territories_bordering_enemy()
        tm.populate_enemy_attack_options(moved_one, tm.defend_territories())
        self._determine_if_move_territories_can_be_held()
        prioritized = self._prioritize_defend_options()

        capital = pd.my_capital
        enemy_distance = math.inf
        if capital is not None:
            enemy_distance = closest_enemy_land_territory_distance(pd, player, capital)
            self._move_units_to_defend_territories(prioritized, enemy_distance)

        cant_hold = tm.cant_hold_territories()
        values = find_territory_values(pd, player, cant_hold, [], tm.defend_territories())
        move_map = tm.defend_options.territory_map
        for t in tm.defend_territories():
            move_map[t].value = values[t]
        self._move_units_to_best_territories()

        # ProNonCombatMoveAi then re-runs the step above on a copy with a capital defence range when
        # the capital lacks local land superiority. The executed moves come from the original map
        # (captured before the copy), so the only visible effect is that AA moves, which are planned
        # on the copy, are not executed. This port reproduces that behaviour.
        defensive_rerun = (capital is not None and 2 <= enemy_distance <= 3
                           and not territory_has_local_land_superiority_after_moves(
                               pd, capital, enemy_distance, player, move_map))
        if defensive_rerun:
            self.log("noncombat: capital lacks local land superiority (defensive re-run, AA stays)")
        else:
            self._move_infra_units(infra_move_map)
        return move_map

    # ---------------------------------------------------------------------------------------
    def _find_units_that_cant_move(self, purchase_territories):
        pd, player, d = self.pd, self.player, self.pd.d
        move_map = self.tm.defend_options.territory_map
        unit_move_map = self.tm.defend_options.unit_move_map
        for t, pt in move_map.items():
            pt.cant_move_units.update_all(
                u for u in pd.units_in(t)
                if (u.owner == player and pd.movement_left(u) <= 0)
                or (u.owner != player and d.is_allied(player, u.owner)))
        for u in list(unit_move_map):
            ts = unit_move_map[u]
            if len(ts) == 1 and next(iter(ts)) == u.loc:
                move_map[u.loc].cant_move_units.add(u)
                del unit_move_map[u]
        if purchase_territories is not None:
            for ppt in purchase_territories.values():
                for place in ppt.can_place:
                    if place.t in move_map:
                        move_map[place.t].cant_move_units.update_all(place.place_units)
        else:
            pus = pd.s.balances[player][d.treasury_resource]
            for t, pt in move_map.items():
                if pd.has_factory_not_conquered_owned_land(player, t):
                    pt.cant_move_units.update_all(find_max_purchase_defenders(
                        pd, player, t, pd.purchase_options.land_options(), pus))

    def _find_infra_units_that_can_move(self) -> Dict[object, OrderedSet]:
        pd = self.pd
        unit_move_map = self.tm.defend_options.unit_move_map
        infra = {}
        for u in list(unit_move_map):
            if u.owner == self.player and pd.movement_left(u) > 0 and pd.ut(u).is_infrastructure:
                infra[u] = unit_move_map.pop(u)
        return infra

    def _move_one_defender_to_land_territories_bordering_enemy(self) -> List[int]:
        pd, player, d = self.pd, self.player, self.pd.d
        move_map = self.tm.defend_options.territory_map
        unit_move_map = self.tm.defend_options.unit_move_map
        enemies = pd.potential_enemy_players(player)
        to_defend = []
        for t, pt in move_map.items():
            if d.is_water[t]:
                continue
            has_allied_land = any(pd.ut(u).is_land and not pd.ut(u).is_infrastructure
                                  and d.is_allied(player, u.owner) for u in pt.cant_move_units)
            if not has_allied_land and pd.has_neighbor_owned_by_with_land_unit(t, enemies):
                to_defend.append(t)
        for unit, ts in sort_unit_move_options(pd, unit_move_map).items():
            if not pd.ut(unit).is_land:
                continue
            for t in ts:
                if t in to_defend and (pd.value_of(unit) <= d.production[t] + 3 or pd.owned_units(player, t)):
                    move_map[t].units.append(unit)
                    unit_move_map.pop(unit, None)
                    to_defend.remove(t)
                    break
            if not to_defend:
                break
        return to_defend

    def _determine_if_move_territories_can_be_held(self):
        pd, calc = self.pd, self.calc
        enemy_attack = self.tm.enemy_attack_options
        for t, pt in self.tm.defend_options.territory_map.items():
            enemy_max = enemy_attack.get_max(t)
            if enemy_max is None:
                continue
            attackers = list(enemy_max.max_units)
            pt.max_enemy_units = attackers
            min_defenders = [u for u in pt.cant_move_units if not pd.ut(u).is_aa]
            min_result = calc.calculate(pd, t, attackers, min_defenders)
            pt.min_battle_result = min_result
            if min_result.tuv_swing <= 0 and min_defenders:
                continue
            defenders = [u for u in dict.fromkeys(list(pt.max_units) + list(pt.cant_move_units))
                         if not pd.ut(u).is_aa]
            result = calc.calculate(pd, t, attackers, defenders)
            is_factory = 1 if pd.has_infra_factory(t) else 0
            is_capital = 1 if t == pd.my_capital else 0
            extra = [u for u in defenders if u not in set(min_defenders)]
            hold_value = pd.tuv(extra) / 8 * (1 + 0.5 * is_factory) * (1 + 2.0 * is_capital)
            if len(min_defenders) != len(defenders) and result.tuv_swing - hold_value < min_result.tuv_swing:
                continue
            pt.can_hold = False

    def _prioritize_defend_options(self) -> List[ProTerritory]:
        pd, player, d = self.pd, self.player, self.pd.d
        move_map = self.tm.defend_options.territory_map
        enemies = pd.potential_enemy_players(player)
        for t, pt in move_map.items():
            is_my_capital = 1 if t == pd.my_capital else 0
            is_factory = 1 if pd.has_infra_factory(t) else 0
            production = 0 if is_my_capital else d.production[t]
            is_capital = 1 if d.capital_of[t] != -1 else 0
            neighbor_value = 0.0
            if pd.is_land(t):
                for n in pd.neighbors(t, pd.is_land):
                    p = float(d.production[n])
                    if pd.is_allied_territory(player, n):
                        p *= 0.1
                    neighbor_value += p
            cant_move_value = pd.tuv(pt.cant_move_units)
            owner_multiplier = 1.0
            if not any(u.owner == player for u in pt.cant_move_units):
                owner_multiplier = 0.0 if d.is_water[t] else 0.5
            pt.value = (owner_multiplier * (2.0 * production + 10.0 * is_factory + 0.5 * cant_move_value
                                            + 0.5 * neighbor_value)
                        * (1 + 10.0 * is_my_capital) * (1 + 4.0 * is_capital))
        prioritized = sorted(move_map.values(), key=lambda p: -p.value)
        out = []
        for pt in prioritized:
            t = pt.t
            has_factory = pd.has_infra_factory(t)
            min_result = pt.min_battle_result
            cant_move_value = pd.tuv(pt.cant_move_units)
            only_air = pd.is_land(t) and pt.max_enemy_units and all(pd.ut(u).is_air for u in pt.max_enemy_units)
            not_factory_should_hold = not has_factory and (min_result.tuv_swing <= 0
                                                           or not min_result.has_land_unit_remaining)
            already_held = min_result.tuv_swing <= 0 and min_result.win_percentage < 100 - pd.win_percentage
            no_enemy_neighbors = pd.is_land(t) and not has_factory and \
                not pd.has_neighbor_owned_by_with_land_unit(t, enemies)
            only_amphib = pd.is_land(t) and not has_factory and \
                not any(pd.ut(u).is_land for u in pt.max_units) and cant_move_value < 5
            if not pt.can_hold or pt.value <= 0 or only_air or not_factory_should_hold or already_held \
                    or no_enemy_neighbors or only_amphib:
                continue
            out.append(pt)
        return out

    def _units_to_add(self, unit) -> List:
        used = set()
        for p in self.tm.defend_options.territory_map.values():
            used.update(p.all_defenders())
        return [] if unit in used else [unit]

    def _move_units_to_defend_territories(self, prioritized: List[ProTerritory], enemy_distance):
        pd, player, calc, d = self.pd, self.player, self.calc, self.pd.d
        if not prioritized:
            return
        move_map = self.tm.defend_options.territory_map
        unit_move_map = self.tm.defend_options.unit_move_map
        num_to_defend = 1
        while True:
            for pt in move_map.values():
                pt.temp_units.clear()
                pt.set_battle_result(None)
            if num_to_defend <= 0:
                break
            to_defend = prioritized[:num_to_defend]
            defend_options = {}
            for unit, ts in unit_move_map.items():
                defend_options[unit] = OrderedSet((p.t, None) for p in to_defend if p.t in ts)
            sorted_options = sort_unit_move_options(pd, defend_options)
            added = []

            # Enough units in territories to have at least a chance of winning
            for unit in list(sorted_options):
                ut = pd.ut(unit)
                if ut.is_air or unit in added:
                    continue
                estimates = {}
                for t in sorted_options[unit]:
                    pt = move_map[t]
                    estimates[estimate_strength_difference(pd, t, pt.max_enemy_units,
                                                           pt.eligible_defenders(player))] = t
                if estimates and max(estimates) > 60:
                    t = estimates[max(estimates)]
                    to_add = self._units_to_add(unit)
                    move_map[t].temp_units.extend(to_add)
                    added.extend(to_add)
            for u in added:
                sorted_options.pop(u, None)

            # Non-air units (carrier-capable air waits for the air pass)
            for unit in list(sorted_options):
                if can_land_on_carrier(pd.ut(unit)) or unit in added:
                    continue
                best_t = self._best_defend_territory(unit, sorted_options[unit], air_pass=False)
                if best_t is not None:
                    pt = move_map[best_t]
                    pt.set_battle_result(None)
                    to_add = self._units_to_add(unit)
                    pt.temp_units.extend(to_add)
                    added.extend(to_add)
            for u in added:
                sorted_options.pop(u, None)

            # Air units (and anything left)
            for unit in list(sorted_options):
                best_t = self._best_defend_territory(unit, sorted_options[unit], air_pass=True)
                if best_t is not None:
                    move_map[best_t].temp_units.append(unit)
                    move_map[best_t].set_battle_result(None)
                    added.append(unit)
            for u in added:
                sorted_options.pop(u, None)

            # Determine if all defenses are successful
            successful = True
            contains_capital = False
            to_check = set()
            for pt in to_defend:
                to_check.add(pt.t)
                to_check.update(u.loc for u in pt.temp_units if not pd.ut(u).is_air)
            values = find_territory_values(pd, player, self.tm.cant_hold_territories(), [], to_check)
            for pt in to_defend:
                t = pt.t
                result = calc.calculate(pd, t, pt.max_enemy_units, pt.all_defenders())
                pt.set_battle_result(result)
                is_factory = 1 if pd.has_infra_factory(t) else 0
                is_capital = 0
                if t == pd.my_capital:
                    is_capital = 1
                    contains_capital = True
                extra_value = pd.tuv(pt.temp_units)
                hold_value = extra_value / 8 * (1 + 0.5 * is_factory) * (1 + 2.0 * is_capital)
                higher_strategic_value = True
                if pd.is_land(t) and t != pd.my_capital and not pd.has_infra_factory(t):
                    non_air = [u for u in pt.temp_units if not pd.ut(u).is_air]
                    average = sum(values[u.loc] for u in non_air) / len(non_air) if non_air else math.nan
                    if values[t] < average:
                        higher_strategic_value = False
                if result.tuv_swing - hold_value > max(0.0, pt.min_battle_result.tuv_swing) or (
                        not higher_strategic_value
                        and result.tuv_swing + extra_value / 2 >= pt.min_battle_result.tuv_swing):
                    successful = False

            current = prioritized[num_to_defend - 1].t
            capital = pd.my_capital
            if capital is not None:
                cap_pt = move_map.get(capital)
                if contains_capital and current != capital and cap_pt is not None \
                        and cap_pt.battle_result is not None \
                        and cap_pt.battle_result.win_percentage > 100 - pd.win_percentage \
                        and set(move_map[current].all_defenders()) & set(cap_pt.max_defenders()):
                    successful = False
                if pd.is_land(current) and 2 <= enemy_distance <= 3:
                    distance = pd.distance(capital, current, lambda x: pd.can_move_land(player, x, True))
                    if distance > 0 and enemy_distance in (distance, distance - 1) and \
                            not territory_has_local_land_superiority_after_moves(pd, capital, enemy_distance,
                                                                                 player, move_map):
                        successful = False

            if successful:
                num_to_defend += 1
                for pt in to_defend:
                    pt.can_attack = True
                if num_to_defend > len(prioritized):
                    break
            else:
                prioritized[num_to_defend - 1].can_hold = False
                del prioritized[num_to_defend - 1]
                if num_to_defend > len(prioritized):
                    num_to_defend -= 1

        for pt in move_map.values():
            for u in list(pt.temp_units):
                if u.owner != player:
                    pt.cant_move_units.add(u)
                    pt.temp_units.remove(u)
            pt.units.extend(pt.temp_units)
            for u in pt.temp_units:
                unit_move_map.pop(u, None)
            pt.temp_units.clear()

    def _best_defend_territory(self, unit, territories, air_pass: bool) -> Optional[int]:
        pd, player, calc = self.pd, self.player, self.calc
        move_map = self.tm.defend_options.territory_map
        best_t, best_win = None, -1.0
        for t in territories:
            pt = move_map[t]
            if air_pass and pd.ut(unit).is_air:
                if pd.d.is_water[t]:
                    continue  # no carriers in the V1 profile
                if pd.s.owner[t] != player and not pd.has_infra_factory(t):
                    continue  # don't move air units to allied land without a factory
            if pt.battle_result is None:
                pt.set_battle_result(calc.estimate_defend(pd, t, pt.max_enemy_units, pt.eligible_defenders(player)))
            result = pt.battle_result
            has_factory = pd.has_infra_factory(t)
            if result.win_percentage > best_win and (
                    (t == pd.my_capital and result.win_percentage > 100 - pd.win_percentage)
                    or (has_factory and result.win_percentage > 100 - pd.min_win_percentage)
                    or result.tuv_swing >= 0):
                best_t, best_win = t, result.win_percentage
        return best_t

    def _move_units_to_best_territories(self):
        pd, player, calc, d = self.pd, self.player, self.calc, self.pd.d
        move_map = self.tm.defend_options.territory_map
        unit_move_map = self.tm.defend_options.unit_move_map
        for pt in move_map.values():  # the (transport/fleet) loop's reset; nothing else applies to V1
            pt.temp_units.clear()
            pt.set_battle_result(None)
        water_neighbors = lambda t: [n for n in d.neighbors[t] if d.is_water[n] and not d.impassable[n]]  # noqa: E731
        factory_by_sea = lambda t: (pd.has_infra_factory(t) and pd.s.owner[t] == player  # noqa: E731
                                    and bool(water_neighbors(t)))

        # Land units to the territory with the highest value (and most need for transported units)
        added = []
        for u in list(unit_move_map):
            if not pd.ut(u).is_land or u in added:
                continue
            best_t, best_value, best_need = None, 0.0, -math.inf
            for t in unit_move_map[u]:
                pt = move_map[t]
                if pt.can_hold and pt.value >= best_value:
                    need = (1 + 10 * (1 if factory_by_sea(t) else 0)) * len(water_neighbors(t))
                    if pt.value > best_value or need > best_need:
                        best_t, best_value, best_need = t, pt.value, need
            if best_t is not None:
                to_add = self._units_to_add(u)
                move_map[best_t].units.extend(to_add)
                added.extend(to_add)
        for u in added:
            unit_move_map.pop(u, None)

        # Land units towards the nearest owned factory next to the sea
        factories = [t for t in range(pd.n) if factory_by_sea(t)]
        land_combat = lambda x: pd.can_move_land(player, x, True)  # noqa: E731
        for u in list(unit_move_map):
            if not pd.ut(u).is_land or u in added:
                continue
            min_t, min_distance = None, math.inf
            for t in unit_move_map[u]:
                if not move_map[t].can_hold:
                    continue
                for f in factories:
                    dist = pd.distance(t, f, land_combat)
                    if dist < 0:
                        dist = 10 * pd.distance(t, f)
                    if 0 <= dist < min_distance:
                        min_t, min_distance = t, dist
            if min_t is not None:
                to_add = self._units_to_add(u)
                move_map[min_t].units.extend(to_add)
                added.extend(to_add)
        for u in added:
            unit_move_map.pop(u, None)

        # Remaining land units to the safest territory
        for u in list(unit_move_map):
            if not pd.ut(u).is_land or u in added:
                continue
            moved = {x for pt in move_map.values() for x in pt.units}
            best_t = self._safest(u, unit_move_map[u], moved)
            if best_t is not None:
                to_add = self._units_to_add(u)
                move_map[best_t].units.extend(to_add)
                added.extend(to_add)
        for u in added:
            unit_move_map.pop(u, None)

        # Air units to the safe territory with the most attack options
        cant_hold = [t for t, pt in move_map.items() if not pt.can_hold]
        for u in list(unit_move_map):
            if not pd.ut(u).is_air:
                continue
            best_t, best_value = None, 0.0
            for t in unit_move_map[u]:
                pt = move_map[t]
                if not pt.can_hold or d.is_water[t]:
                    continue
                defenders = pt.all_defenders() + [u]
                if pt.battle_result is None:
                    pt.set_battle_result(calc.calculate(pd, t, pt.max_enemy_units, defenders))
                result = pt.battle_result
                if result.win_percentage >= pd.min_win_percentage or result.tuv_swing > 0:
                    pt.can_hold = False
                    continue
                mine = [x for x in defenders if x.owner == player]
                result2 = calc.calculate(pd, t, pt.max_enemy_units, mine)
                cant_hold_without_allies = 1 if (result2.win_percentage >= pd.min_win_percentage
                                                 or result2.tuv_swing > 0) else 0
                rng = pd.ut(u).movement
                fly = pd.can_move_air
                attack_range = pd.neighbors_within(t, rng // 2, fly)
                n_enemy = sum(1 for x in attack_range if pd.is_enemy_not_passive_neutral_land(player, x))
                n_land = sum(1 for x in attack_range
                             if pd.is_land(x) and (pd.is_enemy_territory(player, x) or x in cant_hold)
                             and any(pd.ut(y).is_land for n in d.neighbors[x] for y in pd.owned_units(player, n)))
                n_nearby = sum(1 for x in pd.neighbors_within(t, rng, fly)
                               if pd.is_enemy_not_passive_neutral_land(player, x))
                isnt_factory = 0 if pd.has_infra_factory(t) else 1
                air_value = ((100.0 * n_land + 10.0 * n_enemy + n_nearby) / (1 + cant_hold_without_allies)
                             / (1 + cant_hold_without_allies * isnt_factory))
                if air_value > best_value:
                    best_t, best_value = t, air_value
            if best_t is not None:
                move_map[best_t].units.append(u)
                move_map[best_t].set_battle_result(None)
                del unit_move_map[u]

        # Remaining air units to the safest territory
        for u in list(unit_move_map):
            if not pd.ut(u).is_air:
                continue
            best_t, best_diff = None, math.inf
            for t in unit_move_map[u]:
                if d.is_water[t]:
                    continue
                pt = move_map[t]
                diff = estimate_strength_difference(pd, t, pt.max_enemy_units, pt.all_defenders() + [u])
                if diff < best_diff:
                    best_t, best_diff = t, diff
            if best_t is not None:
                move_map[best_t].units.append(u)
                del unit_move_map[u]

    def _safest(self, unit, territories, already_moved) -> Optional[int]:
        pd = self.pd
        move_map = self.tm.defend_options.territory_map
        best_t, best_diff = None, math.inf
        for t in territories:
            pt = move_map[t]
            defenders = [x for x in pt.max_defenders() if x not in already_moved] + pt.units
            diff = estimate_strength_difference(pd, t, pt.max_enemy_units, defenders)
            if diff < best_diff:
                best_t, best_diff = t, diff
        return best_t

    def _move_infra_units(self, infra_move_map):
        pd, player = self.pd, self.player
        move_map = self.tm.defend_options.territory_map
        for u, ts in infra_move_map.items():
            current = u.loc
            if not pd.ut(u).is_aa or move_map[current].can_hold or pd.has_infra_factory(current):
                continue
            best_t, best_value = None, 0.0
            for t in ts:
                pt = move_map[t]
                if not pt.can_hold:
                    continue
                has_aa = any(pd.ut(x).is_aa for x in list(pt.cant_move_units) + pt.units)
                value = pt.value * (0.01 if has_aa else 1.0)
                if value > best_value:
                    best_t, best_value = t, value
            if best_t is not None:
                move_map[best_t].units.append(u)


def closest_enemy_land_territory_distance(pd: ProData, player: int, t: int) -> int:
    """ProUtils.getClosestEnemyLandTerritoryDistance (-1 if none within 9)."""
    land = pd.can_potentially_move_land
    enemies = set(pd.potential_enemy_players(player))
    best = 10
    for x in pd.neighbors_within(t, 9, land):
        if pd.s.owner[x] in enemies:
            dist = pd.distance(t, x, land)
            if 0 <= dist < best:
                best = dist
    return best if best < 10 else -1
