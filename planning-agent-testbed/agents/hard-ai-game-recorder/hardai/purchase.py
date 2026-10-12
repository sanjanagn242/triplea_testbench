"""Purchase and placement (ProPurchaseAi) for land, air and AA units.

purchase() runs on the simulated end-of-turn state (see ai.HardAi.purchase) and decides both what
to buy and where each unit will be placed. place() places the bought units as planned, then places
anything left over (placeRemainingUnits). Fleets, factories and repairs are outside the V1 profile.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

import triplea_fm as fm

from .combat_move import unit_production
from .data import (BattleResult, PlaceTerritory, ProData, PurchaseOption, PurchaseTerritory, ResourceTracker,
                   TempUnit)
from .odds import SHORT_RANGE, territory_has_local_land_superiority
from .territory_manager import TerritoryManager
from .values import find_territory_values


def find_purchase_territories(pd: ProData, player: int) -> Dict[int, PurchaseTerritory]:
    out = {}
    for t in range(pd.n):
        if pd.has_factory_not_conquered_owned_land(player, t) and pd.can_move_land(player, t, False):
            out[t] = PurchaseTerritory(t, unit_production(pd, player, t), [PlaceTerritory(t)])
    return out


class PurchaseAi:
    def __init__(self, ai):
        self.ai = ai
        self.calc = ai.calc
        self.pd: ProData = None
        self.player = None
        self.resources: ResourceTracker = None
        self.tm: TerritoryManager = None

    def log(self, msg):
        self.ai.log(msg)

    def randomize(self, efficiencies: Dict[PurchaseOption, float]) -> Optional[PurchaseOption]:
        """ProPurchaseUtils.randomizePurchaseOption (weighted by efficiency)."""
        total = sum(efficiencies.values())
        if total == 0:
            return None
        upper, bounds = 0.0, []
        for ppo, eff in efficiencies.items():
            upper += eff / total * 100
            bounds.append((ppo, upper))
        r = self.ai.rnd.random() * 100
        for ppo, bound in bounds:
            if r <= bound:
                return ppo
        return bounds[0][0]

    # ---------------------------------------------------------------------------------------
    def purchase(self, pd: ProData, pus: int, held_units: List) -> Tuple[Dict[int, PurchaseTerritory], Dict[int, int]]:
        """Returns the purchase territories with planned placements and {rule id: count}."""
        self.pd = pd
        self.player = player = pd.player
        self.resources = ResourceTracker(pus)
        self.tm = TerritoryManager(self.calc, pd)
        options = pd.purchase_options
        purchase_territories = find_purchase_territories(pd, player)
        place_territories = {t for t in range(pd.n) if pd.s.owner[t] == player and pd.is_land(t)}
        for ppt in purchase_territories.values():
            place_territories.update(p.t for p in ppt.can_place)
        self.tm.populate_enemy_attack_options([], place_territories)
        self._find_defenders_in_place_territories(purchase_territories)

        need = self._prioritize_territories_to_defend(purchase_territories)
        self._purchase_defenders(purchase_territories, need, options.land_fodder, options.land_zero_move,
                                 options.air)

        to_check = {p.t for ppt in purchase_territories.values() for p in ppt.can_place}
        values = find_territory_values(pd, player, [], [], to_check)
        for ppt in purchase_territories.values():
            for place in ppt.can_place:
                place.strategic_value = values[place.t]

        prioritized_land = self._prioritize_land_territories(purchase_territories)
        self._purchase_aa_units(purchase_territories, prioritized_land, options.aa)
        self._purchase_land_units(purchase_territories, prioritized_land, held_units)
        self._purchase_units_with_remaining_production(purchase_territories, options.land_options(), options.air)
        self._upgrade_units_with_remaining_pus(purchase_territories)

        counts: Dict[int, int] = {}
        for ppo in options.all_options():
            n = sum(1 for ppt in purchase_territories.values() for p in ppt.can_place for u in p.place_units
                    if isinstance(u, TempUnit) and u.type == ppo.type_id)
            if n > 0:
                counts[ppo.rule_id] = n // ppo.quantity
        return purchase_territories, counts

    def _find_defenders_in_place_territories(self, purchase_territories):
        for ppt in purchase_territories.values():
            for place in ppt.can_place:
                place.defending_units = self.pd.allied_units(self.player, place.t)

    def _prioritize_territories_to_defend(self, purchase_territories) -> List[PlaceTerritory]:
        pd, player, calc, d = self.pd, self.player, self.calc, self.pd.d
        enemy_attack = self.tm.enemy_attack_options
        need = {}
        for ppt in purchase_territories.values():
            for place in ppt.can_place:
                t = place.t
                enemy_max = enemy_attack.get_max(t)
                if enemy_max is None or d.is_water[t]:
                    continue
                attackers = list(enemy_max.max_units)
                result = calc.calculate(pd, t, attackers, place.defending_units)
                place.min_battle_result = result
                only_air = attackers and all(pd.ut(u).is_air for u in attackers)
                if result.has_land_unit_remaining or result.tuv_swing > 0 or (
                        t == pd.my_capital and not only_air and result.win_percentage > 100 - pd.win_percentage):
                    need[place] = None
        for place in need:
            t = place.t
            is_my_capital = 1 if t == pd.my_capital else 0
            is_factory = 1 if pd.has_infra_factory(t) and pd.s.owner[t] == player else 0
            defending_value = pd.tuv(place.defending_units)
            place.defense_value = ((2.0 * d.production[t] + 4.0 * is_factory + 0.5 * defending_value)
                                   * (1 + is_factory) * (1 + 10.0 * is_my_capital))
        out = [p for p in need if p.defense_value > 0]
        out.sort(key=lambda p: -p.defense_value)
        return out

    def _remove_invalid(self, options: List[PurchaseOption], remaining_production: int):
        options[:] = [p for p in options if self.resources.has_enough(p) and p.quantity <= remaining_production]

    def _is_held(self, t, result: BattleResult) -> bool:
        pd = self.pd
        if t != pd.my_capital:
            return not result.has_land_unit_remaining and result.tuv_swing <= 0
        return result.win_percentage < 100 - pd.win_percentage and result.tuv_swing <= 0

    def _purchase_defenders(self, purchase_territories, need, defense_options, zero_move_options, air_options):
        if self.resources.is_empty():
            return
        pd, player, calc = self.pd, self.player, self.calc
        enemy_attack = self.tm.enemy_attack_options
        for place in need:
            t = place.t
            owned_local = pd.owned_units(player, t)
            units_to_place: List = []
            final = BattleResult()
            for ppt in [p for p in purchase_territories.values() if place in p.can_place]:
                remaining = ppt.remaining_unit_production()
                if remaining <= 0:
                    continue
                options = list(defense_options) + list(zero_move_options) + list(air_options)
                while True:
                    self._remove_invalid(options, remaining)
                    efficiencies = {p: p.defense_efficiency_at(1, owned_local, units_to_place) for p in options}
                    selected = self.randomize(efficiencies)
                    if selected is None:
                        break
                    self.resources.temp_purchase(selected)
                    remaining -= selected.quantity
                    units_to_place.extend(selected.create_temp_units())
                    attackers = list(enemy_attack.get_max(t).max_units)
                    final = calc.calculate(pd, t, attackers, place.defending_units + units_to_place)
                    if self._is_held(t, final):
                        break
            superiority = territory_has_local_land_superiority(pd, t, SHORT_RANGE, player, purchase_territories)
            if not final.has_land_unit_remaining \
                    or final.tuv_swing - self.resources.temp_pus() / 2 < place.min_battle_result.tuv_swing \
                    or t == pd.my_capital or (pd.is_land(t) and superiority):
                self.resources.confirm_temp_purchases()
                self._add_units_to_place_territory(place, units_to_place, purchase_territories)
                self.log(f"  defend {pd.d.territory_names[t]} with {len(units_to_place)} units")
            else:
                self.resources.clear_temp_purchases()
                for ppt in purchase_territories.values():
                    for p in ppt.can_place:
                        if p == place:
                            p.can_hold = False

    def _prioritize_land_territories(self, purchase_territories) -> List[PlaceTerritory]:
        pd, player, d = self.pd, self.player, self.pd.d
        enemies = set(pd.potential_enemy_players(player))
        out = []
        for ppt in purchase_territories.values():
            for place in ppt.can_place:
                t = place.t
                if d.is_water[t] or place.strategic_value < 1 or not place.can_hold:
                    continue
                has_enemy_neighbors = any(pd.is_enemy_land(player, n) for n in d.neighbors[t])
                nearby = pd.neighbors_within(t, 9, pd.can_potentially_move_land)
                n_enemy = sum(1 for x in nearby if pd.s.owner[x] in enemies)
                superiority = territory_has_local_land_superiority(pd, t, SHORT_RANGE, player)
                if has_enemy_neighbors or n_enemy >= 3 or not superiority:
                    out.append(place)
        out.sort(key=lambda p: -p.strategic_value)
        return out

    def _purchase_aa_units(self, purchase_territories, prioritized_land, aa_options):
        if self.resources.is_empty():
            return
        pd = self.pd
        enemy_attack = self.tm.enemy_attack_options
        for place in prioritized_land:
            t = place.t
            enemy_max = enemy_attack.get_max(t)
            if enemy_max is None:
                continue
            remaining = purchase_territories[t].remaining_unit_production()
            if remaining <= 0:
                continue
            enemy_can_bomb = any(pd.ut(u).is_strategic_bomber for u in enemy_max.max_units)
            can_be_bombed = any(pd.ut(u).can_produce and pd.ut(u).can_be_damaged for u in pd.units_in(t))
            has_aa = any(pd.ut(u).is_aa for u in pd.units_in(t))
            if not enemy_can_bomb or not can_be_bombed or has_aa:
                continue
            options = list(aa_options)
            self._remove_invalid(options, remaining)
            if not options:
                continue
            best = min(options, key=lambda p: p.cost)
            self.resources.purchase(best)
            place.place_units.extend(best.create_temp_units())

    def _purchase_land_units(self, purchase_territories, prioritized_land, held_units):
        pd, player = self.pd, self.player
        options = pd.purchase_options
        unplaced = [u for u in held_units if pd.ut(u).domain != 2]
        if self.resources.is_empty() and not unplaced:
            return
        enemy_players = set(pd.enemy_players(player))
        to_check = set()
        for place in prioritized_land:
            to_check.update(x for x in pd.neighbors_within(place.t, 9, pd.can_potentially_move_land)
                            if pd.s.owner[x] in enemy_players)
        values = find_territory_values(pd, player, [], [], to_check)
        for place in prioritized_land:
            t = place.t
            remaining = purchase_territories[t].remaining_unit_production()
            if remaining <= 0:
                continue
            fodder, attack, defense = list(options.land_fodder), list(options.land_attack), list(options.land_defense)
            enemy_distance = closest_enemy_or_neutral_land_distance(pd, player, t, values)
            if enemy_distance <= 0:
                enemy_distance = 10
            fodder_percent = 80 - enemy_distance * 5
            near = pd.neighbors_within(t, 2, lambda x: pd.can_move_land(player, x, False)) | {t}
            owned_local = [u for x in sorted(near) for u in pd.owned_units(player, x)]
            units_to_place: List = []
            while unplaced and remaining > 0:
                units_to_place.append(unplaced.pop(0))
                remaining -= 1
            added_fodder, attack_minus_defense, select_fodder = 0, 0.0, True
            while True:
                for lst in (fodder, attack, defense):
                    self._remove_invalid(lst, remaining)
                selected = None
                if not select_fodder and attack_minus_defense > 0 and defense:
                    selected = self.randomize({p: p.defense_efficiency_at(enemy_distance, owned_local, units_to_place)
                                               for p in defense})
                elif not select_fodder and attack:
                    selected = self.randomize({p: p.attack_efficiency_at(enemy_distance, owned_local, units_to_place)
                                               for p in attack})
                elif fodder:
                    selected = self.randomize({p: p.fodder_efficiency(enemy_distance, owned_local, units_to_place)
                                               for p in fodder})
                    if selected is not None:
                        added_fodder += selected.quantity
                if selected is None:
                    break
                self.resources.purchase(selected)
                remaining -= selected.quantity
                units_to_place.extend(selected.create_temp_units())
                attack_minus_defense += selected.attack - selected.defense
                select_fodder = added_fodder / len(units_to_place) * 100 <= fodder_percent
            place.place_units.extend(units_to_place)

    def _purchase_units_with_remaining_production(self, purchase_territories, land_options, air_options):
        if self.resources.is_empty():
            return
        pd, player, d = self.pd, self.player, self.pd.d
        can_hold, cant_hold = [], []
        for ppt in purchase_territories.values():
            for place in ppt.can_place:
                t = place.t
                if d.is_water[t] or purchase_territories[t].remaining_unit_production() <= 0:
                    continue
                (can_hold if place.can_hold else cant_hold).append(place)
        can_hold.sort(key=lambda p: -p.strategic_value)
        for place in can_hold:
            t = place.t
            options = list(air_options) + list(land_options)
            remaining = purchase_territories[t].remaining_unit_production()
            while True:
                self._remove_invalid(options, remaining)
                if not options:
                    break
                best, best_eff = None, 0.0
                for p in options:
                    eff = p.attack_efficiency * p.movement / p.quantity * (10 if p.is_air else 1)
                    if eff > best_eff:
                        best, best_eff = p, eff
                if best is None:
                    break
                self.resources.purchase(best)
                remaining -= best.quantity
                place.place_units.extend(best.create_temp_units())
        cant_hold.sort(key=lambda p: -p.defense_value)
        for place in cant_hold:
            t = place.t
            owned_local = pd.owned_units(player, t)
            options = list(air_options) + list(land_options)
            remaining = purchase_territories[t].remaining_unit_production()
            while True:
                self._remove_invalid(options, remaining)
                selected = self.randomize({p: p.cost ** 2 * p.defense_efficiency_at(1, owned_local, place.place_units)
                                           for p in options})
                if selected is None:
                    break
                self.resources.purchase(selected)
                remaining -= selected.quantity
                place.place_units.extend(selected.create_temp_units())

    def _upgrade_units_with_remaining_pus(self, purchase_territories):
        if self.resources.is_empty():
            return
        pd, d = self.pd, self.pd.d
        options = pd.purchase_options
        places = [p for ppt in purchase_territories.values() for p in ppt.can_place
                  if not d.is_water[p.t] and p.can_hold]
        places.sort(key=lambda p: p.strategic_value)  # far away territories first
        for place in places:
            t = place.t
            all_options = list(options.air) + list(options.land_options())
            for_territory = list(all_options)
            remaining_upgrades = purchase_territories[t].unit_production // 3
            while remaining_upgrades > 0:
                min_option = None
                for u in place.place_units:
                    for p in all_options:
                        if u.type == p.type_id and (min_option is None or p.cost < min_option.cost):
                            min_option = p
                if min_option is None:
                    break
                self.resources.remove_temp_purchase(min_option)
                self._remove_invalid(for_territory, 1)
                self.resources.clear_temp_purchases()
                if not for_territory:
                    break
                best, best_eff = None, _upgrade_efficiency(min_option, place.strategic_value)
                for p in for_territory:
                    if p.cost > min_option.cost and (p.is_air or place.strategic_value >= 0.25
                                                     or _transport_cost(p) <= _transport_cost(min_option)):
                        eff = _upgrade_efficiency(p, place.strategic_value) * (10 if p.is_air else 1)
                        if eff > best_eff:
                            best, best_eff = p, eff
                if best is None:
                    all_options.remove(min_option)
                    continue
                to_remove = [u for u in place.place_units if u.type == min_option.type_id][:min_option.quantity]
                if len(to_remove) < min_option.quantity:
                    all_options.remove(min_option)
                    continue
                self.resources.remove_purchase(min_option)
                remaining_upgrades -= min_option.quantity
                for u in to_remove:
                    place.place_units.remove(u)
                for _ in to_remove:
                    if self.resources.has_enough(best):
                        self.resources.purchase(best)
                        place.place_units.extend(best.create_temp_units())

    def _add_units_to_place_territory(self, place: PlaceTerritory, units: List, purchase_territories):
        for ppt in purchase_territories.values():
            for p in ppt.can_place:
                if p == place and ppt.remaining_unit_production() > 0:
                    n = min(ppt.remaining_unit_production(), len(units))
                    p.place_units.extend(units[:n])
                    del units[:n]

    # ---------------------------------------------------------------------------------------
    def place(self, pd: ProData, purchase_territories: Optional[Dict[int, PurchaseTerritory]], rng) -> List:
        """Placement actions for the PLACE decision of pd.s, checked by the kernel on a copy."""
        self.pd = pd
        self.player = player = pd.player
        d = pd.d
        w = pd.s.clone()
        actions: List = []
        if purchase_territories:
            for ppt in purchase_territories.values():
                for place in ppt.can_place:
                    held = [u for u in w.held_units(player)]
                    chosen = []
                    for planned in place.place_units:
                        match = next((u for u in held if u.type == planned.type and u not in chosen), None)
                        if match is not None:
                            chosen.append(match)
                    self._do_place(d, w, place.t, chosen, rng, actions)
        if w.held_units(player):
            self._place_remaining_units(ProData(d, w, player), rng, actions)
        return actions

    def _do_place(self, d, w, t, units, rng, actions):
        """Place units one by one, as doPlace does; the kernel rejects what does not fit."""
        placed = []
        for u in units:
            a = fm.PlaceAction((u.id,), t)
            if fm.validate(d, w, a).ok:
                fm.apply_in_place(d, w, a, rng)
                placed.append(u.id)
        if placed:
            actions.append(fm.PlaceAction(tuple(placed), t))

    def _placeable(self, w, t) -> Tuple[List, int]:
        """placeDelegate.getPlaceableUnits: the held units that fit at t and the room left."""
        cap = (w.placement_cap or {}).get(t)
        if cap is None or w.owner[t] != self.player:
            return [], 0
        held = [u for u in w.held_units(self.player) if not self.pd.ut(u).is_factory]
        room = len(held) if cap < 0 else max(0, cap - w.placement_used.get(t, 0))
        return held, room

    def _place_remaining_units(self, pd: ProData, rng, actions):
        self.pd = pd
        self.tm = TerritoryManager(self.calc, pd)
        player, calc, d, w = self.player, self.calc, pd.d, pd.s
        purchase_territories = find_purchase_territories(pd, player)
        place_territories = {p.t for ppt in purchase_territories.values() for p in ppt.can_place}
        self.tm.populate_enemy_attack_options([], place_territories)
        self._find_defenders_in_place_territories(purchase_territories)
        need = self._prioritize_territories_to_defend(purchase_territories)
        enemy_attack = self.tm.enemy_attack_options
        for place in need:
            t = place.t
            held, room = self._placeable(w, t)
            to_place, final = [], BattleResult()
            for u in held[:room]:
                to_place.append(u)
                final = calc.calculate(pd, t, list(enemy_attack.get_max(t).max_units),
                                       place.defending_units + to_place)
                if self._is_held(t, final):
                    break
            if not final.has_land_unit_remaining or final.tuv_swing < place.min_battle_result.tuv_swing \
                    or t == pd.my_capital:
                self._do_place(d, w, t, to_place, rng, actions)
            else:
                place.can_hold = False
        values = find_territory_values(pd, player, [], [], place_territories)
        for ppt in purchase_territories.values():
            for place in ppt.can_place:
                place.strategic_value = values[place.t]
        prioritized = self._prioritize_land_territories(purchase_territories)
        for ppt in purchase_territories.values():
            for place in ppt.can_place:
                if place not in prioritized:
                    prioritized.append(place)
        for place in prioritized:
            held, room = self._placeable(w, place.t)
            if held and room > 0:
                self._do_place(d, w, place.t, held[:room], rng, actions)


def _upgrade_efficiency(p: PurchaseOption, strategic_value: float) -> float:
    multiplier = p.defense_efficiency if strategic_value >= 1 else p.movement
    return p.attack_efficiency * multiplier * p.cost / p.quantity


def _transport_cost(p: PurchaseOption) -> int:
    cost = next((int(v) for k, v in p.unit_type.deferred if k == "transportCost"), -1)
    return cost * p.quantity


def closest_enemy_or_neutral_land_distance(pd: ProData, player: int, t: int, values: Dict[int, float]) -> int:
    """ProUtils.getClosestEnemyOrNeutralLandTerritoryDistance (-1 if none within 9)."""
    land = pd.can_potentially_move_land
    enemies = set(pd.enemy_players(player))
    best = 10
    for x in pd.neighbors_within(t, 9, land):
        if pd.s.owner[x] not in enemies or values.get(x, 0.0) <= 0:
            continue
        dist = pd.distance(t, x, land)
        if pd.is_neutral_land(x):
            dist += 1
        if dist < best:
            best = dist
    return best if best < 10 else -1
