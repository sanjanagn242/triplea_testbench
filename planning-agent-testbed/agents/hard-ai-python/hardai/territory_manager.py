"""Move option analysis (ProTerritoryManager) for land and air units.

Which territories each unit can reach comes from the forward model's route generator
(triplea_fm.toolkit.unit_routes); this module only applies the Hard AI's filters on top of it.
Naval, transport, bombard and scramble options do not exist in the V1 profile and are omitted.
"""

from __future__ import annotations

from typing import Collection, List, Optional, Sequence, Set

from triplea_fm import toolkit

from .data import MyMoveOptions, OrderedSet, OtherMoveOptions, ProData, ProTerritory
from .odds import OddsCalculator


class TerritoryManager:
    def __init__(self, calc: OddsCalculator, pd: ProData, other: Optional["TerritoryManager"] = None):
        self.calc = calc
        self.pd = pd
        self.player = pd.player
        self.attack_options = MyMoveOptions()
        self.defend_options = MyMoveOptions()
        self.allied_attack_options = OtherMoveOptions()
        self.enemy_defend_options = OtherMoveOptions()
        self.enemy_attack_options = OtherMoveOptions()
        if other is not None:  # copy constructor
            self.attack_options = other.attack_options.copy(pd)
            self.defend_options = other.defend_options.copy(pd)
            self.allied_attack_options = other.allied_attack_options
            self.enemy_defend_options = other.enemy_defend_options
            self.enemy_attack_options = other.enemy_attack_options

    # ---- populate ----
    def populate_attack_options(self):
        find_attack_options(self.pd, self.player, self.pd.my_unit_territories, self.attack_options, [], [], [],
                            False, False)
        for u, ts in self.attack_options.unit_move_map.items():
            if self.pd.ut(u).is_strategic_bomber:
                self.attack_options.bomber_move_map[u] = OrderedSet(ts)
        self.allied_attack_options = self._find_allied_attack_options()

    def populate_defense_options(self, cleared: Sequence[int]):
        find_defend_options(self.pd, self.player, self.pd.my_unit_territories, self.defend_options,
                            list(cleared), False)

    def populate_enemy_attack_options(self, cleared: Collection[int], to_check: Collection[int]):
        self.enemy_attack_options = find_enemy_attack_options(self.pd, self.player, cleared, to_check)

    def populate_enemy_defense_options(self):
        pd = self.pd
        cleared = [t for t in range(pd.n) if pd.is_allied_territory(self.player, t)]
        maps = []
        for e in pd.enemy_players_in_turn_order(self.player):
            opts = MyMoveOptions()
            find_defend_options(pd, e, _unit_territories(pd, e), opts, cleared, True)
            maps.append(opts.territory_map)
        self.enemy_defend_options = OtherMoveOptions(pd, maps, self.player, False)

    def _find_allied_attack_options(self) -> OtherMoveOptions:
        pd = self.pd
        maps = []
        for ally in pd.allied_players_in_turn_order(self.player):
            opts = MyMoveOptions()
            find_attack_options(pd, ally, _unit_territories(pd, ally), opts, [], [], [], False, False)
            maps.append(opts.territory_map)
        return OtherMoveOptions(pd, maps, self.player, True)

    # ---- queries ----
    def remove_territories_that_cant_be_conquered(self) -> List[ProTerritory]:
        pd, calc, player = self.pd, self.calc, self.player
        d = pd.d
        attack_map = self.attack_options.territory_map
        to_remove = []
        for t, patd in attack_map.items():
            defenders = patd.max_enemy_defenders(player)
            patd.max_battle_result = calc.estimate_attack(pd, t, list(patd.max_units), defenders)
            is_enemy_capital_or_factory = not pd.is_neutral_land(t) and (
                d.capital_of[t] != -1 or pd.has_infra_factory(t))
            allied_attack = self.allied_attack_options.get_max(t)
            if patd.max_battle_result.win_percentage < pd.min_win_percentage and is_enemy_capital_or_factory \
                    and allied_attack is not None:
                self._check_strafing(t, patd, defenders, allied_attack)
            result = patd.max_battle_result
            if result.win_percentage < pd.min_win_percentage or (
                    patd.strafing and (result.win_percentage < pd.win_percentage
                                       or not result.has_land_unit_remaining)):
                to_remove.append(t)
        result = [p for t, p in attack_map.items() if t not in to_remove]
        for ts in self.attack_options.unit_move_map.values():
            for t in to_remove:
                ts.discard(t)
        return result

    def _check_strafing(self, t, patd: ProTerritory, defenders, allied_attack: ProTerritory):
        """Attack an enemy capital/factory that allies can only take after this player weakens it."""
        pd, calc, player = self.pd, self.calc, self.player
        allied_units = list(allied_attack.max_units)
        if not allied_units:
            return
        allied_player = allied_units[0].owner
        cap = pd._first_owned_capital(allied_player)
        if cap is None or t in pd.d.neighbors[cap]:
            return
        additional = []
        players = pd.other_players_in_turn_order(player)
        for option in self.enemy_defend_options.get_all(t):
            units = list(option.max_units)
            if units and pd.is_players_turn_first(players, units[0].owner, allied_player):
                additional.extend(units)
        before = list(dict.fromkeys(list(defenders) + additional))
        result = calc.estimate_attack(pd, t, allied_units, before)
        if result.win_percentage < pd.win_percentage:
            patd.strafing = True
            strafe = calc.call_battle_calc(pd, t, list(patd.max_units), defenders, retreat_when_only_air=True)
            after = list(dict.fromkeys(list(strafe.average_defenders_remaining) + additional))
            patd.max_battle_result = calc.estimate_attack(pd, t, allied_units, after)

    def defend_territories(self) -> List[int]:
        return list(self.defend_options.territory_map)

    def strafing_territories(self) -> List[int]:
        return [t for t, p in self.attack_options.territory_map.items() if p.strafing]

    def cant_hold_territories(self) -> List[int]:
        return [t for t, p in self.defend_options.territory_map.items() if not p.can_hold]


def _unit_territories(pd: ProData, player: int) -> List[int]:
    return sorted({u.loc for u in pd.s.units.values() if u.owner == player and u.loc >= 0})


# ---- option finders (static helpers of ProTerritoryManager) ----

def find_attack_options(pd: ProData, player: int, unit_territories: Sequence[int], options: MyMoveOptions,
                        enemy_territories: Sequence[int], allied_territories: Sequence[int],
                        to_check: Sequence[int], checking_enemy: bool, ignoring_relationships: bool):
    cant_be_held = set(enemy_territories) | set(to_check)
    find_land_move_options(pd, player, unit_territories, options,
                           lambda t: pd.is_enemy_territory(player, t) or t in cant_be_held,
                           enemy_territories, allied_territories, True, checking_enemy, ignoring_relationships)
    find_air_move_options(pd, player, unit_territories, options,
                          lambda t: pd.has_enemy_units(player, t) or t in cant_be_held,
                          enemy_territories, allied_territories, True, checking_enemy, ignoring_relationships)


def find_defend_options(pd: ProData, player: int, unit_territories: Sequence[int], options: MyMoveOptions,
                        cleared: Sequence[int], checking_enemy: bool):
    find_land_move_options(pd, player, unit_territories, options,
                           lambda t: pd.is_allied_territory(player, t), [], cleared, False, checking_enemy, False)
    find_air_move_options(pd, player, unit_territories, options,
                          lambda t: pd.can_land_air(player, t, False) and not pd.is_neutral_land(t),
                          [], [], False, checking_enemy, False)


def find_enemy_attack_options(pd: ProData, player: int, cleared: Collection[int],
                              to_check: Collection[int]) -> OtherMoveOptions:
    allied_territories: Set[int] = set()
    enemy_territories = list(cleared)
    maps = []
    for e in pd.enemy_players_in_turn_order(player):
        terrs = [t for t in _unit_territories(pd, e) if t not in cleared]
        opts = MyMoveOptions()
        find_attack_options(pd, e, terrs, opts, list(enemy_territories), sorted(allied_territories),
                            list(to_check), True, True)
        maps.append(opts.territory_map)
        allied_territories.update(t for t in opts.territory_map if pd.is_land(t))
        enemy_territories = [t for t in enemy_territories if t not in allied_territories]
    return OtherMoveOptions(pd, maps, player, True)


def find_land_move_options(pd: ProData, player: int, unit_territories, options: MyMoveOptions, move_to_match,
                           enemy_territories, cleared_territories, combat: bool, checking_enemy: bool,
                           ignoring_relationships: bool):
    d, s = pd.d, pd.s
    for start in unit_territories:
        for u in sorted(s.units_in(start), key=lambda x: x.id):
            ut = pd.ut(u)
            if u.owner != player or not ut.is_land or not pd.can_move(u, combat):
                continue
            rng = pd.movement_left(u)
            routes = toolkit.unit_routes(d, s, player, start, rng, False, ut.can_blitz and combat, combat,
                                         ignore_enemy_units=checking_enemy,
                                         cleared=cleared_territories if checking_enemy else (),
                                         blocked=enemy_territories)
            if ignoring_relationships:
                passable = pd.can_potentially_move_land
            else:
                passable = lambda t: pd.can_move_land(player, t, combat)  # noqa: E731
            potential = [t for t in [start] + sorted(routes) if passable(t) and move_to_match(t)]
            if not combat and start not in potential:
                potential.insert(0, start)
            for t in potential:
                if combat and ut.is_infrastructure:
                    continue  # can't take part in combat
                pd.get_pro_territory(options.territory_map, t).max_units.add(u)
                options.unit_move_map.setdefault(u, OrderedSet()).add(t)


def find_air_move_options(pd: ProData, player: int, unit_territories, options: MyMoveOptions, move_to_match,
                          enemy_territories, allied_territories, combat: bool, checking_enemy: bool,
                          ignoring_relationships: bool):
    d, s = pd.d, pd.s
    aa_blocked = () if checking_enemy else [t for t in range(pd.n) if pd.has_enemy_aa(player, t)]
    for start in unit_territories:
        for u in sorted(s.units_in(start), key=lambda x: x.id):
            ut = pd.ut(u)
            if u.owner != player or not ut.is_air or not pd.can_move(u, combat):
                continue
            rng = ut.movement if checking_enemy else pd.movement_left(u)
            routes = toolkit.unit_routes(d, s, player, start, rng, True, False, combat, blocked=aa_blocked)
            routes[start] = (start,)
            for t in sorted(routes):
                if not pd.can_move_air(t) or not move_to_match(t):
                    continue
                length = len(routes[t]) - 1
                remaining = rng - length
                if remaining < 0:
                    continue
                if combat and (remaining < length or d.is_water[start]):
                    fly = (lambda x: pd.can_move_air(x)) if checking_enemy else \
                        (lambda x: pd.can_move_air_no_aa(player, x))
                    landing = pd.neighbors_within(t, remaining, fly)
                    if not any(pd.can_land_air(player, x, combat, enemy_territories, allied_territories)
                               for x in landing):
                        continue
                pd.get_pro_territory(options.territory_map, t).max_units.add(u)
                options.unit_move_map.setdefault(u, OrderedSet()).add(t)
