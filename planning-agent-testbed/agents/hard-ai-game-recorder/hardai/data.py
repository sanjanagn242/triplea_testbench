"""Analysis data shared by every phase (ProData, ProTerritory, ProPurchaseOption, ProBattleResult).

Everything here reads a forward-model SimulationState; nothing changes it. Units are the state's
own Unit objects, plus TempUnit objects for hypothetical units (planned purchases), so the same
identity-based bookkeeping as the Java AI works on both.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Set

from triplea_fm import rules
from triplea_fm.definition import COMBAT_MOVE, NULL_PLAYER, GameDefinition
from triplea_fm.state import HELD, SimulationState

Pred = Callable[[int], bool]


class TempUnit:
    """A hypothetical unit (UnitType.createTemp): a planned purchase or a max-purchase defender."""

    __slots__ = ("id", "type", "owner", "orig_owner", "loc", "moved", "hits", "in_combat")
    _next_id = -1

    def __init__(self, type_id: int, owner: int):
        self.id = TempUnit._next_id
        TempUnit._next_id -= 1
        self.type = type_id
        self.owner = owner
        self.orig_owner = owner
        self.loc = HELD
        self.moved = 0
        self.hits = 0
        self.in_combat = False

    def __repr__(self):
        return f"TempUnit({self.type}, {self.owner})"


@dataclass
class BattleResult:
    """ProBattleResult. The default is the 'no battle' result."""

    win_percentage: float = 0.0
    tuv_swing: float = -1.0
    has_land_unit_remaining: bool = False
    average_attackers_remaining: Sequence = ()
    average_defenders_remaining: Sequence = ()
    battle_rounds: float = 0.0


class OrderedSet(dict):
    """Insertion-ordered set (deterministic replacement for Java's HashSet iteration)."""

    def add(self, x):
        self[x] = None

    def update_all(self, xs: Iterable):
        for x in xs:
            self[x] = None

    def discard(self, x):
        self.pop(x, None)


class ProTerritory:
    """The result of an AI territory analysis (ProTerritory), land/air fields only."""

    def __init__(self, t: int, pd: "ProData"):
        self.t = t
        self.pd = pd
        self.max_units = OrderedSet()
        self.units: List = []
        self.bombers: List = []
        self.max_battle_result = BattleResult()
        self.value = 0.0
        self.can_hold = True
        self.can_attack = False
        self.strength_estimate = math.inf
        self.strafing = False
        self.currently_wins = False
        self.battle_result: Optional[BattleResult] = None
        self.cant_move_units = OrderedSet()
        self.max_enemy_units: List = []
        self.min_battle_result = BattleResult()
        self.temp_units: List = []

    def copy(self, pd: "ProData") -> "ProTerritory":
        c = ProTerritory(self.t, pd)
        c.max_units = OrderedSet(self.max_units)
        c.units = list(self.units)
        c.bombers = list(self.bombers)
        c.max_battle_result = self.max_battle_result
        c.value = self.value
        c.can_hold = self.can_hold
        c.can_attack = self.can_attack
        c.strength_estimate = self.strength_estimate
        c.strafing = self.strafing
        c.currently_wins = self.currently_wins
        c.battle_result = self.battle_result
        c.cant_move_units = OrderedSet(self.cant_move_units)
        c.max_enemy_units = list(self.max_enemy_units)
        c.min_battle_result = self.min_battle_result
        c.temp_units = list(self.temp_units)
        return c

    def all_defenders(self) -> List:
        return list(dict.fromkeys(list(self.units) + list(self.cant_move_units) + self.temp_units))

    def eligible_defenders(self, player: int) -> List:
        # Allied (not owned) air does not defend land it was moved to (unitIsAlliedNotOwnedAir).
        d = self.pd.d
        return [u for u in self.all_defenders()
                if not (u.owner != player and d.unit_types[u.type].is_air)]

    def max_defenders(self) -> List:
        return list(dict.fromkeys(list(self.max_units) + list(self.cant_move_units)))

    def max_enemy_defenders(self, player: int) -> List:
        return self.pd.enemy_units(player, self.t)

    def set_battle_result(self, result: Optional[BattleResult]):
        self.battle_result = result
        if result is None:
            self.currently_wins = False
        elif result.win_percentage >= self.pd.win_percentage and result.has_land_unit_remaining:
            self.currently_wins = True

    def estimate_battle_result(self, calc, player: int):
        self.set_battle_result(calc.estimate_attack(self.pd, self.t, self.units,
                                                    self.max_enemy_defenders(player)))

    def __repr__(self):
        return self.pd.d.territory_names[self.t]


class MyMoveOptions:
    """ProMyMoveOptions: per-territory analysis plus each unit's possible destinations."""

    def __init__(self):
        self.territory_map: Dict[int, ProTerritory] = {}
        self.unit_move_map: Dict[object, OrderedSet] = {}
        self.bomber_move_map: Dict[object, OrderedSet] = {}

    def copy(self, pd: "ProData") -> "MyMoveOptions":
        c = MyMoveOptions()
        c.territory_map = {t: p.copy(pd) for t, p in self.territory_map.items()}
        c.unit_move_map = {u: OrderedSet(ts) for u, ts in self.unit_move_map.items()}
        c.bomber_move_map = {u: OrderedSet(ts) for u, ts in self.bomber_move_map.items()}
        return c


class OtherMoveOptions:
    """ProOtherMoveOptions: the strongest single-player attack (or defence) per territory."""

    def __init__(self, pd: "ProData" = None, move_maps: Sequence[Dict[int, ProTerritory]] = (),
                 player: int = NULL_PLAYER, is_attacker: bool = True):
        self.max_move_map: Dict[int, ProTerritory] = {}
        self.move_maps: Dict[int, List[ProTerritory]] = {}
        if pd is None:
            return
        from .odds import estimate_strength  # noqa: avoid an import cycle
        d = pd.d
        players = pd.other_players_in_turn_order(player)
        for move_map in move_maps:
            for t, pt in move_map.items():
                self.move_maps.setdefault(t, []).append(pt)
                current = list(pt.max_units)
                if not current:
                    continue
                mover = current[0].owner
                if d.is_allied(player, mover) and not pd.is_players_turn_first(players, mover, pd.s.owner[t]):
                    continue
                if t not in self.max_move_map:
                    self.max_move_map[t] = pt
                    continue
                best = list(self.max_move_map[t].max_units)
                max_strength = estimate_strength(pd, t, best, [], is_attacker) if best else 0.0
                strength = estimate_strength(pd, t, current, [], is_attacker)
                cur_land = any(d.unit_types[u.type].is_land for u in current)
                max_land = any(d.unit_types[u.type].is_land for u in best)
                water = d.is_water[t]
                if (cur_land and ((not max_land and not water) or strength > max_strength)) or \
                        ((not max_land or water) and strength > max_strength):
                    self.max_move_map[t] = pt

    def get_max(self, t: int) -> Optional[ProTerritory]:
        return self.max_move_map.get(t)

    def get_all(self, t: int) -> List[ProTerritory]:
        return self.move_maps.get(t, [])


class PurchaseOption:
    """ProPurchaseOption for a single-unit-type production rule paid in PUs."""

    def __init__(self, d: GameDefinition, rule_id: int, player: int):
        rule = d.production_rules[rule_id]
        self.d = d
        self.player = player
        self.rule_id = rule_id
        self.type_id = rule.results[0][0]
        ut = d.unit_types[self.type_id]
        self.unit_type = ut
        self.cost = sum(a for r, a in rule.costs if r == d.treasury_resource)
        self.quantity = sum(q for _, q in rule.results)
        self.movement = ut.movement
        self.hit_points = 0 if ut.is_infrastructure else ut.hit_points * self.quantity
        self.attack = float(ut.attack * self.quantity)
        self.defense = float(ut.defense * self.quantity)
        self.is_air = ut.is_air
        self.is_attack_support = ut.is_artillery
        self.is_defense_support = False
        self.cost_per_hit_point = math.inf if self.hit_points == 0 else self.cost / self.hit_points
        six = 6.0 / d.dice_sides
        cost = max(self.cost, 1)
        self.attack_efficiency = (1 + self.hit_points) * (
            self.hit_points + self.attack * six + 0.5 * self.defense * six) / cost
        self.defense_efficiency = (1 + self.hit_points) * (
            self.hit_points + 0.5 * self.attack * six + self.defense * six) / cost

    def __repr__(self):
        return f"{self.d.production_rules[self.rule_id].name}(cost={self.cost})"

    def create_temp_units(self) -> List[TempUnit]:
        return [TempUnit(self.type_id, self.player) for _ in range(self.quantity)]

    # ---- efficiencies (ProPurchaseOption.get*Efficiency) ----
    def fodder_efficiency(self, enemy_distance, owned_local_units, units_to_place) -> float:
        return self._efficiency(0.25, 0.25, self._support_factor(owned_local_units, units_to_place, False),
                                0.0, math.sqrt(self._land_distance_factor(enemy_distance)))

    def attack_efficiency_at(self, enemy_distance, owned_local_units, units_to_place) -> float:
        return self._efficiency(1.25, 0.75, self._support_factor(owned_local_units, units_to_place, False),
                                0.0, self._land_distance_factor(enemy_distance))

    def defense_efficiency_at(self, enemy_distance, owned_local_units, units_to_place) -> float:
        return self._efficiency(0.75, 1.25, self._support_factor(owned_local_units, units_to_place, False),
                                0.0, self._land_distance_factor(enemy_distance))

    def _land_distance_factor(self, enemy_distance: int) -> float:
        if self.movement <= 0:
            return 0.1
        distance = max(0.0, enemy_distance - 1.5)
        move = self.movement
        move_factor = 1.0 + 2.0 * (2 ** (move - 1.0) - 1.0) / 2 ** (move - 1.0)
        return move_factor ** (distance / 5)

    def _support_factor(self, owned_local_units, units_to_place, defense: bool) -> float:
        """Artillery support (+1 attack to one supportable unit per artillery), the only support rule
        in the V1 profile. Mirrors calculateSupportFactor's 2:1 supportable-to-support ratio."""
        if defense or not self.is_attack_support:
            return 0.0
        d = self.d
        units = list(units_to_place) + [None] + list(owned_local_units)  # None = this unit
        support = sum(1 for u in units if (self.unit_type if u is None else d.unit_types[u.type]).is_artillery)
        supportable = sum(1 for u in units
                          if (self.unit_type if u is None else d.unit_types[u.type]).is_artillery_supportable)
        added = 1
        provided = support - added
        extra = max(0, supportable - provided)
        ratio = min(1.0, 2.0 * extra / (supportable + added))
        return (added * 0.9) ** 0.9 * 1.0 * ratio

    def _efficiency(self, attack_factor, defense_factor, support_attack, support_defense, distance_factor) -> float:
        six = 6.0 / self.d.dice_sides
        hp_per_unit_factor = 3 + self.hit_points / self.quantity
        attack_value = attack_factor * (self.attack + support_attack * self.quantity) * six
        defense_value = defense_factor * (self.defense + support_defense * self.quantity) * six
        base = (2 * self.hit_points + attack_value + defense_value) * hp_per_unit_factor * distance_factor
        return (base / max(self.cost, 1)) ** 30 / self.quantity


class PurchaseOptionMap:
    """ProPurchaseOptionMap restricted to what the V1 profile can build (land, air, AA)."""

    def __init__(self, d: GameDefinition, player: int):
        self.land_fodder: List[PurchaseOption] = []
        self.land_attack: List[PurchaseOption] = []
        self.land_defense: List[PurchaseOption] = []
        self.land_zero_move: List[PurchaseOption] = []
        self.air: List[PurchaseOption] = []
        self.aa: List[PurchaseOption] = []
        for rid in d.frontier[player]:
            rule = d.production_rules[rid]
            if len(rule.results) != 1 or any(r != d.treasury_resource for r, _ in rule.costs):
                continue
            ut = d.unit_types[rule.results[0][0]]
            if ut.domain == 2 or (ut.can_produce and ut.is_infrastructure):
                continue  # sea units and factories are outside the V1 profile
            ppo = PurchaseOption(d, rid, player)
            if ut.movement <= 0 and ut.is_land:
                self.land_zero_move.append(ppo)
            elif ut.is_land:
                if not ut.is_infrastructure:
                    self.land_fodder.append(ppo)
                if (ppo.attack > 0 or ppo.is_attack_support) and (ppo.attack >= ppo.defense or ppo.movement > 1):
                    self.land_attack.append(ppo)
                if (ppo.defense > 0 or ppo.is_defense_support) and (ppo.defense >= ppo.attack or ppo.movement > 1):
                    self.land_defense.append(ppo)
                if ut.is_aa:
                    self.aa.append(ppo)
            elif ut.is_air:
                self.air.append(ppo)
        if not self.land_attack:
            self.land_attack = list(self.land_defense)
        if not self.land_defense:
            self.land_defense = list(self.land_attack)

    def land_options(self) -> List[PurchaseOption]:
        return list(dict.fromkeys(self.land_fodder + self.land_attack + self.land_defense))

    def all_options(self) -> List[PurchaseOption]:
        return list(dict.fromkeys(self.land_options() + self.land_zero_move + self.air + self.aa))


class ProData:
    """ProData for one analysed state: the player, its capital, unit values and purchase options."""

    win_percentage = 90.0  # V1 refuses Low Luck, so the dice values of ProData always apply
    min_win_percentage = 65.0

    def __init__(self, d: GameDefinition, s: SimulationState, player: int):
        self.d = d
        self.s = s
        self.player = player
        self.n = len(d.territory_names)
        self.my_capital = self._first_owned_capital(player)
        self.my_unit_territories = sorted({u.loc for u in s.units.values() if u.owner == player and u.loc >= 0})
        self.unit_value = [d.unit_cost(ut.id) for ut in d.unit_types]  # TuvCostsCalculator
        self.purchase_options = PurchaseOptionMap(d, player)
        land = [p.cost_per_hit_point for p in self.purchase_options.land_options()]
        self.min_cost_per_hit_point = min(land) if land else math.inf

    # ---- players (ProUtils) ----
    def _first_owned_capital(self, player) -> Optional[int]:
        caps = [t for t, c in enumerate(self.d.capital_of) if c == player]
        owned = [t for t in caps if self.s.owner[t] == player]
        return owned[0] if owned else (caps[0] if caps else None)

    def is_passive_neutral(self, p: int) -> bool:
        return p == NULL_PLAYER or not any(st.actor == p and st.kind == COMBAT_MOVE for st in self.d.steps)

    def other_players_in_turn_order(self, player: int) -> List[int]:
        steps = self.d.steps
        out = []
        for i in range(len(steps)):
            st = steps[(self.s.step + i) % len(steps)]
            if st.kind == COMBAT_MOVE and st.actor not in (NULL_PLAYER, player) and st.actor not in out:
                out.append(st.actor)
        return out

    def enemy_players_in_turn_order(self, player: int) -> List[int]:
        return [p for p in self.other_players_in_turn_order(player) if not self.d.is_allied(player, p)]

    def allied_players_in_turn_order(self, player: int) -> List[int]:
        return [p for p in self.other_players_in_turn_order(player) if self.d.is_allied(player, p)]

    @staticmethod
    def is_players_turn_first(players_in_order: List[int], p1: int, p2: int) -> bool:
        for p in players_in_order:
            if p == p1:
                return True
            if p == p2:
                return False
        return True

    def enemy_players(self, player: int) -> List[int]:
        return [p for p in range(len(self.d.player_names)) if not self.d.is_allied(player, p)]

    def potential_enemy_players(self, player: int) -> List[int]:
        return [p for p in self.enemy_players(player) if not self.is_passive_neutral(p)]

    def is_ffa(self, player: int) -> bool:
        enemies = [p for p in self.enemy_players(player) if not self.is_passive_neutral(p)]
        return any(self.d.is_at_war(a, b) for a in enemies for b in enemies if a != b)

    def player_production(self, p: int) -> float:
        return float(sum(self.d.production[t] for t in range(self.n) if self.s.owner[t] == p) * self.d.pu_multiplier)

    def live_enemy_capitals(self, player: int) -> List[int]:
        d, s = self.d, self.s
        potential = set(self.potential_enemy_players(player))
        return [t for t, c in enumerate(d.capital_of)
                if c != NULL_PLAYER and not d.is_allied(player, c) and s.owner[t] == c
                and not d.impassable[t] and s.owner[t] in potential]

    def live_allied_capitals(self, player: int) -> List[int]:
        d, s = self.d, self.s
        return [t for t, c in enumerate(d.capital_of)
                if c != NULL_PLAYER and d.is_allied(player, c) and s.owner[t] == c and not d.impassable[t]]

    # ---- units ----
    def ut(self, u):
        return self.d.unit_types[u.type]

    def value_of(self, u) -> int:
        return self.unit_value[u.type]

    def tuv(self, units: Iterable) -> int:
        return sum(self.unit_value[u.type] for u in units)

    def units_in(self, t: int) -> List:
        return self.s.units_in(t)

    def enemy_units(self, player: int, t: int) -> List:
        return [u for u in self.s.units_in(t) if self.d.is_at_war(player, u.owner)]

    def allied_units(self, player: int, t: int) -> List:
        return [u for u in self.s.units_in(t) if self.d.is_allied(player, u.owner)]

    def owned_units(self, player: int, t: int) -> List:
        return [u for u in self.s.units_in(t) if u.owner == player]

    def movement_left(self, u) -> int:
        return 0 if isinstance(u, TempUnit) else rules.movement_left(self.d, u)

    def can_move(self, u, combat: bool) -> bool:
        ut = self.ut(u)
        return self.movement_left(u) > 0 and (not combat or ut.can_move_in_combat)

    # ---- territories (Matches / ProMatches for the land/air profile) ----
    def is_land(self, t) -> bool:
        return not self.d.is_water[t]

    def is_neutral_land(self, t) -> bool:
        return not self.d.is_water[t] and self.s.owner[t] == NULL_PLAYER

    def is_allied_territory(self, player, t) -> bool:
        o = self.s.owner[t]
        return o != NULL_PLAYER and self.d.is_allied(player, o)

    def is_enemy_territory(self, player, t) -> bool:
        """Matches.isTerritoryEnemyAndNotUnownedWater (unowned land counts as enemy)."""
        o = self.s.owner[t]
        if o == NULL_PLAYER and self.d.is_water[t]:
            return False
        return o != player and self.d.is_at_war(player, o)

    def is_enemy_land(self, player, t) -> bool:
        return self.is_land(t) and self.is_enemy_territory(player, t)

    def is_enemy_not_neutral_land(self, player, t) -> bool:
        return self.is_enemy_land(player, t) and not self.is_neutral_land(t)

    def is_enemy_not_passive_neutral_land(self, player, t) -> bool:
        return self.is_enemy_not_neutral_land(player, t) and not self.is_passive_neutral(self.s.owner[t])

    def has_enemy_units(self, player, t) -> bool:
        return bool(self.enemy_units(player, t))

    def has_infra_factory(self, t) -> bool:
        return self.is_land(t) and any(self.ut(u).can_produce and self.ut(u).is_infrastructure
                                       for u in self.s.units_in(t))

    def has_infra_factory_enemy_land(self, player, t) -> bool:
        return self.has_infra_factory(t) and self.is_enemy_land(player, t)

    def has_infra_factory_allied_land(self, player, t) -> bool:
        return self.has_infra_factory(t) and self.is_allied_territory(player, t)

    def was_conquered(self, t) -> bool:
        """Conquered this turn (the state's battle tracker belongs to the player on move)."""
        return t in self.s.conquered

    def has_factory_not_conquered_owned_land(self, player, t) -> bool:
        return (self.is_land(t) and self.s.owner[t] == player and not self.was_conquered(t)
                and any(u.owner == player and self.ut(u).can_produce for u in self.s.units_in(t)))

    def can_move_land(self, player, t, combat: bool) -> bool:
        """territoryCanMoveLandUnits: passable land, and not enemy/neutral outside combat."""
        if self.d.is_water[t] or self.d.impassable[t]:
            return False
        return combat or self.is_allied_territory(player, t)

    def can_potentially_move_land(self, t) -> bool:
        return not self.d.is_water[t] and not self.d.impassable[t]

    def can_move_air(self, t) -> bool:
        return not self.d.impassable[t] and not self.is_neutral_land(t)

    def has_enemy_aa(self, player, t) -> bool:
        return any(self.ut(u).is_aa for u in self.enemy_units(player, t))

    def can_move_air_no_aa(self, player, t) -> bool:
        return self.can_move_air(t) and not self.has_enemy_aa(player, t)

    def can_land_air(self, player, t, combat: bool, enemy_territories=(), allied_territories=()) -> bool:
        """territoryCanLandAirUnits."""
        if t in allied_territories:
            return True
        if t in enemy_territories or self.d.impassable[t]:
            return False
        return rules.can_land_here(self.d, self.s, player, t)

    def has_neighbor_owned_by_with_land_unit(self, t, players: Sequence[int]) -> bool:
        ps = set(players)
        return any(self.s.owner[n] in ps and any(self.ut(u).is_land for u in self.s.units_in(n))
                   for n in self.d.neighbors[t])

    # ---- graph (GameMap) ----
    def neighbors(self, t: int, cond: Optional[Pred] = None) -> List[int]:
        return [n for n in self.d.neighbors[t] if cond is None or cond(n)]

    def neighbors_within(self, t: int, distance: int, cond: Optional[Pred] = None) -> Set[int]:
        """GameMap.getNeighbors(t, distance, cond): reachable through matching territories,
        excluding t itself."""
        seen = {t: 0}
        q = deque([t])
        while q:
            x = q.popleft()
            if seen[x] >= distance:
                continue
            for n in self.d.neighbors[x]:
                if n not in seen and (cond is None or cond(n)):
                    seen[n] = seen[x] + 1
                    q.append(n)
        seen.pop(t)
        return set(seen)

    def distance(self, a: int, b: int, cond: Optional[Pred] = None, ignore_end: bool = False) -> int:
        """GameMap.getDistance / getDistanceIgnoreEndForCondition (-1 if unreachable)."""
        if a == b:
            return 0
        seen = {a: 0}
        q = deque([a])
        while q:
            x = q.popleft()
            for n in self.d.neighbors[x]:
                if n in seen:
                    continue
                if n == b and (ignore_end or cond is None or cond(n)):
                    return seen[x] + 1
                if cond is None or cond(n):
                    seen[n] = seen[x] + 1
                    q.append(n)
        return -1

    def get_pro_territory(self, move_map: Dict[int, ProTerritory], t: int) -> ProTerritory:
        pt = move_map.get(t)
        if pt is None:
            pt = move_map[t] = ProTerritory(t, self)
        return pt


@dataclass
class ResourceTracker:
    """ProResourceTracker for the single PU resource of the V1 profile."""

    pus: int
    temp: int = 0

    def is_empty(self) -> bool:
        return self.pus - self.temp <= 0

    def has_enough(self, ppo: PurchaseOption) -> bool:
        return self.pus - self.temp >= ppo.cost

    def purchase(self, ppo: PurchaseOption):
        self.pus -= ppo.cost

    def remove_purchase(self, ppo: PurchaseOption):
        self.pus += ppo.cost

    def temp_purchase(self, ppo: PurchaseOption):
        self.temp += ppo.cost

    def remove_temp_purchase(self, ppo: PurchaseOption):
        self.temp -= ppo.cost

    def confirm_temp_purchases(self):
        self.pus -= self.temp
        self.temp = 0

    def clear_temp_purchases(self):
        self.temp = 0

    def temp_pus(self) -> int:
        return self.temp


@dataclass
class PlaceTerritory:
    """ProPlaceTerritory."""

    t: int
    place_units: List = field(default_factory=list)
    defending_units: List = field(default_factory=list)
    min_battle_result: BattleResult = field(default_factory=BattleResult)
    defense_value: float = 0.0
    strategic_value: float = 0.0
    can_hold: bool = True

    def __hash__(self):
        return hash(self.t)

    def __eq__(self, other):
        return isinstance(other, PlaceTerritory) and other.t == self.t


@dataclass
class PurchaseTerritory:
    """ProPurchaseTerritory: a factory territory and where its production can be placed."""

    t: int
    unit_production: int
    can_place: List[PlaceTerritory] = field(default_factory=list)

    def remaining_unit_production(self) -> int:
        return self.unit_production - sum(len(p.place_units) for p in self.can_place)
