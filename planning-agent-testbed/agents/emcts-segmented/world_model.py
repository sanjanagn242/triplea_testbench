"""Roxo's forward model (WorldModel, NewAction / NewActionSkip / NewActionAdvance and ActionUtils
in his MCTSGrandStrategy fork) on top of the testbench kernel (triplea_fm).

A WorldModel is one tree node's world: a kernel state paused at a move decision, the territories
already decided in this move phase (`excluded`), the acting player's territories and the enemy
territories next to them, and the node's action list. Actions are territory based:

  * combat move: for the next enemy territory next to the player's territories that its units
    can take (Weak AI strength > 1.1x the defence), one action per unit set of
    getUnitsUpToStrengthSet (largest first), then skip; territories that cannot be taken are
    dropped;
  * noncombat move: for the next territory the player owns, one action per non-empty combination
    of the unit groups (one group per territory the units stand in) that can reach it, largest
    combinations first, then skip;
  * when no territory is left: Advance, which ends the phase, resolves the battles and stops at
    the next move phase, whoever's it is. The tree therefore goes on through later phases and
    the opponent's turns.
"""

from __future__ import annotations

import itertools
import random
from dataclasses import dataclass
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

import triplea_fm as fm
from triplea_fm import NULL_PLAYER, EndPhaseAction, MoveAction, RetreatAction, SelectBattleAction
from triplea_fm import rules, toolkit
from triplea_fm.actions import CONTINUE
from triplea_fm.definition import COMBAT_MOVE
from triplea_fm.state import BATTLE_SELECTION_D, CASUALTIES_D, MOVE_D, RETREAT_D

import fast_battle
import weak_ai
from land_graph import LandGraph

fast_battle.install()

TERRITORY, SKIP, ADVANCE = "territory", "skip", "advance"


@dataclass(frozen=True)
class TerritoryAction:
    """NewAction (units sent to territory), NewActionSkip or NewActionAdvance."""
    kind: str
    territory: Optional[int] = None
    units: Tuple[int, ...] = ()


@dataclass
class PlayoutResult:
    reward: float
    phases: int  # move phases the playout played
    rounds: int  # rounds it advanced
    terminal: bool


class WorldModel:
    """One node's world (his WorldModel)."""

    def __init__(self, game: "Game", state, excluded: FrozenSet[int] = frozenset(),
                 allied: Optional[List[int]] = None, enemy: Optional[List[int]] = None):
        self.game = game
        self.state = state
        self.excluded = set(excluded)
        self.allied = allied if allied is not None else game.allied_territories(state)
        self.enemy = enemy if enemy is not None else game.enemy_territories(state, self.allied)
        self.actions: List[TerritoryAction] = []
        self.action_index = 0

    @property
    def terminal(self) -> bool:
        return self.state.terminal is not None

    def child(self) -> "WorldModel":
        """generateChildWorldModel: a copy of the state, the excluded set and the territory lists."""
        return WorldModel(self.game, self.state.clone(), frozenset(self.excluded), self.allied, self.enemy)

    def next_action(self) -> Optional[TerritoryAction]:
        if self.action_index < len(self.actions):
            self.action_index += 1
            return self.actions[self.action_index - 1]
        return None

    def generate_actions(self):
        self.actions = [] if self.terminal else self.game.actions(self)
        self.action_index = 0

    def apply(self, action: TerritoryAction):
        if action.kind == TERRITORY:
            self.game.move_units(self.state, action.territory, action.units)
            self.excluded.add(action.territory)
        elif action.kind == SKIP:
            self.excluded.add(action.territory)
        else:
            self.game.advance(self.state)
            self.excluded = set()
            self.allied = self.game.allied_territories(self.state)
            self.enemy = self.game.enemy_territories(self.state, self.allied)


class Game:
    """Rules and heuristics shared by every node of one game (his ActionUtils, ForwardModel and
    ModelTerritoryManager helpers, the playout and the reward)."""

    def __init__(self, d, player: int, playout_phases: int = 40, seed: int = 0):
        self.d = d
        self.player = player  # the searching player: rewards are always from its point of view
        self.playout_phases = playout_phases
        self.graph = LandGraph(d)
        self.rng = fm.SeededRng(0)  # no dice are drawn: battles are deterministic
        self.rnd = random.Random(seed)  # the Weak AI's shuffle

    # ---- the acting player and its territories ----
    @staticmethod
    def actor(s) -> int:
        return s.decision.actor

    def is_combat(self, s) -> bool:
        return self.d.steps[s.step].kind == COMBAT_MOVE

    def allied_territories(self, s) -> List[int]:
        """generateAlliedTerritories: territories the acting player owns, in map order."""
        p = self.actor(s) if s.terminal is None else self.player
        return [t for t in range(len(self.d.territory_names)) if s.owner[t] == p]

    def enemy_territories(self, s, allied: Sequence[int]) -> List[int]:
        """getNeighboringEnemyTerritories: enemy land next to each allied territory, in that order
        (a territory next to several of them is listed several times)."""
        if s.terminal is not None:
            return []
        d, p = self.d, self.actor(s)
        return [x for t in allied for x in d.neighbors[t]
                if not d.is_water[x] and not d.impassable[x] and rules.is_enemy_territory(d, s, p, x)]

    # ---- action generation (WorldModel.generateActions) ----
    def actions(self, w: WorldModel) -> List[TerritoryAction]:
        d, s = self.d, w.state
        if self.is_combat(s):
            for t in w.enemy:
                if t in w.excluded:
                    continue
                units = self.move_options(s, t, combat=True)
                if self.can_attack_successfully(s, units, t):
                    sets = self.units_up_to_strength_sets(weak_ai.strength(d, s.units_in(t), False), units)
                    return ([TerritoryAction(TERRITORY, t, tuple(u.id for u in us)) for us in sets]
                            + [TerritoryAction(SKIP, t)])
                w.excluded.add(t)
            return [TerritoryAction(ADVANCE)]
        for t in w.allied:
            if t in w.excluded:
                continue
            groups = self.move_option_groups(s, t)
            combos = [tuple(u.id for g in combo for u in g)
                      for r in range(len(groups), 0, -1) for combo in itertools.combinations(groups, r)]
            return [TerritoryAction(TERRITORY, t, units) for units in combos] + [TerritoryAction(SKIP, t)]
        return [TerritoryAction(ADVANCE)]

    def _movable_land_units(self, s, origin: int, combat: bool) -> List:
        """ProMatches.unitCanBeMovedAndIsOwnedLand: the acting player's land units in origin with
        movement left (in a combat move, not those that cannot move in combat)."""
        d, p = self.d, self.actor(s)
        return [u for u in s.units_in(origin)
                if u.owner == p and d.unit_types[u.type].is_land and rules.movement_left(d, u) > 0
                and (not combat or d.unit_types[u.type].can_move_in_combat)]

    def _route(self, s, u, t: int, max_steps: int, cache=None) -> Optional[Tuple[int, ...]]:
        """getRouteForUnit with territoryCanMoveLandUnitsThrough: a shortest route for u to t
        within max_steps. `cache` shares the search between units of one kind in one state."""
        ut = self.d.unit_types[u.type]
        key = (u.loc, max_steps, ut.can_blitz)
        routes = cache.get(key) if cache is not None else None
        if routes is None:
            routes = toolkit.unit_routes(self.d, s, self.actor(s), u.loc, max_steps, False, ut.can_blitz,
                                         self.is_combat(s))
            if cache is not None:
                cache[key] = routes
        return routes.get(t)

    def _reaches(self, s, u, t: int, cache) -> bool:
        return self._route(s, u, t, rules.movement_left(self.d, u), cache) is not None

    def _unit_territories(self, s) -> List[int]:
        p = self.actor(s)
        return [t for t in range(len(self.d.territory_names)) if any(u.owner == p for u in s.units_in(t))]

    def move_options(self, s, t: int, combat: bool) -> List:
        """findLandMoveOptionsSingular: every unit of the acting player that can reach t this move
        (in his code a HashSet; here in map order, then unit order)."""
        out, cache = [], {}
        for origin in self._unit_territories(s):
            if origin != t:
                out.extend(u for u in self._movable_land_units(s, origin, combat) if self._reaches(s, u, t, cache))
        return out

    def move_option_groups(self, s, t: int) -> List[List]:
        """findLandMoveOptionsSingularSet: the same units, one group per territory they stand in."""
        groups, cache = [], {}
        for origin in self._unit_territories(s):
            if origin != t:
                group = [u for u in self._movable_land_units(s, origin, False) if self._reaches(s, u, t, cache)]
                if group:
                    groups.append(group)
        return groups

    def can_attack_successfully(self, s, units, t: int) -> bool:
        """ModelTerritoryManager.canAttackSuccessfully: Weak AI strength > 1.1x the territory's."""
        return weak_ai.strength(self.d, units, True) > 1.1 * weak_ai.strength(self.d, s.units_in(t), False)

    def units_up_to_strength_sets(self, max_strength: float, units: List) -> List[List]:
        """WorldModel.getUnitsUpToStrengthSet: growing prefixes of units kept when their strength
        passes a threshold that starts at 1.1x the defence and rises by 0.1 for each kept prefix,
        staying below 3.5x + 4; if the full set was not kept, it is added. Largest first."""
        sets, threshold, prefix, full = [], 1.1, [], False
        for u in units:
            prefix.append(u)
            full = False
            ours = weak_ai.strength(self.d, prefix, True)
            if threshold * max_strength < ours < 3.5 * max_strength + 4:
                full = True
                sets.append(list(prefix))
                threshold += 0.1
        if not full:
            sets.append(list(prefix))
        return list(reversed(sets))

    # ---- applying actions ----
    def routed_moves(self, s, t: int, unit_ids: Sequence[int]) -> List[MoveAction]:
        """ForwardModel.calculateMoveRoutesNew + doMove: one route per unit (no movement limit;
        the kernel checks it), units sharing a route merged into one move."""
        by_route: Dict[Tuple[int, ...], List[int]] = {}
        cache = {}
        for uid in unit_ids:
            u = s.units.get(uid)
            if u is None or u.loc == t or u.loc < 0:
                continue
            route = self._route(s, u, t, len(self.d.territory_names), cache)
            if route is not None:
                by_route.setdefault(route, []).append(uid)
        return [MoveAction(tuple(ids), r) for r, ids in by_route.items()]

    def move_units(self, s, t: int, unit_ids: Sequence[int]):
        for m in self.routed_moves(s, t, unit_ids):
            fm.apply_in_place(self.d, s, m, self.rng)  # a rejected move is skipped, as performMove

    def advance(self, s):
        """ActionUtils.advance: end this move phase, resolve its battles, stop at the next move
        decision (or the end of the game)."""
        fm.apply_in_place(self.d, s, EndPhaseAction(), self.rng)
        while s.terminal is None and s.decision.kind != MOVE_D:
            self._automatic_decision(s)

    def _automatic_decision(self, s):
        dec = s.decision
        if dec.kind == BATTLE_SELECTION_D:
            action = SelectBattleAction(min(b.id for b in s.battles.values() if b.status == "PENDING"))
        elif dec.kind == CASUALTIES_D:  # raids only; land battles never ask
            action = toolkit.default_casualties(self.d, s)
        elif dec.kind == RETREAT_D:
            action = RetreatAction(CONTINUE)
        else:  # purchase / place: no production in this setting
            action = EndPhaseAction()
        if fm.apply_in_place(self.d, s, action, self.rng) == fm.REJECTED:
            candidates = toolkit.candidates(self.d, s)
            fm.apply_in_place(self.d, s, candidates[-1] if candidates else EndPhaseAction(), self.rng)

    # ---- playout and reward (NonExploringMcts.Playout) ----
    def playout(self, s) -> PlayoutResult:
        """From a node's state (consumed): up to 40 times, the acting player makes its Weak AI moves
        for this move phase and the game advances to the next move phase."""
        start_round = s.round
        phases = 0
        while s.terminal is None and phases < self.playout_phases:
            phases += 1
            if s.decision.kind == MOVE_D:
                p = self.actor(s)
                if self.is_combat(s):
                    moves = weak_ai.combat_moves(self.d, s, p, self.rnd)
                else:
                    moves = weak_ai.noncombat_moves(self.d, s, p, self.graph)
                for m in moves:
                    fm.apply_in_place(self.d, s, m, self.rng)
            self.advance(s)
        return PlayoutResult(self.reward(s, phases), phases, s.round - start_round, s.terminal is not None)

    def reward(self, s, phases: int) -> float:
        """Wins score 0.6 + 0.4 (40 - phases) / 40, losses the negative; otherwise
        (own units - all other units) / (2.5 x all units), counting every unit on the map."""
        d, p = self.d, self.player
        if s.terminal is not None:
            won = d.alliance_of[p] in s.terminal.winners  # winners are alliance ids
            value = 0.6 + 0.4 * (self.playout_phases - phases) / self.playout_phases
            return value if won else -value
        units = [u for u in s.units.values() if u.loc >= 0]
        own = sum(1 for u in units if u.owner == p)
        return (own - (len(units) - own)) / (len(units) * 2.5) if units else 0.0
