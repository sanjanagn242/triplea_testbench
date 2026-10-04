"""Baseline heuristic policy and a generic state evaluation (library-provided, not kernel).

`HeuristicPolicy` can play every decision kind for any player. The testbench uses it as a
reference opponent in offline arenas (the forward-model counterpart of TripleA's built-in AIs),
and agents may use it for rollouts or opponent modelling. `evaluate` scores a state for a player
in [0, 1]. Both are agent-side helpers built on the public kernel API, like the ActionToolkit;
the kernel does not import this module.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional

from . import kernel as fm
from . import rules, toolkit
from .actions import (CONTINUE, RAID, RETREAT, EndPhaseAction, MoveAction, PlaceAction,
                                PurchaseAction, RetreatAction, SelectBattleAction)
from .definition import COMBAT_MOVE, NULL_PLAYER
from .state import (BATTLE_SELECTION_D, CASUALTIES_D, MOVE_D, PLACE_D, PURCHASE_D,
                              RETREAT_D)


class Precomp:
    """Static per-definition tables."""

    def __init__(self, d):
        self.d = d
        n = len(d.territory_names)
        self.dist = []
        for a in range(n):
            dist = [10 ** 6] * n
            dist[a] = 0
            q = deque([a])
            while q:
                t = q.popleft()
                for m in d.neighbors[t]:
                    if dist[m] > dist[t] + 1 and not d.impassable[m]:
                        dist[m] = dist[t] + 1
                        q.append(m)
            self.dist.append(dist)
        self.cost = [max(1, d.unit_cost(u.id)) for u in d.unit_types]
        # Territory value used by heuristics: victory cities and capitals dominate.
        self.value = []
        for t in range(n):
            v = d.production[t] + 12 * d.victory_value[t]
            if d.capital_of[t] != NULL_PLAYER:
                v += 15
            self.value.append(v)


def _att(d, units) -> int:
    support = sum(1 for u in units if d.unit_types[u.type].is_artillery)
    total = 0
    for u in units:
        ut = d.unit_types[u.type]
        if ut.is_infrastructure:
            continue
        a = ut.attack
        if ut.is_artillery_supportable and support > 0:
            a += 1
            support -= 1
        total += a
    return total


def _def(d, units) -> int:
    return sum(d.unit_types[u.type].defense for u in units if not d.unit_types[u.type].is_infrastructure)


def evaluate(d, pc: Precomp, s, player: int) -> float:
    """Value in [0, 1] for `player`: 1/0 at terminal states, otherwise its share of a material and
    territorial score across all players."""
    if s.terminal is not None:
        return 1.0 if d.alliance_of[player] in s.terminal.winners else 0.0
    n = len(d.player_names)
    score = [1.0] * n
    for t, o in enumerate(s.owner):
        if o != NULL_PLAYER:
            score[o] += 3 * d.production[t] + 30 * d.victory_value[t]
    for t, c in enumerate(d.capital_of):
        if c != NULL_PLAYER and s.owner[t] == c:
            score[c] += 25
    for u in s.units.values():
        if u.owner != NULL_PLAYER and not d.unit_types[u.type].is_infrastructure:
            score[u.owner] += pc.cost[u.type]
    for p in range(n):
        score[p] += s.balances[p][d.treasury_resource]
    total = sum(score)
    return score[player] / total


class HeuristicPolicy:
    def __init__(self, d, pc: Optional[Precomp] = None, aggression: float = 1.2):
        self.d = d
        self.pc = pc or Precomp(d)
        self.aggression = aggression

    # ---- dispatch ----
    def act(self, s, rnd):
        k = s.decision.kind
        if k == PURCHASE_D:
            return self.purchase(s)
        if k == MOVE_D:
            if self.d.steps[s.step].kind == COMBAT_MOVE:
                return self.combat_move(s, rnd)
            return self.noncombat_move(s)
        if k == PLACE_D:
            return self.place(s)
        if k == BATTLE_SELECTION_D:
            return SelectBattleAction(min(b.id for b in s.battles.values()
                                          if b.attacker == s.decision.actor and b.status == "PENDING"))
        if k == CASUALTIES_D:
            return toolkit.default_casualties(self.d, s)
        if k == RETREAT_D:
            return self.retreat(s)
        raise ValueError(k)

    # ---- purchase ----
    def purchase(self, s):
        d = self.d
        p = s.decision.actor
        if s.held_units(p):
            return EndPhaseAction()
        r = d.treasury_resource
        budget = s.balances[p][r]
        options = []
        for rid in d.frontier[p]:
            rule = d.production_rules[rid]
            if len(rule.results) != 1 or any(res != r for res, _ in rule.costs):
                continue
            cost = sum(a for _, a in rule.costs)
            ut = d.unit_types[rule.results[0][0]]
            if ut.is_infrastructure or ut.domain == 2 or cost <= 0:
                continue
            options.append((rid, cost, ut))
        if not options:
            return EndPhaseAction()
        defensive = max(options, key=lambda o: (o[2].defense / o[1], -o[1]))
        offensive = max(options, key=lambda o: ((o[2].attack + 0.5 * o[2].movement) / o[1], -o[1]))
        counts: Dict[int, int] = {}
        toggle = True
        while True:
            pick = offensive if toggle else defensive
            if budget < pick[1]:
                pick = defensive if budget >= defensive[1] else None
            if pick is None:
                break
            counts[pick[0]] = counts.get(pick[0], 0) + 1
            budget -= pick[1]
            toggle = not toggle
        if not counts:
            return EndPhaseAction()
        a = PurchaseAction(tuple(sorted(counts.items())))
        return a if fm.validate(d, s, a).ok else EndPhaseAction()

    # ---- placement ----
    def place(self, s):
        d = self.d
        p = s.decision.actor
        held = [u.id for u in s.held_units(p) if not d.unit_types[u.type].is_factory]
        if not held:
            return EndPhaseAction()
        caps = sorted((s.placement_cap or {}).items(), key=lambda kv: (kv[1] >= 0, -self.pc.value[kv[0]]))
        for t, cap in caps:
            room = len(held) if cap < 0 else cap - s.placement_used.get(t, 0)
            if room > 0:
                a = PlaceAction(tuple(held[:room]), t)
                if fm.validate(d, s, a).ok:
                    return a
        return EndPhaseAction()

    # ---- battles ----
    def retreat(self, s):
        d = self.d
        b = s.battles[s.decision.battle_id]
        att = [s.units[u] for u in b.attackers if u in s.units]
        dfd = [u for u in rules.enemy_units(d, s, b.attacker, b.site)]
        if _att(d, att) >= 0.6 * _def(d, dfd):
            return RetreatAction(CONTINUE)
        dests = rules.retreat_destinations(d, s, b)
        return RetreatAction(RETREAT, dests[0]) if dests else RetreatAction(CONTINUE)

    # ---- targets ----
    def _targets(self, s, p) -> List[int]:
        d = self.d
        return [t for t in range(len(d.territory_names))
                if not d.is_water[t] and s.owner[t] != p and not d.is_allied(p, s.owner[t])
                and self.pc.value[t] > 0]

    def _target_distance(self, s, p) -> List[int]:
        targets = self._targets(s, p)
        n = len(self.d.territory_names)
        if not targets:
            return [0] * n
        return [min(self.pc.dist[t][x] for x in targets) for t in range(n)]

    # ---- combat move ----
    def combat_move(self, s, rnd):
        d = self.d
        p = s.decision.actor
        movable: Dict[int, List] = {}
        for u in s.units.values():
            if u.owner == p and u.loc >= 0 and rules.movement_left(d, u) > 0:
                ut = d.unit_types[u.type]
                if ut.is_land and ut.can_move_in_combat:
                    movable.setdefault(u.loc, []).append(u)
        if not movable:
            return EndPhaseAction()
        best = None
        for t, units in movable.items():
            for n in d.neighbors[t]:
                if d.is_water[n] or s.owner[n] == p or d.is_allied(p, s.owner[n]):
                    continue
                enemies = rules.enemy_combat_units(d, s, p, n)
                defense = _def(d, enemies)
                already = [u for u in s.units_in(n) if u.owner == p]
                if not enemies:
                    if already:
                        continue
                    pick = min(units, key=lambda u: (self.pc.cost[u.type], u.id))
                    score = self.pc.value[n] + 1
                    cand = (score, MoveAction((pick.id,), (t, n)))
                else:
                    pool = []
                    for src in d.neighbors[n]:
                        pool.extend(movable.get(src, []))
                    power = _att(d, pool + already)
                    if power < self.aggression * defense + 1:
                        continue
                    score = self.pc.value[n] + 2 * (power - defense)
                    cand = (score, MoveAction(tuple(sorted(u.id for u in units)), (t, n)))
                if best is None or cand[0] > best[0]:
                    if fm.validate(d, s, cand[1]).ok:
                        best = cand
        return best[1] if best else EndPhaseAction()

    # ---- noncombat move ----
    def noncombat_move(self, s):
        d = self.d
        p = s.decision.actor
        dist = self._target_distance(s, p)
        for u in sorted(s.units.values(), key=lambda x: x.id):
            if u.owner != p or u.loc < 0 or rules.movement_left(d, u) <= 0:
                continue
            ut = d.unit_types[u.type]
            t = u.loc
            if ut.is_air:
                if rules.can_land_here(d, s, p, t):
                    continue
                spots = rules._air_distances(d, s, t, rules.movement_left(d, u))
                for dest in sorted(spots, key=lambda x: (spots[x], dist[x])):
                    if dest != t and rules.can_land_here(d, s, p, dest):
                        route = self._route(t, dest, lambda x: not rules.is_neutral_land(d, s, x))
                        if route:
                            a = MoveAction((u.id,), route)
                            if fm.validate(d, s, a).ok:
                                return a
                continue
            if not ut.is_land:
                continue
            capital_guard = d.capital_of[t] == p and sum(
                1 for x in s.units_in(t) if x.owner == p and not d.unit_types[x.type].is_infrastructure) <= 2
            if capital_guard:
                continue
            best = None
            for n in d.neighbors[t]:
                if s.owner[n] == p and dist[n] < dist[t] and not rules.enemy_units(d, s, p, n):
                    if best is None or dist[n] < dist[best]:
                        best = n
            if best is not None:
                group = tuple(sorted(x.id for x in s.units_in(t) if x.owner == p and x.type == u.type
                                     and rules.movement_left(d, x) > 0))
                a = MoveAction(group, (t, best))
                if fm.validate(d, s, a).ok:
                    return a
        return EndPhaseAction()

    def _route(self, start, end, passable):
        d = self.d
        parent = {start: None}
        q = deque([start])
        while q:
            t = q.popleft()
            if t == end:
                break
            for n in d.neighbors[t]:
                if n not in parent and (n == end or passable(n)):
                    parent[n] = t
                    q.append(n)
        if end not in parent:
            return None
        path = [end]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        return tuple(reversed(path))

    # ---- priors for MCTS ----
    def prior(self, s, a) -> float:
        d = self.d
        p = s.decision.actor
        k = a.kind
        if k == "end_phase":
            return 0.5
        if k == "move":
            end = a.route[-1]
            units = [s.units[u] for u in a.units]
            if d.steps[s.step].kind == COMBAT_MOVE and s.owner[end] != p and not d.is_allied(p, s.owner[end]):
                defense = _def(d, rules.enemy_combat_units(d, s, p, end))
                power = _att(d, units + [u for u in s.units_in(end) if u.owner == p])
                if a.intent == RAID:
                    return 0.3
                if defense == 0:
                    land = any(d.unit_types[u.type].is_land for u in units)
                    return 1.0 + self.pc.value[end] / 10 if land else 0.05
                return (power / (defense + 1)) + self.pc.value[end] / 20 - 0.5
            dist = self._target_distance(s, p)
            return 0.4 + 0.1 * (dist[a.route[0]] - dist[end])
        if k == "purchase":
            return 0.5 + 0.01 * sum(c for _, c in a.quantities)
        return 0.5
