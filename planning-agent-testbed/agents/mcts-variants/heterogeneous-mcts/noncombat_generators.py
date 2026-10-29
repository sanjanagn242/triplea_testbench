"""Non-combat move options for the non-combat MCTS (noncombat_mcts.py), one territory at a time.

The greedy action_generators.noncombat_move plans the whole phase at once: (1) land the air the
engine would destroy, (2) fill each territory's defensive shortfall, most important first, and
(3) push the spare units toward the frontier. Here step 2 becomes a search, the way the combat
move search decides one target at a time (action_generators.combat_actions):

  land_air_moves       step 1, forced, played once when a player's non-combat move starts.
  noncombat_actions    step 2 for the most important territory still short and not excluded:
                       reinforcement sets of growing strength (strongest first), then a skip.
                       When no such territory is left, a single end-of-phase action whose moves
                       are step 3 (positioning_moves).
  positioning_moves    step 3, played when the phase ends.

Values, threats and wanted cover are those of noncombat_move (defense_value, _enemy_threat,
THREAT_COVER, HOME_COVER, HOME_KEEP, FRONTIER_MIN); units already moved this phase (`moved`, forward-
model unit ids) are never moved again. `game` must come from GameState.from_sim so units carry
forward-model ids. Helpers come from `ag`, the Cython build when available.
"""

from generators import ag

# Reinforcement sets offered for a territory, as shares of its shortfall; the last set offered is
# the first to cover the largest share, or every unit that can be spared if none covers it.
COVER_LEVELS = (0.5, 1.0, 1.5)
_EPS = 1e-9


class NoncombatAction:
    """A reinforcement of `to_terr` (moves = fm.MoveActions, strength = defence value sent), a skip
    of `to_terr` (moves = [], strength 0), or the end of the phase (end_phase=True, moves = the
    positioning moves played as the phase ends). Same fields as action_generators.CombatAction,
    so the search, its metrics and the game records treat both alike."""
    __slots__ = ("to_terr", "moves", "strength", "end_phase")

    def __init__(self, to_terr=None, moves=None, end_phase=False, strength=0):
        self.to_terr = to_terr
        self.moves = moves if moves is not None else []
        self.end_phase = end_phase
        self.strength = strength

    def __repr__(self):
        if self.end_phase:
            return f"noncombat: END PHASE ({len(self.moves)} positioning moves)"
        if not self.moves:
            return f"noncombat: skip {self.to_terr}"
        n = sum(len(mv.units) for mv in self.moves)
        return f"noncombat to {self.to_terr}: {n} units, strength={self.strength:.1f}"


class _Plan:
    """noncombat_move's bookkeeping for the current state: cover each territory has and wants,
    and the units that may still move (not in `moved`)."""

    def __init__(self, game, definition, player, moved):
        m = game.map
        self.game = game
        self.definition = definition
        self.player = player
        self.m = m
        territories = game.territories

        def is_combat_unit(u):
            ut = m.unit_types.get(u.unit_type)
            return not (ut and ut.is_infrastructure)

        self.my_names = [name for name, t in territories.items() if t.owner == player]
        self.mine = set(self.my_names)
        self.landing = ag._landing_spots(game, definition, player)
        self.frontier = {name for name in self.my_names
                         if any(nb not in self.mine for nb in m.neighbors(name, "land"))}
        self.home_factory = ag._home_factory(game, player)
        self.strength = {name: sum(ag.defense_value(u) for u in territories[name].units
                                   if u.owner == player and is_combat_unit(u))
                         for name in self.my_names}
        total = sum(self.strength.values())
        threat = ag._enemy_threat(game, definition, player, self.my_names)
        self.want = {}
        for name in self.my_names:
            value = ag.THREAT_COVER * threat[name]
            if name == self.home_factory:
                value = max(ag.HOME_COVER * threat[name], ag.HOME_KEEP * total)
            if name in self.frontier:
                value = max(value, ag.FRONTIER_MIN)
            self.want[name] = value

        self.movable = {}
        for name, t in territories.items():
            units = [u for u in t.units if u.owner == player and is_combat_unit(u)
                     and u.movement_left > 0 and u.domain in ("land", "air")
                     and u.uid not in moved]
            if units:
                self.movable[name] = units
        self._air_cache, self._dest_cache = {}, {}

    def air_routes(self, u, src):
        key = (src, u.movement_left)
        if key not in self._air_cache:
            self._air_cache[key] = ag._air_routes(self.game, src, u.movement_left)
        return self._air_cache[key]

    def destinations(self, u, src):
        """{territory of mine: route} the unit can end on this phase."""
        key = (src, u.domain, u.movement_left)
        if key not in self._dest_cache:
            if u.domain == "air":
                routes = self.air_routes(u, src)
                self._dest_cache[key] = {t: r for t, r in routes.items()
                                         if t in self.mine and t in self.landing}
            else:
                routes = ag._reach_routes(self.game, self.player, src, u.movement_left, "land")
                self._dest_cache[key] = {t: r for t, r in routes.items() if t in self.mine}
        return self._dest_cache[key]

    def shortfall(self, name):
        return max(0.0, self.want[name] - self.strength[name])

    def priority(self, name):
        production = self.m.G.nodes[name].get("production", 0) or 0
        return (1.5 if name == self.home_factory else 1.0) * (1.0 + min(production, 10) / 10.0)

    def defense_gain(self, dst, value):
        return self.priority(dst) * min(value, self.shortfall(dst))

    def donor_loss(self, src, value):
        before = max(0.0, self.want.get(src, 0.0) - self.strength.get(src, 0.0))
        after = max(0.0, self.want.get(src, 0.0) - (self.strength.get(src, 0.0) - value))
        return self.priority(src) * (after - before)

    def can_spare(self, src, value):
        return src not in self.want or self.strength[src] - value >= self.want[src] - _EPS

    def send(self, u, src, route):
        """Plan a move: update the cover of both ends."""
        self.movable[src].remove(u)
        value = ag.defense_value(u)
        if src in self.strength:
            self.strength[src] -= value
        dst = route[len(route) - 1]
        if dst in self.strength:
            self.strength[dst] += value


def land_air_moves(game, definition, player, moved=frozenset()):
    """noncombat_move step 1: air where the engine would destroy it at the end of the phase flies
    to the reachable landing spot most short of its value. Returns [fm.MoveAction]."""
    plan = _Plan(game, definition, player, moved)
    planned = []
    for src in list(plan.movable):
        if src in plan.landing:
            continue
        for u in [u for u in plan.movable[src] if u.domain == "air"]:
            spots = [(t, r) for t, r in plan.air_routes(u, src).items() if t in plan.landing]
            if not spots:
                continue  # lost at the end of the phase, as in the engine
            t, r = max(spots, key=lambda s: (plan.want.get(s[0], 0.0)
                                             - plan.strength.get(s[0], 0.0), -len(s[1])))
            plan.send(u, src, r)
            planned.append((u, r))
    return ag._group_moves(definition, planned)


def noncombat_actions(game, definition, excluded, moved, player=None):
    """Reinforcement options for the most important territory short of cover, not excluded.

    Returns (actions, excluded). Actions are reinforcement sets strongest first, then a skip of the
    territory; it (and every short territory no spare unit can reach, passed over) is added to
    `excluded`. When no territory is left, actions is [end of phase with the positioning moves].
    """
    player = player or game.whoAmI
    excluded = set(excluded)
    plan = _Plan(game, definition, player, moved)

    targets = sorted((t for t in plan.my_names
                      if t not in excluded and plan.shortfall(t) > _EPS),
                     key=lambda t: -plan.priority(t) * plan.shortfall(t))
    for target in targets:
        short = plan.shortfall(target)
        # Donors as noncombat_move ranks them: defence gained here less cover lost at the source.
        candidates = []
        for src, units in plan.movable.items():
            if src == target:
                continue
            for u in units:
                route = plan.destinations(u, src).get(target)
                if route is None:
                    continue
                value = ag.defense_value(u)
                score = plan.defense_gain(target, value) - plan.donor_loss(src, value)
                candidates.append(((score, -len(route)), u, src, route, value))
        candidates.sort(key=lambda c: c[0], reverse=True)

        # Incremental buildup, keeping each donor territory at its own wanted cover.
        sets, chosen, given, sent = [], [], {}, 0.0
        levels = list(COVER_LEVELS)
        for _, u, src, route, value in candidates:
            if not plan.can_spare(src, given.get(src, 0.0) + value):
                continue
            chosen.append((u, route))
            given[src] = given.get(src, 0.0) + value
            sent += value
            if sent >= levels[0] * short - _EPS:
                sets.append((list(chosen), sent))
                while levels and sent >= levels[0] * short - _EPS:
                    levels.pop(0)
                if not levels:
                    break
        if chosen and (not sets or sets[-1][1] != sent):
            sets.append((list(chosen), sent))  # everything that can be spared
        excluded.add(target)
        if not sets:
            continue  # no spare unit reaches it: pass over

        sets.reverse()
        actions = [NoncombatAction(to_terr=target, moves=ag._group_moves(definition, units),
                                   strength=s) for units, s in sets]
        actions.append(NoncombatAction(to_terr=target, moves=[], strength=0))
        return actions, excluded

    return [NoncombatAction(end_phase=True,
                            moves=positioning_moves(game, definition, player, moved))], excluded


def positioning_moves(game, definition, player=None, moved=frozenset(), objectives=None):
    """noncombat_move step 3: spare units move closer to the frontier, to the least covered of the
    reachable territories that close the most distance. Returns [fm.MoveAction]."""
    player = player or game.whoAmI
    plan = _Plan(game, definition, player, moved)
    m = plan.m
    objectives = objectives if objectives is not None else m.objectives()
    dist_to_frontier = m.distance_from_set(plan.frontier, "land") if plan.frontier else {}
    dist_to_objective = m.distance_from_set(objectives, "land") if objectives else {}
    inf = float("inf")
    planned = []
    for src in sorted((n for n in plan.my_names if n in plan.movable),
                      key=lambda n: -dist_to_frontier.get(n, inf)):
        here = dist_to_frontier.get(src, inf)
        for u in list(plan.movable[src]):
            value = ag.defense_value(u)
            if not plan.can_spare(src, value):
                continue
            best = None
            for dst, route in plan.destinations(u, src).items():
                advance = here - dist_to_frontier.get(dst, inf)
                if advance <= 0 or advance == inf:
                    continue
                positioning_gain = advance + 0.2 * max(
                    0.0, 5.0 - dist_to_objective.get(dst, inf))
                score = (plan.defense_gain(dst, value) + positioning_gain
                         - plan.donor_loss(src, value))
                key = (score, -len(route))
                if best is None or key > best[0]:
                    best = (key, route)
            if best is not None and best[0][0] > _EPS:
                plan.send(u, src, best[1])
                planned.append((u, best[1]))
    return ag._group_moves(definition, planned)
