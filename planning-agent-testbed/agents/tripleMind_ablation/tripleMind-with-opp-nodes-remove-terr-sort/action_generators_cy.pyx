# cython: language_level=3, boundscheck=False, wraparound=False, initializedcheck=False
"""Cython build of action_generators.py: same functions, same arguments, same results.

action_generators.py stays the readable reference (and the pure-Python fallback); this module
must return exactly what it returns. The speed comes from:
  * adjacency lists precomputed once per map, in the exact order MapGraph.neighbors returns them
    (networkx subgraph views were most of the generators' time),
  * static map facts (victory cities, capitals, unit-type flags, production) cached per map,
  * typed loops, and attack strength kept as a running total for the incremental attack sets
    (calculate_attack_strength == sum of attacks + min(#supportable, #artillery)).
Build with scripts/build_cython.sh; generators.py picks this module when it is built.
"""

import heapq

import triplea_fm as fm


# Extra weight given to unit roles as the game goes on: (round limit, roles to repeat).
PURCHASE_BIAS = (
    (5, ("infantry",)),
    (10, ("infantry", "blitz")),
    (15, ("blitz", "artillery")),
)

cdef long long INF_INT = 2147483647


# ---- per-map caches (the map graph never changes during a game) ----------------------------

cdef class _MapFacts:
    cdef public object m
    cdef public dict adj            # domain -> {territory: tuple(neighbors in MapGraph order)}
    cdef public dict members        # domain -> territories in that domain's graph view
    cdef public set victory_cities
    cdef public dict capital_of     # player -> capital territory
    cdef public set capitals
    cdef public dict production     # territory -> production
    cdef public set water
    cdef public set infra_types     # unit types with is_infrastructure
    cdef public set static_types    # infrastructure or AA (never moved by the generators)
    cdef public set producer_types  # can_produce
    cdef public set artillery_types
    cdef public set supportable_types
    cdef public dict unit_cost      # unit type -> cost (None if not buildable)

    def __init__(self, m):
        self.m = m
        self.adj = {}
        self.members = {}
        self.victory_cities = set(m.victory_cities())
        self.capital_of = m.capitals()
        self.capitals = set(self.capital_of.values())
        self.production = {t: d["production"] for t, d in m.G.nodes(data=True)}
        self.water = {t for t in m.G.nodes if m.is_water(t)}
        uts = m.unit_types
        self.infra_types = {n for n, ut in uts.items() if ut.is_infrastructure}
        self.static_types = {n for n, ut in uts.items() if ut.is_infrastructure or ut.is_aa}
        self.producer_types = {n for n, ut in uts.items() if ut.can_produce}
        self.artillery_types = {n for n, ut in uts.items() if ut.is_artillery}
        self.supportable_types = {n for n, ut in uts.items() if ut.artillery_supportable}
        self.unit_cost = {}

    cdef dict neighbors(self, object domain):
        cdef dict per = self.adj.get(domain)
        if per is None:
            per = {t: tuple(self.m.neighbors(t, domain)) for t in self.m.G.nodes}
            self.adj[domain] = per
        return per

    cdef set member_set(self, object domain):
        cdef set nodes = self.members.get(domain)
        if nodes is None:
            nodes = set(self.m.view(domain).nodes)
            self.members[domain] = nodes
        return nodes

    cdef object cost(self, str unit_type):
        if unit_type not in self.unit_cost:
            self.unit_cost[unit_type] = self.m.unit_cost(unit_type)
        return self.unit_cost[unit_type]


cdef dict _FACTS = {}


cdef _MapFacts _facts(object m):
    cdef _MapFacts f = _FACTS.get(id(m))
    if f is None or f.m is not m:
        f = _MapFacts(m)
        _FACTS[id(m)] = f
    return f


cdef inline str _domain(object u):
    cdef dict p = u.properties
    if p.get("air"):
        return "air"
    if p.get("sea"):
        return "sea"
    return "land"


cdef dict _bfs_distances(dict adj, set members, object sources):
    """Multi-source BFS distances over a cached adjacency (same values as
    MapGraph.distance_from_set: unweighted shortest path lengths to the nearest source; sources
    outside the domain's graph are ignored, as there)."""
    cdef dict dist = {}
    cdef list frontier = []
    cdef Py_ssize_t i = 0
    cdef long long d
    for s in sources:
        if s in members and s not in dist:
            dist[s] = 0
            frontier.append(s)
    while i < len(frontier):
        cur = frontier[i]
        i += 1
        d = dist[cur] + 1
        for nb in adj[cur]:
            if nb not in dist:
                dist[nb] = d
                frontier.append(nb)
    return dist


def _roles(unit_type):
    roles = set()
    if unit_type.is_infantry:
        roles.add("infantry")
    if unit_type.can_blitz:
        roles.add("blitz")
    if unit_type.is_artillery:
        roles.add("artillery")
    return roles


# ---- purchase -------------------------------------------------------------------------------

def purchase_actions(game, definition, player=None):
    """Spend all PUs round-robin over the cheapest-first land combat units the player can buy.

    Returns [fm.PurchaseAction], or [] when the player has no factory or cannot afford anything.
    """
    m = game.map
    cdef _MapFacts f = _facts(m)
    player = player or game.whoAmI
    cdef long long remaining = game.players[player].PU

    cdef bint has_factory = False
    for terr in game.owned_by(player):
        for u in terr.units:
            if u.owner == player and u.unit_type in f.producer_types:
                has_factory = True
                break
        if has_factory:
            break
    if not has_factory:
        return []

    # Land combat units from the player's frontier: (rule name, cost, unit type).
    cdef list rules = []
    for rule in m.purchasable(player):
        for unit_name in rule.results:
            ut = m.unit_types.get(unit_name)
            if (ut and ut.domain == "land" and not ut.is_infrastructure and not ut.is_aa
                    and set(rule.costs) <= {"PUs"} and rule.cost > 0):
                rules.append((rule.name, rule.cost, ut))
                break

    # Bias rotation based on round
    for limit, roles in PURCHASE_BIAS:
        if game.round < limit:
            rules = rules + [r for r in rules if _roles(r[2]) & set(roles)]
            break
    rules.sort(key=lambda r: r[1])

    cdef dict purchase_dict = {}
    cdef Py_ssize_t i = 0
    cdef list affordable
    while True:
        affordable = [(name, cost) for name, cost, _ in rules if cost <= remaining]
        if not affordable:
            break
        name, cost = affordable[i % len(affordable)]
        purchase_dict[name] = purchase_dict.get(name, 0) + 1
        remaining -= cost
        i += 1

    if not purchase_dict:
        return []

    rule_id = definition.production_rule_index
    return [fm.PurchaseAction(tuple(sorted((rule_id[name], count)
                                           for name, count in purchase_dict.items())))]


def _home_factory(game, player):
    """The player's capital if it still holds it, else any owned territory with its factory."""
    cdef _MapFacts f = _facts(game.map)
    capital = f.capital_of.get(player)
    if capital and game.territories[capital].owner == player:
        return capital
    for terr in game.territories.values():
        if terr.owner != player:
            continue
        for u in terr.units:
            if u.owner == player and u.unit_type in f.producer_types:
                return terr.name
    return None


# ---- non-combat moves (the live turn and the rollouts play the same policy) ------------------

# A stack's value weighs its unit count and its combat points (defence for my units, attack for
# the enemy's), so a few strong units and many weak ones compare sensibly.
COUNT_WEIGHT = 1.0
POINT_WEIGHT = 0.5
THREAT_COVER = 1.0   # defence value wanted per point of enemy threat on a territory
HOME_COVER = 1.5     # the same on the home factory, whose loss costs the production
HOME_KEEP = 0.08     # share of my army's value the home factory keeps regardless
FRONTIER_MIN = 3.0   # value a frontier territory keeps regardless
_EPS = 1e-9


def defense_value(u):
    return COUNT_WEIGHT + POINT_WEIGHT * u.defense


def attack_value(u):
    return COUNT_WEIGHT + POINT_WEIGHT * u.attack


def _route_to(parent, t):
    path = [t]
    step = parent[t]
    while step is not None:
        path.append(step)
        step = parent[step]
    return tuple(reversed(path))


def _air_routes(game, start, moves):
    """{territory: route} within `moves` flights of start (see action_generators._air_routes)."""
    cdef _MapFacts f = _facts(game.map)
    cdef dict air = f.neighbors("air")
    cdef set water = f.water
    terrs = game.territories
    parent = {start: None}
    queue = [(start, 0)]
    i = 0
    while i < len(queue):
        current, dist = queue[i]
        i += 1
        if dist >= moves:
            continue
        for n in air[current]:
            if n in parent or (terrs[n].owner == "Neutral" and n not in water):
                continue
            parent[n] = current
            queue.append((n, dist + 1))
    return {t: _route_to(parent, t) for t in parent if t != start}


def _landing_spots(game, definition, player):
    """Territories where the player's air may end the phase (see action_generators._landing_spots)."""
    cdef set water = _facts(game.map).water
    me = definition.player_index[player]
    conquered = getattr(game, "conquered", ())
    return {name for name, t in game.territories.items()
            if name not in water and name not in conquered and t.owner != "Neutral"
            and definition.is_allied(me, definition.player_index[t.owner])}


def _enemy_threat(game, definition, player, targets):
    """{target: attack value of the enemy units that can reach it next turn} (see
    action_generators._enemy_threat)."""
    m = game.map
    cdef set static = _facts(m).static_types
    me = definition.player_index[player]
    stacks = {}  # (territory, domain, reach) -> attack value
    for name, terr in game.territories.items():
        for u in terr.units:
            if (u.owner == player or u.owner == "Neutral" or _domain(u) == "sea" or u.unit_type in static
                    or not definition.is_at_war(me, definition.player_index[u.owner])):
                continue
            dom = _domain(u)
            reach = u.range - 1 if dom == "air" else u.range
            if reach > 0:
                key = (name, dom, reach)
                stacks[key] = stacks.get(key, 0.0) + attack_value(u)
    threat = {t: 0.0 for t in targets}
    for (src, domain, reach), value in stacks.items():
        for t in targets:
            d = m.distance(src, t, domain)
            if d is not None and d <= reach:
                threat[t] += value
    return threat


def noncombat_move(game, definition, player=None, objectives=None):
    """Non-combat moves for the live turn and the rollouts (see
    action_generators.noncombat_move for the full description)."""
    m = game.map
    cdef _MapFacts f = _facts(m)
    cdef dict land = f.neighbors("land")
    infra = f.infra_types
    player = player or game.whoAmI
    territories = game.territories
    objectives = objectives if objectives is not None else m.objectives()

    my_names = [name for name, t in territories.items() if t.owner == player]
    mine = set(my_names)
    landing = _landing_spots(game, definition, player)
    frontier = {name for name in my_names
                if any(nb not in mine for nb in land[name])}
    home_factory = _home_factory(game, player)

    # Values change as moves are planned, so keep them outside the observation.
    strength = {name: sum(defense_value(u) for u in territories[name].units
                          if u.owner == player and u.unit_type not in infra) for name in my_names}
    total = sum(strength.values())
    threat = _enemy_threat(game, definition, player, my_names)

    def wanted(name):
        value = THREAT_COVER * threat[name]
        if name == home_factory:
            value = max(HOME_COVER * threat[name], HOME_KEEP * total)
        if name in frontier:
            value = max(value, FRONTIER_MIN)
        return value

    want = {name: wanted(name) for name in my_names}

    # Movable land/air units of mine, wherever they are (each unit is moved at most once).
    movable = {}
    for name, t in territories.items():
        units = [u for u in t.units if u.owner == player and u.unit_type not in infra
                 and u.movement_left > 0 and _domain(u) in ("land", "air")]
        if units:
            movable[name] = units

    air_cache, dest_cache = {}, {}

    def air_routes(u, src):
        key = (src, u.movement_left)
        if key not in air_cache:
            air_cache[key] = _air_routes(game, src, u.movement_left)
        return air_cache[key]

    def destinations(u, src):
        """{territory of mine: route} the unit can end on this phase."""
        dom = _domain(u)
        key = (src, dom, u.movement_left)
        if key not in dest_cache:
            if dom == "air":
                routes = air_routes(u, src)
                dest_cache[key] = {t: r for t, r in routes.items() if t in mine and t in landing}
            else:
                routes = _reach_routes(game, player, src, u.movement_left, "land")
                dest_cache[key] = {t: r for t, r in routes.items() if t in mine}
        return dest_cache[key]

    planned = []  # (unit, route)

    def send(u, src, route):
        movable[src].remove(u)
        value = defense_value(u)
        if src in strength:
            strength[src] -= value
        dst = route[len(route) - 1]
        if dst in strength:
            strength[dst] += value
        planned.append((u, route))

    def can_spare(src, u):
        return src not in want or strength[src] - defense_value(u) >= want[src] - _EPS

    # ---------- 1: land the air the engine would destroy ----------
    for src in list(movable):
        if src in landing:
            continue
        for u in [u for u in movable[src] if _domain(u) == "air"]:
            spots = [(t, r) for t, r in air_routes(u, src).items() if t in landing]
            if not spots:
                continue  # lost at the end of the phase, as in the engine
            t, r = max(spots, key=lambda s: (want.get(s[0], 0.0) - strength.get(s[0], 0.0),
                                             -len(s[1])))
            send(u, src, r)

    # ---------- 2: greedy marginal-value defensive reinforcement ----------
    # Only actual arrivals count as defensive reinforcement.
    def shortfall(name):
        return max(0.0, want[name] - strength[name])

    def priority(name):
        production = f.production.get(name, 0) or 0
        return (1.5 if name == home_factory else 1.0) * (1.0 + min(production, 10) / 10.0)

    def defense_gain(dst, value):
        return priority(dst) * min(value, shortfall(dst))

    def donor_loss(src, value):
        before = max(0.0, want.get(src, 0.0) - strength.get(src, 0.0))
        after = max(0.0, want.get(src, 0.0) - (strength.get(src, 0.0) - value))
        return priority(src) * (after - before)

    heap = [(-priority(t) * shortfall(t), t) for t in my_names if shortfall(t) > _EPS]
    heapq.heapify(heap)
    while heap:
        _, target = heapq.heappop(heap)
        if shortfall(target) <= _EPS:
            continue
        best = None
        for src, units in movable.items():
            if src == target:
                continue
            for u in units:
                if not can_spare(src, u):
                    continue
                route = destinations(u, src).get(target)
                if route is None:
                    continue
                value = defense_value(u)
                score = defense_gain(target, value) - donor_loss(src, value)
                key = (score, -max(0.0, value - shortfall(target)), -len(route))
                if best is None or key > best[0]:
                    best = (key, u, src, route)
        if best is None or best[0][0] <= _EPS:
            continue
        _, u, src, route = best
        send(u, src, route)
        if shortfall(target) > _EPS:
            heapq.heappush(heap, (-priority(target) * shortfall(target), target))

    # ---------- 3: greedy positioning of spare units ----------
    dist_to_frontier = _bfs_distances(land, f.member_set("land"), frontier) if frontier else {}
    dist_to_objective = _bfs_distances(land, f.member_set("land"), objectives) if objectives else {}
    inf = float("inf")
    for src in sorted((n for n in my_names if n in movable),
                      key=lambda n: -dist_to_frontier.get(n, inf)):
        here = dist_to_frontier.get(src, inf)
        for u in list(movable[src]):
            if not can_spare(src, u):
                continue
            value = defense_value(u)
            best = None
            for dst, route in destinations(u, src).items():
                advance = here - dist_to_frontier.get(dst, inf)
                if advance <= 0 or advance == inf:
                    continue
                positioning_gain = advance + 0.2 * max(
                    0.0, 5.0 - dist_to_objective.get(dst, inf))
                score = defense_gain(dst, value) + positioning_gain - donor_loss(src, value)
                key = (score, -len(route))
                if best is None or key > best[0]:
                    best = (key, route)
            if best is not None and best[0][0] > _EPS:
                send(u, src, best[1])

    return _group_moves(definition, planned)


# ---- shared helpers (ports of move_generator.pyx) --------------------------------------------

def _is_static(m, u):
    """Units the move generators never move (factories, AA guns, other infrastructure)."""
    return u.unit_type in _facts(m).static_types


cdef frozenset _NO_BLOCK = frozenset()


cdef dict _reach(dict adj, dict territories, object player, object start, long long move_range,
                 object blocked):
    """can_reach_cache_helper as a BFS that also keeps one route per reachable territory: a unit
    may pass only through territories owned by `player` and not in `blocked`; the destination
    itself may be anything."""
    if move_range <= 0:
        return {}
    cdef dict parent = {start: None}
    cdef list queue = [start]
    cdef list depth = [0]
    cdef list reached = []
    cdef Py_ssize_t i = 0
    cdef long long dist
    while i < len(queue):
        current = queue[i]
        dist = depth[i]
        i += 1
        if dist >= move_range:
            continue
        for neighbor in adj[current]:
            if neighbor in parent:
                continue
            parent[neighbor] = current
            reached.append(neighbor)
            if territories[neighbor].owner != player or neighbor in blocked:
                continue  # Can't pass through enemy territory (or a blocked one)
            if dist + 1 < move_range:
                queue.append(neighbor)
                depth.append(dist + 1)
    cdef dict routes = {}
    cdef list path
    for t in reached:
        path = [t]
        step = parent[t]
        while step is not None:
            path.append(step)
            step = parent[step]
        path.reverse()
        routes[t] = tuple(path)
    return routes


def _reach_routes(game, player, start, move_range, domain, blocked=frozenset()):
    """Python-callable wrapper of the BFS above: {territory: route of names}."""
    return _reach(_facts(game.map).neighbors(domain), game.territories, player, start, move_range,
                  blocked)


def combat_blocked(game):
    """Territories conquered this turn without blitzing: in combat movement, land units cannot
    move through them (Matches.territoryIsBlitzable, rules.is_blitzable)."""
    return game.conquered - game.blitzed


def _group_moves(definition, moves):
    """[(unit, route)] -> [fm.MoveAction], one per route (form_attacks)."""
    t_index = definition.territory_index
    cdef dict grouped = {}
    for u, route in moves:
        grouped.setdefault(route, []).append(u.uid)
    return [fm.MoveAction(tuple(uids), tuple(t_index[t] for t in route))
            for route, uids in grouped.items()]


# ---- combat moves ----------------------------------------------------------------------------

cdef class CombatAction:
    """Move(delegate="combat") of the original: an attack on `to_terr` (moves = fm.MoveActions),
    a skip of `to_terr` (moves = [], strength 0), or the end of the phase (end_phase=True)."""
    cdef public object to_terr
    cdef public object moves
    cdef public object end_phase
    cdef public object strength

    def __init__(self, to_terr=None, moves=None, end_phase=False, strength=0):
        self.to_terr = to_terr
        self.moves = moves if moves is not None else []
        self.end_phase = end_phase
        self.strength = strength

    def __repr__(self):
        if self.end_phase:
            return "combat: END PHASE"
        if not self.moves:
            return f"combat: skip {self.to_terr}"
        n = sum(len(mv.units) for mv in self.moves)
        return f"combat on {self.to_terr}: {n} units, strength={self.strength}"


def calculate_attack_strength(m, units):
    """Supportable units (infantry) paired 1:1 with artillery get +1 attack."""
    cdef _MapFacts f = _facts(m)
    cdef long long total = 0, infantry = 0, artillery = 0
    for u in units:
        total += u.attack
        if u.unit_type in f.supportable_types:
            infantry += 1
        if u.unit_type in f.artillery_types:
            artillery += 1
    return total + (infantry if infantry < artillery else artillery)


def defender_strength(m, territory, player):
    """Total defense strength of enemy units on a territory."""
    cdef set infra = _facts(m).infra_types
    return sum(u.defense for u in territory.units if u.owner != player
               and u.unit_type not in infra)


def enemy_territories_reachable(game, player):
    """MCTSGameState.set_terr_lists: enemy land territories bordering mine, most important first."""
    cdef _MapFacts f = _facts(game.map)
    cdef dict raw = f.neighbors(None)
    cdef dict terrs = game.territories
    cdef set vc = f.victory_cities
    cdef set mine = {name for name, t in terrs.items() if t.owner == player}
    cdef list reachable = []
    cdef dict priority = {}
    cdef long long score, next_to_factory, surrounded_by_me, enemy_neighbors

    for name, t in terrs.items():
        if t.owner == player or name in f.water:
            continue
        nbs = raw[name]
        if not any(n in mine for n in nbs):
            continue
        reachable.append(name)
    return reachable


cdef list _reachable_units(_MapFacts f, dict terrs, object player, object target, dict routes,
                           object blocked):
    """[(unit, from, route)]: my unmoved mobile units, in my territories, that can reach target.
    Land routes avoid `blocked` (territories conquered this turn without blitzing)."""
    cdef list reachable = []
    cdef str dom
    for terr in terrs.values():
        if terr.owner != player:
            continue
        for u in terr.units:
            if u.owner != player or u.unit_type in f.static_types or u.moved:
                continue
            dom = _domain(u)
            if dom == "sea":
                continue
            key = (terr.name, u.range, dom)
            r = routes.get(key)
            if r is None:
                r = _reach(f.neighbors(dom), terrs, player, terr.name, u.range,
                           blocked if dom == "land" else _NO_BLOCK)
                routes[key] = r
            if target in r:
                reachable.append((u, terr.name, r[target]))
    return reachable


def get_reachable_units(game, target, player):
    """[(unit, from, route)]: my unmoved mobile units, in my territories, that can reach target."""
    return _reachable_units(_facts(game.map), game.territories, player, target, {},
                            game.conquered - game.blitzed)


def combat_actions(game, definition, excluded, player=None):
    """heuristic_combat_legal_moves: attack options for the most important target not excluded.

    Returns (actions, excluded). Actions are the attack sets strongest first, then a skip of the
    target; the target (and every unreachable one passed over) is added to `excluded`. Actions are
    [] when no target is left. `game` must come from GameState.from_sim (units carry fm ids).
    """
    m = game.map
    cdef _MapFacts f = _facts(m)
    cdef dict raw = f.neighbors(None)
    cdef dict terrs = game.territories
    player = player or game.whoAmI
    excluded = set(excluded)
    cdef list actions = []
    cdef dict routes = {}   # reach routes per (from, range, domain); the state does not change here
    blocked = game.conquered - game.blitzed
    cdef list reachable, keys, order, all_units, sets
    cdef long long def_strength, total, infantry, artillery
    cdef long long current = 0
    cdef double threshold
    cdef double max_threshold = 3.5
    cdef Py_ssize_t n, j

    for name in enemy_territories_reachable(game, player):
        if name in excluded:
            continue
        # ── gather attackers ──────────────────────────────────────────────
        reachable = _reachable_units(f, terrs, player, name, routes, blocked)
        if not reachable:
            excluded.add(name)
            continue

        # ── sort by donor_score (desc): adjacent first, then attack power ──
        keys = []
        for u, frm, _ in reachable:
            keys.append((100.0 if name in raw[frm] else 0.0) + 10.0 * u.attack)
        order = sorted(range(len(reachable)), key=keys.__getitem__, reverse=True)
        all_units = [reachable[j] for j in order]

        # ── defender strength ─────────────────────────────────────────────
        def_strength = 0
        for u in terrs[name].units:
            if u.owner != player and u.unit_type not in f.infra_types:
                def_strength += u.defense

        # ── incremental buildup (strength kept as a running total) ────────
        threshold = 1.5
        sets = []
        total = 0
        infantry = 0
        artillery = 0
        n = len(all_units)
        for j in range(n):
            u = all_units[j][0]
            total += u.attack
            if u.unit_type in f.supportable_types:
                infantry += 1
            if u.unit_type in f.artillery_types:
                artillery += 1
            current = total + (infantry if infantry < artillery else artillery)
            if current > threshold * def_strength and current < (max_threshold * def_strength) + 4:
                sets.append((j + 1, current))
                threshold += 0.1

        # always include full force if it beats defender
        if n > 0 and current > def_strength:
            if not sets or sets[len(sets) - 1][1] != current:
                sets.append((n, current))

        # -- fallback: undefended, cheapest single unit capture
        if not sets and def_strength == 0:
            best = min(all_units, key=lambda x: _cost_or_default(f, x[0]))
            actions.append(CombatAction(
                to_terr=name,
                moves=_group_moves(definition, [(best[0], best[2])]),
                strength=calculate_attack_strength(m, [best[0]]),
            ))

        # ── append moves (strongest first) ───────────────────────────────
        sets.reverse()
        for count, s in sets:
            actions.append(CombatAction(
                to_terr=name,
                moves=_group_moves(definition, [(x[0], x[2]) for x in all_units[:count]]),
                strength=s,
            ))

        # skip move always included
        actions.append(CombatAction(to_terr=name, moves=[], strength=0))
        # excluded.add(name)
        # break
    return actions, excluded


cdef object _cost_or_default(_MapFacts f, object u):
    c = f.cost(u.unit_type)
    return c if c is not None else 9999
