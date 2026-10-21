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


# ---- live non-combat (called once per turn) ------------------------------------------------

def noncombat_move(game, definition, player=None, objectives=None):
    """Push units from the rear toward the frontier and fill territories up to a quota.
    See action_generators.noncombat_move for the full description."""
    m = game.map
    cdef _MapFacts f = _facts(m)
    cdef dict land = f.neighbors("land")
    player = player or game.whoAmI
    territories = game.territories
    objectives = objectives if objectives is not None else m.objectives()
    victory_cities = f.victory_cities
    capitals = f.capitals
    infra = f.infra_types

    # Unit counts change as moves are planned, so keep them outside the observation.
    cdef dict strength = {name: sum(1 for u in t.units if u.unit_type not in infra)
                          for name, t in territories.items()}
    # Movable land/air units of mine, per territory (each unit is moved at most once).
    my_territories = [t for t in territories.values() if t.owner == player]
    cdef dict movable = {
        t.name: [u for u in t.units if u.owner == player and u.unit_type not in infra
                 and u.movement_left > 0 and _domain(u) in ("land", "air")]
        for t in my_territories
    }

    my_names = {t.name for t in my_territories}
    not_mine = set(territories) - my_names

    my_frontier = {t.name for t in my_territories if any(n in not_mine for n in land[t.name])}

    home_factory = _home_factory(game, player)

    def has_enemy_neighbor(name):
        return any(n in not_mine and n not in capitals for n in land[name])

    staging_score = {t.name: sum(1 for nb in land[t.name] if nb in my_frontier)
                     for t in my_territories}
    factory_adj_score = {t.name: sum(1 for nb in land[t.name] if nb in victory_cities)
                         for t in my_territories}

    inf = float("inf")
    dist_to_frontier = _bfs_distances(land, f.member_set("land"), my_frontier)
    dist_to_objective = _bfs_distances(land, f.member_set("land"), objectives)

    def territory_priority(name):
        score = 0
        if name in my_frontier:
            score += 1000
            if strength[name] == 0:
                score += 500
        score += staging_score.get(name, 0) * 200
        score += factory_adj_score.get(name, 0) * 100
        score += max(0, 5 - dist_to_objective.get(name, inf)) * 20
        return score

    targets = sorted(my_names, key=territory_priority, reverse=True)

    # Noncombat: land units may only pass through my territories; air may overfly anything.
    avoid = {"land": not_mine, "air": set()}
    _reach_cache = {}

    def reach(unit, src):
        dom = _domain(unit)
        key = (src, dom, unit.movement_left)
        if key not in _reach_cache:
            _reach_cache[key] = m.reachable(src, unit.movement_left, dom, avoid=avoid[dom])
        return _reach_cache[key]

    def can_reach(unit, src, dst):
        return dst != src and dst in my_names and dst in reach(unit, src)

    def best_step_toward_frontier(unit, from_name):
        best, best_score = None, -10 ** 9
        from_d = dist_to_frontier.get(from_name, inf)
        for dest_name in targets:
            if not can_reach(unit, from_name, dest_name):
                continue
            improvement = from_d - dist_to_frontier.get(dest_name, inf)
            score = improvement * 100
            if dest_name in my_frontier:
                score += 50
            score += staging_score.get(dest_name, 0) * 10
            score += factory_adj_score.get(dest_name, 0) * 5
            if score > best_score:
                best_score, best = score, dest_name
        return best, best_score

    total_my_units = sum(strength[n] for n in my_names)

    def quota(name):
        if name == home_factory:
            base = 0.08
        elif name in my_frontier:
            enemy_threat = sum(strength[n] for n in land[name] if n in not_mine)
            return max(3, min(enemy_threat + 2, int(total_my_units * 0.15)))
        elif staging_score.get(name, 0) > 0 or factory_adj_score.get(name, 0) > 0:
            base = 0.03
        elif name in objectives:
            base = 0.02
        else:
            base = 0.01
        return max(1, int(total_my_units * base))

    def frontier_excess(name):
        max_enemy_threat = max((strength[n] for n in land[name] if n in not_mine), default=0)
        return max(0, strength[name] - (max_enemy_threat + 2))

    frontier_excess_map = {t: frontier_excess(t) for t in my_frontier}

    planned = {}  # (from, to) -> [Unit], insertion-ordered

    def send(unit, src, dst):
        movable[src].remove(unit)
        strength[src] -= 1
        strength[dst] += 1
        planned.setdefault((src, dst), []).append(unit)
        if src in frontier_excess_map:
            frontier_excess_map[src] -= 1

    def can_donate(name):
        if not movable.get(name):
            return False
        if name in my_frontier and name != home_factory:
            return frontier_excess_map.get(name, 0) > 0
        return True

    # ---------- PASS 0: evacuate home factory before placement ----------
    if home_factory in movable:
        keep_min = 2 if has_enemy_neighbor(home_factory) else 0
        excess = max(0, strength[home_factory] - keep_min)
        for unit in list(movable[home_factory]):
            if excess <= 0:
                break
            dest, score = best_step_toward_frontier(unit, home_factory)
            if dest is None or score <= 0:
                continue
            send(unit, home_factory, dest)
            excess -= 1

    # ---------- PASS 1: fill targets to quota (with step-to-frontier fallback) ----------
    donors = [t.name for t in my_territories]
    for target in targets:
        for donor in donors:
            if strength[target] >= quota(target):
                break
            if donor == target:
                continue
            while strength[target] < quota(target) and can_donate(donor):
                moved = False
                for unit in list(movable[donor]):
                    if can_reach(unit, donor, target):
                        dest = target
                    else:
                        dest, score = best_step_toward_frontier(unit, donor)
                        if dest is None or score <= 0:
                            continue
                    send(unit, donor, dest)
                    moved = True
                    break
                if not moved:
                    break

    # ---------- encode as forward-model actions ----------
    t_index = definition.territory_index
    actions = []
    for (src, dst), units in planned.items():
        for group in _split_by_domain(units):
            dom = _domain(group[0])
            route = m.path(src, dst, dom, avoid=avoid[dom])
            actions.append(fm.MoveAction(tuple(u.uid for u in group),
                                         tuple(t_index[t] for t in route)))
    return actions


def _split_by_domain(units):
    groups = {}
    for u in units:
        groups.setdefault(_domain(u), []).append(u)
    return list(groups.values())


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


def enemy_territories_by_priority(game, player):
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
        owner = t.owner
        score = 0
        if name in vc:
            score += 1000
        next_to_factory = 0
        surrounded_by_me = 0
        enemy_neighbors = 0
        for n in nbs:
            if n in vc:
                next_to_factory += 1
            if n in mine:
                surrounded_by_me += 1
            if terrs[n].owner == owner:
                enemy_neighbors += 1
        score += next_to_factory * 500
        score += f.production[name] * 100
        score += surrounded_by_me * 50
        score -= enemy_neighbors * 30
        priority[name] = score
    reachable.sort(key=priority.__getitem__, reverse=True)
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

    for name in enemy_territories_by_priority(game, player):
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
        excluded.add(name)
        break
    return actions, excluded


cdef object _cost_or_default(_MapFacts f, object u):
    c = f.cost(u.unit_type)
    return c if c is not None else 9999


# ---- noncombat moves used in rollouts (heuristic_non_combat_legal_moves of the .pyx) ---------

def noncombat_move_rollout(game, definition, player=None):
    """Rollout noncombat policy. Returns [fm.MoveAction], one per (from, route)."""
    m = game.map
    cdef _MapFacts f = _facts(m)
    cdef dict raw = f.neighbors(None)
    player = player or game.whoAmI
    cdef dict territories = game.territories
    cdef list my_territories = [t for t in territories.values() if t.owner == player]
    cdef set capitals = f.capitals
    cdef set victory_cities = f.victory_cities
    cdef set infra = f.infra_types
    cdef set static = f.static_types

    # ── quantity snapshot (the only thing mutated): {terr: {(unit_type, owner): qty}} ──
    # plus count[t] = snapshot units in t that are not infrastructure (count_snap of the original)
    cdef dict qty_snap = {}
    cdef dict count = {}
    cdef dict snap
    cdef long long c
    for name, terr in territories.items():
        snap = {}
        c = 0
        for u in terr.units:
            k = (u.unit_type, u.owner)
            snap[k] = snap.get(k, 0) + 1
            if u.unit_type not in infra:
                c += 1
        qty_snap[name] = snap
        count[name] = c
    # real units still free to move, per territory (each unit moves at most once)
    cdef dict free_units = {t.name: [u for u in t.units if u.owner == player] for t in my_territories}

    # ── frontier detection ──
    cdef set my_frontier = set()
    for terr in my_territories:
        for n in raw[terr.name]:
            if territories[n].owner != player:
                my_frontier.add(terr.name)
                break

    home_factory = _home_factory(game, player)

    # ── pre-compute scoring maps ──
    cdef dict staging_score = {}
    cdef dict factory_adj_score = {}
    cdef long long score
    for terr in my_territories:
        score = 0
        for nb in raw[terr.name]:
            if nb in my_frontier:
                score += 1
        staging_score[terr.name] = score
        score = 0
        for nb in raw[terr.name]:
            if nb == home_factory:
                score += 3
            elif nb in victory_cities:
                score += 1
        factory_adj_score[terr.name] = score

    # ── BFS distance to frontier ──
    cdef dict dist_to_frontier = _bfs_distances(raw, f.member_set(None), my_frontier)

    cdef long long total_my_units = 0
    for t in my_territories:
        total_my_units += count[t.name]

    cdef dict frontier_excess_map = {}
    cdef long long max_threat, excess
    for name in my_frontier:
        max_threat = 0
        for nb in raw[name]:
            if territories[nb].owner != player and count[nb] > max_threat:
                max_threat = count[nb]
        excess = count[name] - (max_threat + 2)
        frontier_excess_map[name] = excess if excess > 0 else 0

    cdef dict prio = {}
    cdef long long real
    for t in my_territories:
        score = 0
        if t.name in my_frontier:
            score += 1000
            real = 0
            for u in t.units:
                if u.unit_type not in infra:
                    real += 1
            if real == 0:
                score += 500
        score += staging_score.get(t.name, 0) * 200
        score += factory_adj_score.get(t.name, 0) * 100
        prio[t.name] = score
    cdef list targets = sorted(my_territories, key=lambda t: prio[t.name], reverse=True)

    # ── donor list ──
    cdef list donor_territories = []
    for t in my_territories:
        if count[t.name] <= 0:
            continue
        if t.name in my_frontier and t.name != home_factory:
            if frontier_excess_map.get(t.name, 0) <= 0:
                continue
        donor_territories.append(t)
    donor_territories.sort(
        key=lambda t: count[t.name] - _rollout_quota(t.name, home_factory, my_frontier, raw,
                                                     territories, player, count, staging_score,
                                                     factory_adj_score, total_my_units),
        reverse=True)

    cdef dict route_cache = {}
    cdef list moves = []  # (unit, route)
    cdef long long keep_min, excess_units, sent, min_keep, donor_idx
    cdef dict factory_snap

    # ── PASS 0: evacuate home factory ──
    if home_factory in free_units:
        keep_min = 2 if _has_enemy_neighbor(home_factory, raw, territories, player, capitals) else 0
        excess_units = count[home_factory] - keep_min
        if excess_units < 0:
            excess_units = 0
        sent = 0
        factory_snap = qty_snap[home_factory]
        for u in list(free_units[home_factory]):
            k = (u.unit_type, u.owner)
            if u.unit_type in static or u.range <= 0 or factory_snap.get(k, 0) <= 0:
                continue
            if sent >= excess_units:
                break
            r = _rollout_reach(f, route_cache, territories, player, u, home_factory)
            dest, step_score = _best_step(u, home_factory, r, targets, dist_to_frontier,
                                          my_frontier, staging_score, factory_adj_score)
            if dest is None or step_score <= 0:
                continue
            free_units[home_factory].remove(u)
            moves.append((u, r[dest]))
            factory_snap[k] -= 1
            count[home_factory] -= 1
            sent += 1

    # ── PASS 1: fill targets to quota ──
    for terr in targets:
        terr_name = terr.name
        if count[terr_name] >= _rollout_quota(terr_name, home_factory, my_frontier, raw,
                                              territories, player, count, staging_score,
                                              factory_adj_score, total_my_units):
            continue
        donor_idx = 0
        while count[terr_name] < _rollout_quota(terr_name, home_factory, my_frontier, raw,
                                                territories, player, count, staging_score,
                                                factory_adj_score, total_my_units):
            if donor_idx >= len(donor_territories):
                break
            donor = donor_territories[donor_idx]
            donor_name = donor.name

            # drain guard
            if donor_name in my_frontier:
                if count[donor_name] <= frontier_excess_map.get(donor_name, 0):
                    donor_idx += 1
                    continue
            else:
                min_keep = 2 if _has_enemy_neighbor(donor_name, raw, territories, player,
                                                    capitals) else 1
                if count[donor_name] <= min_keep:
                    donor_idx += 1
                    continue

            # pick first movable unit that still has snapshot qty > 0
            picked = None
            snap = qty_snap[donor_name]
            for u in free_units.get(donor_name, []):
                if u.unit_type in static or u.range <= 0:
                    continue
                if snap.get((u.unit_type, u.owner), 0) <= 0:
                    continue
                picked = u
                break
            if picked is None:
                donor_idx += 1
                continue

            r = _rollout_reach(f, route_cache, territories, player, picked, donor_name)
            dest = None
            if terr_name in r:
                dest = terr_name
            else:
                step, step_score = _best_step(picked, donor_name, r, targets, dist_to_frontier,
                                              my_frontier, staging_score, factory_adj_score)
                if step is not None and step_score > 0:
                    dest = step
            if dest is None:
                donor_idx += 1
                continue

            free_units[donor_name].remove(picked)
            moves.append((picked, r[dest]))
            k = (picked.unit_type, picked.owner)
            snap[k] -= 1
            count[donor_name] -= 1
            # also credit the destination so it counts toward quota
            k = (picked.unit_type, player)
            qty_snap[dest][k] = qty_snap[dest].get(k, 0) + 1
            count[dest] += 1
            if donor_name in my_frontier:
                excess = frontier_excess_map.get(donor_name, 0) - 1
                frontier_excess_map[donor_name] = excess if excess > 0 else 0

    # Grouped by (from, route) like grouped_moves in heuristic_non_combat_legal_moves.
    return _group_moves(definition, moves)


cdef dict _rollout_reach(_MapFacts f, dict route_cache, dict territories, object player,
                         object u, object from_name):
    cdef str dom = _domain(u)
    key = (from_name, u.range, dom)
    r = route_cache.get(key)
    if r is None:
        r = _reach(f.neighbors(dom), territories, player, from_name, u.range, _NO_BLOCK)
        route_cache[key] = r
    return r


cdef tuple _best_step(object u, object from_name, dict r, list targets, dict dist_to_frontier,
                      set my_frontier, dict staging_score, dict factory_adj_score):
    best = None
    cdef long long best_score = -2147483647
    cdef long long from_d = dist_to_frontier.get(from_name, INF_INT)
    cdef long long improvement, score
    for dest in targets:
        dest_name = dest.name
        if dest_name == from_name or dest_name not in r:
            continue
        improvement = from_d - <long long>dist_to_frontier.get(dest_name, INF_INT)
        score = improvement * 100
        if dest_name in my_frontier:
            score += 50
        score += <long long>staging_score.get(dest_name, 0) * 10
        score += <long long>factory_adj_score.get(dest_name, 0) * 5
        if score > best_score:
            best_score = score
            best = dest_name
    return best, best_score


cdef bint _has_enemy_neighbor(object name, dict raw, dict territories, object player,
                              set capitals):
    for n in raw[name]:
        if territories[n].owner != player and n not in capitals:
            return True
    return False


cdef long long _rollout_quota(object name, object home_factory, set my_frontier, dict raw,
                              dict territories, object player, dict count, dict staging_score,
                              dict factory_adj_score, long long total_my_units):
    cdef long long enemy_threat, cap, v
    if name == home_factory:
        v = <long long>(total_my_units * 0.08)
        return v if v > 1 else 1
    if name in my_frontier:
        enemy_threat = 0
        for nb in raw[name]:
            if territories[nb].owner != player:
                enemy_threat += count[nb]
        cap = <long long>(total_my_units * 0.15)
        v = enemy_threat + 2 if enemy_threat + 2 < cap else cap
        return v if v > 3 else 3
    if staging_score.get(name, 0) > 0 or factory_adj_score.get(name, 0) > 0:
        v = <long long>(total_my_units * 0.03)
        return v if v > 1 else 1
    v = <long long>(total_my_units * 0.01)
    return v if v > 1 else 1
