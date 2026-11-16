"""Candidate actions per phase, built from a map_graph.GameState.

Each generator returns a list of triplea_fm actions (PurchaseAction, MoveAction, PlaceAction), so
the agent can encode them for the engine with LiveBridge and the same actions can be fed to the
forward model during search. Unit roles come from the map's unit attachments, never from unit
names, so the generators work on any map.
"""

import heapq

import triplea_fm as fm


# Extra weight given to unit roles as the game goes on: (round limit, roles to repeat).
PURCHASE_BIAS = (
    (5, ("infantry",)),
    (10, ("infantry", "blitz")),
    (15, ("blitz", "artillery")),
)


def _roles(unit_type):
    roles = set()
    if unit_type.is_infantry:
        roles.add("infantry")
    if unit_type.can_blitz:
        roles.add("blitz")
    if unit_type.is_artillery:
        roles.add("artillery")
    return roles


def purchase_actions(game, definition, player=None):
    """Spend all PUs round-robin over the cheapest-first land combat units the player can buy.

    Returns [fm.PurchaseAction], or [] when the player has no factory or cannot afford anything.
    `definition` is the triplea_fm GameDefinition, used to turn rule names into rule ids.
    """
    m = game.map
    player = player or game.whoAmI
    resources = game.players[player].PU

    has_factory = any(
        u.owner == player and m.unit_types.get(u.unit_type) and m.unit_types[u.unit_type].can_produce
        for terr in game.owned_by(player)
        for u in terr.units
    )
    if not has_factory:
        return []

    # Land combat units from the player's frontier: (rule name, cost, unit type).
    rules = []
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

    purchase_dict = {}
    remaining = resources
    i = 0

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
    m = game.map
    capital = m.capitals().get(player)
    if capital and game.owner(capital) == player:
        return capital
    for terr in game.owned_by(player):
        if any(u.owner == player and m.unit_types.get(u.unit_type)
               and m.unit_types[u.unit_type].can_produce for u in terr.units):
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
    """{territory: route} within `moves` flights of start, as the engine lets air fly in non
    combat: over anything passable but neutral land (rules._validate_noncombat_route). Enemy
    territory and units do not block an all-air move."""
    m = game.map
    terrs = game.territories
    parent = {start: None}
    queue = [(start, 0)]
    i = 0
    while i < len(queue):
        current, dist = queue[i]
        i += 1
        if dist >= moves:
            continue
        for n in m.neighbors(current, "air"):
            if n in parent or (terrs[n].owner == "Neutral" and not m.is_water(n)):
                continue
            parent[n] = current
            queue.append((n, dist + 1))
    return {t: _route_to(parent, t) for t in parent if t != start}


def _landing_spots(game, definition, player):
    """Territories where the player's air may end the phase (rules.can_land_here, the engine's
    Matches.airCanLandOnThisAlliedNonConqueredLandTerritory): land, allied owner, not conquered
    this turn. Air anywhere else is destroyed at the end of non combat (AirThatCantLandUtil)."""
    m = game.map
    me = definition.player_index[player]
    conquered = getattr(game, "conquered", ())
    return {name for name, t in game.territories.items()
            if not m.is_water(name) and name not in conquered and t.owner != "Neutral"
            and definition.is_allied(me, definition.player_index[t.owner])}


def _enemy_threat(game, definition, player, targets):
    """{target: attack value of the enemy units that can reach it next turn}. Land units reach
    within their movement over land; air within one less, keeping a move to land elsewhere (air
    may not land where it attacked). Blocking is ignored, so this errs on the high side."""
    m = game.map
    me = definition.player_index[player]
    stacks = {}  # (territory, domain, reach) -> attack value
    for name, terr in game.territories.items():
        for u in terr.units:
            if (u.owner == player or u.owner == "Neutral" or u.domain == "sea" or _is_static(m, u)
                    or not definition.is_at_war(me, definition.player_index[u.owner])):
                continue
            reach = u.range - 1 if u.domain == "air" else u.range
            if reach > 0:
                key = (name, u.domain, reach)
                stacks[key] = stacks.get(key, 0.0) + attack_value(u)
    threat = {t: 0.0 for t in targets}
    for (src, domain, reach), value in stacks.items():
        for t in targets:
            d = m.distance(src, t, domain)
            if d is not None and d <= reach:
                threat[t] += value
    return threat


def noncombat_move(game, definition, player=None, objectives=None):
    """Non-combat moves, for the live turn and the rollouts alike: land the air the engine would
    destroy, cover each territory against the enemy threat on it, then push spare units forward.

    Each territory of mine wants a defence value of THREAT_COVER x the attack value of the enemy
    units that can reach it next turn (_enemy_threat); the home factory HOME_COVER x that and at
    least HOME_KEEP of my army, a frontier territory at least FRONTIER_MIN. Values weigh unit
    count and combat points (defense_value, attack_value).
      1. Air where the engine would destroy it at the end of the phase (water, land not allied,
         land conquered this turn) flies to the reachable landing spot most short of its value.
      2. Territories short of their value, home factory first then most short first, are filled
         from the units other territories can spare (each keeps its own value), best defender
         first. A unit that cannot reach the territory this turn steps toward it and counts
         against its shortfall, so one unreachable territory does not empty the rear.
      3. Spare units move closer to the frontier, to the least covered of the reachable
         territories that close the most distance (ties: closer to `objectives`, by default the
         map's highest-production territories), best defender first.
    Land units move only through my territories; air flies over anything but neutral land and
    ends only on a landing spot (_landing_spots). `game` must come from GameState.from_sim so
    units carry forward-model ids. Returns [fm.MoveAction], one per route, in the order decided.
    """
    m = game.map
    player = player or game.whoAmI
    territories = game.territories
    objectives = objectives if objectives is not None else m.objectives()

    def is_combat_unit(u):
        ut = m.unit_types.get(u.unit_type)
        return not (ut and ut.is_infrastructure)

    my_names = [name for name, t in territories.items() if t.owner == player]
    mine = set(my_names)
    landing = _landing_spots(game, definition, player)
    frontier = {name for name in my_names
                if any(nb not in mine for nb in m.neighbors(name, "land"))}
    home_factory = _home_factory(game, player)

    # Values change as moves are planned, so keep them outside the observation.
    strength = {name: sum(defense_value(u) for u in territories[name].units
                          if u.owner == player and is_combat_unit(u)) for name in my_names}
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
        units = [u for u in t.units if u.owner == player and is_combat_unit(u)
                 and u.movement_left > 0 and u.domain in ("land", "air")]
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
        key = (src, u.domain, u.movement_left)
        if key not in dest_cache:
            if u.domain == "air":
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
        for u in [u for u in movable[src] if u.domain == "air"]:
            spots = [(t, r) for t, r in air_routes(u, src).items() if t in landing]
            if not spots:
                continue  # lost at the end of the phase, as in the engine
            t, r = max(spots, key=lambda s: (want.get(s[0], 0.0) - strength.get(s[0], 0.0),
                                             -len(s[1])))
            send(u, src, r)

    # ---------- 2: greedily fill the most important defensive shortfalls ----------
    # Only actual arrivals increase strength. En-route moves are useful positioning,
    # but are not counted as defense at their eventual target.
    def shortfall(name):
        return max(0.0, want[name] - strength[name])

    def priority(name):
        # Relative shortage matters; home/factory and valuable production are protected.
        production = m.G.nodes[name].get("production", 0) or 0
        return (1.5 if name == home_factory else 1.0) * (1.0 + min(production, 10) / 10.0)

    def defense_gain(dst, value):
        return priority(dst) * min(value, shortfall(dst))

    def donor_loss(src, value):
        # Penalize removing useful cover even when the donor remains above its floor.
        before = max(0.0, want.get(src, 0.0) - strength.get(src, 0.0))
        after = max(0.0, want.get(src, 0.0) - (strength.get(src, 0.0) - value))
        return priority(src) * (after - before)

    # Heap holds territories by initial shortfall priority. A territory is requeued
    # only if it can still benefit from another direct reinforcement.
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
                # Shorter routes and smaller over-allocation win close ties.
                key = (score, -max(0.0, value - shortfall(target)), -len(route))
                if best is None or key > best[0]:
                    best = (key, u, src, route)
        if best is None or best[0][0] <= _EPS:
            continue
        _, u, src, route = best
        send(u, src, route)
        if shortfall(target) > _EPS:
            heapq.heappush(heap, (-priority(target) * shortfall(target), target))

    # ---------- 3: position spare units, without a lookahead search ----------
    dist_to_frontier = m.distance_from_set(frontier, "land") if frontier else {}
    dist_to_objective = m.distance_from_set(objectives, "land") if objectives else {}
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
                # Prefer useful frontier positions; avoid overcrowding safe areas.
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
    ut = m.unit_types.get(u.unit_type)
    return bool(ut and (ut.is_infrastructure or ut.is_aa))


def _reach_routes(game, player, start, move_range, domain, blocked=frozenset()):
    """can_reach_cache_helper as a BFS that also keeps one route per reachable territory: a unit
    may pass only through territories owned by `player` and not in `blocked`; the destination
    itself may be anything. Returns {territory: route of names}."""
    m = game.map
    if move_range <= 0:
        return {}
    parent = {start: None}
    queue = [(start, 0)]
    reached = []
    while queue:
        current, dist = queue.pop(0)
        if dist >= move_range:
            continue
        for neighbor in m.neighbors(current, domain):
            if neighbor in parent:
                continue
            parent[neighbor] = current
            reached.append(neighbor)
            if game.territories[neighbor].owner != player or neighbor in blocked:
                continue  # Can't pass through enemy territory (or a blocked one)
            if dist + 1 < move_range:
                queue.append((neighbor, dist + 1))
    routes = {}
    for t in reached:
        path = [t]
        while parent[path[-1]] is not None:
            path.append(parent[path[-1]])
        routes[t] = tuple(reversed(path))
    return routes


def _group_moves(definition, moves):
    """[(unit, route)] -> [fm.MoveAction], one per route (form_attacks)."""
    t_index = definition.territory_index
    grouped = {}
    for u, route in moves:
        grouped.setdefault(route, []).append(u.uid)
    return [fm.MoveAction(tuple(uids), tuple(t_index[t] for t in route))
            for route, uids in grouped.items()]


# ---- combat moves ----------------------------------------------------------------------------

class CombatAction:
    """Move(delegate="combat") of the original: an attack on `to_terr` (moves = fm.MoveActions),
    a skip of `to_terr` (moves = [], strength 0), or the end of the phase (end_phase=True)."""
    __slots__ = ("to_terr", "moves", "strength", "end_phase")

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
    def is_art(u):
        ut = m.unit_types.get(u.unit_type)
        return bool(ut and ut.is_artillery)

    def supportable(u):
        ut = m.unit_types.get(u.unit_type)
        return bool(ut and ut.artillery_supportable)

    infantry = sum(1 for u in units if supportable(u))
    artillery = sum(1 for u in units if is_art(u))
    supported_inf = min(infantry, artillery)
    strength = 0
    for u in units:
        if supportable(u):
            if supported_inf > 0:
                strength += u.attack + 1
                supported_inf -= 1
            else:
                strength += u.attack
        else:
            strength += u.attack
    return strength


def defender_strength(m, territory, player):
    """Total defense strength of enemy units on a territory."""
    return sum(u.defense for u in territory.units if u.owner != player
               and not (m.unit_types.get(u.unit_type)
                        and m.unit_types[u.unit_type].is_infrastructure))


def enemy_territories_by_priority(game, player):
    """MCTSGameState.set_terr_lists: enemy land territories bordering mine, most important first."""
    m = game.map
    terrs = game.territories
    victory_cities = set(m.victory_cities())
    mine = {name for name, t in terrs.items() if t.owner == player}

    def territory_priority(name):
        priority_score = 0
        # Victory city = highest priority
        if name in victory_cities:
            priority_score += 1000
        next_to_factory = sum(1 for n in m.neighbors(name) if n in victory_cities)
        priority_score += next_to_factory * 500
        # Production value
        priority_score += m.G.nodes[name]["production"] * 100
        # Number of bordering friendly territories (easier to attack/hold)
        surrounded_by_me = sum(1 for n in m.neighbors(name) if n in mine)
        priority_score += surrounded_by_me * 50
        # Number of bordering territories of the same owner
        enemy_neighbors = sum(1 for n in m.neighbors(name) if terrs[n].owner == terrs[name].owner)
        priority_score -= enemy_neighbors * 30
        return priority_score

    reachable = [name for name, t in terrs.items()
                 if t.owner != player and not m.is_water(name)
                 and any(n in mine for n in m.neighbors(name))]
    reachable.sort(key=territory_priority, reverse=True)
    return reachable


def combat_blocked(game):
    """Territories conquered this turn without blitzing: in combat movement, land units cannot
    move through them (Matches.territoryIsBlitzable, rules.is_blitzable)."""
    return game.conquered - game.blitzed


def _can_land_here(game, player, t):
    """rules.can_land_here with the view's owners: my land territory, not conquered this turn.
    (The view has no alliances, so allied territories do not count.)"""
    return (not game.map.is_water(t) and t not in game.conquered
            and game.territories[t].owner == player)


def _air_can_land(game, player, target, moves_left):
    """AirMovementValidator.validateAirCanLand as rules.air_can_find_land: an air unit ending its
    attack on `target` with `moves_left` must land on target or reach a landing territory, flying
    over no neutral land."""
    if _can_land_here(game, player, target):
        return True
    if moves_left <= 0:
        return False
    m = game.map
    dist = {target: 0}
    queue = [target]
    while queue:
        t = queue.pop(0)
        if dist[t] >= moves_left:
            continue
        for n in m.neighbors(t, "air"):
            if n in dist or (not m.is_water(n) and game.territories[n].owner == "Neutral"):
                continue
            if _can_land_here(game, player, n):
                return True
            dist[n] = dist[t] + 1
            queue.append(n)
    return False


def get_reachable_units(game, target, player):
    """[(unit, from, route)]: my unmoved mobile units, in my territories, that can reach target;
    an air unit only if it can still land after the attack."""
    m = game.map
    reachable = []
    routes = {}
    landing = {}  # moves left at the target -> whether air can land
    blocked = combat_blocked(game)
    for terr in game.owned_by(player):
        for u in terr.units:
            if u.owner != player or _is_static(m, u) or u.moved or u.domain == "sea":
                continue
            key = (terr.name, u.range, u.domain)
            if key not in routes:
                routes[key] = _reach_routes(game, player, terr.name, u.range, u.domain,
                                            blocked if u.domain == "land" else frozenset())
            if target in routes[key]:
                route = routes[key][target]
                if u.domain == "air":
                    left = u.range - (len(route) - 1)
                    if left not in landing:
                        landing[left] = _air_can_land(game, player, target, left)
                    if not landing[left]:
                        continue
                reachable.append((u, terr.name, route))
    return reachable


def combat_actions(game, definition, excluded, player=None):
    """heuristic_combat_legal_moves: attack options for the most important target not excluded.

    Returns (actions, excluded). Actions are the attack sets strongest first, then a skip of the
    target; the target (and every unreachable one passed over) is added to `excluded`. Actions are
    [] when no target is left. `game` must come from GameState.from_sim (units carry fm ids).
    """
    m = game.map
    player = player or game.whoAmI
    excluded = set(excluded)
    actions = []

    def unit_cost(u):
        c = m.unit_cost(u.unit_type)
        return c if c is not None else 9999

    for enemy_territory_name in enemy_territories_by_priority(game, player):
        if enemy_territory_name in excluded:
            continue
        # ── gather attackers ──────────────────────────────────────────────
        reachable = get_reachable_units(game, enemy_territory_name, player)
        if not reachable:
            excluded.add(enemy_territory_name)
            continue

        # ── sort by donor_score (desc): adjacent first, then attack power ──
        def donor_score(info):
            u, frm, _ = info
            score = 0.0
            if enemy_territory_name in m.neighbors(frm):
                score += 100.0
            score += 10.0 * u.attack
            return score

        sorted_reachable = sorted(reachable, key=donor_score, reverse=True)

        # ── defender strength ─────────────────────────────────────────────
        def_strength = defender_strength(m, game.territories[enemy_territory_name], player)

        # ── incremental buildup ───────────────────────────────────────────
        strengthThreshold = 1.5
        maxThreshold = 3.5
        sets = []
        unitsUpToStrength = []
        all_units = list(sorted_reachable)

        for unit_info in all_units:
            unitsUpToStrength.append(unit_info)
            currentStrength = calculate_attack_strength(m, [x[0] for x in unitsUpToStrength])
            if (currentStrength > strengthThreshold * def_strength
                    and currentStrength < (maxThreshold * def_strength) + 4):
                sets.append((list(unitsUpToStrength), currentStrength))
                strengthThreshold += 0.1

        # always include full force if it beats defender
        if all_units:
            currentStrength = calculate_attack_strength(m, [x[0] for x in all_units])
            if currentStrength > def_strength:
                if not sets or sets[-1][1] != currentStrength:
                    sets.append((list(all_units), currentStrength))

        # -- fallback: undefended, cheapest single unit capture
        if not sets and def_strength == 0:
            best = min(sorted_reachable, key=lambda x: unit_cost(x[0]))
            actions.append(CombatAction(
                to_terr=enemy_territory_name,
                moves=_group_moves(definition, [(best[0], best[2])]),
                strength=calculate_attack_strength(m, [best[0]]),
            ))

        # ── append moves (strongest first) ───────────────────────────────
        sets.reverse()
        for unit_infos, s in sets:
            actions.append(CombatAction(
                to_terr=enemy_territory_name,
                moves=_group_moves(definition, [(u, route) for u, _, route in unit_infos]),
                strength=s,
            ))

        # skip move always included
        actions.append(CombatAction(to_terr=enemy_territory_name, moves=[], strength=0))
        excluded.add(enemy_territory_name)
        break
    return actions, excluded
