"""Candidate actions per phase, built from a map_graph.GameState.

Each generator returns a list of triplea_fm actions (PurchaseAction, MoveAction, PlaceAction), so
the agent can encode them for the engine with LiveBridge and the same actions can be fed to the
forward model during search. Unit roles come from the map's unit attachments, never from unit
names, so the generators work on any map.
"""

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


def noncombat_move(game, definition, player=None, objectives=None):
    """Push units from the rear toward the frontier and fill territories up to a quota.

    Pass 0 evacuates the home factory so placement has room; pass 1 visits owned territories by
    priority (frontier, staging, victory-city ring, closeness to objectives) and fills each to its
    quota from donor territories, stepping toward the frontier when the target is out of reach.

    `objectives` defaults to the map's highest-production territories (the flag on CTF).
    `game` must come from GameState.from_sim so units carry forward-model ids. Returns
    [fm.MoveAction], one per (from, to) pair and unit domain, in the order decided.
    """
    m = game.map
    player = player or game.whoAmI
    territories = game.territories
    objectives = objectives if objectives is not None else m.objectives()
    victory_cities = set(m.victory_cities())
    capitals = set(m.capitals().values())

    def neighbors(name):
        return m.neighbors(name, "land")

    def is_combat_unit(u):
        ut = m.unit_types.get(u.unit_type)
        return not (ut and ut.is_infrastructure)

    # Unit counts change as moves are planned, so keep them outside the observation.
    strength = {name: sum(1 for u in t.units if is_combat_unit(u)) for name, t in territories.items()}
    # Movable land/air units of mine, per territory (each unit is moved at most once).
    movable = {
        t.name: [u for u in t.units if u.owner == player and is_combat_unit(u)
                 and u.movement_left > 0 and u.domain in ("land", "air")]
        for t in game.owned_by(player)
    }

    # ---------- basics ----------
    my_territories = game.owned_by(player)
    my_names = {t.name for t in my_territories}
    not_mine = set(territories) - my_names

    # frontier = my territory adjacent to non-owned
    my_frontier = {t.name for t in my_territories if any(n in not_mine for n in neighbors(t.name))}

    home_factory = _home_factory(game, player)

    def has_enemy_neighbor(name):
        return any(n in not_mine and n not in capitals for n in neighbors(name))

    # staging score: how many frontier neighbors this tile borders
    staging_score = {t.name: sum(1 for nb in neighbors(t.name) if nb in my_frontier)
                     for t in my_territories}

    # victory-city ring: how many victory cities are adjacent
    factory_adj_score = {t.name: sum(1 for nb in neighbors(t.name) if nb in victory_cities)
                         for t in my_territories}

    inf = float("inf")
    dist_to_frontier = m.distance_from_set(my_frontier, "land")
    dist_to_objective = m.distance_from_set(objectives, "land")

    # ---------- target priority ----------
    def territory_priority(name):
        score = 0
        # frontier first; an empty frontier gets a big bump
        if name in my_frontier:
            score += 1000
            if strength[name] == 0:
                score += 500
        score += staging_score.get(name, 0) * 200
        score += factory_adj_score.get(name, 0) * 100
        score += max(0, 5 - dist_to_objective.get(name, inf)) * 20
        return score

    targets = sorted(my_names, key=territory_priority, reverse=True)

    # ---------- reachability ----------
    # Noncombat: land units may only pass through my territories; air may overfly anything.
    avoid = {"land": not_mine, "air": set()}
    _reach_cache = {}

    def reach(unit, src):
        key = (src, unit.domain, unit.movement_left)
        if key not in _reach_cache:
            _reach_cache[key] = m.reachable(src, unit.movement_left, unit.domain,
                                            avoid=avoid[unit.domain])
        return _reach_cache[key]

    def can_reach(unit, src, dst):
        return dst != src and dst in my_names and dst in reach(unit, src)

    def best_step_toward_frontier(unit, from_name):
        """Reachable owned destination that most reduces the distance to the frontier."""
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
            base = 0.08  # always keep ~8% of army here
        elif name in my_frontier:
            # scale with enemy pressure on that tile
            enemy_threat = sum(strength[n] for n in neighbors(name) if n in not_mine)
            return max(3, min(enemy_threat + 2, int(total_my_units * 0.15)))
        elif staging_score.get(name, 0) > 0 or factory_adj_score.get(name, 0) > 0:
            base = 0.03
        elif name in objectives:
            base = 0.02
        else:
            base = 0.01
        return max(1, int(total_my_units * base))

    def frontier_excess(name):
        max_enemy_threat = max((strength[n] for n in neighbors(name) if n in not_mine), default=0)
        buffer = 2  # surplus beyond a safety buffer
        return max(0, strength[name] - (max_enemy_threat + buffer))

    frontier_excess_map = {t: frontier_excess(t) for t in my_frontier}

    planned = {}  # (from, to) -> [Unit], insertion-ordered

    def send(unit, src, dst):
        movable[src].remove(unit)
        strength[src] -= 1
        strength[dst] += 1
        planned.setdefault((src, dst), []).append(unit)
        if src in frontier_excess_map:
            frontier_excess_map[src] -= 1

    # ---------- donors ----------
    # Don't drain frontier tiles below their safety buffer (the home factory is handled in pass 0).
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
            domain = group[0].domain
            route = m.path(src, dst, domain, avoid=avoid[domain])
            actions.append(fm.MoveAction(tuple(u.uid for u in group),
                                         tuple(t_index[t] for t in route)))
    return actions


def _split_by_domain(units):
    groups = {}
    for u in units:
        groups.setdefault(u.domain, []).append(u)
    return list(groups.values())


# ---- shared helpers (ports of move_generator.pyx) --------------------------------------------

def _is_static(m, u):
    """Units the move generators never move (factories, AA guns, other infrastructure)."""
    ut = m.unit_types.get(u.unit_type)
    return bool(ut and (ut.is_infrastructure or ut.is_aa))


def _reach_routes(game, player, start, move_range, domain):
    """can_reach_cache_helper as a BFS that also keeps one route per reachable territory: a unit
    may pass only through territories owned by `player`; the destination itself may be anything.
    Returns {territory: route of names}."""
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
            if game.territories[neighbor].owner != player:
                continue  # Can't pass through enemy territory
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


def get_reachable_units(game, target, player):
    """[(unit, from, route)]: my unmoved mobile units, in my territories, that can reach target."""
    m = game.map
    reachable = []
    routes = {}
    for terr in game.owned_by(player):
        for u in terr.units:
            if u.owner != player or _is_static(m, u) or u.moved or u.domain == "sea":
                continue
            key = (terr.name, u.range, u.domain)
            if key not in routes:
                routes[key] = _reach_routes(game, player, terr.name, u.range, u.domain)
            if target in routes[key]:
                reachable.append((u, terr.name, routes[key][target]))
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


# ---- noncombat moves used in rollouts (heuristic_non_combat_legal_moves of the .pyx) ---------

def noncombat_move_rollout(game, definition, player=None):
    """Rollout noncombat policy. Returns [fm.MoveAction], one per (from, route)."""
    m = game.map
    player = player or game.whoAmI
    territories = game.territories
    my_territories = game.owned_by(player)
    capitals = set(m.capitals().values())
    victory_cities = set(m.victory_cities())

    # ── quantity snapshot (the only thing mutated): {terr: {(unit_type, owner): qty}} ──
    qty_snap = {}
    for name, terr in territories.items():
        qty_snap[name] = {}
        for u in terr.units:
            k = (u.unit_type, u.owner)
            qty_snap[name][k] = qty_snap[name].get(k, 0) + 1
    # real units still free to move, per territory (each unit moves at most once)
    free_units = {t.name: [u for u in t.units if u.owner == player] for t in my_territories}

    def count_snap(name):
        return sum(q for (ut, _), q in qty_snap.get(name, {}).items()
                   if not (m.unit_types.get(ut) and m.unit_types[ut].is_infrastructure))

    def count_real(terr):
        return sum(1 for u in terr.units if not (m.unit_types.get(u.unit_type)
                                                 and m.unit_types[u.unit_type].is_infrastructure))

    # ── frontier detection ──
    my_frontier_territories = set()
    for terr in my_territories:
        for n in m.neighbors(terr.name):
            if territories[n].owner != player:
                my_frontier_territories.add(terr.name)
                break

    home_factory = _home_factory(game, player)

    # ── pre-compute scoring maps ──
    staging_score = {}
    factory_adj_score = {}
    for terr in my_territories:
        staging_score[terr.name] = sum(1 for nb in m.neighbors(terr.name)
                                       if nb in my_frontier_territories)
        score = 0
        for nb in m.neighbors(terr.name):
            if nb == home_factory:
                score += 3
            elif nb in victory_cities:
                score += 1
        factory_adj_score[terr.name] = score

    # ── BFS distance to frontier ──
    dist_to_frontier = m.distance_from_set(my_frontier_territories)
    INF = 2147483647

    total_my_units = sum(count_snap(t.name) for t in my_territories)

    def frontier_excess(name, buffer=2):
        my_strength = count_snap(name)
        max_enemy_threat = 0
        for nb in m.neighbors(name):
            if territories[nb].owner != player:
                max_enemy_threat = max(max_enemy_threat, count_snap(nb))
        excess = my_strength - (max_enemy_threat + buffer)
        return excess if excess > 0 else 0

    frontier_excess_map = {t: frontier_excess(t) for t in my_frontier_territories}

    def territory_priority(t):
        score = 0
        if t.name in my_frontier_territories:
            score += 1000
            if count_real(t) == 0:
                score += 500
        score += staging_score.get(t.name, 0) * 200
        score += factory_adj_score.get(t.name, 0) * 100
        return score

    def quota(name):
        if name == home_factory:
            return max(1, int(total_my_units * 0.08))
        if name in my_frontier_territories:
            enemy_threat = sum(count_snap(nb) for nb in m.neighbors(name)
                               if territories[nb].owner != player)
            return max(3, min(enemy_threat + 2, int(total_my_units * 0.15)))
        if staging_score.get(name, 0) > 0 or factory_adj_score.get(name, 0) > 0:
            return max(1, int(total_my_units * 0.03))
        return max(1, int(total_my_units * 0.01))

    def has_enemy_neighbor(name):
        return any(territories[n].owner != player and n not in capitals for n in m.neighbors(name))

    targets = sorted(my_territories, key=territory_priority, reverse=True)

    # ── donor list ──
    donor_territories = []
    for t in my_territories:
        if count_snap(t.name) <= 0:
            continue
        if t.name in my_frontier_territories and t.name != home_factory:
            if frontier_excess_map.get(t.name, 0) <= 0:
                continue
        donor_territories.append(t)
    donor_territories.sort(key=lambda t: count_snap(t.name) - quota(t.name), reverse=True)

    route_cache = {}

    def reach(u, from_name):
        key = (from_name, u.range, u.domain)
        if key not in route_cache:
            route_cache[key] = _reach_routes(game, player, from_name, u.range, u.domain)
        return route_cache[key]

    def best_step_toward_frontier(u, from_name):
        best, best_score = None, -2147483647
        from_d = dist_to_frontier.get(from_name, INF)
        for dest in targets:
            if dest.name == from_name or dest.name not in reach(u, from_name):
                continue
            improvement = from_d - dist_to_frontier.get(dest.name, INF)
            score = improvement * 100
            if dest.name in my_frontier_territories:
                score += 50
            score += staging_score.get(dest.name, 0) * 10
            score += factory_adj_score.get(dest.name, 0) * 5
            if score > best_score:
                best_score, best = score, dest.name
        return best, best_score

    moves = []  # (unit, route)

    def take(from_name, unit_type):
        for u in free_units.get(from_name, []):
            if u.unit_type == unit_type:
                free_units[from_name].remove(u)
                return u
        return None

    # ── PASS 0: evacuate home factory ──
    if home_factory in free_units:
        keep_min = 2 if has_enemy_neighbor(home_factory) else 0
        excess_units = max(0, count_snap(home_factory) - keep_min)
        sent = 0
        factory_snap = qty_snap[home_factory]
        for u in list(free_units[home_factory]):
            k = (u.unit_type, u.owner)
            if _is_static(m, u) or u.range <= 0 or factory_snap.get(k, 0) <= 0:
                continue
            if sent >= excess_units:
                break
            dest, score = best_step_toward_frontier(u, home_factory)
            if dest is None or score <= 0:
                continue
            free_units[home_factory].remove(u)
            moves.append((u, reach(u, home_factory)[dest]))
            factory_snap[k] -= 1
            sent += 1

    # ── PASS 1: fill targets to quota ──
    for terr in targets:
        terr_name = terr.name
        if count_snap(terr_name) >= quota(terr_name):
            continue
        donor_idx = 0
        while count_snap(terr_name) < quota(terr_name):
            if donor_idx >= len(donor_territories):
                break
            donor = donor_territories[donor_idx]

            # drain guard
            if donor.name in my_frontier_territories:
                if count_snap(donor.name) <= frontier_excess_map.get(donor.name, 0):
                    donor_idx += 1
                    continue
            else:
                min_keep = 2 if has_enemy_neighbor(donor.name) else 1
                if count_snap(donor.name) <= min_keep:
                    donor_idx += 1
                    continue

            # pick first movable unit that still has snapshot qty > 0
            picked = None
            for u in free_units.get(donor.name, []):
                if _is_static(m, u) or u.range <= 0:
                    continue
                if qty_snap[donor.name].get((u.unit_type, u.owner), 0) <= 0:
                    continue
                picked = u
                break
            if picked is None:
                donor_idx += 1
                continue

            dest = None
            if terr_name in reach(picked, donor.name):
                dest = terr_name
            else:
                step, step_score = best_step_toward_frontier(picked, donor.name)
                if step is not None and step_score > 0:
                    dest = step
            if dest is None:
                donor_idx += 1
                continue

            take(donor.name, picked.unit_type)
            moves.append((picked, reach(picked, donor.name)[dest]))
            k = (picked.unit_type, picked.owner)
            qty_snap[donor.name][k] -= 1
            # also credit the destination so it counts toward quota
            qty_snap[dest][(picked.unit_type, player)] = qty_snap[dest].get((picked.unit_type, player), 0) + 1
            if donor.name in my_frontier_territories:
                frontier_excess_map[donor.name] = max(0, frontier_excess_map.get(donor.name, 0) - 1)

    # Grouped by (from, route) like grouped_moves in heuristic_non_combat_legal_moves.
    return _group_moves(definition, moves)
