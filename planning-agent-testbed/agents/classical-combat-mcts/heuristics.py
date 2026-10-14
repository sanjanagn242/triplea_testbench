"""Heuristic actions for the phases the classical-combat MCTS agent does not search.

Ports of tripleMind's purchase, noncombat-move and placement heuristics onto the triplea_fm
forward model. Each takes (definition, state, player) and returns a list of triplea_fm actions.
"""

from collections import deque

import triplea_fm as fm
from triplea_fm.definition import AIR, LAND
from triplea_fm.rules import movement_left

# Purchase heuristic: extra weight given to unit roles as the game goes on (round limit, roles).
PURCHASE_BIAS = (
    (5, ("infantry",)),
    (10, ("infantry", "blitz")),
    (15, ("blitz", "artillery")),
)


def _roles(ut) -> set:
    """Unit roles from the forward-model unit type (artillery-supportable stands in for infantry)."""
    roles = set()
    if ut.is_artillery_supportable:
        roles.add("infantry")
    if ut.can_blitz:
        roles.add("blitz")
    if ut.is_artillery:
        roles.add("artillery")
    return roles


def purchase(d, root, p):
    """Spend all PUs round-robin over the cheapest-first land combat units the player can buy.

    Returns [fm.PurchaseAction], or [] when the player has no factory or cannot afford anything.
    """
    pus = d.treasury_resource
    resources = root.balances[p][pus]

    has_factory = any(
        u.owner == p and u.loc >= 0 and root.owner[u.loc] == p and d.unit_types[u.type].can_produce
        for u in root.units.values()
    )
    if not has_factory:
        return []

    # Land combat units from the player's frontier: (rule id, cost, unit type).
    rules = []
    for rid in d.frontier[p]:
        rule = d.production_rules[rid]
        cost = sum(a for _, a in rule.costs)
        if any(res != pus for res, _ in rule.costs) or cost <= 0:
            continue
        for type_id, _ in rule.results:
            ut = d.unit_types[type_id]
            if ut.is_land and not ut.is_infrastructure and not ut.is_aa:
                rules.append((rid, cost, ut))
                break

    # Bias rotation based on round
    for limit, roles in PURCHASE_BIAS:
        if root.round < limit:
            rules = rules + [r for r in rules if _roles(r[2]) & set(roles)]
            break
    rules.sort(key=lambda r: r[1])

    purchase_dict = {}
    remaining = resources
    i = 0

    while True:
        affordable = [(rid, cost) for rid, cost, _ in rules if cost <= remaining]
        if not affordable:
            break
        rid, cost = affordable[i % len(affordable)]
        purchase_dict[rid] = purchase_dict.get(rid, 0) + 1
        remaining -= cost
        i += 1

    if not purchase_dict:
        return []

    return [fm.PurchaseAction(tuple(sorted(purchase_dict.items())))]


def _bfs(d, sources, domain, avoid=frozenset(), cutoff=None):
    """Multi-source BFS over territories a unit of `domain` may enter (land: no water; any:
    not impassable), never entering `avoid` (sources are always allowed).
    Returns ({territory: distance}, {territory: parent})."""
    dist, parent = {}, {}
    queue = deque()
    for t in sources:
        if not d.impassable[t] and (domain == AIR or not d.is_water[t]):
            dist[t] = 0
            queue.append(t)
    while queue:
        t = queue.popleft()
        if cutoff is not None and dist[t] >= cutoff:
            continue
        for n in d.neighbors[t]:
            if n in dist or n in avoid or d.impassable[n] or (domain != AIR and d.is_water[n]):
                continue
            dist[n] = dist[t] + 1
            parent[n] = t
            queue.append(n)
    return dist, parent


def _home_factory(d, s, p):
    """The player's capital if it still holds it, else any owned territory with its factory."""
    for t, cap in enumerate(d.capital_of):
        if cap == p and s.owner[t] == p:
            return t
    for t in range(len(d.territory_names)):
        if s.owner[t] == p and any(u.owner == p and d.unit_types[u.type].can_produce
                                   for u in s.units_in(t)):
            return t
    return None


def noncombat_move(d, s, p):
    """Push units from the rear toward the frontier and fill territories up to a quota.

    Pass 0 evacuates the home factory so placement has room; pass 1 visits owned territories by
    priority (frontier, staging, victory-city ring, closeness to objectives) and fills each to its
    quota from donor territories, stepping toward the frontier when the target is out of reach.

    Objectives are the map's highest-production territories (the flag on CTF). Returns
    [fm.MoveAction], one per (from, to) pair and unit domain, in the order decided.
    """
    n_terr = len(d.territory_names)
    best_prod = max(d.production, default=0)
    objectives = {t for t in range(n_terr) if best_prod and d.production[t] == best_prod}
    victory_cities = {t for t in range(n_terr) if d.victory_value[t] > 0}
    capitals = {t for t in range(n_terr) if d.capital_of[t] != fm.NULL_PLAYER}

    def land(t):
        return not d.is_water[t] and not d.impassable[t]

    def neighbors(t):
        return [n for n in d.neighbors[t] if land(n)]

    def is_combat_unit(u):
        return not d.unit_types[u.type].is_infrastructure

    # Unit counts change as moves are planned, so keep them outside the state.
    strength = {t: sum(1 for u in s.units_in(t) if is_combat_unit(u)) for t in range(n_terr)}

    # ---------- basics ----------
    my_names = {t for t in range(n_terr) if s.owner[t] == p}
    my_territories = sorted(my_names)
    not_mine = set(range(n_terr)) - my_names

    # Movable land/air units of mine, per territory (each unit is moved at most once).
    movable = {
        t: [u for u in s.units_in(t) if u.owner == p and is_combat_unit(u)
            and movement_left(d, u) > 0 and d.unit_types[u.type].domain in (LAND, AIR)]
        for t in my_territories
    }

    # frontier = my territory adjacent to non-owned
    my_frontier = {t for t in my_territories if any(n in not_mine for n in neighbors(t))}

    home_factory = _home_factory(d, s, p)

    def has_enemy_neighbor(t):
        return any(n in not_mine and n not in capitals for n in neighbors(t))

    # staging score: how many frontier neighbors this tile borders
    staging_score = {t: sum(1 for nb in neighbors(t) if nb in my_frontier) for t in my_territories}

    # victory-city ring: how many victory cities are adjacent
    factory_adj_score = {t: sum(1 for nb in neighbors(t) if nb in victory_cities) for t in my_territories}

    inf = float("inf")
    dist_to_frontier, _ = _bfs(d, my_frontier, LAND)
    dist_to_objective, _ = _bfs(d, objectives, LAND)

    # ---------- target priority ----------
    def territory_priority(t):
        score = 0
        # frontier first; an empty frontier gets a big bump
        if t in my_frontier:
            score += 1000
            if strength[t] == 0:
                score += 500
        score += staging_score.get(t, 0) * 200
        score += factory_adj_score.get(t, 0) * 100
        score += max(0, 5 - dist_to_objective.get(t, inf)) * 20
        return score

    targets = sorted(my_names, key=territory_priority, reverse=True)

    # ---------- reachability ----------
    # Noncombat: land units may only pass through my territories; air may overfly anything.
    avoid = {LAND: not_mine, AIR: frozenset()}
    _reach_cache = {}

    def reach(u, src):
        domain, left = d.unit_types[u.type].domain, movement_left(d, u)
        key = (src, domain, left)
        if key not in _reach_cache:
            _reach_cache[key] = _bfs(d, [src], domain, avoid=avoid[domain], cutoff=left)
        return _reach_cache[key]

    def can_reach(u, src, dst):
        return dst != src and dst in my_names and dst in reach(u, src)[0]

    def route(u, src, dst):
        parent = reach(u, src)[1]
        path = [dst]
        while path[-1] != src:
            path.append(parent[path[-1]])
        return tuple(reversed(path))

    def best_step_toward_frontier(u, src):
        """Reachable owned destination that most reduces the distance to the frontier."""
        best, best_score = None, -10 ** 9
        from_d = dist_to_frontier.get(src, inf)
        for dest in targets:
            if not can_reach(u, src, dest):
                continue
            improvement = from_d - dist_to_frontier.get(dest, inf)
            score = improvement * 100
            if dest in my_frontier:
                score += 50
            score += staging_score.get(dest, 0) * 10
            score += factory_adj_score.get(dest, 0) * 5
            if score > best_score:
                best_score, best = score, dest
        return best, best_score

    total_my_units = sum(strength[t] for t in my_names)

    def quota(t):
        if t == home_factory:
            base = 0.08  # always keep ~8% of army here
        elif t in my_frontier:
            # scale with enemy pressure on that tile
            enemy_threat = sum(strength[n] for n in neighbors(t) if n in not_mine)
            return max(3, min(enemy_threat + 2, int(total_my_units * 0.15)))
        elif staging_score.get(t, 0) > 0 or factory_adj_score.get(t, 0) > 0:
            base = 0.03
        elif t in objectives:
            base = 0.02
        else:
            base = 0.01
        return max(1, int(total_my_units * base))

    def frontier_excess(t):
        max_enemy_threat = max((strength[n] for n in neighbors(t) if n in not_mine), default=0)
        buffer = 2  # surplus beyond a safety buffer
        return max(0, strength[t] - (max_enemy_threat + buffer))

    frontier_excess_map = {t: frontier_excess(t) for t in my_frontier}

    planned = {}  # (from, to) -> [Unit], insertion-ordered

    def send(u, src, dst):
        movable[src].remove(u)
        strength[src] -= 1
        strength[dst] += 1
        planned.setdefault((src, dst), []).append(u)
        if src in frontier_excess_map:
            frontier_excess_map[src] -= 1

    # ---------- donors ----------
    # Don't drain frontier tiles below their safety buffer (the home factory is handled in pass 0).
    def can_donate(t):
        if not movable.get(t):
            return False
        if t in my_frontier and t != home_factory:
            return frontier_excess_map.get(t, 0) > 0
        return True

    # ---------- PASS 0: evacuate home factory before placement ----------
    if home_factory in movable:
        keep_min = 2 if has_enemy_neighbor(home_factory) else 0
        excess = max(0, strength[home_factory] - keep_min)
        for u in list(movable[home_factory]):
            if excess <= 0:
                break
            dest, score = best_step_toward_frontier(u, home_factory)
            if dest is None or score <= 0:
                continue
            send(u, home_factory, dest)
            excess -= 1

    # ---------- PASS 1: fill targets to quota (with step-to-frontier fallback) ----------
    for target in targets:
        for donor in my_territories:
            if strength[target] >= quota(target):
                break
            if donor == target:
                continue
            while strength[target] < quota(target) and can_donate(donor):
                moved = False
                for u in list(movable[donor]):
                    if can_reach(u, donor, target):
                        dest = target
                    else:
                        dest, score = best_step_toward_frontier(u, donor)
                        if dest is None or score <= 0:
                            continue
                    send(u, donor, dest)
                    moved = True
                    break
                if not moved:
                    break

    # ---------- encode as forward-model actions ----------
    actions = []
    for (src, dst), units in planned.items():
        by_domain = {}
        for u in units:
            by_domain.setdefault(d.unit_types[u.type].domain, []).append(u)
        for group in by_domain.values():
            # Every unit in the group reached dst, so the first one's route works for all.
            actions.append(fm.MoveAction(tuple(u.id for u in group), route(group[0], src, dst)))
    return actions


def place(d, root, p, rng):
    """Put everything bought at the home factory; over the production limit, place what fits.

    Returns [fm.PlaceAction], or [] when nothing is held or the player has no factory.
    """
    s = root.clone()
    held = s.held_units(p)
    home = _home_factory(d, s, p)
    if not held or home is None:
        return []
    everything = fm.PlaceAction(tuple(u.id for u in held), home)
    if fm.apply_in_place(d, s, everything, rng) != fm.REJECTED:
        actions = [everything]
    else:
        actions = []
        for u in held:  # over the production limit: place what fits
            single = fm.PlaceAction((u.id,), home)
            if fm.apply_in_place(d, s, single, rng) == fm.REJECTED:
                break
            actions.append(single)
    return actions
