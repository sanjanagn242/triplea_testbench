"""The actions of the roxo_paper agents: Roxo's territory-based action generation (M. Roxo, "MCTS-based
Planning for Grand Strategy Games", IST 2021, Sec. 5.3.1), plus a purchase phase built the same
way, for any player of the forward model.

Shared by roxo_paper/ne_mcts, roxo_paper/bb_mcts and roxo_paper/oep as identical copies (keep them
byte-identical, see ../check-shared-files.sh): the agents differ only in how they select.

A phase is a list of decision *items*, each decided once by one action: the units sent to a
territory, the units bought of one rule, or its skip. Items whose only option is the skip are
skipped without asking. Once every item is decided, END ends the phase.

  purchase      items: the player's land combat units bought with PUs, most attack + defence per
                PU first. Options per rule: buy all that is affordable and placeable, half of that,
                or skip (the non-combat move's {all, half, none}). No factory: nothing to buy.
  combat move   items: the enemy land territories next to the player's, in map order (each once).
                The units that can reach one (land, not moved this phase) must beat its defence by
                ATTACK_MARGIN (Weak AI strength, canAttackSuccessfully), else it is skipped (the
                pruning of unwinnable battles). Options: the growing unit sets of
                getUnitsUpToStrengthSet (Roxo's ATTACKCOMBINATIONS), largest first, then skip.
  non-combat    the air the engine would destroy lands first (forced, not an item). Items: the
                player's territories, capitals first, then by land distance to the enemy. Options:
                from each territory with units that can reach it, none, the first half or all of
                them, every combination (Algorithm 5), largest first; the all-none one is the skip.

Units moved this phase are never offered again, so a unit belongs to one action of a sequence and
an item's options only shrink as the phase goes on (which makes skipping single-option items safe).

A Phase works for a `search` object that provides d (GameDefinition), view(sim, player_name) and
apply(sim, action, rng).
"""

import itertools

import triplea_fm as fm
from triplea_fm import NULL_PLAYER, rules, toolkit
from triplea_fm.definition import COMBAT_MOVE, NONCOMBAT_MOVE, PURCHASE

import noncombat_generators as ncg
import weak_ai
from land_graph import LandGraph, UNREACHABLE

ATTACK_MARGIN = 1.1  # canAttackSuccessfully: attack strength > 1.1 x defence
MAX_NONCOMBAT_OPTIONS = 243  # cap on a territory's combinations (3^5): sources beyond it send all/none
END_KEY = "END"  # Phase.action_key of every END


def actor(d, sim):
    return d.steps[sim.step].actor


class PhaseState:
    """The forward-model state part-way through a phase. info: (frozenset of the items decided,
    frozenset of the unit ids moved this phase) in a move phase, (items decided, ((rule id, units
    bought), ...)) in the purchase."""
    __slots__ = ("sim", "depth", "info")

    def __init__(self, sim, depth, info):
        self.sim = sim
        self.depth = depth  # actions taken in the phase
        self.info = info

    def clone(self):
        return PhaseState(self.sim.clone(), self.depth, self.info)


class RoxoAction:
    """One option of an item, or END (end_phase). Moves: units (ids) and the routed moves sending
    them to territory `item`; purchase: `count` units of rule `item`. A skip has no units / count 0.
    `skipped` are the single-option items passed over to reach this one (decided with it)."""
    __slots__ = ("item", "units", "moves", "count", "strength", "end_phase", "skipped")

    def __init__(self, item=None, units=(), moves=(), count=0, strength=0.0, end_phase=False,
                 skipped=frozenset()):
        self.item = item
        self.units = units
        self.moves = moves
        self.count = count
        self.strength = strength
        self.end_phase = end_phase
        self.skipped = skipped

    @property
    def is_skip(self):
        return not self.end_phase and not self.units and not self.count

    def with_skipped(self, skipped):
        return RoxoAction(self.item, self.units, self.moves, self.count, self.strength,
                          self.end_phase, skipped)


END = RoxoAction(end_phase=True)


class Phase:
    name = None
    step_kind = None

    def __init__(self, search):
        self.search = search
        self.d = search.d

    # ---- the items and their options (what the subclasses define) ----

    def start(self, sim):
        """PhaseState at the start of the phase, and the forced moves applied to reach it."""
        return PhaseState(sim, 0, (frozenset(), frozenset() if self.moves_units else ())), []

    moves_units = True

    def items(self, state):
        """Every item of the phase, in its order (decided ones included)."""
        raise NotImplementedError

    def options(self, state, item):
        """The item's options in state, largest first, the skip last."""
        raise NotImplementedError

    def rebuild(self, state, action):
        """`action` (from another state, e.g. another genome) as an option of its item in state, or
        None when it is no longer legal there (its units used or out of reach, or unaffordable)."""
        raise NotImplementedError

    # ---- the sequence of actions the searches walk ----

    def pending(self, state):
        decided = state.info[0]
        return [i for i in self.items(state) if i not in decided]

    def actions(self, state):
        """Options of the first undecided item that has a choice (the items before it, with a skip
        only, are decided with it); [END] once none is left."""
        skipped = []
        for item in self.pending(state):
            opts = self.options(state, item)
            if len(opts) > 1:
                frozen = frozenset(skipped)
                return [a.with_skipped(frozen) for a in opts] if frozen else opts
            skipped.append(item)
        return [END]

    def step(self, state, action, rng):
        """Apply a non-END action to state in place; False if the forward model rejects it."""
        decided, rest = state.info
        if not action.is_skip and not self._apply(state, action, rng):
            return False
        decided = decided | action.skipped | {action.item}
        state.info = (decided, self._after(rest, action))
        state.depth += 1
        return True

    def _apply(self, state, action, rng):
        return all(self.search.apply(state.sim, mv, rng) for mv in action.moves)

    def _after(self, rest, action):
        return rest | frozenset(action.units)

    def key(self, state):
        """Transposition key: the items decided and where the moved units stand."""
        decided, moved = state.info
        units = state.sim.units
        return decided, tuple(sorted((uid, units[uid].loc) for uid in moved if uid in units))

    def action_key(self, action):
        if action.end_phase:
            return END_KEY
        return action.item, action.units

    def end_of(self, state):
        return END

    def finish(self, state, end, rng):
        return self.search.apply(state.sim, fm.EndPhaseAction(), rng)

    def levels_left(self, state, actions):
        """Upper bound on the actions still to choose: the undecided items."""
        return len(self.pending(state))

    def label(self, action):
        d = self.d
        if action.end_phase:
            return "END"
        name = d.territory_names[action.item]
        return f"skip {name}" if action.is_skip else f"{name} | {len(action.units)} units"

    def response_actions(self, sequence):
        return [mv for a in sequence for mv in a.moves]

    # ---- shared unit helpers ----

    def routes(self):
        """Per-search cache of land routes: {(player, start, steps, blitz, combat): {dest: route}}.
        Enemy units do not move during a move phase, so the routes stay valid; the forward model
        still validates every move."""
        cache = getattr(self.search, "route_cache", None)
        if cache is None:
            cache = self.search.route_cache = {}
        return cache

    def route(self, sim, p, u, t, combat):
        d = self.d
        ut = d.unit_types[u.type]
        steps = rules.movement_left(d, u)
        key = (p, u.loc, steps, ut.can_blitz, combat)
        cache = self.routes()
        found = cache.get(key)
        if found is None:
            found = cache[key] = toolkit.unit_routes(d, sim, p, u.loc, steps, False, ut.can_blitz,
                                                     combat)
        return found.get(t)

    def movable(self, sim, p, moved, combat):
        """p's land units with movement left that did not move this phase, by territory then id
        (findLandMoveOptionsSingular's units)."""
        d = self.d
        out = []
        for u in sim.units.values():
            if u.owner != p or u.loc < 0 or u.id in moved:
                continue
            ut = d.unit_types[u.type]
            if (not ut.is_land or ut.is_infrastructure or rules.movement_left(d, u) <= 0
                    or (combat and not ut.can_move_in_combat)):
                continue
            out.append(u)
        out.sort(key=lambda u: (u.loc, u.id))
        return out

    def reaching(self, sim, p, moved, t, combat):
        """(unit, route) for every movable unit outside t that can reach t this move."""
        out = []
        for u in self.movable(sim, p, moved, combat):
            if u.loc != t:
                r = self.route(sim, p, u, t, combat)
                if r is not None:
                    out.append((u, r))
        return out

    @staticmethod
    def moves_of(pairs):
        """Units sharing a route merged into one move."""
        by_route = {}
        for u, r in pairs:
            by_route.setdefault(tuple(r), []).append(u.id)
        return tuple(fm.MoveAction(tuple(ids), r) for r, ids in by_route.items())

    def rebuild_units(self, state, action, combat):
        if action.is_skip:
            return RoxoAction(action.item)
        sim = state.sim
        reach = {u.id: (u, r) for u, r in self.reaching(sim, actor(self.d, sim), state.info[1],
                                                          action.item, combat)}
        if not all(uid in reach for uid in action.units):
            return None
        pairs = [reach[uid] for uid in action.units]
        return RoxoAction(action.item, action.units, self.moves_of(pairs),
                          strength=action.strength)


class PurchasePhase(Phase):
    """Items: rule ids. info: (rules decided, ((rule id, units bought), ...))."""
    name = "purchase"
    step_kind = PURCHASE
    moves_units = False

    def candidates(self, p):
        """p's rules for one land combat unit for PUs, most attack + defence per PU first
        (frontiers are fixed: computed once per player)."""
        cache = getattr(self, "_candidates", None)
        if cache is None:
            cache = self._candidates = {}
        if p not in cache:
            d = self.d
            pus = d.treasury_resource
            rows = []
            for rid in d.frontier[p]:
                rule = d.production_rules[rid]
                if len(rule.results) != 1 or [r for r, _ in rule.costs] != [pus]:
                    continue
                type_id, qty = rule.results[0]
                cost = rule.costs[0][1]
                ut = d.unit_types[type_id]
                if qty == 1 and cost > 0 and ut.is_land and not ut.is_infrastructure and not ut.is_aa:
                    rows.append((-(ut.attack + ut.defense) / cost, rid, cost))
            rows.sort()
            cache[p] = [(rid, cost) for _, rid, cost in rows]
        return cache[p]

    def capacity(self, sim, p):
        """Units p can place this turn (rules.placement_snapshot); None: unlimited."""
        caps = rules.placement_snapshot(self.d, sim, p)
        if any(c < 0 for c in caps.values()):
            return None
        return sum(caps.values())

    def items(self, state):
        sim = state.sim
        p = actor(self.d, sim)
        if not rules.placement_snapshot(self.d, sim, p):
            return []  # no factory: nothing to buy
        return [rid for rid, _ in self.candidates(p)]

    def options(self, state, item):
        sim = state.sim
        p = actor(self.d, sim)
        cost = dict(self.candidates(p))[item]
        most = sim.balances[p][self.d.treasury_resource] // cost
        cap = self.capacity(sim, p)
        if cap is not None:
            most = min(most, cap - sum(n for _, n in state.info[1]))
        counts = [most] if most > 0 else []
        if 0 < most // 2 < most:
            counts.append(most // 2)
        return [RoxoAction(item, count=n) for n in counts] + [RoxoAction(item)]

    def rebuild(self, state, action):
        if action.is_skip:
            return RoxoAction(action.item)
        most = self.options(state, action.item)[0].count
        return RoxoAction(action.item, count=action.count) if action.count <= most else None

    def _apply(self, state, action, rng):
        return self.search.apply(state.sim, fm.PurchaseAction(((action.item, action.count),)), rng)

    def _after(self, bought, action):
        if not action.count:
            return bought
        b = dict(bought)
        b[action.item] = b.get(action.item, 0) + action.count
        return tuple(sorted(b.items()))

    def key(self, state):
        return state.info

    def action_key(self, action):
        return END_KEY if action.end_phase else (action.item, action.count)

    def label(self, action):
        if action.end_phase:
            return "END"
        name = self.d.unit_types[self.d.production_rules[action.item].results[0][0]].name
        return f"skip {name}" if action.is_skip else f"buy {action.count} {name}"

    def response_actions(self, sequence):
        bought = ()
        for a in sequence:
            if a.count:
                bought = self._after(bought, a)
        return [fm.PurchaseAction(bought)] if bought else []


class CombatPhase(Phase):
    """Items: enemy territory indices."""
    name = "combatMove"
    step_kind = COMBAT_MOVE

    def items(self, state):
        """getNeighboringEnemyTerritories: enemy land next to the player's territories, each once."""
        d, sim = self.d, state.sim
        p = actor(d, sim)
        seen = []
        for t in range(len(d.territory_names)):
            if sim.owner[t] != p:
                continue
            for x in d.neighbors[t]:
                if (x not in seen and not d.is_water[x] and not d.impassable[x]
                        and rules.is_enemy_territory(d, sim, p, x)):
                    seen.append(x)
        return seen

    def options(self, state, item):
        d, sim = self.d, state.sim
        p = actor(d, sim)
        reach = self.reaching(sim, p, state.info[1], item, True)
        units = [u for u, _ in reach]
        defence = weak_ai.strength(d, sim.units_in(item), False)
        if not units or weak_ai.strength(d, units, True) <= ATTACK_MARGIN * defence:
            return [RoxoAction(item)]  # cannot be taken: skip only
        routes = dict((u.id, r) for u, r in reach)
        out = []
        for unit_set in units_up_to_strength_sets(d, defence, units):
            out.append(RoxoAction(item, tuple(u.id for u in unit_set),
                                  self.moves_of([(u, routes[u.id]) for u in unit_set]),
                                  strength=weak_ai.strength(d, unit_set, True)))
        return out + [RoxoAction(item)]

    def rebuild(self, state, action):
        return self.rebuild_units(state, action, True)


def units_up_to_strength_sets(d, defence, units):
    """getUnitsUpToStrengthSet (Roxo's ATTACKCOMBINATIONS): growing prefixes of units kept when
    their strength passes a threshold that starts at ATTACK_MARGIN x defence and rises by 0.1 for
    each kept prefix, staying below 3.5 x defence + 4; the full set is added if it was not kept.
    Largest first."""
    sets, threshold, prefix, full = [], ATTACK_MARGIN, [], False
    for u in units:
        prefix.append(u)
        full = False
        ours = weak_ai.strength(d, prefix, True)
        if threshold * defence < ours < 3.5 * defence + 4:
            full = True
            sets.append(list(prefix))
            threshold += 0.1
    if not full:
        sets.append(list(prefix))
    return list(reversed(sets))


class NoncombatPhase(Phase):
    """Items: the player's territory indices."""
    name = "nonCombatMove"
    step_kind = NONCOMBAT_MOVE

    def graph(self):
        g = getattr(self.search, "land_graph", None)
        if g is None:
            g = self.search.land_graph = LandGraph(self.d)
        return g

    def start(self, sim):
        """The air the engine would destroy lands first: forced, so not an action."""
        search = self.search
        player = self.d.player_names[actor(self.d, sim)]
        moved, applied = set(), []
        rng = fm.SeededRng(0)  # moves roll no dice
        for mv in ncg.land_air_moves(search.view(sim, player), self.d, player):
            if search.apply(sim, mv, rng):
                moved.update(mv.units)
                applied.append(mv)
        return PhaseState(sim, 0, (frozenset(), frozenset(moved))), applied

    def items(self, state):
        """The player's land territories: capitals first, then nearest the enemy (Sec. 5.3.1)."""
        d, sim = self.d, state.sim
        p = actor(d, sim)
        g = self.graph()
        owned = [t for t in g.land if sim.owner[t] == p]
        enemy = [x for x in g.land if sim.owner[x] != NULL_PLAYER and d.is_at_war(p, sim.owner[x])]

        def distance(t):
            return min((g.dist[e][t] for e in enemy), default=UNREACHABLE)
        return sorted(owned, key=lambda t: (d.capital_of[t] != p, distance(t), t))

    def options(self, state, item):
        sim = state.sim
        p = actor(self.d, sim)
        groups = {}
        for u, r in self.reaching(sim, p, state.info[1], item, False):
            groups.setdefault(u.loc, []).append((u, r))
        # per source territory: all, first half, none (largest sources first, capped)
        choices, product = [], 1
        for src in sorted(groups, key=lambda s: (-len(groups[s]), s)):
            g = groups[src]
            half = g[:len(g) // 2]
            opts = [g, half, []] if half else [g, []]
            if product * len(opts) > MAX_NONCOMBAT_OPTIONS:
                opts = [g, []]
                if product * 2 > MAX_NONCOMBAT_OPTIONS:
                    continue
            product *= len(opts)
            choices.append(opts)
        combos = []
        for pick in itertools.product(*choices):
            pairs = [x for part in pick for x in part]
            if pairs:
                combos.append(pairs)
        combos.sort(key=lambda pairs: -len(pairs))  # stable: all-of-everything first
        out = [RoxoAction(item, tuple(u.id for u, _ in pairs), self.moves_of(pairs),
                          strength=float(len(pairs)))
               for pairs in combos]
        return out + [RoxoAction(item)]

    def rebuild(self, state, action):
        return self.rebuild_units(state, action, False)


def make_phases(search):
    """{step kind: Phase} for the three searched phases."""
    return {p.step_kind: p for p in (PurchasePhase(search), CombatPhase(search),
                                     NoncombatPhase(search))}
