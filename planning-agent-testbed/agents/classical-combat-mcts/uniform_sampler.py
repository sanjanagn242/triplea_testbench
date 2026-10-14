"""A uniformly random legal action for one playout step, without listing every legal action.

UniformSampler.action(s) has exactly the distribution of rnd.choice(list(legal_actions(d, s)))
(triplea_fm.actionspace, EndPhase included where it is legal), but it is much cheaper: it draws
uniformly from the candidates the exhaustive generator would produce (rejection sampling, using
counts instead of the list) and checks only the drawn candidate with kernel.validate, redrawing
when the kernel rejects it. Rejection sampling from a uniform proposal is uniform over the
accepted set, so the result is uniform over the legal actions.

Battle selection, retreats and zero-hit casualties have tiny action sets and are enumerated. If
MAX_PROPOSALS draws find nothing legal, the step falls back to enumeration too (counted in
`fallbacks`), so a legal action is never missed.
"""

from __future__ import annotations

from math import prod

from triplea_fm import kernel, rules
from triplea_fm.actions import NORMAL, RAID, EndPhaseAction, MoveAction, PlaceAction, PurchaseAction, \
    SelectCasualtiesAction
from triplea_fm.actionspace import _classes, _walks, legal_actions
from triplea_fm.definition import COMBAT_MOVE
from triplea_fm.state import CASUALTIES_D, MOVE_D, PLACE_D, PURCHASE_D

MAX_PROPOSALS = 2000  # candidate draws per step before falling back to enumeration


class UniformSampler:
    def __init__(self, d, rnd, max_proposals: int = MAX_PROPOSALS):
        self.d = d
        self.rnd = rnd
        self.max_proposals = max_proposals
        self.fallbacks = 0  # steps that had to enumerate after max_proposals failed draws
        self._walk_cache = {}  # routes depend only on the map

    def action(self, s):
        """A uniformly random legal action for s.decision, or None if there is none."""
        dec = s.decision
        if dec is None or s.terminal is not None:
            return None
        proposer = {PURCHASE_D: self._purchase, MOVE_D: self._move, PLACE_D: self._place,
                    CASUALTIES_D: self._casualties}.get(dec.kind)
        draw = proposer(s) if proposer else None
        if draw is not None:
            for _ in range(self.max_proposals):
                a = draw()
                if a is not None and kernel.validate(self.d, s, a).ok:
                    return a
            self.fallbacks += 1
        actions = list(legal_actions(self.d, s))
        return self.rnd.choice(actions) if actions else None

    # ---- helpers ----

    def _walks(self, start, max_len, any_land):
        key = (start, max_len, any_land)
        if key not in self._walk_cache:
            self._walk_cache[key] = _walks(self.d, start, max_len, any_land)
        return self._walk_cache[key]

    def _weighted_index(self, weights):
        """Index i with probability weights[i] / sum(weights) (integer weights)."""
        x = self.rnd.randrange(sum(weights))
        for i, w in enumerate(weights):
            if x < w:
                return i
            x -= w
        raise AssertionError("unreachable")

    def _box_counts(self, sizes):
        """A uniformly random count vector with 0 <= counts[i] <= sizes[i]."""
        return [self.rnd.randint(0, n) for n in sizes]

    @staticmethod
    def _ids(classes, counts):
        return tuple(sorted(u for c, k in zip(classes, counts) for u in c[:k]))

    # ---- proposers: each returns draw(), a uniform candidate (or EndPhase) or None to redraw ----

    def _purchase(self, s):
        """Every affordable count vector over the frontier rules; the zero vector stands for
        EndPhase, which the exhaustive set lists once in its place."""
        d, p = self.d, s.decision.actor
        if not rules.has_enough_capitals(d, s, p):
            return EndPhaseAction
        rule_ids = sorted(set(d.frontier[p]))
        costs = []
        for rid in rule_ids:
            c = [0] * len(s.balances[p])
            for r, a in d.production_rules[rid].costs:
                c[r] += a
            if not any(c):
                return None  # free rule: let enumeration raise as the generator does
            costs.append(c)
        memo = {}

        def most(i, left):
            return min(left[r] // c for r, c in enumerate(costs[i]) if c > 0)

        def after(i, left, k):
            return tuple(left[r] - k * costs[i][r] for r in range(len(left)))

        def count(i, left):
            """Affordable count vectors over rules i.. with `left` resources (zero included)."""
            if i == len(rule_ids):
                return 1
            key = (i, left)
            if key not in memo:
                memo[key] = sum(count(i + 1, after(i, left, k)) for k in range(most(i, left) + 1))
            return memo[key]

        def draw():
            left = tuple(s.balances[p])
            counts = []
            for i in range(len(rule_ids)):
                k = self._weighted_index([count(i + 1, after(i, left, k))
                                          for k in range(most(i, left) + 1)])
                counts.append(k)
                left = after(i, left, k)
            if not any(counts):
                return EndPhaseAction()
            return PurchaseAction(tuple((rid, k) for rid, k in zip(rule_ids, counts) if k))

        return draw

    def _move(self, s):
        """Every nonzero count vector of one territory's movable units along every route within
        the group's movement (NORMAL, and RAID for all-bomber groups in combat), plus EndPhase.
        Proposal: territory weighted by its count-vector box x an upper bound on its routes, a
        uniform vector from the box, a uniform route slot under the bound; the zero vector and slots
        past the group's real routes are redrawn, so every candidate has the same proposal mass."""
        d, p = self.d, s.decision.actor
        combat = d.steps[s.step].kind == COMBAT_MOVE
        by_loc = {}
        for u in s.units.values():
            if (u.owner == p and u.loc >= 0 and rules.movement_left(d, u) > 0
                    and (not combat or d.unit_types[u.type].can_move_in_combat)):
                by_loc.setdefault(u.loc, []).append(u)
        stacks, weights = [], []
        for start, units in by_loc.items():
            classes = []
            for ids in _classes(s, units):
                u = s.units[ids[0]]
                ut = d.unit_types[u.type]
                classes.append((ids, rules.movement_left(d, u), ut.is_land, combat and ut.is_strategic_bomber))
            # Any group's routes are a subset of these: longest movement, water allowed unless
            # every class is land; doubled when some group could raid.
            bound = len(self._walks(start, max(c[1] for c in classes), all(c[2] for c in classes)))
            bound *= 2 if any(c[3] for c in classes) else 1
            box = prod(len(c[0]) + 1 for c in classes)  # count vectors, zero included (redrawn)
            if bound and box > 1:
                stacks.append((start, classes, bound))
                weights.append(box * bound)
        weights.append(1)  # EndPhase

        def draw():
            i = self._weighted_index(weights)
            if i == len(stacks):
                return EndPhaseAction()
            start, classes, bound = stacks[i]
            counts = self._box_counts([len(c[0]) for c in classes])
            chosen = [c for c, k in zip(classes, counts) if k]
            if not chosen:
                return None
            routes = self._walks(start, min(c[1] for c in chosen), any(c[2] for c in chosen))
            raid = all(c[3] for c in chosen)
            slot = self.rnd.randrange(bound)
            if slot >= len(routes) * (2 if raid else 1):
                return None
            ids = self._ids([c[0] for c in classes], counts)
            if raid:  # the generator lists NORMAL then RAID for each route
                return MoveAction(ids, routes[slot // 2], RAID if slot % 2 else NORMAL)
            return MoveAction(ids, routes[slot], NORMAL)

        return draw

    def _place(self, s):
        """Every count vector of 1..room held units at every producer with room, plus EndPhase."""
        d, p = self.d, s.decision.actor
        held = [u for u in s.held_units(p)
                if not d.unit_types[u.type].is_factory and d.unit_types[u.type].domain != 2]
        if not held:
            return EndPhaseAction
        classes = _classes(s, held)
        sizes = [len(c) for c in classes]
        sites = []
        for t, cap in (s.placement_cap or {}).items():
            room = len(held) if cap < 0 else cap - s.placement_used.get(t, 0)
            if room > 0:
                sites.append((t, room))
        box = prod(n + 1 for n in sizes)  # count vectors, zero included (redrawn)
        total = len(sites) * box + 1  # + EndPhase

        def draw():
            x = self.rnd.randrange(total)
            if x == 0:
                return EndPhaseAction()
            t, room = sites[(x - 1) // box]
            counts = self._box_counts(sizes)
            if not 1 <= sum(counts) <= room:
                return None
            return PlaceAction(self._ids(classes, counts), t)

        return draw

    def _casualties(self, s):
        """Every count vector of exactly `need` eligible units (drawn exactly by counting)."""
        hb = s.decision.hit_batch
        need = min(hb.hits, len(hb.eligible))
        if need <= 0:
            return None  # the generator lists nothing here; enumeration handles it
        classes = _classes(s, [s.units[u] for u in hb.eligible])
        sizes = [len(c) for c in classes]
        # ways[i][r]: count vectors over classes i.. summing to r
        ways = [[0] * (need + 1) for _ in range(len(sizes) + 1)]
        ways[len(sizes)][0] = 1
        for i in range(len(sizes) - 1, -1, -1):
            for r in range(need + 1):
                ways[i][r] = sum(ways[i + 1][r - k] for k in range(min(sizes[i], r) + 1))
        if ways[0][need] == 0:
            return None

        def draw():
            r, counts = need, []
            for i, n in enumerate(sizes):
                k = self._weighted_index([ways[i + 1][r - k] for k in range(min(n, r) + 1)])
                counts.append(k)
                r -= k
            return SelectCasualtiesAction(hb.id, self._ids(classes, counts))

        return draw
