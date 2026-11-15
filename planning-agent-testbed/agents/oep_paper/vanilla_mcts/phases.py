"""The actions of the all-phase OEP paper agents: the purchase, combat move and non-combat move as
sequences of actions, with their transitions, sorting heuristic and keys.

Shared by greedy_action, greedy_turn, vanilla_mcts, ne_mcts, bb_mcts and oep as identical copies
(keep them byte-identical): the agents differ only in how they select among these actions.

  purchase      buy one unit of a candidate rule (land combat units bought with PUs, as
                TripleMind's purchase_actions); rated by attack + defence per PU, END last. A state
                is the units bought so far.
  combat move   one attack per enemy territory: every unit of mine that can reach it; rated:
                attacks with strength > ROLLOUT_STRENGTH_MARGIN x defence in TripleMind priority
                order, then END, then the other attacks. A state is where my moved units stand.
  non-combat    the air the engine would destroy lands first (forced, not an action); then, one
                territory short of cover at a time, its reinforcement sets strongest first and a
                skip (noncombat_generators); once none is left, END plays the positioning moves.
                A state is the territories decided and where the units moved this phase stand.

Phase.actions lists the legal actions highest-rated first (the sorting heuristic). END is always
offered in the purchase and the combat move; in the non-combat move only once every territory is
decided, but Phase.end_of gives the END of any state (there: the positioning moves from it).

A Phase works for a `search` object that provides d (GameDefinition), map (MapGraph), me (player
index), whoAmI (player name), view(sim), apply(sim, action, rng) and move_rng.
"""

import triplea_fm as fm
from triplea_fm.definition import COMBAT_MOVE, NONCOMBAT_MOVE, PURCHASE

from generators import ag  # compiled Cython generators when built, else action_generators.py
import noncombat_generators as ncg

ROLLOUT_STRENGTH_MARGIN = 1.5  # TripleMind's first attack threshold (combat_actions)
END_KEY = "END"  # Phase.action_key of every END


class PhaseState:
    """The forward-model state part-way through a phase, with the phase's own bookkeeping
    (`info`, immutable: units bought, territories attacked, or territories decided + units
    moved)."""
    __slots__ = ("sim", "depth", "info")

    def __init__(self, sim, depth, info):
        self.sim = sim
        self.depth = depth  # actions taken in the phase
        self.info = info

    def clone(self):
        return PhaseState(self.sim.clone(), self.depth, self.info)


class PurchaseUnit:
    """One unit of production rule `rule_id`, or END."""
    __slots__ = ("rule_id", "unit_type", "costs", "strength", "end_phase")

    def __init__(self, rule_id=None, unit_type=None, costs=(), strength=0.0, end_phase=False):
        self.rule_id = rule_id
        self.unit_type = unit_type
        self.costs = costs  # ((resource id, amount), ...)
        self.strength = strength  # attack + defence per PU
        self.end_phase = end_phase


class Phase:
    """One searched phase. `info` of a PhaseState is the phase's own, see the subclasses."""
    name = None
    step_kind = None
    end = None  # the END action, when it does not depend on the state

    def __init__(self, search):
        self.search = search
        self.d = search.d

    def start(self, sim):
        """PhaseState at the start of the phase, and the forced moves applied to reach it."""
        raise NotImplementedError

    def actions(self, state):
        """Legal actions, highest-rated first (END included where it is offered)."""
        raise NotImplementedError

    def step(self, state, action, rng):
        """Apply a non-END action to state in place; False if the forward model rejects it."""
        raise NotImplementedError

    def key(self, state):
        """Transposition key: states reached by different orders of the same actions share it."""
        raise NotImplementedError

    def action_key(self, action):
        """What makes an action the same action in another state (END: END_KEY)."""
        raise NotImplementedError

    def end_of(self, state):
        """The END action of state, offered or not."""
        return self.end

    def finish(self, state, end, rng):
        """Play END (its moves, if any) and end the phase."""
        return self.search.apply(state.sim, fm.EndPhaseAction(), rng)

    def label(self, action):
        raise NotImplementedError

    def levels_left(self, state, actions):
        """Upper bound on the actions still to choose before END: each action attacks or decides
        something the actions offered here."""
        return sum(1 for a in actions if not a.end_phase)

    def response_actions(self, sequence):
        """The forward-model actions the chosen sequence sends to the engine."""
        return [mv for a in sequence for mv in a.moves]


class PurchasePhase(Phase):
    """info: ((rule id, units bought), ...)."""
    name = "purchase"
    step_kind = PURCHASE
    end = PurchaseUnit(end_phase=True)

    def __init__(self, search):
        super().__init__(search)
        self._units = None

    def units(self):
        """Candidate rules, as purchase_actions picks them (land combat units for PUs only), most
        attack + defence per PU first. Frontiers are fixed, so this is computed once."""
        if self._units is None:
            d, m = self.d, self.search.map
            pus = d.resource_names.index("PUs")
            units = []
            for rid in d.frontier[self.search.me]:
                rule = d.production_rules[rid]
                if len(rule.results) != 1 or [r for r, _ in rule.costs] != [pus]:
                    continue
                type_id, qty = rule.results[0]
                cost = rule.costs[0][1]
                ut = m.unit_types.get(d.unit_types[type_id].name)
                if (qty == 1 and cost > 0 and ut and ut.domain == "land"
                        and not ut.is_infrastructure and not ut.is_aa):
                    units.append(PurchaseUnit(rid, ut.name, rule.costs,
                                              (ut.attack + ut.defense) / cost))
            units.sort(key=lambda a: (-a.strength, a.rule_id))
            self._units = units
        return self._units

    def start(self, sim):
        return PhaseState(sim, 0, ()), []

    def actions(self, state):
        d, m, me = self.d, self.search.map, self.search.me
        sim = state.sim
        has_factory = any(
            u.owner == me and u.loc >= 0 and sim.owner[u.loc] == me
            and d.unit_types[u.type].name in m.unit_types
            and m.unit_types[d.unit_types[u.type].name].can_produce
            for u in sim.units.values())
        if not has_factory:
            return [self.end]
        balance = sim.balances[me]
        return [a for a in self.units()
                if all(balance[r] >= amount for r, amount in a.costs)] + [self.end]

    def step(self, state, action, rng):
        if not self.search.apply(state.sim, fm.PurchaseAction(((action.rule_id, 1),)), rng):
            return False
        bought = dict(state.info)
        bought[action.rule_id] = bought.get(action.rule_id, 0) + 1
        state.info = tuple(sorted(bought.items()))
        state.depth += 1
        return True

    def key(self, state):
        return state.info

    def action_key(self, action):
        return END_KEY if action.end_phase else action.rule_id

    def label(self, action):
        return "END" if action.end_phase else f"buy {action.unit_type}"

    def levels_left(self, state, actions):
        """Units still affordable: the most of any one offered rule the balance pays for."""
        balance = state.sim.balances[self.search.me]
        return max((min(balance[r] // amount for r, amount in a.costs)
                    for a in actions if not a.end_phase), default=0)

    def response_actions(self, sequence):
        bought = {}
        for a in sequence:
            if not a.end_phase:
                bought[a.rule_id] = bought.get(a.rule_id, 0) + 1
        return [fm.PurchaseAction(tuple(sorted(bought.items())))] if bought else []


class CombatPhase(Phase):
    """info: frozenset of the territories attacked."""
    name = "combatMove"
    step_kind = COMBAT_MOVE
    end = ag.CombatAction(end_phase=True, strength=0)

    def start(self, sim):
        return PhaseState(sim, 0, frozenset()), []

    def actions(self, state):
        """One attack per enemy target not attacked yet: all my unmoved units that can reach it.
        Rated: attacks beating the defence by ROLLOUT_STRENGTH_MARGIN (TripleMind priority
        order), END, then the other attacks."""
        game = self.search.view(state.sim)
        m = game.map
        me = self.search.whoAmI
        strong, weak = [], []
        for target in ag.enemy_territories_by_priority(game, me):
            if target in state.info:
                continue
            reachable = ag.get_reachable_units(game, target, me)
            if not reachable:
                continue
            action = ag.CombatAction(
                to_terr=target,
                moves=ag._group_moves(self.d, [(u, route) for u, _, route in reachable]),
                strength=ag.calculate_attack_strength(m, [u for u, _, _ in reachable]))
            defence = ag.defender_strength(m, game.territories[target], me)
            (strong if action.strength > ROLLOUT_STRENGTH_MARGIN * defence else weak).append(action)
        return strong + [self.end] + weak

    def step(self, state, action, rng):
        if not all(self.search.apply(state.sim, mv, rng) for mv in action.moves):
            return False
        state.info = state.info | {action.to_terr}
        state.depth += 1
        return True

    def key(self, state):
        me = self.search.me
        return tuple(sorted((u.id, u.loc) for u in state.sim.units.values()
                            if u.owner == me and u.moved))

    def action_key(self, action):
        return END_KEY if action.end_phase else action.to_terr

    def label(self, action):
        return "END" if action.end_phase else f"{action.to_terr} | s={action.strength}"


class NoncombatPhase(Phase):
    """info: (frozenset of the territories decided, frozenset of the unit ids moved this phase)."""
    name = "nonCombatMove"
    step_kind = NONCOMBAT_MOVE

    def start(self, sim):
        """The air the engine would destroy lands first: forced, so not an action."""
        search = self.search
        moves = ncg.land_air_moves(search.view(sim), self.d, search.whoAmI)
        moved = set()
        rng = search.move_rng
        applied = []
        for mv in moves:
            if search.apply(sim, mv, rng):
                moved.update(mv.units)
                applied.append(mv)
        return PhaseState(sim, 0, (frozenset(), frozenset(moved))), applied

    def actions(self, state):
        """Reinforcement sets for the most important territory short of cover, strongest first,
        then its skip; END (with the positioning moves) once no territory is left."""
        excluded, moved = state.info
        actions, _ = ncg.noncombat_actions(self.search.view(state.sim), self.d, excluded, moved,
                                           self.search.whoAmI)
        return actions

    def step(self, state, action, rng):
        moved = set(state.info[1])
        for mv in action.moves:
            if not self.search.apply(state.sim, mv, rng):
                return False
            moved.update(mv.units)
        state.info = (action.excluded, frozenset(moved))
        state.depth += 1
        return True

    def key(self, state):
        excluded, moved = state.info
        units = state.sim.units
        return excluded, tuple(sorted((uid, units[uid].loc) for uid in moved if uid in units))

    def action_key(self, action):
        """The territory and the units sent there (none for its skip)."""
        if action.end_phase:
            return END_KEY
        return action.to_terr, tuple(sorted(u for mv in action.moves for u in mv.units))

    def end_of(self, state):
        """END with the positioning moves from state, also while territories are undecided."""
        excluded, moved = state.info
        search = self.search
        return ncg.NoncombatAction(
            end_phase=True, excluded=excluded,
            moves=ncg.positioning_moves(search.view(state.sim), self.d, search.whoAmI, moved))

    def finish(self, state, end, rng):
        sim = state.sim
        for mv in end.moves:
            self.search.apply(sim, mv, rng)  # a rejected positioning move is skipped, as live
        return self.search.apply(sim, fm.EndPhaseAction(), rng)

    def levels_left(self, state, actions):
        """Territories still short of cover (noncombat_actions decides one per action)."""
        excluded, moved = state.info
        plan = ncg._Plan(self.search.view(state.sim), self.d, self.search.whoAmI, moved)
        return sum(1 for t in plan.my_names if t not in excluded and plan.shortfall(t) > ncg._EPS)

    def label(self, action):
        if action.end_phase:
            return f"END ({len(action.moves)} positioning moves)"
        if not action.moves:
            return f"skip {action.to_terr}"
        return f"{action.to_terr} | s={action.strength:.1f}"


def make_phases(search):
    """{phase name: Phase} for the three searched phases."""
    return {p.name: p for p in (PurchasePhase(search), CombatPhase(search), NoncombatPhase(search))}


def complete_greedily(phase, state, rng, limit=1000):
    """The sorting heuristic's highest-rated actions from state (played on it) until END, END
    included: what a search plays from a state it did not get to choose in."""
    sequence = []
    for _ in range(limit):
        actions = phase.actions(state)
        if actions[0].end_phase or not phase.step(state, actions[0], rng):
            break
        sequence.append(actions[0])
    sequence.append(phase.end_of(state))
    return sequence
