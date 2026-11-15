"""Non-exploring MCTS (Justesen et al., "Playing Multiaction Adversarial Games: Online Evolutionary
Planning Versus Tree Search", IEEE ToG 2018, Sec. V-VI) for the TripleA purchase, combat move and
non-combat move: the Vanilla MCTS of ../vanilla_mcts with C = 0 and epsilon = 1.

  * one tree ply is one action, not one turn: a node is the state part-way through the phase, an
    edge (node, action) holds the visits and the value of the descents through it;
  * tree policy, non-exploring: UCT X_j + 2C sqrt(2 ln n / n_j) with C = 0, i.e. the edge with the
    best mean (ties: the sorting heuristic's order); every child of a node is visited once (untried
    actions are expanded first, in the order of the sorting heuristic) before any of them is
    expanded further; an END edge ends the descent;
  * default policy, epsilon-greedy with epsilon = 1: always the action the sorting heuristic rates
    highest. Battles roll dice, so the dice are fixed for the search: every rollout of one search
    uses the same seed, which makes rollouts fully deterministic, as C = 0 needs (one visit values
    a child). A rollout is therefore computed once per (node, END) and its reward reused;
  * rollouts are depth-limited to one turn: the rest of the phase, then my remaining phases (each
    searched phase with the same default policy, battles with default casualties and no retreat,
    placement at the home factory) until my turn ends, rated with the heuristic (evaluate_state);
  * transposition table: a state reached by different orders of the same actions is one node,
    but backpropagation follows the descent path only;
  * backpropagation, multi-action BackupNegamax (paper's Algorithm 2): every edge on the path gets
    +delta if its player is the max player (whoAmI), otherwise -delta. Within one phase every edge
    is whoAmI's, so this is +delta throughout; the rule is kept so the tree stays correct if a
    phase ever hands the move over.

After the time budget the tree is followed from the root along the most visited edges (ties: best
mean) until END; a path that stops at a node the search never expanded is completed with the
sorting heuristic's highest-rated actions.

Actions per phase (each phase's sorting heuristic lists them highest-rated first):

  purchase      buy one unit of a candidate rule (land combat units bought with PUs, as
                TripleMind's purchase_actions); rated by attack + defence per PU, END last. A node
                is the units bought so far.
  combat move   one attack per enemy territory: every unit of mine that can reach it (as the other
                OEP agents); rated: attacks with strength > 1.5 x defence in TripleMind priority
                order, then END, then the other attacks. A node is where my moved units stand.
  non-combat    the air the engine would destroy lands first (forced, not searched); then, one
                territory short of cover at a time, reinforcement sets strongest first and a skip
                (noncombat_generators); when none is left, END plays the positioning moves. A node
                is the territories decided and where the units moved this phase stand.
"""

import math
import os
import random
import time

import triplea_fm as fm
from triplea_fm import toolkit
from triplea_fm.actions import CONTINUE, RetreatAction, SelectBattleAction
from triplea_fm.definition import (BATTLE, COMBAT_MOVE, END_ROUND, END_TURN, NONCOMBAT_MOVE,
                                   PLACE, PURCHASE)
from triplea_fm.state import BATTLE_SELECTION_D, CASUALTIES_D, RETREAT_D

from generators import ag  # compiled Cython generators when built, else action_generators.py
import map_graph
import noncombat_generators as ncg
from helper import EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, ACTION_SEQ_HEADER, MetricLogger

# TripleMind's columns first (the plot scripts read them by name), then the columns the other OEP
# agents share, then this agent's own.
NE_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["reached_end_of_turn", "num_actions_per_level",
                                            "best_value_per_level", "states_visited",
                                            "transpositions", "phase"]
NE_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "rollout_actions", "phase"]
NE_ACTION_SEQ_HEADER = ACTION_SEQ_HEADER + ["value_gap", "end_phase_value", "states_visited",
                                            "level_seconds", "phase"]
SIBLING_VALUES_HEADER = ["player", "game", "round", "tl", "action", "chosen", "samples",
                         "mean_value", "std_value", "diff_to_best", "phase"]

ROLLOUT_STRENGTH_MARGIN = 1.5  # TripleMind's first attack threshold (combat_actions)
MAX_TURN_STEPS = 50  # guard for the rollout's walk through the rest of the turn


# =========================
# Tree
# =========================

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


class Node:
    """A state of the phase, shared by every order of actions that reaches it."""
    __slots__ = ("state", "player", "actions", "edges", "visits")

    def __init__(self, state, player):
        self.state = state
        self.player = player  # who chooses the action here
        self.actions = None  # rated actions, END included, generated on the first visit
        self.edges = []  # one per expanded action, in self.actions order
        self.visits = 0  # descents through this node, from any parent

    def fully_expanded(self):
        return self.actions is not None and len(self.edges) == len(self.actions)


class Edge:
    """(node, action): visits and value of the descents through it, from `player`'s side."""
    __slots__ = ("action", "child", "player", "visits", "value", "value_sq")

    def __init__(self, action, child, player):
        self.action = action
        self.child = child  # None for END
        self.player = player
        self.visits = 0
        self.value = 0.0
        self.value_sq = 0.0

    @property
    def mean(self):
        return self.value / self.visits if self.visits else -math.inf

    @property
    def std(self):
        if not self.visits:
            return 0.0
        m = self.value / self.visits
        return math.sqrt(max(self.value_sq / self.visits - m * m, 0.0))


# =========================
# Phases: actions (sorting heuristic order), transitions, keys
# =========================

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
        """Legal actions, END included, highest-rated first."""
        raise NotImplementedError

    def step(self, state, action, rng):
        """Apply a non-END action to state in place; False if the forward model rejects it."""
        raise NotImplementedError

    def key(self, state):
        raise NotImplementedError

    def finish(self, state, end, rng):
        """Play END (its moves, if any) and end the phase."""
        return self.search.apply(state.sim, fm.EndPhaseAction(), rng)

    def label(self, action):
        raise NotImplementedError

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

    def label(self, action):
        return "END" if action.end_phase else f"buy {action.unit_type}"

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

    def label(self, action):
        return "END" if action.end_phase else f"{action.to_terr} | s={action.strength}"


class NoncombatPhase(Phase):
    """info: (frozenset of the territories decided, frozenset of the unit ids moved this phase)."""
    name = "nonCombatMove"
    step_kind = NONCOMBAT_MOVE

    def start(self, sim):
        """The air the engine would destroy lands first: forced, so not a tree ply."""
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

    def finish(self, state, end, rng):
        sim = state.sim
        for mv in end.moves:
            self.search.apply(sim, mv, rng)  # a rejected positioning move is skipped, as live
        return self.search.apply(sim, fm.EndPhaseAction(), rng)

    def label(self, action):
        if action.end_phase:
            return f"END ({len(action.moves)} positioning moves)"
        if not action.moves:
            return f"skip {action.to_terr}"
        return f"{action.to_terr} | s={action.strength:.1f}"


def end_action(actions):
    return next(a for a in actions if a.end_phase)


# =========================
# Search
# =========================

class NEMCTS:
    def __init__(self, definition, map_graph_, whoAmI, time_budget, seed=0, metrics_dir="metrics"):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI
        self.me = definition.player_index[whoAmI]
        self.time_budget = time_budget  # seconds per searched decision
        self.rnd = random.Random(seed)
        self.move_rng = self.new_rng()  # moves and purchases roll no dice
        self.dice_seed = self.rnd.getrandbits(62)  # every rollout's dice, redrawn per search
        self.rewards = {}  # (id(node), took END) -> rollout reward, per search
        self.phases = {p.step_kind: p for p in (PurchasePhase(self), CombatPhase(self),
                                                NoncombatPhase(self))}
        self.by_name = {p.name: p for p in self.phases.values()}
        self.states_visited = 0
        self.iteration = 0
        self.last_stats = None

        def logger(name, header):
            return MetricLogger(os.path.join(metrics_dir, name), header=header)

        # efficiency.csv holds the combat-move searches only, so it compares row for row with the
        # other OEP agents (combat search only); the other phases have their own files.
        self.efficiency_metric = {
            "combatMove": logger("efficiency.csv", NE_EFFICIENCY_HEADER),
            "purchase": logger("efficiency_purchase.csv", NE_EFFICIENCY_HEADER),
            "nonCombatMove": logger("efficiency_noncombat.csv", NE_EFFICIENCY_HEADER),
        }
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = logger("rollout_stats.csv", NE_ROLLOUT_STATS_HEADER)
        self.action_seq_stats_metric = logger("action_seq_stats.csv", NE_ACTION_SEQ_HEADER)
        self.sibling_metric = logger("sibling_values.csv", SIBLING_VALUES_HEADER)

    # ---- forward model -------------------------------------------------------------------

    def apply(self, sim, action, rng):
        return fm.apply_in_place(self.d, sim, action, rng) != fm.REJECTED

    def view(self, sim):
        return map_graph.GameState.from_sim(self.map, self.d, sim, self.whoAmI)

    def new_rng(self):
        return fm.SeededRng(self.rnd.getrandbits(62))

    def actor(self, sim):
        return self.d.steps[sim.step].actor

    # ---- default policy ------------------------------------------------------------------

    def default_action(self, actions):
        """epsilon-greedy with epsilon = 1: always the highest-rated action."""
        return actions[0]

    def play_phase(self, phase, state, rng):
        """Rest of the phase with the default policy, then END. Returns the actions played."""
        played = 0
        while True:
            actions = phase.actions(state)
            action = self.default_action(actions)
            if action.end_phase:
                return played, phase.finish(state, action, rng)
            if not phase.step(state, action, rng):
                # rejected: stop the phase here
                return played, phase.finish(state, end_action(phase.actions(state)), rng)
            played += 1

    def play_rest_of_turn(self, sim, rng):
        """My remaining phases after the current one, until my turn ends. Returns the actions the
        default policy played in the searched phases."""
        played = 0
        for _ in range(MAX_TURN_STEPS):
            if sim.terminal is not None or sim.decision is None:
                break
            step = self.d.steps[sim.step]
            if step.actor != self.me or step.kind in (END_TURN, END_ROUND):
                break
            before = (sim.step, sim.round)
            phase = self.phases.get(step.kind)
            if phase is not None:
                state, _ = phase.start(sim)
                n, ok = self.play_phase(phase, state, rng)
                played += n
            elif step.kind == BATTLE:
                ok = self.fight_battles(sim, rng)
            elif step.kind == PLACE:
                ok = self.place_at_home(sim, rng)
            else:
                ok = self.apply(sim, fm.EndPhaseAction(), rng)
            if not ok or (sim.step, sim.round) == before:
                break
        return played

    def fight_battles(self, sim, rng, max_steps=2000):
        """The battle phase as the live adapter plays it: every battle in order, default
        casualties, never retreat."""
        steps = 0
        ok = True
        while (sim.terminal is None and sim.decision is not None and steps < max_steps
               and self.d.steps[sim.step].kind == BATTLE):
            steps += 1
            kind = sim.decision.kind
            if kind == BATTLE_SELECTION_D:
                pending = [b.id for b in sim.battles.values()
                           if b.attacker == sim.decision.actor and b.status == "PENDING"]
                ok = (self.apply(sim, SelectBattleAction(min(pending)), rng) if pending
                      else self.apply(sim, fm.EndPhaseAction(), rng))
            elif kind == CASUALTIES_D:
                ok = self.apply(sim, toolkit.default_casualties(self.d, sim), rng)
            elif kind == RETREAT_D:
                ok = self.apply(sim, RetreatAction(CONTINUE), rng)
            else:
                ok = self.apply(sim, fm.EndPhaseAction(), rng)
            if not ok:
                break
        return ok

    def place_at_home(self, sim, rng):
        """Everything bought goes to the home factory (TripleMind's placement policy)."""
        held = sim.held_units(sim.decision.actor)
        home = ag._home_factory(self.view(sim), self.whoAmI)
        if held and home is not None:
            territory = self.d.territory_index[home]
            if not self.apply(sim, fm.PlaceAction(tuple(u.id for u in held), territory), rng):
                for u in held:  # over the production limit: place what fits
                    if not self.apply(sim, fm.PlaceAction((u.id,), territory), rng):
                        break
        return self.apply(sim, fm.EndPhaseAction(), rng)

    def rollout(self, phase, node, end, game_number, round_):
        """Reward of node's state (from whoAmI's side): the rest of the phase (just `end` when
        the descent took an END edge) and the rest of my turn with the default policy, the
        search's fixed dice, rated with evaluate_state. Deterministic, so computed once per
        (node, end) and cached for the search."""
        cache_key = (id(node), end is not None)
        if cache_key in self.rewards:
            return self.rewards[cache_key]
        state = node.state.clone()
        rng = fm.SeededRng(self.dice_seed)
        try:
            if end is not None:
                played, ok = 0, phase.finish(state, end, rng)
            else:
                played, ok = self.play_phase(phase, state, rng)
            if ok:
                played += self.play_rest_of_turn(state.sim, rng)
            reward, terminal = evaluate_state(self, state.sim, self.whoAmI)
        except Exception:
            reward, terminal = -0.5, False
        self.states_visited += 1
        self.rollout_stats_metric.buffer(self.whoAmI, game_number, round_, self.iteration, 0,
                                         reward, 1 if terminal else 0, node.state.depth + 1,
                                         played, phase.name)
        self.rewards[cache_key] = reward
        return reward

    # ---- tree policy ---------------------------------------------------------------------

    def iterate(self, phase, root, table, game_number, round_, stats):
        """One MCTS iteration: selection, expansion, rollout, BackupNegamax."""
        self.iteration += 1
        node, path, nodes, end = root, [], [root], None
        # --- selection: UCT with C = 0, the best-mean edge, once every child has been visited ---
        while True:
            if node.actions is None:
                node.actions = phase.actions(node.state)
            if not node.fully_expanded():
                break
            edge = max(node.edges, key=lambda e: e.mean)  # ties: the highest-rated action
            path.append(edge)
            if edge.child is None:
                end = edge.action
                break
            node = edge.child
            nodes.append(node)
        # --- expansion: the node's next untried action, in sorting-heuristic order ---
        if end is None:
            action = node.actions[len(node.edges)]
            if action.end_phase:
                edge = Edge(action, None, node.player)
                end = action
            else:
                state = node.state.clone()
                if not phase.step(state, action, self.move_rng):
                    node.actions.remove(action)  # rejected by the forward model
                    return
                key = phase.key(state)
                child = table.get(key)
                if child is None:
                    child = Node(state, self.actor(state.sim))
                    table[key] = child
                    stats["max_depth"] = max(stats["max_depth"], state.depth)
                else:
                    stats["transpositions"] += 1
                edge = Edge(action, child, node.player)
            node.edges.append(edge)
            path.append(edge)
            if end is None:
                node = edge.child
                nodes.append(node)
        # --- rollout ---
        delta = self.rollout(phase, node, end, game_number, round_)
        # --- BackupNegamax (Algorithm 2), along the descent path only ---
        for n in nodes:
            n.visits += 1
        for e in path:
            signed = delta if e.player == self.me else -delta
            e.visits += 1
            e.value += signed
            e.value_sq += signed * signed

    # ---- search --------------------------------------------------------------------------

    def search(self, phase_name, root_sim):
        """The phase's chosen actions, as forward-model actions for the engine (forced ones
        first)."""
        phase = self.by_name[phase_name]
        started = time.time()
        deadline = started + self.time_budget
        game_number, round_ = root_sim.game_number, root_sim.round
        self.states_visited = 0
        self.iteration = 0
        self.move_rng = self.new_rng()
        self.dice_seed = self.rnd.getrandbits(62)  # the same dice for every rollout of this search
        self.rewards = {}

        root_state, forced = phase.start(root_sim.clone())
        root = Node(root_state, self.actor(root_state.sim))
        root.actions = phase.actions(root.state)
        table = {phase.key(root.state): root}
        stats = {"transpositions": 0, "max_depth": 0}
        if len(root.actions) > 1:  # a single action (END, or one forced option) is no choice
            while time.time() < deadline:
                self.iterate(phase, root, table, game_number, round_, stats)

        # --- the most visited path, completed with the highest-rated actions ---
        sequence, levels = [], []
        node, reached_end = root, False
        while node is not None:
            if node.edges:
                best = max(node.edges, key=lambda e: (e.visits, e.mean))
                if len(node.edges) > 1:
                    levels.append((node, best))
                sequence.append(best.action)
                if best.child is None:
                    reached_end = True
                    break
                node = best.child
                continue
            # never expanded: the sorting heuristic from here on, on a copy of the state
            state = node.state.clone()
            for _ in range(1000):
                actions = phase.actions(state)
                if actions[0].end_phase:
                    sequence.append(actions[0])
                    break
                if not phase.step(state, actions[0], self.move_rng):
                    break
                sequence.append(actions[0])
            break

        for tl, (n, best) in enumerate(levels, 1):
            self.log_level(phase, game_number, round_, tl, n, best)
        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        chosen = [a for a in sequence if not a.end_phase]
        self.last_stats = {"phase": phase.name, "iterations": self.iteration, "nodes": len(table),
                           "transpositions": stats["transpositions"],
                           "reached_end_of_turn": reached_end, "seconds": round(search_time, 3)}
        visits = sum(e.visits for e in root.edges)
        self.efficiency_metric[phase.name].log(
            self.whoAmI,
            game_number,
            round_,
            search_time,
            self.iteration,  # num_iterations
            sum(e.value for e in root.edges) / visits if visits else "",  # root_value
            len(root.edges),  # num_children_at_root
            0,  # max_rollout: rollouts end with this turn (0 rounds)
            count_leaves(table),  # num_action_sequences: sequences at the tree's frontier
            len(chosen),  # num_of_terr_attacked: actions chosen (attacks, units, reinforcements)
            stats["max_depth"],  # max_tree_depth: actions in the deepest state in the tree
            1 if reached_end else 0,  # the most visited path ends with an END edge
            ";".join(str(len(n.edges)) for n, _ in levels),
            ";".join(f"{b.mean:.4f}" for _, b in levels),
            self.states_visited,
            stats["transpositions"],
            phase.name)
        self.last_sequence = [phase.label(a) for a in sequence]
        return forced + phase.response_actions(sequence)

    def log_level(self, phase, game_number, round_, tl, node, best):
        """One action_seq_stats row for the level and one sibling_values row per edge."""
        edges = node.edges
        by_value = sorted(edges, key=lambda e: e.mean, reverse=True)
        by_value.remove(best)
        second = by_value[0] if by_value else None
        ranked = [best] + by_value
        end = next((e for e in edges if e.child is None), None)
        visits = [e.visits for e in edges]
        means = [e.mean if e.visits else 0.0 for e in edges]
        mean_v, var_v = _mean_var(visits)
        mean_a, var_a = _mean_var(means)
        z = lambda x, m, var: (x - m) / math.sqrt(var) if var > 1e-12 else ""  # blank if all equal
        self.action_seq_stats_metric.buffer(
            self.whoAmI, game_number, round_, tl, self.whoAmI, phase.label(best.action),
            len(edges), best.visits, best.visits / sum(visits) if sum(visits) else 0.0,
            second.visits if second else "", mean_v, var_v, math.sqrt(var_v),
            z(best.visits, mean_v, var_v),
            best.mean, second.mean if second else "", mean_a, math.sqrt(var_a),
            z(best.mean, mean_a, var_a),
            best.mean - second.mean if second else "",
            end.mean if end else "", node.visits, "", phase.name)
        for e in ranked:
            self.sibling_metric.buffer(self.whoAmI, game_number, round_, tl,
                                       phase.label(e.action), 1 if e is best else 0, e.visits,
                                       e.mean, e.std, best.mean - e.mean, phase.name)


def count_leaves(table):
    """Sequences at the tree's frontier: END edges plus edges into nodes never expanded."""
    leaves = 0
    for node in table.values():
        for e in node.edges:
            if e.child is None or not e.child.edges:
                leaves += 1
    return leaves


def _mean_var(xs):
    """Mean and population variance of a non-empty list."""
    m = sum(xs) / len(xs)
    return m, sum((x - m) ** 2 for x in xs) / len(xs)


# =========================
# State evaluation: TripleMind's evaluate_state (mcts.py), as the other OEP agents use it: on a
# bare forward-model state, without the rollout-depth discount and scaled to [-1, 1]:
# 1 / -1 for a won / lost game, 2 * (0.5 TUV share + 0.5 income share) - 1 otherwise.
# =========================

def evaluate_state(search, sim, whoAmI):
    me = whoAmI

    terminal = sim.terminal
    if terminal is not None:
        d = search.d
        if d.alliance_of[d.player_index[me]] in terminal.winners:
            return 1.0, True
        return -1.0, True

    game = search.view(sim)
    m = game.map
    territories = game.territories

    # production counts the territories each player owns now (not the PU balance, which lags
    # captures and includes unspent PUs)
    my_tuv = 0.0
    enemy_tuv = 0.0
    income = 0
    income_denom = 0
    nodes = m.G.nodes
    for name, terr in territories.items():
        if terr.owner != "Neutral":
            prod = nodes[name]["production"]
            income_denom += prod
            if terr.owner == me:
                income += prod
        for u in terr.units:
            ut = m.unit_types.get(u.unit_type)
            if ut and ut.is_infrastructure:
                continue
            cost = m.unit_cost(u.unit_type) or 0
            if u.owner == me:
                my_tuv += cost
            elif u.owner != "Neutral":
                enemy_tuv += cost

    tuv_denom = my_tuv + enemy_tuv
    if tuv_denom < 1e-9:
        tuv_denom = 1e-9
    tuv_term = my_tuv / tuv_denom

    income_term = income / income_denom if income_denom > 0 else 0

    score = 0.50 * tuv_term + 0.50 * income_term  # in [0, 1]
    return 2.0 * score - 1.0, False  # in [-1, 1], the range of the terminal values
