"""TripleMind purchase MCTS with opponent nodes: mcts.MCTS searching the purchase.

A node's children buy one unit each (one production rule, count 1) that the player can still
afford. Buying the same units in another order reaches the same state, so the tree is a DAG: a
transposition table maps (the node where that player's purchase started, units bought so far) to
one shared node, whose visits and value every path through it updates. Backpropagation follows
the path the iteration took, not parent links. When nothing is affordable any more, the phase
ends and the tree passes on to the next player's purchase (opponent nodes); the phases in between
are played by the rollout policies. Rollouts buy epsilon-greedily, one unit at a time: a random
affordable unit with probability rollout_epsilon, else the one with the most attack + defense per
PU.

Candidate units are those of action_generators.purchase_actions: land combat units bought with
PUs only, since the placement policy puts everything at the home factory.
"""

import os
import random
import time

import triplea_fm as fm
from triplea_fm.definition import PURCHASE
from triplea_fm.state import PURCHASE_D

from helper import save_mcts_tree_png
from mcts import MCTS, MCTSGameState, MCTSNode, no_target


class PurchaseUnit:
    """A purchase tree edge: one unit of production rule `rule_id` (end_phase: nothing to buy)."""
    __slots__ = ("rule_id", "unit_type", "costs", "strength", "end_phase")

    def __init__(self, rule_id=None, unit_type=None, costs=(), strength=0.0, end_phase=False):
        self.rule_id = rule_id
        self.unit_type = unit_type
        self.costs = costs  # ((resource id, amount), ...)
        self.strength = strength  # attack + defense per PU
        self.end_phase = end_phase

    def action(self):
        return fm.PurchaseAction(((self.rule_id, 1),))

    def __repr__(self):
        return "purchase: END PHASE" if self.end_phase else f"purchase: {self.unit_type}"


class PurchaseMCTSGameState(MCTSGameState):
    """MCTSGameState whose searched moves are single-unit purchases. `bought` counts the units
    bought this phase per rule id; with the phase's start it identifies the state."""

    def __init__(self, search, sim, current_player, played_by=None):
        super().__init__(search, sim, current_player, played_by)
        self.bought = {}

    def clone(self):
        cloned = PurchaseMCTSGameState(self.search, self.sim.clone(), self.current_player,
                                       self.played_by)
        cloned.actions = list(self.actions)
        cloned.num_actions = len(self.actions)
        cloned.actionIndex = self.actionIndex
        cloned.bought = dict(self.bought)
        return cloned

    def bought_key(self):
        return tuple(sorted(self.bought.items()))

    def legal_moves(self):
        self.actionIndex = 0
        self.actions = self.search.affordable_units(self.sim, self.current_player)
        if not self.actions:
            self.actions = [PurchaseUnit(end_phase=True)]
        self.num_actions = len(self.actions)

    def apply_move(self, action):
        if action.end_phase:
            return
        self.search.apply(self.sim, action.action())
        self.bought[action.rule_id] = self.bought.get(action.rule_id, 0) + 1


class PurchaseNode(MCTSNode):
    """MCTSNode of the purchase DAG. `parent` is the first parent (for tree_level and drawing);
    `anchor` is the node where its player's purchase started, the transposition key's scope."""

    def __init__(self, state, parent=None, action=None, sign=1):
        super().__init__(state, parent, action, sign)
        self.anchor = self
        self.child_actions = {}  # id(child) -> the unit this node buys to reach it

    def action_to(self, child):
        return self.child_actions[id(child)]


class PurchaseMCTS(MCTS):
    search_kind = PURCHASE

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.table = {}  # (id(anchor), units bought) -> PurchaseNode
        self._unit_rules = {}  # player -> [PurchaseUnit] of every candidate rule

    # ---- actions -------------------------------------------------------------------------

    def unit_rules(self, player):
        """The player's candidate rules, as purchase_actions picks them (frontiers are fixed)."""
        if player not in self._unit_rules:
            d, m = self.d, self.map
            pus = d.resource_names.index("PUs")
            units = []
            for rid in d.frontier[d.player_index[player]]:
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
            self._unit_rules[player] = units
        return self._unit_rules[player]

    def affordable_units(self, sim, player):
        """Candidate units the player can pay for now; none without a factory of its own."""
        d, m = self.d, self.map
        p = d.player_index[player]
        has_factory = any(
            u.owner == p and u.loc >= 0 and sim.owner[u.loc] == p
            and m.unit_types[d.unit_types[u.type].name].can_produce
            for u in sim.units.values() if d.unit_types[u.type].name in m.unit_types)
        if not has_factory:
            return []
        balance = sim.balances[p]
        return [a for a in self.unit_rules(player)
                if all(balance[r] >= amount for r, amount in a.costs)]

    def pick_rollout_unit(self, actions):
        """With probability self.rollout_epsilon a random unit, else the most attack + defense per
        PU (ties broken at random)."""
        if random.random() < self.rollout_epsilon:
            return random.choice(actions)
        best = max(a.strength for a in actions)
        return random.choice([a for a in actions if a.strength == best])

    # ---- phases --------------------------------------------------------------------------

    def at_searched_phase(self, sim):
        return sim.decision.kind == PURCHASE_D and self.d.steps[sim.step].kind == PURCHASE

    def start_phase(self, state):
        state.bought = {}
        return []

    def _rollout_purchase(self, current_state):
        """Every purchase of a rollout, the searched player's included: epsilon-greedy units until
        nothing is affordable."""
        sim = current_state.sim
        player = current_state.current_player
        while True:
            actions = self.affordable_units(sim, player)
            if not actions or not self.apply(sim, self.pick_rollout_unit(actions).action()):
                break
        return self.end_phase(sim)

    def _rollout_searched(self, current_state):
        return self._rollout_purchase(current_state)

    # ---- tree ----------------------------------------------------------------------------

    def expand(self, node):
        """Adds node's next untried unit as a child, shared through the transposition table.
        Returns (child, True) for a new node, (child, False) for one already in the table, and
        (node, False) when node has nothing left to try."""
        action = node.state.getNextAction()
        if action is None or action.end_phase:
            node.untried_actions = []
            return node, False
        new_state = node.state.clone()
        new_state.played_by = node.state.next_moves_by
        new_state.apply_move(action)
        key = (id(node.anchor), new_state.bought_key())
        child = self.table.get(key)
        if child is not None:
            node.children.append(child)
            node.child_actions[id(child)] = action
            return child, False
        new_state.legal_moves()
        passes_turn = no_target(new_state)
        if passes_turn:
            self.pass_turn(new_state)
        child = PurchaseNode(new_state, parent=node, action=action,
                             sign=self.sign_of(new_state.played_by))
        child.passes_turn = passes_turn
        if not passes_turn:
            child.anchor = node.anchor
        self.table[key] = child
        node.children.append(child)
        node.child_actions[id(child)] = action
        return child, True

    def mcts_search(self, initial_state):
        """Returns the units to buy, as PurchaseUnits: the most visited path from the root up to
        the node that passes the turn on, completed greedily if the search stopped short of it."""
        initial_state.legal_moves()
        root = PurchaseNode(initial_state)
        self.table = {(id(root), ()): root}

        self.iteration = 0
        self.max_depth = 0
        start_time = time.time()
        while time.time() - start_time < self.time_budget:
            self.iteration += 1
            # Selection and expansion as in MCTS.mcts_search, keeping the path: a node reached
            # through the table has other parents. Reaching a node already in the table adds an
            # edge but no node, so the walk goes on from it.
            node = root
            path = [root]
            while True:
                while node.is_fully_expanded() and not node.is_terminal():
                    best = node.best_child_uct(self.exploration_constant)
                    if best is None:
                        break
                    node = best
                    path.append(node)
                if node.is_terminal():
                    break
                child, new = self.expand(node)
                if child is not node:
                    node = child
                    path.append(node)
                    if new:
                        break  # rollout from the new child
                    continue
                if not node.children:
                    break

            value, depth = self.simulate(node.state)
            self.max_depth = max(self.max_depth, depth)
            for n in path:
                n.visits += 1
                n.value += n.sign * value

        search_time = time.time() - start_time
        self.rollout_stats_metric.flush()
        self.last_root = root
        self.last_stats = {
            "iterations": self.iteration,
            "max_rollout": self.max_depth,
            "seconds": search_time,
        }

        if self.trees_dir is not None:
            os.makedirs(self.trees_dir, exist_ok=True)
            tree_prefix = os.path.join(
                self.trees_dir, f"{self.tree_name}_{self.whoAmI}_g{initial_state.sim.game_number}"
                                f"_r{initial_state.sim.round}")
            try:
                save_mcts_tree_png(root, self.whoAmI, tree_prefix, max_nodes=500, render_img=True)
            except Exception:  # a drawing is never worth losing the turn over
                pass

        units = []
        node = root
        while node.children:
            best_child = max(node.children, key=lambda c: c.visits)
            self.log_choice_stats(initial_state, node, best_child)
            units.append(node.action_to(best_child))
            if best_child.passes_turn:
                break
            node = best_child
        else:
            # The search did not reach the end of the purchase: spend the rest greedily.
            sim = node.state.sim.clone()
            while True:
                actions = self.affordable_units(sim, self.whoAmI)
                if not actions:
                    break
                best = max(actions, key=lambda a: a.strength)
                if not self.apply(sim, best.action()):
                    break
                units.append(best)

        self.efficiency_metric.log(
            self.whoAmI,
            initial_state.sim.game_number,
            initial_state.sim.round,
            search_time,
            self.iteration,
            root.value / root.visits if root.visits > 0 else 0,
            len(root.children),
            self.max_depth,
            sum(1 for n in self.table.values() if not n.children),  # leaves, each counted once
            len(units),  # units bought
            max((n.tree_level for n in self.table.values()), default=0))
        self.action_seq_stats_metric.flush()
        self.table = {}
        return units

    @staticmethod
    def action_label(act):
        return "END" if act.end_phase else f"buy {act.unit_type}"
