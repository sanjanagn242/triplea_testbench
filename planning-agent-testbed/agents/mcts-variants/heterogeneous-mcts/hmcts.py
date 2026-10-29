"""Heterogeneous MCTS: one search tree over the purchase, combat move and non-combat move of every
player, run on the forward model.

The search is called for each purchase, combat-move and non-combat-move request (never for
place). Its root is the requested decision; a node's actions are generated from the decision its
state is at, so the same tree holds edges of every searched kind:

  purchase        one unit each (one production rule, count 1) the player can still afford.
                  Units are offered in candidate order from the last unit bought, so every
                  multiset of units has exactly one path (no transpositions).
  combat move     one attack (or skip) per target territory, action_generators.combat_actions.
  non-combat move one reinforcement (or skip) per territory short of cover,
                  noncombat_generators.noncombat_actions; the forced air landings are played
                  when the phase starts and the positioning moves when it ends.

When a node's phase has nothing left to decide, expand() ends it and plays the decisions in
between (battles, place, end of turn) with the rollout policies up to the next searched decision,
whoever's it is: the tree goes on through this player's later phases and the opponents' (opponent
nodes). Every node is valued by the same reward, evaluate_state, from whoAmI's point of view;
each node keeps it from the view of the player who chose its edge, so UCT maximises for whoever
chooses at the parent.

Rollouts finish the leaf's phase with its own policy, then play every decision with the rollout
policies until depth_budget rounds have ended: depth_budget 1 stops when the round the leaf is in
is over.
"""

import math
import os
import random
import time

import triplea_fm as fm
from triplea_fm import toolkit
from triplea_fm.actions import CONTINUE, RetreatAction, SelectBattleAction
from triplea_fm.definition import COMBAT_MOVE, NONCOMBAT_MOVE, PURCHASE
from triplea_fm.state import (BATTLE_SELECTION_D, CASUALTIES_D, MOVE_D, PLACE_D, PURCHASE_D,
                              RETREAT_D)

from generators import ag  # compiled Cython generators when built, else action_generators.py
import map_graph
import noncombat_generators as ncg
from helper import (EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, ACTION_SEQ_HEADER, MetricLogger,
                    save_mcts_tree_png)

# The searched decisions, by the testbench's phase names.
PURCHASE_PHASE, COMBAT_PHASE, NONCOMBAT_PHASE = "purchase", "combatMove", "nonCombatMove"
SEARCHED_PHASES = (PURCHASE_PHASE, COMBAT_PHASE, NONCOMBAT_PHASE)
# Metric file suffix and tree file name per root phase, as the all-phases-mcts agent names them,
# so the files stay comparable.
METRICS_SUFFIX = {COMBAT_PHASE: "", NONCOMBAT_PHASE: "_noncombat", PURCHASE_PHASE: "_purchase"}
TREE_NAME = {COMBAT_PHASE: "tree", NONCOMBAT_PHASE: "tree_noncombat",
             PURCHASE_PHASE: "tree_purchase"}


class PurchaseUnit:
    """A purchase tree edge: one unit of production rule `rule_id` (end_phase: nothing to buy).
    `index` is its place in the player's candidate list (HeterogeneousMCTS.unit_rules)."""
    __slots__ = ("rule_id", "unit_type", "costs", "strength", "index", "end_phase")

    def __init__(self, rule_id=None, unit_type=None, costs=(), strength=0.0, index=0,
                 end_phase=False):
        self.rule_id = rule_id
        self.unit_type = unit_type
        self.costs = costs  # ((resource id, amount), ...)
        self.strength = strength  # attack + defense per PU
        self.index = index
        self.end_phase = end_phase

    def action(self):
        return fm.PurchaseAction(((self.rule_id, 1),))

    def __repr__(self):
        return "purchase: END PHASE" if self.end_phase else f"purchase: {self.unit_type}"


class HMCTSGameState:
    """A forward-model state plus the bookkeeping of the searched phase it is in. `phase` is the
    searched decision the sim is at (None when it is at none: game over, or the model rejected an
    action on the way)."""

    def __init__(self, search, sim, current_player, played_by=None):
        self.search = search
        self.sim = sim
        self.current_player = current_player
        # The player whose move led to this state: a tree node's value is from its point of view.
        # The root's state was reached by no searched move; it counts as its own player's.
        self.played_by = played_by if played_by is not None else current_player
        self.round = sim.round
        self.phase = search.phase_of(sim)

        self.actions = []  # generated actions of the current decision
        self.num_actions = 0
        self.actionIndex = 0  # next action expand() tries
        self.excluded = set()  # combat / non-combat move: territories already decided
        self.moved = set()  # non-combat move: forward-model ids of the units moved this phase
        self.next_unit = 0  # purchase: candidates before this index are not offered

    def clone(self):
        cloned = HMCTSGameState.__new__(HMCTSGameState)
        cloned.__dict__.update(self.__dict__)
        cloned.sim = self.sim.clone()
        cloned.actions = list(self.actions)
        cloned.excluded = set(self.excluded)
        cloned.moved = set(self.moved)
        return cloned

    @property
    def next_moves_by(self):
        """The player to move in this state: a tree node's actions (children) are this player's."""
        return self.current_player

    def view(self):
        """map_graph.GameState of the forward-model state, seen by the current player."""
        return map_graph.GameState.from_sim(self.search.map, self.search.d, self.sim,
                                            self.current_player)

    # game over (forward-model victory conditions)
    def is_terminal(self):
        return self.sim.terminal is not None

    def legal_moves(self):
        """Generates the current decision's actions; a phase with nothing left to decide offers
        only its end (no_target)."""
        self.actionIndex = 0
        search, player = self.search, self.current_player
        if self.phase == PURCHASE_PHASE:
            self.actions = (search.affordable_units(self.sim, player, self.next_unit)
                            or [PurchaseUnit(end_phase=True)])
        elif self.phase == COMBAT_PHASE:
            actions, self.excluded = ag.combat_actions(self.view(), search.d, self.excluded,
                                                       player)
            self.actions = actions or [ag.CombatAction(end_phase=True, strength=0)]
        elif self.phase == NONCOMBAT_PHASE:
            self.actions, self.excluded = ncg.noncombat_actions(
                self.view(), search.d, self.excluded, self.moved, player)
        else:
            self.actions = []
        self.num_actions = len(self.actions)

    def apply_moves(self, moves):
        for mv in moves:
            self.search.apply(self.sim, mv)
            self.moved.update(mv.units)

    def apply_move(self, action):
        if action.end_phase:
            return
        if self.phase == PURCHASE_PHASE:
            self.search.apply(self.sim, action.action())
            self.next_unit = action.index
        else:
            # Moves roll no dice; the battles are fought in the battle phase.
            self.apply_moves(action.moves)

    def getNextAction(self):
        if self.actionIndex < len(self.actions):
            action = self.actions[self.actionIndex]
            self.actionIndex += 1
            return action
        return None


class MCTSNode:
    def __init__(self, state, parent=None, action=None, sign=1):
        self.state = state
        self.parent = parent
        self.action = action
        # +1 / -1 for whether state.played_by (who picked `action`) is on whoAmI's side. value is
        # kept from that player's point of view, so best_child_uct maximises for whoever chooses
        # at the parent (the parent's state.next_moves_by).
        self.sign = sign
        self.children = []
        self.untried_actions = None
        self.visits = 0
        self.value = 0.0
        self.is_state_terminal = None
        self.tree_level = parent.tree_level + 1 if parent else 0  # edges from the root
        # True when `action` was the chooser's last decision of its phase: expand() then advanced
        # the state to the next searched decision, so this node is that decision.
        self.passes_phase = False

    def action_to(self, child):
        """The action of the edge from this node to child."""
        return child.action

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def is_terminal(self):
        if self.is_state_terminal is None:
            self.is_state_terminal = self.state.is_terminal()
        return self.is_state_terminal

    def best_child_uct(self, c_param=1.414):
        # in case there is still an unvisited child, prioritize it to ensure exploration
        unvisited = [child for child in self.children if child.visits == 0]
        if unvisited:
            return random.choice(unvisited)
        if not self.children:
            return None
        log_visits = math.log(self.visits)
        weights = [child.value / child.visits + c_param * math.sqrt(log_visits / child.visits)
                   for child in self.children]
        max_wt = max(weights)
        best_nodes = [child for child, w in zip(self.children, weights) if w == max_wt]
        # ties: the action with the highest strength
        return max(best_nodes, key=lambda node: node.action.strength)


class HeterogeneousMCTS:
    def __init__(self, definition, map_graph_, whoAmI, time_budget, depth_budget,
                 exploration_constant, metrics_dir="metrics", trees_dir=None,
                 rollout_epsilon=0.1):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI

        # MCTS parameters
        self.time_budget = time_budget  # seconds per decision
        self.depth_budget = depth_budget  # rollout horizon (rounds)
        self.exploration_constant = exploration_constant  # UCB1 exploration parameter
        # random-move probability of the combat-move and purchase rollouts
        self.rollout_epsilon = rollout_epsilon

        # Each search's tree is saved as <trees_dir>/<TREE_NAME[phase]>_<faction>_g<game>_r<round>
        # .dot / .html / .png (helper.save_mcts_tree_png); None saves nothing.
        self.trees_dir = trees_dir

        self.iteration = 0
        self.max_depth = 0
        self.last_root = None
        self.last_stats = None
        self.last_choice = []  # the tree edges the latest search chose
        self._rng = fm.SeededRng(random.getrandbits(62))
        self._unit_rules = {}  # player -> [PurchaseUnit] of every candidate rule

        # Metric files per root phase. The rollout rows are buffered during a search and written
        # once at its end, to keep file I/O out of the loop.
        self.metrics = {
            phase: (MetricLogger(os.path.join(metrics_dir, f"efficiency{suffix}.csv"),
                                 header=EFFICIENCY_HEADER),
                    MetricLogger(os.path.join(metrics_dir, f"rollout_stats{suffix}.csv"),
                                 header=ROLLOUT_STATS_HEADER),
                    MetricLogger(os.path.join(metrics_dir, f"action_seq_stats{suffix}.csv"),
                                 header=ACTION_SEQ_HEADER))
            for phase, suffix in METRICS_SUFFIX.items()}
        (self.efficiency_metric, self.rollout_stats_metric,
         self.action_seq_stats_metric) = self.metrics[COMBAT_PHASE]

    # ---- forward model -------------------------------------------------------------------

    def apply(self, sim, action):
        """Apply an action with the forward model; False if the model rejects it."""
        return fm.apply_in_place(self.d, sim, action, self._rng) != fm.REJECTED

    def end_phase(self, sim):
        return self.apply(sim, fm.EndPhaseAction())

    def phase_of(self, sim):
        """The searched decision sim is at, or None."""
        if sim.terminal is not None:
            return None
        kind, step_kind = sim.decision.kind, self.d.steps[sim.step].kind
        if kind == PURCHASE_D and step_kind == PURCHASE:
            return PURCHASE_PHASE
        if kind == MOVE_D and step_kind == COMBAT_MOVE:
            return COMBAT_PHASE
        if kind == MOVE_D and step_kind == NONCOMBAT_MOVE:
            return NONCOMBAT_PHASE
        return None

    def sign_of(self, player):
        """+1 if `player` is in whoAmI's alliance, else -1."""
        d = self.d
        mine = d.alliance_of[d.player_index[self.whoAmI]]
        return 1 if d.alliance_of[d.player_index[player]] == mine else -1

    # ---- purchase candidates -------------------------------------------------------------

    def unit_rules(self, player):
        """The player's candidate rules, as purchase_actions picks them: land combat units bought
        with PUs only, since the placement policy puts everything at the home factory (frontiers
        are fixed)."""
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
                                              (ut.attack + ut.defense) / cost, len(units)))
            self._unit_rules[player] = units
        return self._unit_rules[player]

    def affordable_units(self, sim, player, first=0):
        """Candidate units from index `first` on that the player can pay for now; none without a
        factory of its own."""
        d, m = self.d, self.map
        p = d.player_index[player]
        has_factory = any(
            u.owner == p and u.loc >= 0 and sim.owner[u.loc] == p
            and m.unit_types[d.unit_types[u.type].name].can_produce
            for u in sim.units.values() if d.unit_types[u.type].name in m.unit_types)
        if not has_factory:
            return []
        balance = sim.balances[p]
        return [a for a in self.unit_rules(player)[first:]
                if all(balance[r] >= amount for r, amount in a.costs)]

    # ---- phases --------------------------------------------------------------------------

    def start_phase(self, state):
        """state has reached a searched decision: it now belongs to that decision's player, with
        nothing decided yet. A non-combat move first lands the air the engine would destroy
        (forced). Returns the moves it applied."""
        sim = state.sim
        state.phase = self.phase_of(sim)
        state.current_player = self.d.player_names[sim.decision.actor]
        state.round = sim.round
        state.excluded = set()
        state.moved = set()
        state.next_unit = 0
        state.actions, state.num_actions, state.actionIndex = [], 0, 0
        if state.phase != NONCOMBAT_PHASE:
            return []
        moves = ncg.land_air_moves(state.view(), self.d, state.current_player)
        state.apply_moves(moves)
        return moves

    def finish_phase(self, state):
        """Ends state's searched phase; a non-combat move plays its positioning moves first."""
        if state.phase == NONCOMBAT_PHASE:
            state.apply_moves(self.positioning_moves(state))
        return self.end_phase(state.sim)

    def positioning_moves(self, state):
        """The end-of-phase moves of state's non-combat move (those of its END action, when it
        has been generated)."""
        end = next((a for a in state.actions if a.end_phase), None)
        return (end.moves if end is not None else
                ncg.positioning_moves(state.view(), self.d, state.current_player, state.moved))

    def advance(self, state):
        """Ends state's searched phase and plays on with the rollout policies to the next searched
        decision, whoever's it is, and starts it. False if the game ended or the model rejected
        an action on the way."""
        sim = state.sim
        ok = self.finish_phase(state)
        steps = 0
        while ok and not state.is_terminal() and steps < 5000:
            steps += 1
            if self.phase_of(sim) is not None:
                self.start_phase(state)
                return True
            ok = self._play_decision(state)
        state.phase = None
        state.round = sim.round
        return False

    def pass_phase(self, state):
        """state's phase has nothing left to decide: advance to the next searched decision with
        something to decide, so the node holding state is that decision. A decision with nothing
        to decide is passed over too, for at most one round's steps (then state stays at a
        decision with only END and the node is a leaf). If the game ends or the model rejects an
        action on the way, state.actions is empty and the node is a leaf."""
        for _ in range(len(self.d.steps)):
            try:
                at_searched = self.advance(state)
                if at_searched:
                    state.legal_moves()
            except Exception:
                at_searched = False
            if not at_searched:
                state.actions, state.num_actions, state.actionIndex = [], 0, 0
                return
            if not no_target(state):
                return

    # ---- rollout policies ----------------------------------------------------------------

    def pick_rollout_move_epsilon_greedy(self, actions):
        """With probability self.rollout_epsilon a random action, else the strongest one (ties
        broken at random)."""
        if random.random() < self.rollout_epsilon:
            return random.choice(actions)
        best = max(a.strength for a in actions)
        return random.choice([a for a in actions if a.strength == best])

    def _rollout_purchase(self, current_state):
        """Epsilon-greedy units until nothing is affordable: a random one with probability
        rollout_epsilon, else the most attack + defense per PU."""
        sim = current_state.sim
        player = current_state.current_player
        while True:
            actions = self.affordable_units(sim, player)
            if not actions or not self.apply(
                    sim, self.pick_rollout_move_epsilon_greedy(actions).action()):
                break
        return self.end_phase(sim)

    def _rollout_combat(self, current_state):
        """Combat move of current_state.current_player, continuing from its excluded set."""
        while True:
            actions, current_state.excluded = ag.combat_actions(
                current_state.view(), self.d, current_state.excluded,
                current_state.current_player)
            if not actions:
                break
            mv = self.pick_rollout_move_epsilon_greedy(actions)
            if mv.end_phase:
                break
            for move in mv.moves:
                self.apply(current_state.sim, move)
        return self.end_phase(current_state.sim)

    def _rollout_noncombat(self, current_state):
        # The greedy whole-phase plan of tripleMind's live non-combat move.
        for mv in ag.noncombat_move(current_state.view(), self.d, current_state.current_player):
            self.apply(current_state.sim, mv)
        return self.end_phase(current_state.sim)

    def _rollout_rest_of_noncombat(self, current_state):
        """Rest of a non-combat move the tree started: a random option per territory, then the
        positioning moves."""
        while True:
            current_state.legal_moves()
            mv = random.choice(current_state.actions)
            if mv.end_phase:
                break
            current_state.apply_move(mv)
        return self.finish_phase(current_state)

    def _place_units(self, current_state):
        """player.place_units(): put everything bought at the home factory."""
        sim = current_state.sim
        held = sim.held_units(sim.decision.actor)
        home = ag._home_factory(current_state.view(), current_state.current_player)
        if held and home is not None:
            territory = self.d.territory_index[home]
            if not self.apply(sim, fm.PlaceAction(tuple(u.id for u in held), territory)):
                for u in held:  # over the production limit: place what fits
                    if not self.apply(sim, fm.PlaceAction((u.id,), territory)):
                        break
        return self.end_phase(sim)

    def _fight_battles(self, current_state):
        """Battle decisions as the live adapter answers them: fight every battle in order, default
        casualties, never retreat."""
        sim = current_state.sim
        kind = sim.decision.kind
        if kind == BATTLE_SELECTION_D:
            pending = [b.id for b in sim.battles.values()
                       if b.attacker == sim.decision.actor and b.status == "PENDING"]
            return self.apply(sim, SelectBattleAction(min(pending))) if pending else self.end_phase(sim)
        if kind == CASUALTIES_D:
            return self.apply(sim, toolkit.default_casualties(self.d, sim))
        if kind == RETREAT_D:
            return self.apply(sim, RetreatAction(CONTINUE))
        return self.end_phase(sim)

    def _play_decision(self, current_state):
        """Plays the sim's current decision, from its start, with the rollout policies."""
        sim = current_state.sim
        dec = sim.decision
        current_state.current_player = self.d.player_names[dec.actor]
        if dec.kind == MOVE_D:
            if self.d.steps[sim.step].kind == COMBAT_MOVE:
                current_state.excluded = set()
                return self._rollout_combat(current_state)
            return self._rollout_noncombat(current_state)
        if dec.kind == PURCHASE_D:
            return self._rollout_purchase(current_state)
        if dec.kind == PLACE_D:
            return self._place_units(current_state)
        return self._fight_battles(current_state)

    def _rollout_phase(self, current_state):
        """Rest of the phase a tree node is in, continuing from its bookkeeping."""
        if current_state.phase == COMBAT_PHASE:
            return self._rollout_combat(current_state)
        if current_state.phase == NONCOMBAT_PHASE:
            return self._rollout_rest_of_noncombat(current_state)
        return self._rollout_purchase(current_state)

    # ---- tree ----------------------------------------------------------------------------

    def expand(self, node):
        action = node.state.getNextAction()
        # END is never a tree edge: it is only generated when nothing is left to decide (a forced
        # move), and the child of the phase's last decision already stands for the next one.
        if action is None or action.end_phase:
            node.untried_actions = []
            return node
        new_state = node.state.clone()
        new_state.played_by = node.state.next_moves_by
        new_state.apply_move(action)
        new_state.legal_moves()
        passes_phase = no_target(new_state)
        if passes_phase:
            self.pass_phase(new_state)
        child = MCTSNode(new_state, parent=node, action=action,
                         sign=self.sign_of(new_state.played_by))
        child.passes_phase = passes_phase
        node.children.append(child)
        return child

    def simulate(self, state):
        current_state = state.clone()
        sim = current_state.sim
        depth = 0
        first = current_state.phase is not None  # the leaf's own phase continues its bookkeeping
        try:
            steps = 0
            while depth < self.depth_budget and not current_state.is_terminal() and steps < 5000:
                steps += 1
                if first:
                    ok = self._rollout_phase(current_state)
                    first = False
                else:
                    ok = self._play_decision(current_state)
                if not ok:
                    break
                depth = sim.round - state.round
        except Exception:
            reward, is_terminal = -0.5, False
        else:
            reward, is_terminal = evaluate_state(current_state, self.whoAmI, depth,
                                                 self.depth_budget)
        self.rollout_stats_metric.buffer(self.whoAmI, state.sim.game_number, state.sim.round,
                                         self.iteration, depth, reward, 1 if is_terminal else 0)
        return reward, depth

    def backpropagate(self, node, reward):
        """reward is whoAmI's; each node adds it from the view of the player who chose it."""
        while node is not None:
            node.visits += 1
            node.value += node.sign * reward
            node = node.parent

    # ---- search --------------------------------------------------------------------------

    def mcts_search(self, initial_state):
        """Searches from initial_state's decision (purchase, combat move or non-combat move) and
        returns the forward-model actions to play for that whole phase: for a purchase one
        PurchaseAction per unit, for a move phase its MoveActions (a non-combat move's forced air
        landings and positioning moves included). The chosen tree edges are in last_choice."""
        phase = initial_state.phase
        (self.efficiency_metric, self.rollout_stats_metric,
         self.action_seq_stats_metric) = self.metrics[phase]
        sim = initial_state.sim
        forced = self.start_phase(initial_state)
        initial_state.legal_moves()
        root = MCTSNode(initial_state)

        self.iteration = 0
        self.max_depth = 0
        start_time = time.time()
        # Nothing to decide: no search, the phase is played as it ends.
        while not no_target(initial_state) and time.time() - start_time < self.time_budget:
            self.iteration += 1
            # --- Selection and expansion: walk down by UCT to a node with an untried action and
            # expand it. expand() returns the node itself when it finds the node has no untried
            # action left (it is now fully expanded); then keep walking down from it, so every
            # iteration still adds a node unless it reaches a terminal or childless node. ---
            node = root
            while True:
                while node.is_fully_expanded() and not node.is_terminal():
                    best = node.best_child_uct(self.exploration_constant)
                    if best is None:
                        break
                    node = best
                if node.is_terminal():
                    break
                child = self.expand(node)
                if child is not node:
                    node = child  # rollout from the new child
                    break
                if not node.children:
                    break  # nothing to expand or walk into: roll out from here

            value, depth = self.simulate(node.state)
            self.max_depth = max(self.max_depth, depth)
            self.backpropagate(node, value)

        search_time = time.time() - start_time
        self.rollout_stats_metric.flush()
        # The latest search tree and its statistics, saved in game records.
        self.last_root = root
        self.last_stats = {
            "iterations": self.iteration,
            "max_rollout": self.max_depth,
            "seconds": search_time,
        }

        if self.trees_dir is not None and root.children:
            os.makedirs(self.trees_dir, exist_ok=True)
            tree_prefix = os.path.join(
                self.trees_dir, f"{TREE_NAME[phase]}_{self.whoAmI}_g{sim.game_number}"
                                f"_r{sim.round}")
            try:
                save_mcts_tree_png(root, self.whoAmI, tree_prefix, max_nodes=500, render_img=True)
            except Exception:  # a drawing is never worth losing the turn over
                pass

        # Our phase: the most visited path from the root up to the child that passes the phase on
        # (the nodes below it are later decisions, ours or the opponents').
        chosen = []
        node = root
        while node.children:
            best_child = max(node.children, key=lambda c: c.visits)
            self.log_choice_stats(initial_state, node, best_child)
            chosen.append(best_child.action)
            if best_child.passes_phase:
                break
            node = best_child
        self.last_choice = chosen
        actions = forced + self.play_phase(initial_state, chosen)

        self.efficiency_metric.log(
            self.whoAmI,
            sim.game_number,
            sim.round,
            search_time,
            self.iteration,
            root.value / root.visits if root.visits > 0 else 0,
            len(root.children),
            self.max_depth,
            count_leaves(root),
            # attacked territories, reinforced territories or units bought
            sum(1 for a in chosen if a.moves) if phase != PURCHASE_PHASE else
            sum(1 for a in actions if a.kind == "purchase"),
            tree_depth(root))
        self.action_seq_stats_metric.flush()
        return actions

    def play_phase(self, initial_state, chosen):
        """The forward-model actions of the chosen edges, played from initial_state to the end of
        its phase: a purchase the search stopped short of spends the rest greedily, a non-combat
        move ends with its positioning moves."""
        state = initial_state.clone()
        played = []
        for edge in chosen:
            if state.phase == PURCHASE_PHASE:
                played.append(edge.action())
            elif state.phase == NONCOMBAT_PHASE or edge.strength != 0:
                played.extend(edge.moves)
            state.apply_move(edge)
        if state.phase == PURCHASE_PHASE:
            while True:
                units = self.affordable_units(state.sim, state.current_player)
                if not units:
                    break
                best = max(units, key=lambda a: a.strength)
                if not self.apply(state.sim, best.action()):
                    break
                self.last_choice.append(best)
                played.append(best.action())
        elif state.phase == NONCOMBAT_PHASE:
            state.legal_moves()
            played.extend(self.positioning_moves(state))
        return played

    def log_choice_stats(self, initial_state, node, chosen):
        """Buffers one action_seq_stats row: how `chosen` stood out among node's children."""
        kids = node.children
        visits = [c.visits for c in kids]
        avgs = [c.value / c.visits if c.visits else 0.0 for c in kids]
        i = kids.index(chosen)
        second = max((k for k in range(len(kids)) if k != i), key=lambda k: visits[k], default=None)
        mean_v, var_v = _mean_var(visits)
        mean_a, var_a = _mean_var(avgs)
        z = lambda x, m, var: (x - m) / math.sqrt(var) if var > 1e-12 else ""  # blank if all equal
        self.action_seq_stats_metric.buffer(
            self.whoAmI, initial_state.sim.game_number, initial_state.sim.round,
            chosen.tree_level, chosen.state.played_by, action_label(chosen.action), len(kids),
            visits[i], visits[i] / sum(visits) if sum(visits) else 0.0,
            visits[second] if second is not None else "", mean_v, var_v, math.sqrt(var_v),
            z(visits[i], mean_v, var_v),
            avgs[i], avgs[second] if second is not None else "", mean_a, math.sqrt(var_a),
            z(avgs[i], mean_a, var_a))


def action_label(act):
    """A tree edge as the action column of action_seq_stats.csv shows it."""
    if act.end_phase:
        return "END"
    if isinstance(act, PurchaseUnit):
        return f"buy {act.unit_type}"
    return f"{act.to_terr} | skip" if not act.moves else f"{act.to_terr} | s={act.strength:g}"


def _mean_var(xs):
    """Mean and population variance of a non-empty list."""
    m = sum(xs) / len(xs)
    return m, sum((x - m) ** 2 for x in xs) / len(xs)


def no_target(state):
    """True when the decision has nothing left to decide (only END, or no action at all)."""
    return all(a.end_phase for a in state.actions)


def count_leaves(root):
    """Number of root-to-leaf paths in the tree, i.e. its leaves (1 for a lone root)."""
    leaves, stack = 0, [root]
    while stack:
        node = stack.pop()
        if node.children:
            stack.extend(node.children)
        else:
            leaves += 1
    return leaves


def tree_depth(root):
    """Deepest node of the tree, in edges from the root (0 for a lone root)."""
    deepest, stack = 0, [root]
    while stack:
        node = stack.pop()
        deepest = max(deepest, node.tree_level)
        stack.extend(node.children)
    return deepest


# =========================
# State evaluation (evaluate_state of move_generator.pyx): the one reward of every node
# =========================

def evaluate_state(state, whoAmI, depth, depth_budget):
    me = whoAmI

    # --- terminal check: the forward model's victory test (EndRoundDelegate.checkVictoryCities,
    # run at the end of each round), whose winners are alliance ids ---
    terminal = state.sim.terminal
    if terminal is not None:
        d = state.search.d
        if d.alliance_of[d.player_index[me]] in terminal.winners:
            return 0.5 + 0.5 * (depth_budget - depth) / depth_budget, True
        return -0.5 - 0.5 * (depth_budget - depth) / depth_budget, True

    game = state.view()
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

    score = 0.50 * tuv_term + 0.50 * income_term
    scaled_score = score - 0.5
    return scaled_score, False
