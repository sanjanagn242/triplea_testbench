"""TripleMind combat-move MCTS (multi front attack), translated to run on the forward model.

The search, node and rollout logic follow the original combat_mcts_agent.py; actions are still
generated in house (action_generators.py), but they are applied with triplea_fm instead of the
hand-written apply_* / update_income / place_units code, so battles, income, placement, turn
order and victory follow the engine's rules. Logging and Cython are removed.
"""

import math
import random
import time
import os

import triplea_fm as fm
from triplea_fm import toolkit
from triplea_fm.actions import CONTINUE, RetreatAction, SelectBattleAction
from triplea_fm.definition import COMBAT_MOVE
from triplea_fm.state import (BATTLE_SELECTION_D, CASUALTIES_D, MOVE_D, PLACE_D, PURCHASE_D,
                              RETREAT_D)

from generators import ag  # compiled Cython generators when built, else action_generators.py
import map_graph
from helper import (EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, ACTION_SEQ_HEADER, MetricLogger,
                    DebugLogger,
                    save_mcts_tree_png)


class MCTSGameState:
    """Game state for MCTS: a forward-model state plus the combat-phase bookkeeping."""

    def __init__(self, search, sim, current_player, played_by=None):
        self.search = search
        self.sim = sim
        self.current_player = current_player
        # The player whose move led to this state: a tree node's value is from its point of view.
        # The root's state was reached by no searched move; it counts as its own player's.
        self.played_by = played_by if played_by is not None else current_player
        self.round = sim.round

        self.actions = []  # Generated actions for current territory
        self.num_actions = len(self.actions)
        self.actionIndex = 0  # Current position in actions list
        self.excluded = set()  # Territories already processed

    def clone(self):
        cloned = MCTSGameState(self.search, self.sim.clone(), self.current_player, self.played_by)
        cloned.actions = list(self.actions)
        cloned.num_actions = len(self.actions)
        cloned.actionIndex = self.actionIndex
        cloned.excluded = set(self.excluded)
        return cloned

    @property
    def next_moves_by(self):
        """The player to move in this state: a tree node's actions (children) are this player's."""
        return self.current_player

    @next_moves_by.setter
    def next_moves_by(self, player):
        self.current_player = player

    def view(self):
        """map_graph.GameState of the forward-model state, seen by the current player."""
        return map_graph.GameState.from_sim(self.search.map, self.search.d, self.sim,
                                            self.current_player)

    # game over (forward-model victory conditions)
    def is_terminal(self):
        return self.sim.terminal is not None

    def heuristic_combat_legal_moves(self):
        self.actions = []
        self.actionIndex = 0
        actions, self.excluded = ag.combat_actions(self.view(), self.search.d, self.excluded,
                                                   self.current_player)
        if actions:
            self.actions = actions
            self.num_actions = len(self.actions)
        else:
            self.actions.append(ag.CombatAction(end_phase=True, strength=0))
            self.num_actions = len(self.actions)

    def apply_combat_move(self, moves):
        for move in moves:
            if move.end_phase:
                return
            for mv in move.moves:
                # Moves roll no dice; the battle is fought in the battle phase.
                self.search.apply(self.sim, mv)

    # The searched phase's moves, as MCTS.expand / pass_turn / mcts_search use them. A subclass
    # searching another phase (noncombat_mcts.NoncombatMCTSGameState) overrides these two.
    def legal_moves(self):
        self.heuristic_combat_legal_moves()

    def apply_move(self, action):
        self.apply_combat_move([action])

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
        self.tree_level = 0  # number of edges from the root before this node
        if parent:
            self.tree_level = parent.tree_level + 1
        # True when `action` was the chooser's last combat move decision: expand() then advanced
        # the state to the next player with a target, so this node is that player's decision.
        self.passes_turn = False

    def action_to(self, child):
        """The action of the edge from this node to child. A node of the purchase DAG
        (purchase_mcts.PurchaseNode) has several parents, and its own `action` is only its first
        parent's edge."""
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

        bestNode = None
        choices_weights = [
            (child.value / child.visits) + c_param * math.sqrt(math.log(self.visits) / child.visits)
            for child in self.children
        ]
        if choices_weights != []:
            max_wt = max(choices_weights)
            best_wts = [idx for idx, w in enumerate(choices_weights) if w == max_wt]
            best_nodes = [self.children[idx] for idx in best_wts]
            if len(best_nodes) == 1:
                bestNode = best_nodes[0]
            else:
                # pick the one with an action with highest strength
                bestNode = max(best_nodes, key=lambda node: self.action_to(node).strength)
        return bestNode


class MCTS:
    # The phase whose moves are tree edges; the tree passes from one player's such phase to the
    # next player's (opponent nodes). noncombat_mcts.NoncombatMCTS searches NONCOMBAT_MOVE.
    search_kind = COMBAT_MOVE

    def __init__(self, definition, map_graph_, whoAmI, time_budget, depth_budget,
                 exploration_constant, metrics_dir="metrics", metrics_suffix="", trees_dir=None,
                 tree_name="tree", rollout_epsilon=0.1):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI
        self.victory_cities = set(map_graph_.victory_cities())

        # MCTS parameters
        self.time_budget = time_budget  # seconds per move
        self.depth_budget = depth_budget  # Maximum playout depth (rounds)
        self.exploration_constant = exploration_constant  # UCB1 exploration parameter
        self.rollout_epsilon = rollout_epsilon  # random-move probability of the combat rollout

        # Each search's tree is saved as <trees_dir>/<tree_name>_<faction>_g<game>_r<round>.dot /
        # .html / .png (helper.save_mcts_tree_png); None saves nothing.
        self.trees_dir = trees_dir
        self.tree_name = tree_name

        self.iteration = 0
        self.max_depth = 0
        self._rng = fm.SeededRng(random.getrandbits(62))

        self.efficiency_metric = MetricLogger(
            os.path.join(metrics_dir, f"efficiency{metrics_suffix}.csv"), header=EFFICIENCY_HEADER)
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = MetricLogger(
            os.path.join(metrics_dir, f"rollout_stats{metrics_suffix}.csv"),
            header=ROLLOUT_STATS_HEADER)
        self.debug_logger = DebugLogger("logs/debug.log", False)
        self.action_seq_stats_metric = MetricLogger(
            os.path.join(metrics_dir, f"action_seq_stats{metrics_suffix}.csv"),
            header=ACTION_SEQ_HEADER)

    # ---- forward model -------------------------------------------------------------------

    def apply(self, sim, action):
        """Apply an action with the forward model; False if the model rejects it."""
        return fm.apply_in_place(self.d, sim, action, self._rng) != fm.REJECTED

    def end_phase(self, sim):
        return self.apply(sim, fm.EndPhaseAction())

    # ---- tree ----------------------------------------------------------------------------

    def sign_of(self, player):
        """+1 if `player` is in whoAmI's alliance, else -1."""
        d = self.d
        mine = d.alliance_of[d.player_index[self.whoAmI]]
        return 1 if d.alliance_of[d.player_index[player]] == mine else -1

    def expand(self, node):
        action = node.state.getNextAction()
        # END is never a tree edge: it is only generated when no target is left (a forced move),
        # and the child of the last attack / skip already stands for the next player's turn.
        if action is None or action.end_phase:
            node.untried_actions = []
            return node
        new_state = node.state.clone()
        new_state.played_by = node.state.next_moves_by
        new_state.apply_move(action)
        new_state.legal_moves()
        passes_turn = no_target(new_state)
        if passes_turn:
            self.pass_turn(new_state)
        child = MCTSNode(new_state, parent=node, action=action,
                         sign=self.sign_of(new_state.played_by))
        child.passes_turn = passes_turn
        node.children.append(child)
        return child

    def pass_turn(self, state):
        """Ends state's combat move (no target left) and plays on to the next combat move with a
        target, so the node holding state is that player's decision. A player with no target is
        passed over too, at most once per player (then state stays at a combat move with only END
        and the node is a leaf). If the game ends or the model rejects an action on the way,
        state.actions is empty and the node is a leaf."""
        for _ in range(len(self.d.player_names)):
            try:
                at_combat = self.advance_to_next_combat_move(state)
                if at_combat:
                    self.start_phase(state)
            except Exception:
                state.actions, state.num_actions, state.actionIndex = [], 0, 0
                return
            if not at_combat:
                return
            state.legal_moves()
            if not no_target(state):
                return

    def pick_rollout_move_adversarial(self, actions, topk=3, epsilon=0.10):
        real = [a for a in actions if not getattr(a, "end_phase", False)]
        real_no_skips = [a for a in actions if (not getattr(a, "end_phase", False)) and getattr(a, "moves", None)]
        endp = [a for a in actions if getattr(a, "end_phase", False)]

        candidates = real_no_skips if real_no_skips else real
        if not candidates:
            return endp[0] if endp else actions[0]

        capital_moves = [a for a in candidates if a.to_terr in self.victory_cities]
        if capital_moves:
            # among capital-capturing moves, take the strongest one (highest strength)
            return max(capital_moves, key=lambda a: a.strength)

        def score(a):
            # crude proxy: prefer higher strength attacks, and attacks that move toward victory_cities
            proximity_bonus = 1.0 if a.to_terr in self.victory_cities else 0.0
            return a.strength + proximity_bonus

        ranked = sorted(candidates, key=score, reverse=True)
        k = min(topk, len(ranked))

        if random.random() < epsilon:
            return random.choice(ranked[:k])
        return ranked[0]

    def pick_rollout_move_random(self, actions):
        return random.choice(actions)

    def pick_rollout_move_epsilon_greedy(self, actions):
        """With probability self.rollout_epsilon a random action, else the strongest one (ties
        broken at random)."""
        if random.random() < self.rollout_epsilon:
            return random.choice(actions)
        best = max(a.strength for a in actions)
        return random.choice([a for a in actions if a.strength == best])

    # ---- rollout -------------------------------------------------------------------------

    def _rollout_combat(self, current_state):
        """Combat move of current_state.current_player, continuing from its excluded set."""
        while True:
            current_state.heuristic_combat_legal_moves()
            mv = self.pick_rollout_move_epsilon_greedy(current_state.actions)
            if mv.end_phase:
                break
            current_state.apply_combat_move([mv])
        return self.end_phase(current_state.sim)

    def _rollout_searched(self, current_state):
        """Rest of the searched phase of current_state.current_player, continuing from the node's
        excluded set (the first phase of a rollout)."""
        return self._rollout_combat(current_state)

    def start_phase(self, state):
        """Called when the tree reaches a player's searched phase, before its moves are generated.
        Returns the moves it applied (none here)."""
        return []

    def finish_phase(self, state):
        """Ends state's searched phase once no target is left."""
        return self.end_phase(state.sim)

    def _rollout_noncombat(self, current_state):
        # The same policy the live turn plays (agent.get_move), so the search models its own moves.
        for mv in ag.noncombat_move(current_state.view(), self.d, current_state.current_player):
            self.apply(current_state.sim, mv)
        return self.end_phase(current_state.sim)

    def _rollout_purchase(self, current_state):
        """The live turn's purchase (agent.get_move): the first of purchase_actions, if any."""
        sim = current_state.sim
        actions = ag.purchase_actions(current_state.view(), self.d, current_state.current_player)
        if actions:
            self.apply(sim, actions[0])
        return self.end_phase(sim)

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
        """Plays the sim's current decision, other than a combat move, with the rollout policies."""
        dec = current_state.sim.decision
        current_state.current_player = self.d.player_names[dec.actor]
        if dec.kind == MOVE_D:
            return self._rollout_noncombat(current_state)
        if dec.kind == PURCHASE_D:
            return self._rollout_purchase(current_state)
        if dec.kind == PLACE_D:
            return self._place_units(current_state)
        return self._fight_battles(current_state)

    def at_searched_phase(self, sim):
        """True when sim's decision is a move in the searched phase (purchase_mcts.PurchaseMCTS
        searches a purchase decision instead)."""
        return sim.decision.kind == MOVE_D and self.d.steps[sim.step].kind == self.search_kind

    def advance_to_next_combat_move(self, state):
        """Ends state's searched phase (combat move, or non-combat move for NoncombatMCTS) and
        plays on with the rollout policies until a player's next searched phase. state then
        belongs to that player, with nothing excluded. False if the game ended or the model
        rejected an action on the way (state.actions is then empty, so the node is not expanded
        further)."""
        sim = state.sim
        ok = self.finish_phase(state)
        state.excluded = set()
        state.actions, state.num_actions, state.actionIndex = [], 0, 0
        steps = 0
        while ok and not state.is_terminal() and steps < 5000:
            steps += 1
            step_kind = self.d.steps[sim.step].kind
            if self.at_searched_phase(sim):
                state.current_player = self.d.player_names[sim.decision.actor]
                state.round = sim.round
                return True
            if sim.decision.kind == MOVE_D and step_kind == COMBAT_MOVE:
                state.current_player = self.d.player_names[sim.decision.actor]
                state.excluded = set()
                ok = self._rollout_combat(state)
                state.excluded = set()
                state.actions, state.num_actions, state.actionIndex = [], 0, 0
                continue
            ok = self._play_decision(state)
        state.round = sim.round
        return False

    def simulate(self, state):
        current_state = state.clone()
        sim = current_state.sim
        depth = 0
        first_sim = True  # the first searched phase continues the node's excluded set
        try:
            steps = 0
            while depth < self.depth_budget and not current_state.is_terminal() and steps < 5000:
                steps += 1
                dec = sim.decision
                # Read before the handler runs: handlers end their phase, which advances sim.step.
                step = self.d.steps[sim.step]
                round_ = sim.round

                if first_sim and self.at_searched_phase(sim):
                    current_state.current_player = self.d.player_names[dec.actor]
                    ok = self._rollout_searched(current_state)
                    first_sim = False
                elif dec.kind == MOVE_D and step.kind == COMBAT_MOVE:
                    current_state.current_player = self.d.player_names[dec.actor]
                    current_state.excluded = set()
                    ok = self._rollout_combat(current_state)
                    first_sim = False
                else:
                    ok = self._play_decision(current_state)
                if self.debug_logger.is_debug:  # skip building the message when not logging
                    if dec.kind == MOVE_D:
                        phase = "COMBAT_MOVE" if step.kind == COMBAT_MOVE else "NONCOMBAT_MOVE"
                    else:
                        phase = dec.kind  # PURCHASE / PLACE / BATTLE_SELECTION / CASUALTIES / RETREAT
                    self.debug_logger.log(f"simulate: round {round_}, step {step.id} {step.name}, "
                                          f"{phase} for {current_state.current_player}, ok={ok}")
                if not ok:
                    break                    
                depth = sim.round - state.round
        except Exception:
            reward, is_terminal = -0.5, False
        else:
            # Evaluate final state
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
        initial_state.legal_moves()
        root = MCTSNode(initial_state)
        root.tree_level = 0

        self.iteration = 0
        self.max_depth = 0
        start_time = time.time()
        while time.time() - start_time < self.time_budget:
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

            # --- Simulation ---
            value, depth = self.simulate(node.state)

            self.max_depth = max(self.max_depth, depth)
            # --- Backprop always ---
            self.backpropagate(node, value)

        search_time = time.time() - start_time
        self.rollout_stats_metric.flush()
        # The latest search tree and its statistics, saved in game records.
        self.last_root = root
        self.last_stats = {
            "iterations": self.iteration,
            "max_rollout": self.max_depth,
            "seconds": time.time() - start_time,
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

        # Our combat move: the most visited path from the root up to the child that passes the
        # turn on (the opponents' nodes below it are not our actions).
        action_seq = []
        node = root
        while node.children != []:
            best_child = max(node.children, key=lambda c: c.visits)
            self.log_choice_stats(initial_state, node, best_child)
            if best_child.action.moves and best_child.action.strength != 0:
                action_seq.append(best_child.action)
            if best_child.passes_turn:
                break
            node = best_child
        self.efficiency_metric.log(
            self.whoAmI,
            initial_state.sim.game_number,
            initial_state.sim.round,
            search_time,
            self.iteration,
            root.value / root.visits if root.visits > 0 else 0,
            len(root.children),
            self.max_depth,
            count_leaves(root),
            len(action_seq),  # one action per attacked territory
            tree_depth(root))
        self.action_seq_stats_metric.flush()
        return action_seq

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
            chosen.tree_level, chosen.state.played_by, self.action_label(node.action_to(chosen)), len(kids),
            visits[i], visits[i] / sum(visits) if sum(visits) else 0.0,
            visits[second] if second is not None else "", mean_v, var_v, math.sqrt(var_v),
            z(visits[i], mean_v, var_v),
            avgs[i], avgs[second] if second is not None else "", mean_a, math.sqrt(var_a),
            z(avgs[i], mean_a, var_a))

    @staticmethod
    def action_label(act):
        """A tree edge as the action column of action_seq_stats.csv shows it."""
        return "END" if act.end_phase else (f"{act.to_terr} | skip" if not act.moves
                                             else f"{act.to_terr} | s={act.strength}")


def _mean_var(xs):
    """Mean and population variance of a non-empty list."""
    m = sum(xs) / len(xs)
    return m, sum((x - m) ** 2 for x in xs) / len(xs)


def no_target(state):
    """True when the combat move generator found no target left (only END is offered)."""
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
# State evaluation (evaluate_state of move_generator.pyx)
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
