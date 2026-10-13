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
from helper import MetricLogger, DebugLogger, save_mcts_tree_png


class MCTSGameState:
    """Game state for MCTS: a forward-model state plus the combat-phase bookkeeping."""

    def __init__(self, search, sim, current_player):
        self.search = search
        self.sim = sim
        self.current_player = current_player
        self.round = sim.round

        self.actions = []  # Generated actions for current territory
        self.actionIndex = 0  # Current position in actions list
        self.excluded = set()  # Territories already processed

    def clone(self):
        cloned = MCTSGameState(self.search, self.sim.clone(), self.current_player)
        cloned.actions = list(self.actions)
        cloned.actionIndex = self.actionIndex
        cloned.excluded = set(self.excluded)
        return cloned

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
        else:
            self.actions.append(ag.CombatAction(end_phase=True, strength=0))

    def apply_combat_move(self, moves):
        for move in moves:
            if move.end_phase:
                return
            for mv in move.moves:
                # Moves roll no dice; the battle is fought in the battle phase.
                self.search.apply(self.sim, mv)

    def getNextAction(self):
        if self.actionIndex < len(self.actions):
            action = self.actions[self.actionIndex]
            self.actionIndex += 1
            return action
        return None


class MCTSNode:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.action = action
        self.children = []
        self.untried_actions = None
        self.visits = 0
        self.value = 0.0
        self.is_state_terminal = None
        self.tree_level = 0  # number of edges from the root before this node
        if parent:
            self.tree_level = parent.tree_level + 1

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
            (child.value / child.visits) + c_param * math.sqrt(2 * math.log(self.visits) / child.visits)
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
                bestNode = max(best_nodes, key=lambda node: node.action.strength)
        return bestNode


class MCTS:
    def __init__(self, definition, map_graph_, whoAmI, time_budget, depth_budget,
                 exploration_constant):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI
        self.victory_cities = set(map_graph_.victory_cities())

        # MCTS parameters
        self.time_budget = time_budget  # seconds per move
        self.depth_budget = depth_budget  # Maximum playout depth (rounds)
        self.exploration_constant = exploration_constant  # UCB1 exploration parameter

        self.iteration = 0
        self.max_depth = 0
        self._rng = fm.SeededRng(random.getrandbits(62))

        self.efficiency_metric = MetricLogger(
            "metrics/efficiency.csv",
            header=["game", "round", "num_iterations", "max_depth", "time_taken", "num_children", "value"]
        )
        self.debug_logger = DebugLogger("logs/debug.log", False)

    # ---- forward model -------------------------------------------------------------------

    def apply(self, sim, action):
        """Apply an action with the forward model; False if the model rejects it."""
        return fm.apply_in_place(self.d, sim, action, self._rng) != fm.REJECTED

    def end_phase(self, sim):
        return self.apply(sim, fm.EndPhaseAction())

    # ---- tree ----------------------------------------------------------------------------

    def expand(self, node):
        action = node.state.getNextAction()
        if action is None or action.end_phase:
            node.untried_actions = []
            return node
        new_state = node.state.clone()
        new_state.apply_combat_move([action])
        new_state.heuristic_combat_legal_moves()
        child = MCTSNode(new_state, parent=node, action=action)
        node.children.append(child)
        return child

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

    # ---- rollout -------------------------------------------------------------------------

    def _rollout_combat(self, current_state):
        """Combat move of current_state.current_player, continuing from its excluded set."""
        while True:
            current_state.heuristic_combat_legal_moves()
            mv = self.pick_rollout_move_random(current_state.actions)
            if mv.end_phase:
                break
            current_state.apply_combat_move([mv])
        return self.end_phase(current_state.sim)

    def _rollout_noncombat(self, current_state):
        for mv in ag.noncombat_move_rollout(current_state.view(), self.d, current_state.current_player):
            self.apply(current_state.sim, mv)
        return self.end_phase(current_state.sim)

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

    def simulate(self, state):
        current_state = state.clone()
        sim = current_state.sim
        depth = 0
        first_sim = True  # the first combat phase continues the node's excluded set
        try:
            steps = 0
            while depth < self.depth_budget and not current_state.is_terminal() and steps < 5000:
                steps += 1
                dec = sim.decision
                # Read before the handler runs: handlers end their phase, which advances sim.step.
                step = self.d.steps[sim.step]
                round_ = sim.round

                current_state.current_player = self.d.player_names[dec.actor]
                if dec.kind == MOVE_D:
                    phase = "COMBAT_MOVE" if step.kind == COMBAT_MOVE else "NONCOMBAT_MOVE"
                else:
                    phase = dec.kind  # PURCHASE / PLACE / BATTLE_SELECTION / CASUALTIES / RETREAT
                if dec.kind == MOVE_D and step.kind == COMBAT_MOVE:
                    if not first_sim:
                        current_state.excluded = set()
                    ok = self._rollout_combat(current_state)
                    first_sim = False
                elif dec.kind == MOVE_D:
                    ok = self._rollout_noncombat(current_state)
                elif dec.kind == PURCHASE_D:
                    ok = self.end_phase(sim)  # purchases are not simulated (as in the original)
                elif dec.kind == PLACE_D:
                    ok = self._place_units(current_state)
                else:
                    ok = self._fight_battles(current_state)
                self.debug_logger.log(f"simulate: round {round_}, step {step.id} {step.name}, {phase} "
                                      f"for {current_state.current_player}, ok={ok}")
                if not ok:
                    break                    
                depth = sim.round - state.round
        except Exception:
            return -0.5, depth

        # Evaluate final state
        reward, is_terminal = evaluate_state(current_state, self.whoAmI, self.victory_cities,
                                             depth, self.depth_budget)
        return reward, depth

    def backpropagate(self, node, reward):
        while node is not None:
            node.visits += 1
            node.value += reward
            node = node.parent

    # ---- search --------------------------------------------------------------------------

    def mcts_search(self, initial_state):
        initial_state.heuristic_combat_legal_moves()
        root = MCTSNode(initial_state)
        root.tree_level = 0

        self.iteration = 0
        self.max_depth = 0
        start_time = time.time()
        while time.time() - start_time < self.time_budget:
            self.iteration += 1
            # --- Selection: walk down until leaf or terminal ---
            node = root
            while node.is_fully_expanded() and not node.is_terminal():
                best = node.best_child_uct(self.exploration_constant)
                if best is None:
                    break
                node = best

            # --- Expand or evaluate terminal ---
            if node.is_terminal():
                value, depth = self.simulate(node.state)  # or use known outcome
            else:
                child = self.expand(node)  # expand leaf
                if child != node:
                    node = child
                    value, depth = self.simulate(node.state)  # rollout from new child
                else:
                    while node.is_fully_expanded() and not node.is_terminal():
                        best = node.best_child_uct(self.exploration_constant)
                        if best is None:
                            break
                        node = best
                    value, depth = self.simulate(node.state)

            self.max_depth = max(self.max_depth, depth)
            # --- Backprop always ---
            self.backpropagate(node, value)
        
        self.efficiency_metric.log(
            initial_state.sim.game_number,
            initial_state.sim.round,
            self.iteration,
            self.max_depth,
            time.time() - start_time,
            len(root.children),
            root.value / root.visits if root.visits > 0 else 0)
        # The latest search tree and its statistics, saved in game records.
        self.last_root = root
        self.last_stats = {
            "iterations": self.iteration,
            "max_rollout": self.max_depth,
            "seconds": time.time() - start_time,
        }

        # tree_prefix = f"trees/tree_g{initial_state.sim.game_number}_r{initial_state.sim.round}"
        # os.makedirs(os.path.dirname(tree_prefix), exist_ok=True)
        # dot_file, png_file = save_mcts_tree_png(root, tree_prefix, max_nodes=500, render_img=True)

        action_seq = []
        node = root
        while node.children != []:
            best_child = max(node.children, key=lambda c: c.visits)
            if best_child is None:
                break
            if best_child.action.moves is not None and best_child.action.end_phase == False and best_child.action.strength != 0:
                action_seq.append(best_child.action)
            if best_child.action.end_phase == True:
                break
            node = best_child
        return action_seq


# =========================
# State evaluation (evaluate_state of move_generator.pyx)
# =========================

def evaluate_state(state, whoAmI, victory_cities, depth, depth_budget):
    me = whoAmI
    game = state.view()
    m = game.map
    territories = game.territories

    # --- terminal check ---
    vc_owner_count = {}
    for name in victory_cities:
        owner = territories[name].owner
        vc_owner_count[owner] = vc_owner_count.get(owner, 0) + 1
    for owner, count in vc_owner_count.items():
        if victory_cities and count == len(victory_cities):
            if owner == me:
                return 0.5 + 0.5 * (depth_budget - depth) / depth_budget, True
            return -0.5 - 0.5 * (depth_budget - depth) / depth_budget, True

    my_tuv = 0.0
    enemy_tuv = 0.0
    for name, terr in territories.items():
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

    players = game.players
    income = players[me].PU
    income_denom = sum(p.PU for p in players.values())
    income_term = income / income_denom if income_denom > 0 else 0

    score = 0.50 * tuv_term + 0.50 * income_term
    scaled_score = 0.5 * score
    return scaled_score, False
