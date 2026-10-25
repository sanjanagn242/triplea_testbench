"""Non-exploring MCTS (Justesen et al., "Playing Multiaction Adversarial Games: Online Evolutionary
Planning Versus Tree Search", IEEE ToG 2018, Sec. V and VI) for the TripleA combat move.

The paper's Non-exploring MCTS is action-based MCTS (one ply = one action) with a tree policy that
does not explore, C = 0, combined with deterministic rollouts: every child of a node is still
visited once before any of them is expanded further, and since a deterministic rollout gives a
child the same value every time, one visit is enough to rank it. Rollouts last until the end of
the turn and are rated with the heuristic; values and visits are kept on edges, with a
transposition table and descent-path-only backpropagation. After the time budget the tree is
followed from the root along the best edges.

Here one action is one attack, as in Greedy Action and Greedy Turn: (every unit of mine that can
reach the target, target territory), for each enemy territory not attacked yet, plus END. So:

  * a node is a plan (its attacks), shared by every order of the same attacks (transposition
    table on where my moved units stand); an edge is (node, action) with its visits and value;
  * selection: from the root, while every action of the node has an edge, take the edge with the
    best mean value (no exploration term); END edges end the descent;
  * expansion: the node's next untried action (TripleMind territory priority order, END first);
  * rollout (deterministic): from the reached plan, keep attacking the highest-priority target my
    reachable units beat by TripleMind's strength margin (strength > 1.5 x defence), until there
    is none; then end the combat move, fight the battles and rate the state with evaluate_state;
  * battles roll dice, so the dice are fixed for the search: every evaluation uses the same
    TRIPLEA_NE_DICE_SEEDS seeds (default 1), which makes evaluations deterministic, as C = 0 needs;
  * backpropagation: the reward is added to the edges of the descent path only.
"""

import math
import os
import random
import time

import triplea_fm as fm
from triplea_fm import toolkit
from triplea_fm.actions import CONTINUE, RetreatAction, SelectBattleAction
from triplea_fm.definition import BATTLE
from triplea_fm.state import BATTLE_SELECTION_D, CASUALTIES_D, RETREAT_D

from generators import ag  # compiled Cython generators when built, else action_generators.py
import map_graph
from helper import EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, ACTION_SEQ_HEADER, MetricLogger

# TripleMind's columns first (the plot scripts read them by name), then the columns the greedy
# OEP agents share, then this agent's own.
NE_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["reached_end_of_turn", "num_actions_per_level",
                                            "best_value_per_level", "states_visited",
                                            "transpositions"]
NE_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "rollout_attacks"]
# One row per level of the chosen sequence: the node's edges; "second" is the second-best edge by
# mean value (TripleMind: the second most visited).
NE_ACTION_SEQ_HEADER = ACTION_SEQ_HEADER + ["value_gap", "end_phase_value", "states_visited",
                                            "level_seconds"]
SIBLING_VALUES_HEADER = ["player", "game", "round", "tl", "action", "chosen", "samples",
                         "mean_value", "std_value", "diff_to_best"]

END = ag.CombatAction(end_phase=True, strength=0)
ROLLOUT_STRENGTH_MARGIN = 1.5  # TripleMind's first attack threshold (combat_actions)


class Node:
    """A plan (attacks so far), shared by every order of the same attacks."""
    __slots__ = ("sim", "depth", "attacked", "actions", "edges")

    def __init__(self, sim, depth, attacked):
        self.sim = sim
        self.depth = depth  # attacks in the plan
        self.attacked = attacked
        self.actions = None  # [END] + attacks, generated on the first visit
        self.edges = []  # one per expanded action, in self.actions order

    def fully_expanded(self):
        return self.actions is not None and len(self.edges) == len(self.actions)


class Edge:
    """(node, action): visits and value of the descents through it."""
    __slots__ = ("action", "child", "visits", "value", "value_sq")

    def __init__(self, action, child):
        self.action = action
        self.child = child  # None for END
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

    def label(self):
        a = self.action
        return "END" if a.end_phase else f"{a.to_terr} | s={a.strength}"


class NonExploringMCTS:
    def __init__(self, definition, map_graph_, whoAmI, time_budget, dice_seeds=1, seed=0,
                 metrics_dir="metrics"):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI
        self.me = definition.player_index[whoAmI]
        self.time_budget = time_budget  # seconds per combat move
        self.dice_seeds = max(1, dice_seeds)  # fixed dice per search, averaged per evaluation
        self.rnd = random.Random(seed)
        self.states_visited = 0
        self.iteration = 0
        self.last_stats = None

        self.efficiency_metric = MetricLogger(
            os.path.join(metrics_dir, "efficiency.csv"), header=NE_EFFICIENCY_HEADER)
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = MetricLogger(
            os.path.join(metrics_dir, "rollout_stats.csv"), header=NE_ROLLOUT_STATS_HEADER)
        self.action_seq_stats_metric = MetricLogger(
            os.path.join(metrics_dir, "action_seq_stats.csv"), header=NE_ACTION_SEQ_HEADER)
        self.sibling_metric = MetricLogger(
            os.path.join(metrics_dir, "sibling_values.csv"), header=SIBLING_VALUES_HEADER)

    # ---- forward model -------------------------------------------------------------------

    def apply(self, sim, action, rng):
        return fm.apply_in_place(self.d, sim, action, rng) != fm.REJECTED

    def apply_attack(self, sim, action, rng):
        """Apply every move of an attack; False (sim partly changed) if one is rejected."""
        return all(self.apply(sim, mv, rng) for mv in action.moves)

    def view(self, sim):
        return map_graph.GameState.from_sim(self.map, self.d, sim, self.whoAmI)

    def state_key(self, sim):
        """Transposition key of a plan: where each of my moved units stands."""
        return tuple(sorted((u.id, u.loc) for u in sim.units.values()
                            if u.owner == self.me and u.moved))

    # ---- actions -------------------------------------------------------------------------

    def candidate_actions(self, sim, attacked):
        """One attack per enemy target not attacked yet: all my unmoved units that can reach it,
        in TripleMind's territory priority order."""
        game = self.view(sim)
        m = game.map
        actions = []
        for target in ag.enemy_territories_by_priority(game, self.whoAmI):
            if target in attacked:
                continue
            reachable = ag.get_reachable_units(game, target, self.whoAmI)
            if not reachable:
                continue
            actions.append(ag.CombatAction(
                to_terr=target,
                moves=ag._group_moves(self.d, [(u, route) for u, _, route in reachable]),
                strength=ag.calculate_attack_strength(m, [u for u, _, _ in reachable])))
        return actions

    def rollout_action(self, sim, attacked):
        """The deterministic rollout policy: the highest-priority attack whose strength beats the
        defence by ROLLOUT_STRENGTH_MARGIN, or None (end the phase)."""
        game = self.view(sim)
        m = game.map
        for target in ag.enemy_territories_by_priority(game, self.whoAmI):
            if target in attacked:
                continue
            reachable = ag.get_reachable_units(game, target, self.whoAmI)
            if not reachable:
                continue
            strength = ag.calculate_attack_strength(m, [u for u, _, _ in reachable])
            defence = ag.defender_strength(m, game.territories[target], self.whoAmI)
            if strength > ROLLOUT_STRENGTH_MARGIN * defence:
                return ag.CombatAction(
                    to_terr=target,
                    moves=ag._group_moves(self.d, [(u, route) for u, _, route in reachable]),
                    strength=strength)
        return None

    # ---- rollout and evaluation ----------------------------------------------------------

    def rollout(self, node, ended, seeds, game_number, round_):
        """Reward of node's plan, continued by the rollout policy unless `ended` (an END edge),
        averaged over the search's fixed dice seeds. Deterministic, so search() computes it once
        per node and reuses it on later visits (only the edge statistics change)."""
        sim = node.sim.clone()
        attacked = set(node.attacked)
        rng = fm.SeededRng(seeds[0])  # moves roll no dice
        attacks = 0
        while not ended:
            action = self.rollout_action(sim, attacked)
            if action is None or not self.apply_attack(sim, action, rng):
                break
            attacked.add(action.to_terr)
            attacks += 1
        values, terminal = [], False
        for seed in seeds:
            value, term = self.evaluate(sim, seed)
            values.append(value)
            terminal = terminal or term
        reward = sum(values) / len(values)
        self.states_visited += len(seeds)
        self.rollout_stats_metric.buffer(self.whoAmI, game_number, round_, self.iteration, 0, reward,
                                         1 if terminal else 0, node.depth + 1, attacks)
        return reward

    def evaluate(self, plan_sim, seed):
        """Value of ending the phase after plan_sim's attacks, after this turn's battles."""
        sim = plan_sim.clone()
        rng = fm.SeededRng(seed)
        try:
            self.apply(sim, fm.EndPhaseAction(), rng)  # end of combat move
            self.fight_battles(sim, rng)
        except Exception:
            return -0.5, False
        return evaluate_state(self, sim, self.whoAmI)

    def fight_battles(self, sim, rng, max_steps=2000):
        """The battle phase as the live adapter plays it: every battle in order, default
        casualties, never retreat."""
        steps = 0
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

    # ---- search --------------------------------------------------------------------------

    def search(self, root_sim):
        """The combat move's attack sequence (a list of CombatActions)."""
        started = time.time()
        deadline = started + self.time_budget
        game_number, round_ = root_sim.game_number, root_sim.round
        self.states_visited = 0
        self.iteration = 0
        seeds = [self.rnd.getrandbits(40) for _ in range(self.dice_seeds)]
        move_rng = fm.SeededRng(seeds[0])  # moves roll no dice

        root = Node(root_sim.clone(), 0, frozenset())
        table = {self.state_key(root.sim): root}
        rewards = {}  # (id(node), ended) -> rollout reward
        transpositions = 0
        max_depth = 0
        while time.time() < deadline:
            self.iteration += 1
            # --- selection: best mean edge (C = 0) while the node is fully expanded ---
            node, path, ended = root, [], False
            while True:
                if node.actions is None:
                    node.actions = [END] + self.candidate_actions(node.sim, node.attacked)
                if not node.fully_expanded():
                    break
                edge = max(node.edges, key=lambda e: e.mean)  # END is first, so it wins ties
                path.append(edge)
                if edge.child is None:
                    ended = True
                    break
                node = edge.child
            # --- expansion: the node's next untried action ---
            if not ended:
                action = node.actions[len(node.edges)]
                if action.end_phase:
                    edge = Edge(action, None)
                    ended = True
                else:
                    sim = node.sim.clone()
                    if not self.apply_attack(sim, action, move_rng):
                        node.actions.remove(action)  # rejected by the forward model
                        continue
                    key = self.state_key(sim)
                    child = table.get(key)
                    if child is None:
                        child = Node(sim, node.depth + 1, node.attacked | {action.to_terr})
                        table[key] = child
                        max_depth = max(max_depth, child.depth)
                    else:
                        transpositions += 1
                    edge = Edge(action, child)
                node.edges.append(edge)
                path.append(edge)
                if not ended:
                    node = edge.child
            # --- rollout (deterministic, so computed once per node) and descent-path backup ---
            key = (id(node), ended)
            if key not in rewards:
                rewards[key] = self.rollout(node, ended, seeds, game_number, round_)
            reward = rewards[key]
            for e in path:
                e.visits += 1
                e.value += reward
                e.value_sq += reward * reward

        # --- the chosen sequence: best edges from the root until END ---
        sequence, levels = [], []
        node, reached_end = root, False
        while node.edges:
            if len(node.actions) == 1:
                reached_end = True  # only END: nothing left to attack, not logged
                break
            best = max(node.edges, key=lambda e: e.mean)
            levels.append((node, best))
            if best.child is None:
                reached_end = True
                break
            sequence.append(best.action)
            node = best.child
        for tl, (n, best) in enumerate(levels, 1):
            self.log_level(game_number, round_, tl, n, best)

        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        root_visits = sum(e.visits for e in root.edges)
        self.last_stats = {"iterations": self.iteration, "states_visited": self.states_visited,
                           "nodes": len(table), "transpositions": transpositions,
                           "reached_end_of_turn": reached_end, "seconds": search_time}
        self.efficiency_metric.log(
            self.whoAmI,
            game_number,
            round_,
            search_time,
            self.iteration,  # num_iterations
            sum(e.value for e in root.edges) / root_visits if root_visits else "",  # root_value
            len(levels[0][0].edges) if levels else 0,  # num_children_at_root
            0,  # max_rollout: rollouts end with this turn's battles (0 rounds)
            count_leaves(table),  # num_action_sequences: plans at the tree's frontier
            len(sequence),  # num_of_terr_attacked
            max_depth,  # max_tree_depth: attacks in the deepest plan in the tree
            1 if reached_end else 0,  # the chosen path ends with END (not at an unexpanded node)
            ";".join(str(len(n.edges)) for n, _ in levels),
            ";".join(f"{b.mean:.4f}" for _, b in levels),
            self.states_visited,
            transpositions)
        return sequence

    def log_level(self, game_number, round_, tl, node, best):
        """One action_seq_stats row for the level and one sibling_values row per edge."""
        edges = node.edges
        ranked = sorted(edges, key=lambda e: e.mean, reverse=True)
        ranked.remove(best)
        ranked.insert(0, best)
        second = ranked[1] if len(ranked) > 1 else None
        end = edges[0]  # END is always the first action
        visits = [e.visits for e in edges]
        means = [e.mean for e in edges]
        mean_v, var_v = _mean_var(visits)
        mean_a, var_a = _mean_var(means)
        z = lambda x, m, var: (x - m) / math.sqrt(var) if var > 1e-12 else ""  # blank if all equal
        self.action_seq_stats_metric.buffer(
            self.whoAmI, game_number, round_, tl, self.whoAmI, best.label(), len(edges),
            best.visits, best.visits / sum(visits) if sum(visits) else 0.0,
            second.visits if second else "", mean_v, var_v, math.sqrt(var_v),
            z(best.visits, mean_v, var_v),
            best.mean, second.mean if second else "", mean_a, math.sqrt(var_a),
            z(best.mean, mean_a, var_a),
            best.mean - second.mean if second else "",
            end.mean, sum(visits), float("nan"))  # level_seconds: the search is not split by level
        for e in ranked:
            self.sibling_metric.buffer(self.whoAmI, game_number, round_, tl, e.label(),
                                       1 if e is best else 0, e.visits, e.mean, e.std,
                                       best.mean - e.mean)


def count_leaves(table):
    """Plans at the tree's frontier: END edges plus edges into nodes that were never expanded."""
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
# State evaluation: TripleMind's evaluate_state (mcts.py), on a bare forward-model state, without
# the rollout-depth discount (every evaluation here is at the same depth) and scaled to [-1, 1]:
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
