"""Non-exploring MCTS as M. Roxo applies it to TripleA ("MCTS-based Planning for Grand Strategy
Games", IST 2021, Sec. 3.10.1 and 5.7, after Justesen et al., IEEE ToG 2018) for the purchase,
combat move and non-combat move.

  * one tree ply is one decision of the phase (phases.py): the options of one territory (or unit
    rule), its skip included; a node is the state part-way through the phase, an edge (node,
    action) holds the visits and the value of the descents through it;
  * tree policy, non-exploring: the edge with the best mean (ties: the generation order, largest
    first); every child of a node is visited once (untried actions expanded in that order) before
    any of them is expanded further; an END edge ends the descent;
  * playout, deterministic (5.4-5.5): the rest of the phase and then every player's phases by
    Roxo's Weak AI heuristic for at most 10 rounds, rated with Roxo's reward (playout.py). The
    heuristic's random choices and the dice are fixed per search (K dice seeds,
    TRIPLEA_ROXO_NE_DICE_SEEDS, averaged; one with fast battles), so a node's playout is computed
    once and reused;
  * transposition table: a state reached by different orders of the same actions is one node,
    but backpropagation follows the descent path only;
  * backpropagation: the reward is added to every edge on the path (all of them are mine within a
    phase; an opponent's edge would get its negative, BackupNegamax).

After the time budget the tree is followed from the root along the most visited edges (ties: best
mean) until END; a path that stops at a node the search never expanded is completed by the playout
heuristic.
"""

import math
import os
import time

import fast_battle
from helper import ACTION_SEQ_HEADER, EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, MetricLogger
from playout import PLAYOUT_ROUNDS, SearchBase
from phases import actor

# TripleMind's columns first (the plot scripts read them by name), then this agent's own.
NE_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["reached_end_of_turn", "num_actions_per_level",
                                            "best_value_per_level", "states_visited",
                                            "transpositions", "phase"]
NE_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "phase"]
NE_ACTION_SEQ_HEADER = ACTION_SEQ_HEADER + ["value_gap", "end_phase_value", "states_visited",
                                            "phase"]
SIBLING_VALUES_HEADER = ["player", "game", "round", "tl", "action", "chosen", "samples",
                         "mean_value", "std_value", "diff_to_best", "phase"]


# =========================
# Tree
# =========================

class Node:
    """A state of the phase, shared by every order of actions that reaches it."""
    __slots__ = ("state", "player", "actions", "edges", "visits")

    def __init__(self, state, player):
        self.state = state
        self.player = player  # who chooses the action here
        self.actions = None  # options, END included, generated on the first visit
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
# Search
# =========================

class NEMCTS(SearchBase):
    def __init__(self, definition, map_graph_, whoAmI, time_budget, playout_rounds=PLAYOUT_ROUNDS,
                 dice_seeds=1, seed=0, metrics_dir="metrics"):
        super().__init__(definition, map_graph_, whoAmI, time_budget, seed, playout_rounds)
        self.k = 1 if fast_battle.DETERMINISTIC else max(1, dice_seeds)
        self.rewards = {}  # (id(node), took END) -> mean playout reward, per search
        self.states_visited = 0
        self.iteration = 0
        self.last_stats = None
        self.last_sequence = []

        def logger(name, header):
            return MetricLogger(os.path.join(metrics_dir, name), header=header)

        # efficiency.csv holds the combat-move searches only, so it compares row for row with the
        # combat-only agents; the other phases have their own files.
        self.efficiency_metric = {
            "combatMove": logger("efficiency.csv", NE_EFFICIENCY_HEADER),
            "purchase": logger("efficiency_purchase.csv", NE_EFFICIENCY_HEADER),
            "nonCombatMove": logger("efficiency_noncombat.csv", NE_EFFICIENCY_HEADER),
        }
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = logger("rollout_stats.csv", NE_ROLLOUT_STATS_HEADER)
        self.action_seq_stats_metric = logger("action_seq_stats.csv", NE_ACTION_SEQ_HEADER)
        self.sibling_metric = logger("sibling_values.csv", SIBLING_VALUES_HEADER)

    def rollout(self, phase, node, end, game_number, start_round):
        """Playout reward of node's state (END first when the descent took an END edge), once per
        fixed dice seed and averaged; deterministic, so computed once per (node, end)."""
        cache_key = (id(node), end is not None)
        if cache_key in self.rewards:
            return self.rewards[cache_key]
        total = 0.0
        for dice in self.dice_seeds:
            reward, rounds, terminal = self.playout(phase, node.state.clone(), end, start_round,
                                                    self.policy_seed, dice)
            total += reward
            self.states_visited += 1
            self.rollout_stats_metric.buffer(self.whoAmI, game_number, start_round,
                                             self.iteration, rounds, reward,
                                             1 if terminal else 0, node.state.depth + 1,
                                             phase.name)
        self.rewards[cache_key] = total / len(self.dice_seeds)
        return self.rewards[cache_key]

    def select(self, node):
        """Non-exploring: the best mean (ties: the first in generation order)."""
        return max(node.edges, key=lambda e: e.mean)

    def iterate(self, phase, root, table, game_number, start_round, stats):
        """One MCTS iteration: selection, expansion, playout, backpropagation."""
        self.iteration += 1
        node, path, nodes, end = root, [], [root], None
        # --- selection, once every child of the node has been visited ---
        while True:
            if node.actions is None:
                node.actions = phase.actions(node.state)
            if not node.fully_expanded():
                break
            edge = self.select(node)
            path.append(edge)
            if edge.child is None:
                end = edge.action
                break
            node = edge.child
            nodes.append(node)
        # --- expansion: the node's next untried action ---
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
                    child = Node(state, actor(self.d, state.sim))
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
        # --- playout ---
        delta = self.rollout(phase, node, end, game_number, start_round)
        # --- backpropagation along the descent path ---
        for n in nodes:
            n.visits += 1
        for e in path:
            signed = delta if e.player == self.me else -delta
            e.visits += 1
            e.value += signed
            e.value_sq += signed * signed

    def most_visited_path(self, phase, node):
        """The most visited edges (ties: best mean) from node until END, and the forward-model
        actions the playout heuristic adds past a node never expanded."""
        sequence, levels = [], []
        while node.edges:
            best = max(node.edges, key=lambda e: (e.visits, e.mean))
            if len(node.edges) > 1:
                levels.append((node, best))
            sequence.append(best.action)
            if best.child is None:
                return sequence, levels, [], True
            node = best.child
        return sequence, levels, self.complete(phase, node.state.clone()), False

    # ---- search --------------------------------------------------------------------------

    def search(self, phase_name, root_sim):
        """The phase's chosen actions, as forward-model actions for the engine (forced ones
        first)."""
        phase = self.by_name[phase_name]
        started = time.time()
        deadline = started + self.time_budget
        game_number, start_round = root_sim.game_number, root_sim.round
        self.new_search()
        self.states_visited = 0
        self.iteration = 0
        self.dice_seeds = [self.new_seed() for _ in range(self.k)]
        self.policy_seed = self.new_seed()
        self.rewards = {}

        root_state, forced = phase.start(root_sim.clone())
        root = Node(root_state, actor(self.d, root_state.sim))
        root.actions = phase.actions(root.state)
        table = {phase.key(root.state): root}
        stats = {"transpositions": 0, "max_depth": 0}
        if len(root.actions) > 1:  # a single action (END) is no choice
            while time.time() < deadline:
                self.iterate(phase, root, table, game_number, start_round, stats)
            sequence, levels, tail, reached_end = self.most_visited_path(phase, root)
        else:
            sequence, levels, tail, reached_end = [], [], [], True

        for tl, (n, best) in enumerate(levels, 1):
            self.log_level(phase, game_number, start_round, tl, n, best)
        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        chosen = [a for a in sequence if not a.end_phase and not a.is_skip]
        self.last_stats = {"phase": phase.name, "iterations": self.iteration, "nodes": len(table),
                           "transpositions": stats["transpositions"],
                           "reached_end_of_turn": reached_end, "seconds": round(search_time, 3)}
        visits = sum(e.visits for e in root.edges)
        self.efficiency_metric[phase.name].log(
            self.whoAmI,
            game_number,
            start_round,
            search_time,
            self.iteration,  # num_iterations
            sum(e.value for e in root.edges) / visits if visits else "",  # root_value
            len(root.edges),  # num_children_at_root
            self.playout_rounds,  # max_rollout: rounds a playout may advance
            count_leaves(table),  # num_action_sequences: sequences at the tree's frontier
            len(chosen),  # num_of_terr_attacked: non-skip actions chosen
            stats["max_depth"],  # max_tree_depth: actions in the deepest state in the tree
            1 if reached_end else 0,  # the most visited path ends with an END edge
            ";".join(str(len(n.edges)) for n, _ in levels),
            ";".join(f"{b.mean:.4f}" for _, b in levels),
            self.states_visited,
            stats["transpositions"],
            phase.name)
        self.last_sequence = [phase.label(a) for a in sequence] + (["(heuristic tail)"] if tail
                                                                    else [])
        return forced + phase.response_actions(sequence) + tail

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
            end.mean if end else "", node.visits, phase.name)
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
