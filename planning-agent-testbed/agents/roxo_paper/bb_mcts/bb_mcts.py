"""Bridge Burning MCTS as M. Roxo applies it to TripleA ("MCTS-based Planning for Grand Strategy
Games", IST 2021, Sec. 3.10.2 and 5.7, after Justesen et al., IEEE ToG 2018) for the purchase,
combat move and non-combat move.

  * one tree ply is one decision of the phase (phases.py): the options of one territory (or unit
    rule), its skip included; a node is the state part-way through the phase, an edge (node,
    action) holds the visits and the value of the descents through it;
  * tree policy, UCT (eq. 2.1): the edge maximising X_j + C sqrt(ln n / n_j), C = 1/sqrt 2, with n
    the node's visits and n_j the edge's; untried actions are expanded first, in generation order
    (largest first); an END edge ends the descent;
  * playout (5.7): the rest of the phase and then every player's phases for at most 10 rounds,
    rated with Roxo's reward (playout.py). Each phase of the playout is, with probability E
    (TRIPLEA_ROXO_BB_EXPLORE, default 0.5), played with random actions of phases.py, otherwise by
    Roxo's Weak AI heuristic. Fresh random choices and dice in every playout;
  * transposition table: a state reached by different orders of the same actions is one node,
    but backpropagation follows the descent path only;
  * backpropagation: the reward is added to every edge on the path (all of them are mine within a
    phase; an opponent's edge would get its negative, BackupNegamax);
  * bridge burning: the time budget is split into sequential burn phases, one per decision of the
    sequence. Within a burn phase the search is ordinary MCTS below the current root; at its end
    every root child except the most promising one (most visited, ties: best mean) is pruned for
    good, and that child is the root of the next burn phase. A TripleA phase has no fixed number of
    decisions, so each burn phase gets the remaining time divided by the items still undecided
    (Phase.levels_left). Committing END stops; the rest of the budget is then unused.

If the time runs out before END is committed, the sequence is completed from the last root along
the most visited edges until END, and past a node never expanded by the playout heuristic.
"""

import math
import os
import time

from helper import ACTION_SEQ_HEADER, EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, MetricLogger
from playout import PLAYOUT_ROUNDS, SearchBase
from phases import actor

C = 1 / math.sqrt(2)  # the exploration constant of eq. 2.1
EXPLORE = 0.5  # E: probability that a playout phase is played randomly

# TripleMind's columns first (the plot scripts read them by name), then this agent's own.
BB_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["reached_end_of_turn", "num_actions_per_level",
                                            "best_value_per_level", "states_visited",
                                            "transpositions", "iterations_per_level", "phase"]
BB_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "phase"]
BB_ACTION_SEQ_HEADER = ACTION_SEQ_HEADER + ["value_gap", "end_phase_value", "states_visited",
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

class BBMCTS(SearchBase):
    def __init__(self, definition, map_graph_, whoAmI, time_budget, c=C, explore=EXPLORE,
                 playout_rounds=PLAYOUT_ROUNDS, seed=0, metrics_dir="metrics"):
        super().__init__(definition, map_graph_, whoAmI, time_budget, seed, playout_rounds)
        self.c = c  # C of X_j + C sqrt(ln n / n_j)
        self.explore = explore  # probability that a playout phase is played randomly
        self.states_visited = 0
        self.iteration = 0
        self.last_stats = None
        self.last_sequence = []

        def logger(name, header):
            return MetricLogger(os.path.join(metrics_dir, name), header=header)

        # efficiency.csv holds the combat-move searches only, so it compares row for row with the
        # combat-only agents; the other phases have their own files.
        self.efficiency_metric = {
            "combatMove": logger("efficiency.csv", BB_EFFICIENCY_HEADER),
            "purchase": logger("efficiency_purchase.csv", BB_EFFICIENCY_HEADER),
            "nonCombatMove": logger("efficiency_noncombat.csv", BB_EFFICIENCY_HEADER),
        }
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = logger("rollout_stats.csv", BB_ROLLOUT_STATS_HEADER)
        self.action_seq_stats_metric = logger("action_seq_stats.csv", BB_ACTION_SEQ_HEADER)
        self.sibling_metric = logger("sibling_values.csv", SIBLING_VALUES_HEADER)

    def rollout(self, phase, node, end, game_number, start_round):
        """Playout reward of node's state (END first when the descent took an END edge), with fresh
        random choices and dice, each phase random with probability `explore`."""
        reward, rounds, terminal = self.playout(phase, node.state.clone(), end, start_round,
                                                self.new_seed(), None, self.explore)
        self.states_visited += 1
        self.rollout_stats_metric.buffer(self.whoAmI, game_number, start_round, self.iteration,
                                         rounds, reward, 1 if terminal else 0,
                                         node.state.depth + 1, phase.name)
        return reward

    def select(self, node):
        """UCT (eq. 2.1): X_j + C sqrt(ln n / n_j)."""
        log_n = math.log(node.visits) if node.visits > 0 else 0.0
        return max(node.edges, key=lambda e: e.mean + self.c * math.sqrt(log_n / e.visits))

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

        root_state, forced = phase.start(root_sim.clone())
        root = first_root = Node(root_state, actor(self.d, root_state.sim))
        table = {phase.key(root.state): root}
        stats = {"transpositions": 0, "max_depth": 0}
        sequence, levels, tail = [], [], []  # levels: (root, committed edge)
        per_level = []  # iterations per burn phase
        reached_end = False
        while True:
            if root.actions is None:
                root.actions = phase.actions(root.state)
            if len(root.actions) == 1 and root.actions[0].end_phase:
                sequence.append(root.actions[0])  # only END is left: no choice
                reached_end = True
                break
            if time.time() >= deadline:
                break  # out of time: completed below
            phase_started = time.time()
            share = max(1, phase.levels_left(root.state, root.actions))
            phase_deadline = phase_started + (deadline - phase_started) / share
            iterations_before = self.iteration
            while time.time() < phase_deadline or not root.edges:
                if time.time() >= deadline and root.edges:
                    break
                self.iterate(phase, root, table, game_number, start_round, stats)
            if not root.edges:
                break
            # burn the bridges: only the most promising root child is kept
            best = max(root.edges, key=lambda e: (e.visits, e.mean))
            levels.append((root, best))
            per_level.append(self.iteration - iterations_before)
            sequence.append(best.action)
            if best.child is None:
                reached_end = True
                break
            root = best.child

        if not reached_end:
            rest, _, tail, reached_end = self.most_visited_path(phase, root)
            sequence += rest

        for tl, (n, best) in enumerate(levels, 1):
            self.log_level(phase, game_number, start_round, tl, n, best)
        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        chosen = [a for a in sequence if not a.end_phase and not a.is_skip]
        self.last_stats = {"phase": phase.name, "iterations": self.iteration, "nodes": len(table),
                           "transpositions": stats["transpositions"],
                           "burn_phases": len(levels), "reached_end_of_turn": reached_end,
                           "seconds": round(search_time, 3)}
        root = first_root
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
            1 if reached_end else 0,  # the sequence ends with an END edge
            ";".join(str(len(n.edges)) for n, _ in levels),
            ";".join(f"{b.mean:.4f}" for _, b in levels),
            self.states_visited,
            stats["transpositions"],
            ";".join(str(i) for i in per_level),
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
