"""Bridge Burning MCTS with all attacks in the tree, for the TripleA combat move (after Justesen
et al., "Playing Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search",
IEEE ToG 2018, Sec. V and VI).

The paper's BB-MCTS is its vanilla MCTS (action-based: one ply = one action; UCT tree policy, the
paper's X + 2C sqrt(2 ln n / n_j); epsilon-greedy rollouts with epsilon = 0.5, depth-limited
to one turn and rated with the heuristic; transposition table with values and visits on edges and
descent-path-only backpropagation) plus the bridge burning strategy: the time budget is split into
as many phases as the turn has actions, and at the end of each phase every root child but the
most promising one is pruned for good, i.e. that child becomes the root of the next phase.

Variant of oep_paper's agent with a different action generation in the tree only: a level offers
every attack on every reachable target, with no territory ordering. For each enemy territory not
attacked yet that some unmoved unit of mine can reach (map order), one action per attack set
TripleMind's combat_actions builds for it (growing prefixes of the attackers, adjacent and
strongest first, above 1.5x the defence and below 3.5x + 4; the full force if it beats the
defence; the cheapest unit for an undefended territory), plus END. The plan is the path through
the tree. Everything else, rollouts included, is as in oep_paper. So:

  * a node is a plan (its attacks), shared by every order of the same attacks (transposition
    table on where my moved units stand); an edge is (node, action) with its visits and value;
  * selection: while every action of the node has an edge, the UCT-best edge, with the UCT
    formula every MCTS agent of the testbed uses: mean + c sqrt(ln n / n_j), c the exploration
    constant as given (the paper's weight 2C sqrt 2 on [0, 1] rewards is c = 4 on these [-1, 1]
    rewards); END edges end the descent;
  * expansion: the node's next untried action (END first, then tree_actions in map order);
  * rollout: from the reached plan, epsilon-greedy until END: with probability epsilon the greedy
    action (the highest-priority attack whose strength beats the defence by TripleMind's 1.5
    margin, END when there is none), otherwise a uniformly random one of [END] + attacks; then end
    the combat move, fight the battles with fresh dice and rate the state with evaluate_state;
  * backpropagation: the reward is added to the edges of the descent path only;
  * bridge burning: the paper split the budget into its fixed number of actions per turn; here the
    number of phases per turn is a fixed estimate, `phases` (TRIPLEA_BB_PHASES, default 4: the
    3-4 territories TripleMind's MCTS attacks in a turn). Each phase gets remaining time / phases
    still estimated, at most one per target left; a turn needing more phases than estimated rolls
    out every root action once per extra phase (slightly over budget) before committing. At its end
    the root's most visited edge (ties: best mean) is committed: END stops, an attack's child
    becomes the root and the rest of the tree above it is never searched again.
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

# TripleMind's columns first (the plot scripts read them by name), then the columns the other OEP
# agents share, then this agent's own.
BB_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["reached_end_of_turn", "num_actions_per_level",
                                            "best_value_per_level", "states_visited",
                                            "transpositions", "iterations_per_level"]
BB_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "rollout_attacks"]
# One row per phase (level of the committed sequence): the root's edges at the end of the phase;
# the committed edge is the most visited, "second" the second-best edge by mean value.
BB_ACTION_SEQ_HEADER = ACTION_SEQ_HEADER + ["value_gap", "end_phase_value", "states_visited",
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


class BridgeBurningMCTS:
    def __init__(self, definition, map_graph_, whoAmI, time_budget, exploration_constant,
                 epsilon, phases=4, seed=0, metrics_dir="metrics"):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI
        self.me = definition.player_index[whoAmI]
        self.time_budget = time_budget  # seconds per combat move
        self.c = exploration_constant  # c of mean + c sqrt(ln n / n_j)
        self.epsilon = epsilon  # rollout: probability of the greedy action
        self.phases = max(1, phases)  # estimated phases (committed actions) per turn
        self.rnd = random.Random(seed)
        self.states_visited = 0
        self.iteration = 0
        self.last_stats = None
        # The last search's tree, for edge_tree.save_edge_tree: the first root (the burnt
        # parts stay reachable from it), the committed edges and each phase's root.
        self.last_root, self.last_path, self.last_phases = None, [], []

        self.efficiency_metric = MetricLogger(
            os.path.join(metrics_dir, "efficiency.csv"), header=BB_EFFICIENCY_HEADER)
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = MetricLogger(
            os.path.join(metrics_dir, "rollout_stats.csv"), header=BB_ROLLOUT_STATS_HEADER)
        self.action_seq_stats_metric = MetricLogger(
            os.path.join(metrics_dir, "action_seq_stats.csv"), header=BB_ACTION_SEQ_HEADER)
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

    def new_rng(self):
        return fm.SeededRng(self.rnd.getrandbits(62))

    # ---- actions -------------------------------------------------------------------------

    def candidate_actions(self, sim, attacked):
        """The rollout's actions (as in oep_paper's BB-MCTS): one attack per enemy target not
        attacked yet, all my unmoved units that can reach it, in TripleMind's territory priority
        order. The tree uses tree_actions."""
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

    def tree_actions(self, sim, attacked):
        """The tree's actions at a level: for every enemy territory not attacked yet that some
        unmoved unit of mine can reach (map order: no territory priority), each of TripleMind's
        attack sets for it (combat_actions' buildup: growing prefixes of the attackers, adjacent and
        strongest first, above 1.5x the defence and below 3.5x + 4; the full force if it beats the
        defence; the cheapest unit for an undefended territory)."""
        game = self.view(sim)
        m = game.map
        actions = []
        for target, terr in game.territories.items():
            if target in attacked or terr.owner == self.whoAmI or m.is_water(target):
                continue
            attackers, defence = ag.target_attackers(game, target, self.whoAmI)
            for unit_infos, strength in ag.attack_sets(m, attackers, defence):
                actions.append(ag.CombatAction(
                    to_terr=target,
                    moves=ag._group_moves(self.d, [(u, route) for u, _, route in unit_infos]),
                    strength=strength))
        return actions

    def greedy_action(self, sim, actions):
        """The rollout's greedy choice: the first (highest-priority) attack whose strength beats
        the defence by ROLLOUT_STRENGTH_MARGIN, or END."""
        game = self.view(sim)
        m = game.map
        for a in actions:
            defence = ag.defender_strength(m, game.territories[a.to_terr], self.whoAmI)
            if a.strength > ROLLOUT_STRENGTH_MARGIN * defence:
                return a
        return END

    # ---- rollout and evaluation ----------------------------------------------------------

    def rollout(self, node, ended, game_number, round_):
        """Reward of node's plan, continued epsilon-greedily unless `ended` (an END edge), with
        fresh dice."""
        sim = node.sim.clone()
        attacked = set(node.attacked)
        rng = self.new_rng()
        attacks = 0
        while not ended:
            actions = self.candidate_actions(sim, attacked)
            if not actions:
                break
            if self.rnd.random() < self.epsilon:
                action = self.greedy_action(sim, actions)
            else:
                action = self.rnd.choice([END] + actions)
            if action.end_phase or not self.apply_attack(sim, action, rng):
                break
            attacked.add(action.to_terr)
            attacks += 1
        reward, terminal = self.evaluate(sim, rng)
        self.states_visited += 1
        self.rollout_stats_metric.buffer(self.whoAmI, game_number, round_, self.iteration, 0, reward,
                                         1 if terminal else 0, node.depth + 1, attacks)
        return reward

    def evaluate(self, plan_sim, rng):
        """Value of ending the phase after plan_sim's attacks, after this turn's battles."""
        sim = plan_sim.clone()
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

    # ---- tree policy ---------------------------------------------------------------------

    def uct(self, edge, parent_visits):
        """mean + c sqrt(ln n / n_j): the UCT formula of every MCTS agent in the testbed."""
        return edge.mean + self.c * math.sqrt(math.log(parent_visits) / edge.visits)

    def iterate(self, root, table, move_rng, game_number, round_, stats):
        """One MCTS iteration below `root`."""
        self.iteration += 1
        # --- selection: UCT-best edge while the node is fully expanded ---
        node, path, ended = root, [], False
        while True:
            if node.actions is None:
                node.actions = [END] + self.tree_actions(node.sim, node.attacked)
            if not node.fully_expanded():
                break
            n = sum(e.visits for e in node.edges)
            edge = max(node.edges, key=lambda e: self.uct(e, n))
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
                    return
                key = self.state_key(sim)
                child = table.get(key)
                if child is None:
                    child = Node(sim, node.depth + 1, node.attacked | {action.to_terr})
                    table[key] = child
                    stats["max_depth"] = max(stats["max_depth"], child.depth)
                else:
                    stats["transpositions"] += 1
                edge = Edge(action, child)
            node.edges.append(edge)
            path.append(edge)
            if not ended:
                node = edge.child
        # --- rollout and descent-path backpropagation ---
        reward = self.rollout(node, ended, game_number, round_)
        for e in path:
            e.visits += 1
            e.value += reward
            e.value_sq += reward * reward

    # ---- search --------------------------------------------------------------------------

    def search(self, root_sim):
        """The combat move's attack sequence (a list of CombatActions)."""
        started = time.time()
        deadline = started + self.time_budget
        game_number, round_ = root_sim.game_number, root_sim.round
        self.states_visited = 0
        self.iteration = 0
        move_rng = self.new_rng()  # moves roll no dice

        root = first_root = Node(root_sim.clone(), 0, frozenset())
        table = {self.state_key(root.sim): root}
        stats = {"transpositions": 0, "max_depth": 0}
        sequence, levels = [], []  # levels: (root, committed edge, iterations, seconds, states)
        reached_end = False
        while True:
            if root.actions is None:
                root.actions = [END] + self.tree_actions(root.sim, root.attacked)
            attacks = len({a.to_terr for a in root.actions if not a.end_phase})  # targets
            if attacks == 0:
                reached_end = True  # nothing left to attack: END, not a choice
                break
            # --- one phase: search below the current root for its share of the time: the
            # remaining time over the phases still estimated (self.phases in all), at most one per
            # target left. A turn needing more phases than estimated has no time left for them:
            # each extra phase rolls out every root action once, then commits ---
            extra = len(levels) >= self.phases
            if time.time() >= deadline and not extra:
                break  # out of time before this phase could search
            phase_started = time.time()
            phases_left = min(self.phases - len(levels), attacks)
            phase_deadline = (phase_started if extra
                              else phase_started + (deadline - phase_started) / phases_left)
            iterations_before, states_before = self.iteration, self.states_visited
            while time.time() < phase_deadline or not root.fully_expanded():
                self.iterate(root, table, move_rng, game_number, round_, stats)
                if time.time() >= deadline and root.fully_expanded():
                    break
            if not root.edges:
                break
            # --- burn the bridges: commit the most promising root edge ---
            best = max(root.edges, key=lambda e: (e.visits, e.mean))
            levels.append((root, best, self.iteration - iterations_before,
                           time.time() - phase_started, self.states_visited - states_before))
            if best.child is None:
                reached_end = True
                break
            sequence.append(best.action)
            root = best.child
        self.last_root = first_root
        self.last_path = [b for _, b, _, _, _ in levels]
        self.last_phases = [n for n, _, _, _, _ in levels]
        for tl, (node, best, _, seconds, states) in enumerate(levels, 1):
            self.log_level(game_number, round_, tl, node, best, states, seconds)

        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        first = levels[0][0] if levels else None
        first_visits = sum(e.visits for e in first.edges) if first else 0
        self.last_stats = {"iterations": self.iteration, "phases": len(levels), "nodes": len(table),
                           "transpositions": stats["transpositions"],
                           "reached_end_of_turn": reached_end, "seconds": search_time}
        self.efficiency_metric.log(
            self.whoAmI,
            game_number,
            round_,
            search_time,
            self.iteration,  # num_iterations
            # root_value: mean reward at the first root, before any bridge was burnt
            sum(e.value for e in first.edges) / first_visits if first_visits else "",
            len(first.edges) if first else 0,  # num_children_at_root
            0,  # max_rollout: rollouts end with this turn's battles (0 rounds)
            count_leaves(table),  # num_action_sequences: plans at the tree's frontier
            len(sequence),  # num_of_terr_attacked
            stats["max_depth"],  # max_tree_depth: attacks in the deepest plan in the tree
            1 if reached_end else 0,  # committed END (or nothing left) before the time ran out
            ";".join(str(len(n.edges)) for n, _, _, _, _ in levels),
            ";".join(f"{b.mean:.4f}" for _, b, _, _, _ in levels),
            self.states_visited,
            stats["transpositions"],
            ";".join(str(i) for _, _, i, _, _ in levels))
        return sequence

    def log_level(self, game_number, round_, tl, node, best, states, seconds):
        """One action_seq_stats row for the phase and one sibling_values row per root edge."""
        edges = node.edges
        by_value = sorted(edges, key=lambda e: e.mean, reverse=True)
        by_value.remove(best)
        second = by_value[0] if by_value else None
        ranked = [best] + by_value
        end = next((e for e in edges if e.child is None), None)
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
            end.mean if end else "", states, seconds)
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
