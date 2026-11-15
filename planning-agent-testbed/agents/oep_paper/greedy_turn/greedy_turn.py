"""Greedy Turn (Justesen et al., "Playing Multiaction Adversarial Games: Online Evolutionary
Planning Versus Tree Search", IEEE ToG 2018, Sec. V) for the TripleA purchase, combat move and
non-combat move.

The paper's Greedy Turn is a depth-first search over a whole turn: every leaf (a complete action
sequence) is rated with the same heuristic as Greedy Action and the sequence leading to the
highest rated state is played. A transposition table skips states already visited, and because
the search is usually not exhaustive the order actions are tried in (action sorting) matters.

It runs once per phase (3 times per turn), on the actions the other all-phase OEP agents use
(phases.py: one unit bought, one attack, or one territory's reinforcement set or skip; END), in
their sorting heuristic's order. A node is a plan (the actions so far this phase); a node where
END is offered is also a leaf: ending the phase there. So:

  1. depth-first, children in the sorting heuristic's order: at each node score its leaf (plan +
     END, then in the combat move this turn's battles, rated by evaluate_state), then expand its
     actions. A child whose state was already reached by another order of the same actions is
     skipped (transposition table: Phase.key);
  2. the DFS runs until the time budget is used or every plan was scored. In the combat move
     with dice battles a finished DFS hands the remaining time to resampling every leaf, with
     the same dice seed for every leaf in one sweep; purchase and non-combat roll no dice, and
     neither do fast battles, so a finished DFS is exact there;
  3. a node's value is the best mean value of a leaf below it; the chosen sequence follows the
     best sibling (END or a child) from the root, the first in the sorting heuristic's order of
     equal values, until END. A path that stops at a node the DFS did not score is completed
     with the highest-rated actions until END. evaluate_state counts neither purchased units in
     hand nor positions, so there the leaves usually tie and the sorting heuristic decides.

The combat move is scored after the battle phase, not after place: non-combat moves and placement
change neither the TUV nor the territory owners evaluate_state counts, so the ranking is the same
and every evaluation is much cheaper.
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

import fast_battle
import map_graph
from phases import complete_greedily, make_phases
from helper import EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, ACTION_SEQ_HEADER, MetricLogger

fast_battle.install()  # the search's battles: dice, or the Fast AI's most likely outcome

# TripleMind's columns first (the plot scripts read them by name), then Greedy Action's extra
# columns (same names, so the two agents' files compare), then this agent's own; `phase`
# (purchase, combatMove, nonCombatMove) last, as one search runs per phase.
GREEDY_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["reached_end_of_turn", "num_actions_per_level",
                                                "best_value_per_level", "states_visited",
                                                "transpositions", "sweeps", "phase"]
GREEDY_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "phase"]
# One row per level of the chosen sequence. The siblings are the node's children (valued by the
# best leaf below them) and END (its own leaf); "second" is the second-best sibling by value and
# visits are the dice samples of the leaves below a sibling.
GREEDY_ACTION_SEQ_HEADER = ACTION_SEQ_HEADER + ["value_gap", "end_phase_value", "states_visited",
                                                "level_seconds", "phase"]
# One row per sibling per level of the chosen sequence.
SIBLING_VALUES_HEADER = ["player", "game", "round", "tl", "action", "chosen", "samples",
                         "mean_value", "std_value", "diff_to_best", "phase"]


class Node:
    """A plan (actions so far this phase) in the phase tree; `values` are the samples of its END
    leaf (none where END is not offered)."""
    __slots__ = ("state", "action", "depth", "actions", "end", "values", "children")

    def __init__(self, state, action, depth):
        self.state = state  # PhaseState after the plan
        self.action = action  # the action leading here (None at the root)
        self.depth = depth  # actions in the plan
        self.actions = None  # actions offered here, in the sorting heuristic's order
        self.end = None  # the END offered here, if any
        self.values = []
        self.children = []

    @property
    def end_mean(self):
        return sum(self.values) / len(self.values) if self.values else -math.inf


class Sibling:
    """One choice at a level of the chosen sequence: END (the node's leaf) or a child subtree."""
    __slots__ = ("label", "node", "is_end", "mean", "samples", "std")

    def __init__(self, label, node, is_end, mean, samples, std):
        self.label, self.node, self.is_end = label, node, is_end
        self.mean, self.samples, self.std = mean, samples, std


class GreedyTurnSearch:
    def __init__(self, definition, map_graph_, whoAmI, time_budget, seed=0, metrics_dir="metrics"):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI
        self.me = definition.player_index[whoAmI]
        self.time_budget = time_budget  # seconds per phase decision
        self.rnd = random.Random(seed)
        self.states_visited = 0
        self.move_rng = fm.SeededRng(self.rnd.getrandbits(62))  # moves roll no dice
        self.phases = make_phases(self)
        self.last_stats = None
        self.last_sequence = []

        self.efficiency_metric = MetricLogger(
            os.path.join(metrics_dir, "efficiency.csv"), header=GREEDY_EFFICIENCY_HEADER)
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = MetricLogger(
            os.path.join(metrics_dir, "rollout_stats.csv"), header=GREEDY_ROLLOUT_STATS_HEADER)
        self.action_seq_stats_metric = MetricLogger(
            os.path.join(metrics_dir, "action_seq_stats.csv"), header=GREEDY_ACTION_SEQ_HEADER)
        self.sibling_metric = MetricLogger(
            os.path.join(metrics_dir, "sibling_values.csv"), header=SIBLING_VALUES_HEADER)

    # ---- forward model -------------------------------------------------------------------

    def apply(self, sim, action, rng):
        return fm.apply_in_place(self.d, sim, action, rng) != fm.REJECTED

    def view(self, sim):
        return map_graph.GameState.from_sim(self.map, self.d, sim, self.whoAmI)

    # ---- evaluation ----------------------------------------------------------------------

    def evaluate(self, phase, node, seed, game_number):
        """Value of ending the phase after node's plan (its END), in the combat move after this
        turn's battles with dice from seed."""
        self.states_visited += 1
        state = node.state.clone()
        rng = fm.SeededRng(seed)
        try:
            phase.finish(state, node.end, rng)
            if phase.name == "combatMove":
                self.fight_battles(state.sim, rng)
        except Exception:
            value, terminal = -0.5, False
        else:
            value, terminal = evaluate_state(self, state.sim, self.whoAmI)
        self.rollout_stats_metric.buffer(self.whoAmI, game_number, state.sim.round,
                                         self.states_visited, 0, value, 1 if terminal else 0,
                                         node.depth + 1, phase.name)
        return value

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

    def search(self, root_sim, phase_name):
        """The best action sequence for root_sim's phase, as forward-model actions for the engine
        (forced ones first)."""
        phase = self.phases[phase_name]
        started = time.time()
        deadline = started + self.time_budget
        game_number, round_ = root_sim.game_number, root_sim.round
        self.states_visited = 0
        self.move_rng = fm.SeededRng(self.rnd.getrandbits(62))  # purchases and moves roll no dice
        base_seed = self.rnd.getrandbits(40)

        # --- depth-first search over the phase, one sample per leaf ---
        root_state, forced = phase.start(root_sim.clone())
        root = Node(root_state, None, 0)
        seen = {phase.key(root.state)}
        leaves = []
        stack = [root]
        transpositions = 0
        max_depth = 0
        while stack and time.time() < deadline:
            node = stack.pop()
            node.actions = phase.actions(node.state)
            node.end = next((a for a in node.actions if a.end_phase), None)
            if node.end is not None:  # ending the phase here is offered: a leaf
                node.values.append(self.evaluate(phase, node, base_seed, game_number))
                leaves.append(node)
            for a in node.actions:
                if a.end_phase:
                    continue
                state = node.state.clone()
                if not phase.step(state, a, self.move_rng):
                    continue
                key = phase.key(state)
                if key in seen:
                    transpositions += 1
                    continue
                seen.add(key)
                node.children.append(Node(state, a, node.depth + 1))
                max_depth = max(max_depth, node.depth + 1)
            stack.extend(reversed(node.children))  # first in the action sorting is searched first
        complete = not stack

        # --- combat move, remaining time: resample every leaf (common dice seed per sweep);
        # fast battles roll no dice, so the one sample is exact ---
        sweeps = 1
        while (complete and phase.name == "combatMove" and not fast_battle.DETERMINISTIC
               and time.time() < deadline):
            for leaf in leaves:
                if time.time() >= deadline:
                    break
                leaf.values.append(self.evaluate(phase, leaf, base_seed + sweeps, game_number))
            sweeps += 1

        # --- the chosen sequence: the best sibling (END or a child) until END ---
        values = {}
        backup(root, values)
        sequence, levels = [], []
        node = root
        while node.actions is not None:
            if len(node.actions) == 1:  # no choice: not logged (as in Greedy Action)
                only = node.actions[0]
                child = node.children[0] if node.children else None
                if only.end_phase:
                    sequence.append(only)
                    break
                if child is None:
                    break  # its state was reached by another order: completed below
                sequence.append(only)
                node = child
                continue
            siblings = self.siblings(phase, node, values)
            if not siblings:
                break
            best = max(siblings, key=lambda s: s.mean)  # the first of equal values
            levels.append((node, siblings, best))
            if best.is_end:
                sequence.append(node.end)
                break
            sequence.append(best.node.action)
            node = best.node
        if not sequence or not sequence[-1].end_phase:
            sequence += complete_greedily(phase, node.state.clone(), self.move_rng)
        for node, siblings, best in levels:
            self.log_level(phase.name, game_number, round_, node.depth + 1, siblings, best)

        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        chosen = [a for a in sequence if not a.end_phase]
        self.last_stats = {"phase": phase.name, "states_visited": self.states_visited,
                           "leaves": len(leaves), "transpositions": transpositions,
                           "sweeps": sweeps, "complete": complete, "seconds": search_time}
        self.last_sequence = [phase.label(a) for a in sequence]
        self.efficiency_metric.log(
            self.whoAmI,
            game_number,
            round_,
            search_time,
            float("nan"),  # num_iterations: no MCTS loop; states_visited is its own column
            values[id(root)][0],  # root_value: value of the chosen sequence
            len(levels[0][1]) if levels else 0,  # num_children_at_root: scored siblings at the root
            float("nan"),  # max_rollout: no rollouts in greedy
            len(leaves),  # num_action_sequences: distinct sequences scored
            len(chosen),  # num_of_terr_attacked: actions chosen (attacks, units, reinforcements)
            max_depth,  # max_tree_depth: deepest plan generated
            1 if complete else 0,  # reached_end_of_turn: every sequence of the phase was searched
            ";".join(str(len(s)) for _, s, _ in levels),
            # value of stopping at each level of the chosen sequence (its END leaf)
            ";".join(f"{n.end_mean:.4f}" for n, _, _ in levels),
            self.states_visited,
            transpositions,
            sweeps,
            phase.name)
        return forced + phase.response_actions(sequence)

    def siblings(self, phase, node, values):
        """The node's choices in the sorting heuristic's order: its END leaf (when offered and
        scored) and every scored child, with their values."""
        children = {id(c.action): c for c in node.children}
        out = []
        for a in node.actions:
            if a.end_phase:
                if node.values:
                    out.append(Sibling(phase.label(a), node, True, node.end_mean,
                                       len(node.values), _std(node.values)))
                continue
            c = children.get(id(a))
            if c is None:
                continue
            mean, samples, best_leaf = values[id(c)]
            if samples:
                out.append(Sibling(phase.label(a), c, False, mean, samples,
                                   _std(best_leaf.values)))
        return out

    def log_level(self, phase, game_number, round_, tl, siblings, best):
        """One action_seq_stats row for the level and one sibling_values row per sibling."""
        ranked = sorted(siblings, key=lambda s: s.mean, reverse=True)
        ranked.remove(best)
        ranked.insert(0, best)
        second = ranked[1] if len(ranked) > 1 else None
        end = next((s for s in siblings if s.is_end), None)  # not offered at every level
        visits = [s.samples for s in siblings]
        means = [s.mean for s in siblings]
        mean_v, var_v = _mean_var(visits)
        mean_a, var_a = _mean_var(means)
        z = lambda x, m, var: (x - m) / math.sqrt(var) if var > 1e-12 else ""  # blank if all equal
        self.action_seq_stats_metric.buffer(
            self.whoAmI, game_number, round_, tl, self.whoAmI, best.label, len(siblings),
            best.samples, best.samples / sum(visits),
            second.samples if second else "", mean_v, var_v, math.sqrt(var_v),
            z(best.samples, mean_v, var_v),
            best.mean, second.mean if second else "", mean_a, math.sqrt(var_a),
            z(best.mean, mean_a, var_a),
            best.mean - second.mean if second else "",
            end.mean if end else "", sum(visits), float("nan"),  # level_seconds: the DFS is not split by level
            phase)
        for s in ranked:
            self.sibling_metric.buffer(self.whoAmI, game_number, round_, tl, s.label,
                                       1 if s is best else 0, s.samples, s.mean, s.std,
                                       best.mean - s.mean, phase)


def backup(node, values):
    """values[id(n)] = (best mean leaf value below n, samples of the leaves below n, that leaf) for
    n and every node below it; nodes the search never reached have no samples (-inf)."""
    best_mean, best_leaf = node.end_mean, node
    samples = len(node.values)
    for c in node.children:
        mean, n, leaf = backup(c, values)
        samples += n
        if mean > best_mean:
            best_mean, best_leaf = mean, leaf
    values[id(node)] = (best_mean, samples, best_leaf)
    return values[id(node)]


def _std(xs):
    return math.sqrt(_mean_var(xs)[1]) if xs else 0.0


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
