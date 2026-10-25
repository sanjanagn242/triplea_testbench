"""Greedy Turn (Justesen et al., "Playing Multiaction Adversarial Games: Online Evolutionary
Planning Versus Tree Search", IEEE ToG 2018, Sec. V) for the TripleA combat move.

The paper's Greedy Turn is a depth-first search over a whole turn: every leaf (a complete action
sequence) is rated with the same heuristic as Greedy Action and the sequence leading to the
highest rated state is played. A transposition table skips states already visited, and because
the search is usually not exhaustive the order actions are tried in (action sorting) matters.

Here one action is one attack, as in Greedy Action: (every unit of mine that can reach the target,
target territory), for each enemy territory not attacked yet. A node is a plan (the attacks so far)
and every node is also a leaf: ending the phase there. So:

  1. depth-first, children in TripleMind's territory priority order (the action sorting): at each
     node score its leaf (plan + END: the plan is played to the end of the battle phase with the
     forward model and evaluate_state rates the result), then expand its attacks. A child whose
     state (where my moved units stand) was already reached by another order of the same attacks
     is skipped (transposition table);
  2. if the search covered every plan before the time budget ran out, the remaining time samples
     every leaf again (battles are random), with the same dice seed for every leaf in one sweep;
  3. a node's value is the best mean value of a leaf below it; the chosen sequence follows the
     best child from the root while it beats ending the phase there (END wins ties).

The state is scored after the battle phase, not after place: non-combat moves and placement
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

from generators import ag  # compiled Cython generators when built, else action_generators.py
import map_graph
from helper import EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, ACTION_SEQ_HEADER, MetricLogger

# TripleMind's columns first (the plot scripts read them by name), then Greedy Action's extra
# columns (same names, so the two agents' files compare), then this agent's own.
GREEDY_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["reached_end_of_turn", "num_actions_per_level",
                                                "best_value_per_level", "states_visited",
                                                "transpositions", "sweeps"]
GREEDY_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl"]
# One row per level of the chosen sequence. The siblings are the node's children (valued by the
# best leaf below them) and END (its own leaf); "second" is the second-best sibling by value and
# visits are the dice samples of the leaves below a sibling.
GREEDY_ACTION_SEQ_HEADER = ACTION_SEQ_HEADER + ["value_gap", "end_phase_value", "states_visited",
                                                "level_seconds"]
# One row per sibling per level of the chosen sequence.
SIBLING_VALUES_HEADER = ["player", "game", "round", "tl", "action", "chosen", "samples",
                         "mean_value", "std_value", "diff_to_best"]

END = ag.CombatAction(end_phase=True, strength=0)


class Node:
    """A plan (attacks so far) in the turn tree; `values` are the samples of its END leaf."""
    __slots__ = ("sim", "action", "depth", "attacked", "values", "children")

    def __init__(self, sim, action, depth, attacked):
        self.sim = sim
        self.action = action  # the attack leading here (None at the root)
        self.depth = depth  # attacks in the plan
        self.attacked = attacked
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
        self.time_budget = time_budget  # seconds per combat move
        self.rnd = random.Random(seed)
        self.states_visited = 0
        self.last_stats = None

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
        in TripleMind's territory priority order (the search's action sorting)."""
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

    # ---- evaluation ----------------------------------------------------------------------

    def evaluate(self, plan_sim, seed, game_number, tl):
        """Value of ending the phase after plan_sim's attacks, after this turn's battles, with
        dice from seed."""
        sim = plan_sim.clone()
        rng = fm.SeededRng(seed)
        self.states_visited += 1
        try:
            self.apply(sim, fm.EndPhaseAction(), rng)  # end of combat move
            self.fight_battles(sim, rng)
        except Exception:
            value, terminal = -0.5, False
        else:
            value, terminal = evaluate_state(self, sim, self.whoAmI)
        self.rollout_stats_metric.buffer(self.whoAmI, game_number, plan_sim.round,
                                         self.states_visited, 0, value, 1 if terminal else 0, tl)
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

    def search(self, root_sim):
        """The best full-turn attack sequence for root_sim's combat move (a list of CombatActions)."""
        started = time.time()
        deadline = started + self.time_budget
        game_number, round_ = root_sim.game_number, root_sim.round
        self.states_visited = 0
        move_rng = fm.SeededRng(self.rnd.getrandbits(62))  # moves roll no dice
        base_seed = self.rnd.getrandbits(40)

        # --- depth-first search over the turn, one sample per leaf ---
        root = Node(root_sim.clone(), None, 0, frozenset())
        seen = {self.state_key(root.sim)}
        leaves = []
        stack = [root]
        transpositions = 0
        max_depth = 0
        while stack and time.time() < deadline:
            node = stack.pop()
            node.values.append(self.evaluate(node.sim, base_seed, game_number, node.depth + 1))
            leaves.append(node)
            for a in self.candidate_actions(node.sim, node.attacked):
                sim = node.sim.clone()
                if not self.apply_attack(sim, a, move_rng):
                    continue
                key = self.state_key(sim)
                if key in seen:
                    transpositions += 1
                    continue
                seen.add(key)
                node.children.append(Node(sim, a, node.depth + 1, node.attacked | {a.to_terr}))
                max_depth = max(max_depth, node.depth + 1)
            stack.extend(reversed(node.children))  # highest-priority attack is searched first
        complete = not stack

        # --- remaining time: resample every leaf (common dice seed per sweep) ---
        sweeps = 1
        while complete and time.time() < deadline:
            for leaf in leaves:
                if time.time() >= deadline:
                    break
                leaf.values.append(self.evaluate(leaf.sim, base_seed + sweeps, game_number,
                                                 leaf.depth + 1))
            sweeps += 1
            # Two sweeps with identical values: no dice involved, more samples change nothing.
            if sweeps >= 2 and all(min(l.values) == max(l.values) for l in leaves):
                break

        # --- the chosen sequence: best child while it beats ending the phase there ---
        values = {}
        backup(root, values)
        sequence, levels = [], []
        node = root
        while True:
            siblings = self.siblings(node, values)
            if len(siblings) == 1:
                break  # nothing left to attack: not a choice, not logged (as in Greedy Action)
            best = max(siblings, key=lambda s: s.mean)  # END is first, so it wins ties
            levels.append((node, siblings, best))
            if best.is_end:
                break
            sequence.append(best.node.action)
            node = best.node
        for node, siblings, best in levels:
            self.log_level(game_number, round_, node.depth + 1, siblings, best)

        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        self.last_stats = {"states_visited": self.states_visited, "leaves": len(leaves),
                           "transpositions": transpositions, "sweeps": sweeps,
                           "complete": complete, "seconds": search_time}
        self.efficiency_metric.log(
            self.whoAmI,
            game_number,
            round_,
            search_time,
            float("nan"),  # num_iterations: no MCTS loop; states_visited is its own column
            values[id(root)][0],  # root_value: value of the chosen sequence
            len(levels[0][1]) if levels else 0,  # num_children_at_root: attacks at the root + END
            float("nan"),  # max_rollout: no rollouts in greedy
            len(leaves),  # num_action_sequences: distinct full-turn sequences scored
            len(sequence),  # num_of_terr_attacked
            max_depth,  # max_tree_depth: deepest plan generated
            1 if complete else 0,  # reached_end_of_turn: every turn sequence was searched
            ";".join(str(len(s)) for _, s, _ in levels),
            # value of stopping at each level of the chosen sequence (its END leaf)
            ";".join(f"{n.end_mean:.4f}" for n, _, _ in levels),
            self.states_visited,
            transpositions,
            sweeps)
        return sequence

    def siblings(self, node, values):
        """END (the node's own leaf) and every scored child, with their values."""
        out = [Sibling("END", node, True, node.end_mean, len(node.values), _std(node.values))]
        for c in node.children:
            mean, samples, best_leaf = values[id(c)]
            if samples:
                out.append(Sibling(f"{c.action.to_terr} | s={c.action.strength}", c, False, mean,
                                   samples, _std(best_leaf.values)))
        return out

    def log_level(self, game_number, round_, tl, siblings, best):
        """One action_seq_stats row for the level and one sibling_values row per sibling."""
        ranked = sorted(siblings, key=lambda s: s.mean, reverse=True)
        ranked.remove(best)
        ranked.insert(0, best)
        second = ranked[1] if len(ranked) > 1 else None
        end = siblings[0]
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
            end.mean, sum(visits), float("nan"))  # level_seconds: the DFS is not split by level
        for s in ranked:
            self.sibling_metric.buffer(self.whoAmI, game_number, round_, tl, s.label,
                                       1 if s is best else 0, s.samples, s.mean, s.std,
                                       best.mean - s.mean)


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
