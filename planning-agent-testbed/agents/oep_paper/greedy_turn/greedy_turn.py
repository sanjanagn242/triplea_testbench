"""Greedy Turn (Justesen et al., "Playing Multiaction Adversarial Games: Online Evolutionary
Planning Versus Tree Search", IEEE ToG 2018, Sec. V) for the TripleA purchase, combat move and
non-combat move.

The paper's Greedy Turn is a depth-first search over a whole turn: every leaf (a complete action
sequence) is rated with the same heuristic as Greedy Action and the sequence leading to the
highest rated state is played. A transposition table skips states already visited, and because
the search is usually not exhaustive the order actions are tried in (action sorting) matters.

It runs once per phase (3 times per turn), with the same atomic actions as Greedy Action, each
list in its action-sorting order:

  purchase       one unit of one land combat rule TripleMind would buy (purchase_rules), its
                 round-robin's next pick first;
  combat move    one attack: (every unit of mine that can reach the target, target territory),
                 for each enemy territory not attacked yet, in TripleMind's territory priority;
  non-combat     one move of TripleMind's noncombat_move plan not applied yet, in its order.

A node is a plan (the actions so far this phase) and every node is also a leaf: ending the phase
there. The number of actions in a phase differs (PUs, targets, plan moves), so the depth is
variable: a branch ends when nothing is left to do. So:

  1. depth-first, children in the action-sorting order: at each node score its leaf (plan + END,
     rated by evaluate_state: in the combat move after the plan is played to the end of the battle
     phase with the forward model, otherwise right after the plan), then expand its actions. A
     child whose state was already reached by another order of the same actions is skipped
     (transposition table: where my moved units stand, and which unit types I hold purchased);
  2. the DFS runs until the time budget is used or every plan was scored. In the combat move a
     finished DFS hands the remaining time to resampling every leaf (battles are random), with
     the same dice seed for every leaf in one sweep; purchase and non-combat roll no dice, so a
     finished DFS is exact;
  3. a node's value is the best mean value of a leaf below it; the chosen sequence follows the
     best child from the root while it beats ending the phase there. In the combat move END wins
     ties (an attack must beat stopping); in purchase and non-combat the child does, so equal
     values go to the action-sorting order. evaluate_state counts neither purchased units in
     hand nor positions, so there the leaves usually tie and the sort order decides.

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
from triplea_fm.state import BATTLE_SELECTION_D, CASUALTIES_D, HELD, RETREAT_D

from generators import ag  # compiled Cython generators when built, else action_generators.py
import map_graph
from helper import EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, ACTION_SEQ_HEADER, MetricLogger

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

PHASES = ("purchase", "combatMove", "nonCombatMove")
END = ag.CombatAction(end_phase=True, strength=0)  # END of every phase


class Node:
    """A plan (actions so far this phase) in the phase tree; `values` are the samples of its END
    leaf."""
    __slots__ = ("sim", "action", "depth", "done", "values", "children")

    def __init__(self, sim, action, depth, done):
        self.sim = sim
        self.action = action  # the action leading here (None at the root)
        self.depth = depth  # actions in the plan
        self.done = done  # those actions, in order
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
        self.noncombat_plan = []
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

    def apply_action(self, sim, action, rng):
        """Apply an action (every move of an attack, or one purchase or move); False (sim partly
        changed) if one is rejected."""
        if isinstance(action, ag.CombatAction):
            return all(self.apply(sim, mv, rng) for mv in action.moves)
        return self.apply(sim, action, rng)

    def view(self, sim):
        return map_graph.GameState.from_sim(self.map, self.d, sim, self.whoAmI)

    def state_key(self, sim):
        """Transposition key of a plan: where each of my moved units stands, and the types of the
        units I hold purchased (their ids follow the purchase order, their types do not)."""
        moved, held = [], []
        for u in sim.units.values():
            if u.owner == self.me:
                if u.loc == HELD:
                    held.append(u.type)
                elif u.moved:
                    moved.append((u.id, u.loc))
        return tuple(sorted(moved)), tuple(sorted(held))

    # ---- actions -------------------------------------------------------------------------

    def candidate_actions(self, phase, sim, done):
        """The actions (END excluded) for the plan sim after the actions `done` this phase, in the
        search's action-sorting order."""
        if phase == "purchase":
            return self.purchase_candidates(sim, len(done))
        if phase == "combatMove":
            return self.combat_candidates(sim, {a.to_terr for a in done})
        return [mv for mv in self.noncombat_plan if mv not in done]

    def purchase_candidates(self, sim, bought):
        """One unit of each affordable rule, in the order TripleMind's round-robin (purchase_actions)
        would buy them after `bought` units: its next pick first."""
        game = self.view(sim)
        pu = game.players[self.whoAmI].PU
        affordable = [name for name, cost, _ in ag.purchase_rules(game, self.whoAmI) if cost <= pu]
        if not affordable:
            return []
        k = bought % len(affordable)
        rule_id = self.d.production_rule_index
        return [fm.PurchaseAction(((rule_id[name], 1),))
                for name in dict.fromkeys(affordable[k:] + affordable[:k])]

    def combat_candidates(self, sim, attacked):
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

    def label(self, a):
        if a is None:
            return "END"
        if isinstance(a, ag.CombatAction):
            return f"{a.to_terr} | s={a.strength}"
        if a.kind == "purchase":
            return " + ".join(f"{self.d.production_rules[rid].name} x{n}" for rid, n in a.quantities)
        names = self.d.territory_names
        return f"{len(a.units)} units {' > '.join(names[t] for t in a.route)}"

    # ---- evaluation ----------------------------------------------------------------------

    def evaluate(self, phase, plan_sim, seed, game_number, tl):
        """Value of ending the phase after plan_sim's actions: in the combat move after this
        turn's battles, with dice from seed, otherwise right after the plan."""
        self.states_visited += 1
        if phase == "combatMove":
            sim = plan_sim.clone()
            rng = fm.SeededRng(seed)
            try:
                self.apply(sim, fm.EndPhaseAction(), rng)  # end of combat move
                self.fight_battles(sim, rng)
            except Exception:
                value, terminal = -0.5, False
            else:
                value, terminal = evaluate_state(self, sim, self.whoAmI)
        else:
            value, terminal = evaluate_state(self, plan_sim, self.whoAmI)
        self.rollout_stats_metric.buffer(self.whoAmI, game_number, plan_sim.round,
                                         self.states_visited, 0, value, 1 if terminal else 0, tl,
                                         phase)
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

    def search(self, root_sim, phase):
        """The best action sequence for root_sim's phase: CombatActions for the combat move,
        fm.PurchaseActions / fm.MoveActions for purchase / non-combat."""
        started = time.time()
        deadline = started + self.time_budget
        game_number, round_ = root_sim.game_number, root_sim.round
        self.states_visited = 0
        if phase == "nonCombatMove":
            self.noncombat_plan = ag.noncombat_move(self.view(root_sim), self.d)
        move_rng = fm.SeededRng(self.rnd.getrandbits(62))  # purchases and moves roll no dice
        base_seed = self.rnd.getrandbits(40)

        # --- depth-first search over the phase, one sample per leaf ---
        root = Node(root_sim.clone(), None, 0, ())
        seen = {self.state_key(root.sim)}
        leaves = []
        stack = [root]
        transpositions = 0
        max_depth = 0
        while stack and time.time() < deadline:
            node = stack.pop()
            node.values.append(self.evaluate(phase, node.sim, base_seed, game_number,
                                             node.depth + 1))
            leaves.append(node)
            for a in self.candidate_actions(phase, node.sim, node.done):
                sim = node.sim.clone()
                if not self.apply_action(sim, a, move_rng):
                    continue
                key = self.state_key(sim)
                if key in seen:
                    transpositions += 1
                    continue
                seen.add(key)
                node.children.append(Node(sim, a, node.depth + 1, node.done + (a,)))
                max_depth = max(max_depth, node.depth + 1)
            stack.extend(reversed(node.children))  # first in the action sorting is searched first
        complete = not stack

        # --- combat move, remaining time: resample every leaf (common dice seed per sweep) ---
        sweeps = 1
        while complete and phase == "combatMove" and time.time() < deadline:
            for leaf in leaves:
                if time.time() >= deadline:
                    break
                leaf.values.append(self.evaluate(phase, leaf.sim, base_seed + sweeps, game_number,
                                                 leaf.depth + 1))
            sweeps += 1

        # --- the chosen sequence: best child while it beats ending the phase there ---
        values = {}
        backup(root, values)
        sequence, levels = [], []
        node = root
        while True:
            siblings = self.siblings(node, values)
            if len(siblings) == 1:
                break  # nothing left to do: not a choice, not logged (as in Greedy Action)
            # The first of equal values: END in the combat move (an attack must beat stopping),
            # the first child in the action sorting otherwise.
            ranked = siblings if phase == "combatMove" else siblings[1:] + siblings[:1]
            best = max(ranked, key=lambda s: s.mean)
            levels.append((node, siblings, best))
            if best.is_end:
                break
            sequence.append(best.node.action)
            node = best.node
        for node, siblings, best in levels:
            self.log_level(phase, game_number, round_, node.depth + 1, siblings, best)

        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        self.last_stats = {"phase": phase, "states_visited": self.states_visited, "leaves": len(leaves),
                           "transpositions": transpositions, "sweeps": sweeps,
                           "complete": complete, "seconds": search_time}
        self.efficiency_metric.log(
            self.whoAmI,
            game_number,
            round_,
            search_time,
            float("nan"),  # num_iterations: no MCTS loop; states_visited is its own column
            values[id(root)][0],  # root_value: value of the chosen sequence
            len(levels[0][1]) if levels else 0,  # num_children_at_root: actions at the root + END
            float("nan"),  # max_rollout: no rollouts in greedy
            len(leaves),  # num_action_sequences: distinct sequences scored
            len(sequence),  # num_of_terr_attacked: sequence length (attacks, units bought, moves)
            max_depth,  # max_tree_depth: deepest plan generated
            1 if complete else 0,  # reached_end_of_turn: every sequence of the phase was searched
            ";".join(str(len(s)) for _, s, _ in levels),
            # value of stopping at each level of the chosen sequence (its END leaf)
            ";".join(f"{n.end_mean:.4f}" for n, _, _ in levels),
            self.states_visited,
            transpositions,
            sweeps,
            phase)
        return sequence

    def siblings(self, node, values):
        """END (the node's own leaf) and every scored child, with their values."""
        out = [Sibling("END", node, True, node.end_mean, len(node.values), _std(node.values))]
        for c in node.children:
            mean, samples, best_leaf = values[id(c)]
            if samples:
                out.append(Sibling(self.label(c.action), c, False, mean, samples,
                                   _std(best_leaf.values)))
        return out

    def log_level(self, phase, game_number, round_, tl, siblings, best):
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
            end.mean, sum(visits), float("nan"),  # level_seconds: the DFS is not split by level
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
