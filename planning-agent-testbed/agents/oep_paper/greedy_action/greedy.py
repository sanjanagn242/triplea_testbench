"""Greedy Action (Justesen et al., "Playing Multiaction Adversarial Games: Online Evolutionary
Planning Versus Tree Search", IEEE ToG 2018, Sec. V) for the TripleA purchase, combat move and
non-combat move.

The paper's Greedy Action does a one-ply search over every legal action, applies the best one,
and repeats until the phase is over. It runs once per phase (3 times per turn), on the actions the
other all-phase OEP agents use (phases.py: one unit bought, one attack, or one territory's
reinforcement set or skip; END), listed in their sorting heuristic's order. Each level of the
sequence:

  1. lists the actions offered by the plan built so far (END included where offered);
  2. scores each one with the forward model and TripleMind's heuristic (evaluate_state): the
     plan + the action, then END (the action's own END, or the END after it), and in the combat
     move this turn's battles (default casualties, no retreat). With dice battles the candidates
     are sampled round-robin until the level's share of the time budget is used (the remaining
     time / Phase.levels_left), with the same dice seed for every candidate in one sweep (common
     random numbers); purchase and non-combat roll no dice, and neither do fast battles, so one
     sweep is exact there;
  3. commits the candidate with the best mean value (the first in the sorting heuristic's order
     of equal values); END ends the sequence. A level whose only action is END is no choice.

If the time runs out before END, the sequence is completed with the sorting heuristic's
highest-rated actions until END. evaluate_state counts neither purchased units in hand nor
positions, so in the purchase and non-combat move the candidates usually tie and the sorting
heuristic decides (comparing whole turns is greedy turn's job).
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

# TripleMind's columns first (the plot scripts read them by name), then this agent's own; `phase`
# (purchase, combatMove, nonCombatMove) last, as one search runs per phase.
GREEDY_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["reached_end_of_turn", "num_actions_per_level",
                                                "best_value_per_level", "phase"]
GREEDY_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "phase"]
# In action_seq_stats.csv "second" is the sibling with the second-best mean value (TripleMind: the
# second most visited); visits are the dice samples of each candidate.
GREEDY_ACTION_SEQ_HEADER = ACTION_SEQ_HEADER + ["value_gap", "end_phase_value", "states_visited",
                                                "level_seconds", "phase"]
# One row per candidate per level: every sibling's mean value and its gap to the best one.
SIBLING_VALUES_HEADER = ["player", "game", "round", "tl", "action", "chosen", "samples",
                         "mean_value", "std_value", "diff_to_best", "phase"]


class Candidate:
    """One action at one level, with the values of its samples."""
    __slots__ = ("action", "values")

    def __init__(self, action):
        self.action = action
        self.values = []

    @property
    def mean(self):
        return sum(self.values) / len(self.values) if self.values else -math.inf


class GreedyActionSearch:
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

    def evaluate(self, phase, state, action, seed, game_number, tl):
        """Value of state + action, then END (the action's own, or the END after it), in the
        combat move after this turn's battles with dice from seed; None if the forward model
        rejects the action."""
        state = state.clone()
        rng = fm.SeededRng(seed)
        if not action.end_phase and not phase.step(state, action, rng):
            return None
        self.states_visited += 1
        try:
            phase.finish(state, action if action.end_phase else phase.end_of(state), rng)
            if phase.name == "combatMove":
                self.fight_battles(state.sim, rng)
        except Exception:
            value, terminal = -0.5, False
        else:
            value, terminal = evaluate_state(self, state.sim, self.whoAmI)
        self.rollout_stats_metric.buffer(self.whoAmI, game_number, state.sim.round,
                                         self.states_visited, 0, value, 1 if terminal else 0, tl,
                                         phase.name)
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
        """The greedy action sequence for root_sim's phase, as forward-model actions for the
        engine (forced ones first)."""
        phase = self.phases[phase_name]
        started = time.time()
        deadline = started + self.time_budget
        game_number, round_ = root_sim.game_number, root_sim.round
        self.states_visited = 0
        self.move_rng = fm.SeededRng(self.rnd.getrandbits(62))  # purchases and moves roll no dice
        dice = phase.name == "combatMove" and not fast_battle.DETERMINISTIC

        state, forced = phase.start(root_sim.clone())
        sequence = []
        levels = []  # per level: (number of actions, best value)
        reached_end = False
        while True:
            tl = len(levels) + 1
            level_started = time.time()
            visited_before = self.states_visited
            actions = phase.actions(state)
            if len(actions) == 1 and actions[0].end_phase:
                sequence.append(actions[0])  # only END: no choice, not logged
                reached_end = True
                break
            if time.time() >= deadline:
                break  # out of time before this level could be scored: completed below
            candidates = [Candidate(a) for a in actions]  # the sorting heuristic's order
            # Dice: this level's share of the remaining time (always at least one full sweep);
            # otherwise one sweep is exact.
            level_deadline = (level_started + (deadline - level_started)
                              / max(1, phase.levels_left(state, actions)) if dice
                              else level_started)
            base_seed = self.rnd.getrandbits(40)
            sweep = 0
            while sweep == 0 or time.time() < level_deadline:
                for c in candidates:
                    if sweep > 0 and time.time() >= level_deadline:
                        break
                    value = self.evaluate(phase, state, c.action, base_seed + sweep,
                                          game_number, tl)
                    if value is not None:
                        c.values.append(value)
                sweep += 1
                if time.time() >= deadline:
                    break
                # Two sweeps with identical values: no dice involved (e.g. undefended targets),
                # more samples cannot change the ranking.
                if sweep >= 2 and all(min(c.values) == max(c.values) for c in candidates if c.values):
                    break
            scored = [c for c in candidates if c.values]
            if not scored:
                break
            best = max(scored, key=lambda c: c.mean)  # the first of equal values
            end = next((c for c in scored if c.action.end_phase), None)
            self.log_level(phase, game_number, round_, tl, scored, best, end,
                           self.states_visited - visited_before, time.time() - level_started)
            levels.append((len(candidates), best.mean))
            sequence.append(best.action)
            if best.action.end_phase:
                reached_end = True
                break
            following = state.clone()
            if not phase.step(following, best.action, self.move_rng):
                sequence.pop()  # cannot happen for a scored candidate; stop rather than loop
                break
            state = following
        if not reached_end:
            sequence += complete_greedily(phase, state, self.move_rng)

        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        self.action_seq_stats_metric.flush()
        self.sibling_metric.flush()
        chosen = [a for a in sequence if not a.end_phase]
        self.last_stats = {"phase": phase.name, "states_visited": self.states_visited,
                           "levels": len(levels), "reached_end_of_turn": reached_end,
                           "seconds": search_time}
        self.last_sequence = [phase.label(a) for a in sequence]
        self.efficiency_metric.log(
            self.whoAmI,
            game_number,
            round_,
            search_time,
            float("nan"),  # num_iterations: no MCTS loop; states_visited is in action_seq_stats
            levels[-1][1] if levels else "",  # root_value: value of the chosen sequence
            levels[0][0] if levels else 0,  # num_children_at_root
            float("nan"),  # max_rollout: no rollouts in greedy
            sum(n for n, _ in levels),  # num_action_sequences: plan prefixes scored
            len(chosen),  # num_of_terr_attacked: actions chosen (attacks, units, reinforcements)
            float("nan"),  # max_tree_depth: no search tree; level count is in last_stats
            1 if reached_end else 0,
            ";".join(str(n) for n, _ in levels),
            ";".join(f"{v:.4f}" for _, v in levels),
            phase.name)
        return forced + phase.response_actions(sequence)

    def log_level(self, phase, game_number, round_, tl, scored, best, end, states, seconds):
        """One action_seq_stats row for the level and one sibling_values row per candidate."""
        ranked = sorted(scored, key=lambda c: c.mean, reverse=True)
        ranked.remove(best)
        ranked.insert(0, best)
        second = ranked[1] if len(ranked) > 1 else None
        visits = [len(c.values) for c in scored]
        means = [c.mean for c in scored]
        mean_v, var_v = _mean_var(visits)
        mean_a, var_a = _mean_var(means)
        z = lambda x, m, var: (x - m) / math.sqrt(var) if var > 1e-12 else ""  # blank if all equal
        self.action_seq_stats_metric.buffer(
            self.whoAmI, game_number, round_, tl, self.whoAmI, phase.label(best.action), len(scored),
            len(best.values), len(best.values) / sum(visits),
            len(second.values) if second else "", mean_v, var_v, math.sqrt(var_v),
            z(len(best.values), mean_v, var_v),
            best.mean, second.mean if second else "", mean_a, math.sqrt(var_a),
            z(best.mean, mean_a, var_a),
            best.mean - second.mean if second else "",
            end.mean if end else "", states, seconds, phase.name)
        for c in ranked:
            _, var_c = _mean_var(c.values)
            self.sibling_metric.buffer(self.whoAmI, game_number, round_, tl, phase.label(c.action),
                                       1 if c is best else 0, len(c.values), c.mean,
                                       math.sqrt(var_c), best.mean - c.mean, phase.name)


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
