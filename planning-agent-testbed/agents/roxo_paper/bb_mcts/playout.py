"""What the roxo_paper searches share besides the actions (phases.py): the forward model helpers, the
playout and the reward of M. Roxo's thesis (Sec. 5.4-5.5).

Shared by roxo_paper/ne_mcts, roxo_paper/bb_mcts and roxo_paper/oep as identical copies (see
../check-shared-files.sh).

  playout   from a state part-way through one of my phases: the rest of that phase by the playout
            policy, then every player's phases by the same policy, for at most PLAYOUT_ROUNDS
            rounds from the decision's round or until the game ends. The policy (Sec. 5.4) is
            Roxo's simplified Weak AI (weak_ai.py): in a combat move, a single unit into each
            undefended enemy territory and just enough units to take the defended ones it beats by
            1.1x; in a non-combat move, one step towards the nearest enemy capital (or enemy
            territory), keeping a threatened capital's units. A purchase buys, rule by rule, the
            first option of phases.PurchasePhase (all it can afford and place of the most attack +
            defence per PU, then the next rule with what is left); placement fills the capital's
            factory first, then the others. Battles: every battle in order, default casualties,
            never retreat. `explore` (BB-MCTS, Sec. 5.7): each phase of the playout is, with that
            probability, played with uniformly random actions of phases.py instead.
  reward    (Algorithm 6) a win 0.6 + 0.4 (R - rounds) / R, a loss the negative, where rounds is how
            many rounds the playout advanced and R = PLAYOUT_ROUNDS; otherwise
            (my units - enemy units) / (2 x all units), units of my alliance against units of
            players at war with me (neutral and unplaced units not counted).
"""

import random

import triplea_fm as fm
from triplea_fm import NULL_PLAYER, rules, toolkit
from triplea_fm.actions import CONTINUE, RetreatAction, SelectBattleAction
from triplea_fm.definition import BATTLE, COMBAT_MOVE, NONCOMBAT_MOVE, PLACE, PURCHASE
from triplea_fm.state import BATTLE_SELECTION_D, CASUALTIES_D, RETREAT_D

import fast_battle
import map_graph
import weak_ai
from phases import actor, make_phases

fast_battle.install()  # the search's battles: dice, or the Fast AI's most likely outcome

PLAYOUT_ROUNDS = 10  # Sec. 5.5: playouts run for at most 10 rounds
MAX_STEPS = 5000  # guard for a playout's walk through the steps


class SearchBase:
    """The forward model, the phases and the playout, for one player of one game."""

    def __init__(self, definition, map_graph_, whoAmI, time_budget, seed=0,
                 playout_rounds=PLAYOUT_ROUNDS):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI
        self.me = definition.player_index[whoAmI]
        self.time_budget = time_budget  # seconds per searched decision
        self.playout_rounds = max(0, playout_rounds)
        self.rnd = random.Random(seed)
        self.move_rng = fm.SeededRng(0)  # moves and purchases roll no dice
        self.route_cache = {}  # phases.Phase.routes, reset per search
        self.land_graph = None  # phases.NoncombatPhase.graph, built on first use
        self.phases = make_phases(self)
        self.by_name = {p.name: p for p in self.phases.values()}

    # ---- forward model -------------------------------------------------------------------

    def apply(self, sim, action, rng):
        return fm.apply_in_place(self.d, sim, action, rng) != fm.REJECTED

    def view(self, sim, player=None):
        return map_graph.GameState.from_sim(self.map, self.d, sim, player or self.whoAmI)

    def new_seed(self):
        return self.rnd.getrandbits(62)

    def new_search(self):
        self.route_cache = {}

    # ---- playout policy -----------------------------------------------------------------

    def heuristic_phase(self, phase, state, rnd, rng):
        """The rest of the phase by the playout heuristic, played on state (not ended). Returns the
        forward-model actions applied."""
        d, sim = self.d, state.sim
        p = actor(d, sim)
        applied = []
        if phase.step_kind == PURCHASE:
            while True:
                action = phase.actions(state)[0]  # the greedy (first) option
                if action.end_phase or not phase.step(state, action, rng):
                    break
                if action.count:
                    applied.append(fm.PurchaseAction(((action.item, action.count),)))
            return applied
        if phase.step_kind == COMBAT_MOVE:
            moves = weak_ai.combat_moves(d, sim, p, rnd)
        else:
            moves = weak_ai.noncombat_moves(d, sim, p, self.phases[NONCOMBAT_MOVE].graph())
        for mv in moves:
            if self.apply(sim, mv, rng):  # a rejected move is skipped
                applied.append(mv)
        return applied

    def random_phase(self, phase, state, rnd, rng):
        """The rest of the phase with uniformly random actions of phases.py (not ended)."""
        for _ in range(MAX_STEPS):
            actions = phase.actions(state)
            action = rnd.choice(actions)
            if action.end_phase or not phase.step(state, action, rng):
                return

    def finish_phase(self, phase, state, rnd, rng, explore=0.0):
        """The rest of the phase by the playout policy, then END."""
        if explore and rnd.random() < explore:
            self.random_phase(phase, state, rnd, rng)
        else:
            self.heuristic_phase(phase, state, rnd, rng)
        return phase.finish(state, phase.end_of(state), rng)

    def play(self, sim, start_round, rnd, rng, explore=0.0):
        """Every player's phases by the playout policy until PLAYOUT_ROUNDS rounds after
        start_round or the end of the game. Returns the rounds advanced."""
        d = self.d
        for _ in range(MAX_STEPS):
            if (sim.terminal is not None or sim.decision is None
                    or sim.round - start_round >= self.playout_rounds):
                break
            step = d.steps[sim.step]
            before = (sim.step, sim.round)
            phase = self.phases.get(step.kind)
            if phase is not None and step.actor != NULL_PLAYER:
                state, _ = phase.start(sim)
                ok = self.finish_phase(phase, state, rnd, rng, explore)
            elif step.kind == BATTLE:
                ok = self.fight_battles(sim, rng)
            elif step.kind == PLACE:
                ok = self.place_all(sim, rng)
            else:
                ok = self.apply(sim, fm.EndPhaseAction(), rng)
            if not ok or (sim.step, sim.round) == before:
                break
        return sim.round - start_round

    def playout(self, phase, state, end, start_round, seed, dice_seed=None, explore=0.0):
        """Reward of state (consumed) from my side: `end` (or the rest of the phase by the policy),
        then the playout. The policy's random choices use `seed`, the battles `dice_seed` (fresh
        when None). Returns (reward, rounds advanced, terminal)."""
        rnd = random.Random(seed)
        rng = fm.SeededRng(dice_seed if dice_seed is not None else self.new_seed())
        try:
            if end is not None:
                ok = phase.finish(state, end, rng)
            else:
                ok = self.finish_phase(phase, state, rnd, rng, explore)
            rounds = self.play(state.sim, start_round, rnd, rng, explore) if ok else 0
            return self.reward(state.sim, rounds), rounds, state.sim.terminal is not None
        except Exception:
            return -0.5, 0, False

    def complete(self, phase, state):
        """What the playout heuristic plays from state for the rest of the phase (fixed shuffle):
        how a search completes a sequence it did not get to choose to the end."""
        return self.heuristic_phase(phase, state, random.Random(0), self.move_rng)

    # ---- battles and placement ----------------------------------------------------------

    def fight_battles(self, sim, rng, max_steps=2000):
        """The battle phase as the live adapter plays it: every battle in order, default
        casualties, never retreat."""
        steps = 0
        ok = True
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
        return ok

    def place_all(self, sim, rng):
        """Everything bought placed: the capital's factory first, then the other factories, as
        many as each takes."""
        d = self.d
        p = sim.decision.actor
        caps = rules.placement_snapshot(d, sim, p)
        sites = sorted(caps, key=lambda t: (d.capital_of[t] != p, t))
        for t in sites:
            held = sim.held_units(p)
            if not held:
                break
            if not self.apply(sim, fm.PlaceAction(tuple(u.id for u in held), t), rng):
                for u in held:  # over the production limit: place what fits
                    if not self.apply(sim, fm.PlaceAction((u.id,), t), rng):
                        break
        return self.apply(sim, fm.EndPhaseAction(), rng)

    # ---- reward (Algorithm 6) -----------------------------------------------------------

    def reward(self, sim, rounds):
        d, me = self.d, self.me
        terminal = sim.terminal
        if terminal is not None:
            value = 0.6 + 0.4 * max(0, self.playout_rounds - rounds) / max(1, self.playout_rounds)
            return value if d.alliance_of[me] in terminal.winners else -value
        mine = enemy = 0
        for u in sim.units.values():
            if u.loc < 0 or u.owner == NULL_PLAYER:
                continue
            if d.is_allied(me, u.owner):
                mine += 1
            else:
                enemy += 1
        total = mine + enemy
        return (mine - enemy) / (2 * total) if total else 0.0
