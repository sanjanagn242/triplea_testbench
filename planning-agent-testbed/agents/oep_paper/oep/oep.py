"""Online Evolutionary Planning (Justesen et al., "Playing Multiaction Adversarial Games: Online
Evolutionary Planning Versus Tree Search", IEEE ToG 2018, Sec. IV) for the TripleA purchase,
combat move and non-combat move, with the actions and heuristic of the other OEP agents
(../vanilla_mcts). One evolution per phase request.

  * genome: a sequence of up to GENOME_LENGTH = 3 actions of the phase (the paper's 5 action
    points, reduced to 3). A gene is one action of the phase, END included: an END gene ends the
    genome early. Genes are kept as keys (Phase.gene) and replayed through the forward model from
    the phase's start state, so a gene is legal when the state the genes before it reach still
    offers it;
  * population 100, survival rate 0.5: each generation the best half is kept and the rest is
    replaced by offspring of two random survivors;
  * initialization: random legal actions through the forward model until the genome's actions are
    used up (or END is drawn);
  * fitness: the genome's actions applied to a clone of the state, then END, then the rest of my
    turn with no further decisions (battles: default casualties, no retreat, the search's fixed
    dice; placement at the home factory; my other phases ended at once), then ROLLOUT_LENGTH
    player turns played out with TripleMind's deterministic policy for every player (0, the
    default: static evaluation, as the paper found rollout-based fitness no better; 2 in a
    two-player game: the opponent's turn, then mine), rated with evaluate_state. Deterministic, so
    the value is cached per resulting state and survivors are never re-evaluated;
  * crossover, uniform: for each gene, a random parent's gene if it is legal in the child's
    simulated state, otherwise the other parent's, then the next genes of the parents' sequences,
    then a random legal action;
  * mutation, probability 0.1 per offspring: one random gene replaced by another legal action;
    if that makes a later gene illegal, the rest of the sequence is regenerated randomly but
    legally.

The evolution runs until the time budget is used; the best genome found is played. A phase whose
only action is END is answered without evolving.

Actions per phase (as ../vanilla_mcts; their order does not matter here):

  purchase      buy one unit of a candidate rule (land combat units bought with PUs). Gene: the rule.
  combat move   one attack per enemy territory: every unmoved unit of mine that can reach it.
                Gene: the target.
  non-combat    the air the engine would destroy lands first (forced, not a gene); then, one
                territory short of cover at a time, a reinforcement set or its skip; END plays the
                positioning moves. Gene: the territory and the units sent.
"""

import os
import random
import time

import triplea_fm as fm
from triplea_fm import toolkit
from triplea_fm.actions import CONTINUE, RetreatAction, SelectBattleAction
from triplea_fm.definition import (BATTLE, COMBAT_MOVE, END_ROUND, END_TURN, NONCOMBAT_MOVE,
                                   PLACE, PURCHASE)
from triplea_fm.state import BATTLE_SELECTION_D, CASUALTIES_D, RETREAT_D

from generators import ag  # compiled Cython generators when built, else action_generators.py
import fast_battle
import map_graph
import noncombat_generators as ncg
from helper import EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, MetricLogger

fast_battle.install()  # the search's battles: dice, or the Fast AI's most likely outcome

# TripleMind's columns first (the plot scripts read them by name), then this agent's own.
OEP_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["evaluations", "cache_hits",
                                             "best_value_per_generation", "best_genome", "phase"]
OEP_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "phase"]

GENOME_LENGTH = 3  # actions per genome (the paper: 5)
POPULATION = 100
SURVIVAL_RATE = 0.5
MUTATION_RATE = 0.1
ROLLOUT_LENGTH = 0  # player turns played out after mine in an evaluation (0: static evaluation)
ROLLOUT_STRENGTH_MARGIN = 1.5  # TripleMind's first attack threshold (combat_actions)
MAX_TURN_STEPS = 50  # guard for the walk through the rest of the turn
END_GENE = "END"


# =========================
# Genomes
# =========================

class PhaseState:
    """The forward-model state part-way through a phase, with the phase's own bookkeeping
    (`info`, immutable: units bought, territories attacked, or territories decided + units
    moved)."""
    __slots__ = ("sim", "depth", "info")

    def __init__(self, sim, depth, info):
        self.sim = sim
        self.depth = depth  # actions taken in the phase
        self.info = info

    def clone(self):
        return PhaseState(self.sim.clone(), self.depth, self.info)



class Genome:
    """Up to GENOME_LENGTH genes (action keys), the last one END when the genome ends early.
    `actions` are the forward-model actions they decode to, END last; `state` the phase state
    they reach (kept until evaluated)."""
    __slots__ = ("genes", "actions", "state", "fitness")

    def __init__(self, genes, actions, state):
        self.genes = genes
        self.actions = actions
        self.state = state
        self.fitness = None

    def gene_at(self, i):
        """Gene i; END past an early END."""
        return self.genes[i] if i < len(self.genes) else END_GENE


# =========================
# Phases: actions, transitions, genes, keys
# =========================

class PurchaseUnit:
    """One unit of production rule `rule_id`, or END."""
    __slots__ = ("rule_id", "unit_type", "costs", "strength", "end_phase")

    def __init__(self, rule_id=None, unit_type=None, costs=(), strength=0.0, end_phase=False):
        self.rule_id = rule_id
        self.unit_type = unit_type
        self.costs = costs  # ((resource id, amount), ...)
        self.strength = strength  # attack + defence per PU
        self.end_phase = end_phase


class Phase:
    """One searched phase. `info` of a PhaseState is the phase's own, see the subclasses."""
    name = None
    step_kind = None
    end = None  # the END action, when it does not depend on the state

    def __init__(self, search):
        self.search = search
        self.d = search.d

    def start(self, sim):
        """PhaseState at the start of the phase, and the forced moves applied to reach it."""
        raise NotImplementedError

    def actions(self, state):
        """Legal actions, END included (in the sorting heuristic's order, unused here)."""
        raise NotImplementedError

    def step(self, state, action, rng):
        """Apply a non-END action to state in place; False if the forward model rejects it."""
        raise NotImplementedError

    def key(self, state):
        raise NotImplementedError

    def gene(self, action):
        """The action's key in a genome: what makes it the same action in another state."""
        raise NotImplementedError

    def finish(self, state, end, rng):
        """Play END (its moves, if any) and end the phase."""
        return self.search.apply(state.sim, fm.EndPhaseAction(), rng)

    def label(self, action):
        raise NotImplementedError

    def response_actions(self, sequence):
        """The forward-model actions the chosen sequence sends to the engine."""
        return [mv for a in sequence for mv in a.moves]


class PurchasePhase(Phase):
    """info: ((rule id, units bought), ...)."""
    name = "purchase"
    step_kind = PURCHASE
    end = PurchaseUnit(end_phase=True)

    def __init__(self, search):
        super().__init__(search)
        self._units = None

    def units(self):
        """Candidate rules, as purchase_actions picks them (land combat units for PUs only), most
        attack + defence per PU first. Frontiers are fixed, so this is computed once."""
        if self._units is None:
            d, m = self.d, self.search.map
            pus = d.resource_names.index("PUs")
            units = []
            for rid in d.frontier[self.search.me]:
                rule = d.production_rules[rid]
                if len(rule.results) != 1 or [r for r, _ in rule.costs] != [pus]:
                    continue
                type_id, qty = rule.results[0]
                cost = rule.costs[0][1]
                ut = m.unit_types.get(d.unit_types[type_id].name)
                if (qty == 1 and cost > 0 and ut and ut.domain == "land"
                        and not ut.is_infrastructure and not ut.is_aa):
                    units.append(PurchaseUnit(rid, ut.name, rule.costs,
                                              (ut.attack + ut.defense) / cost))
            units.sort(key=lambda a: (-a.strength, a.rule_id))
            self._units = units
        return self._units

    def start(self, sim):
        return PhaseState(sim, 0, ()), []

    def actions(self, state):
        d, m, me = self.d, self.search.map, self.search.me
        sim = state.sim
        has_factory = any(
            u.owner == me and u.loc >= 0 and sim.owner[u.loc] == me
            and d.unit_types[u.type].name in m.unit_types
            and m.unit_types[d.unit_types[u.type].name].can_produce
            for u in sim.units.values())
        if not has_factory:
            return [self.end]
        balance = sim.balances[me]
        return [a for a in self.units()
                if all(balance[r] >= amount for r, amount in a.costs)] + [self.end]

    def step(self, state, action, rng):
        if not self.search.apply(state.sim, fm.PurchaseAction(((action.rule_id, 1),)), rng):
            return False
        bought = dict(state.info)
        bought[action.rule_id] = bought.get(action.rule_id, 0) + 1
        state.info = tuple(sorted(bought.items()))
        state.depth += 1
        return True

    def key(self, state):
        return state.info

    def gene(self, action):
        return END_GENE if action.end_phase else action.rule_id

    def label(self, action):
        return "END" if action.end_phase else f"buy {action.unit_type}"

    def response_actions(self, sequence):
        bought = {}
        for a in sequence:
            if not a.end_phase:
                bought[a.rule_id] = bought.get(a.rule_id, 0) + 1
        return [fm.PurchaseAction(tuple(sorted(bought.items())))] if bought else []


class CombatPhase(Phase):
    """info: frozenset of the territories attacked."""
    name = "combatMove"
    step_kind = COMBAT_MOVE
    end = ag.CombatAction(end_phase=True, strength=0)

    def start(self, sim):
        return PhaseState(sim, 0, frozenset()), []

    def actions(self, state):
        """One attack per enemy target not attacked yet: all my unmoved units that can reach it.
        Rated: attacks beating the defence by ROLLOUT_STRENGTH_MARGIN (TripleMind priority
        order), END, then the other attacks."""
        game = self.search.view(state.sim)
        m = game.map
        me = self.search.whoAmI
        strong, weak = [], []
        for target in ag.enemy_territories_by_priority(game, me):
            if target in state.info:
                continue
            reachable = ag.get_reachable_units(game, target, me)
            if not reachable:
                continue
            action = ag.CombatAction(
                to_terr=target,
                moves=ag._group_moves(self.d, [(u, route) for u, _, route in reachable]),
                strength=ag.calculate_attack_strength(m, [u for u, _, _ in reachable]))
            defence = ag.defender_strength(m, game.territories[target], me)
            (strong if action.strength > ROLLOUT_STRENGTH_MARGIN * defence else weak).append(action)
        return strong + [self.end] + weak

    def step(self, state, action, rng):
        if not all(self.search.apply(state.sim, mv, rng) for mv in action.moves):
            return False
        state.info = state.info | {action.to_terr}
        state.depth += 1
        return True

    def key(self, state):
        me = self.search.me
        return tuple(sorted((u.id, u.loc) for u in state.sim.units.values()
                            if u.owner == me and u.moved))

    def gene(self, action):
        return END_GENE if action.end_phase else action.to_terr

    def label(self, action):
        return "END" if action.end_phase else f"{action.to_terr} | s={action.strength}"


class NoncombatPhase(Phase):
    """info: (frozenset of the territories decided, frozenset of the unit ids moved this phase)."""
    name = "nonCombatMove"
    step_kind = NONCOMBAT_MOVE

    def start(self, sim):
        """The air the engine would destroy lands first: forced, so not a tree ply."""
        search = self.search
        moves = ncg.land_air_moves(search.view(sim), self.d, search.whoAmI)
        moved = set()
        rng = search.move_rng
        applied = []
        for mv in moves:
            if search.apply(sim, mv, rng):
                moved.update(mv.units)
                applied.append(mv)
        return PhaseState(sim, 0, (frozenset(), frozenset(moved))), applied

    def actions(self, state):
        """Reinforcement sets for the most important territory short of cover, strongest first,
        then its skip; END (with the positioning moves) once no territory is left."""
        excluded, moved = state.info
        actions, _ = ncg.noncombat_actions(self.search.view(state.sim), self.d, excluded, moved,
                                           self.search.whoAmI)
        return actions

    def step(self, state, action, rng):
        moved = set(state.info[1])
        for mv in action.moves:
            if not self.search.apply(state.sim, mv, rng):
                return False
            moved.update(mv.units)
        state.info = (action.excluded, frozenset(moved))
        state.depth += 1
        return True

    def key(self, state):
        excluded, moved = state.info
        units = state.sim.units
        return excluded, tuple(sorted((uid, units[uid].loc) for uid in moved if uid in units))

    def gene(self, action):
        """The territory and the units sent there (none for its skip)."""
        if action.end_phase:
            return END_GENE
        return action.to_terr, tuple(sorted(u for mv in action.moves for u in mv.units))

    def finish(self, state, end, rng):
        sim = state.sim
        for mv in end.moves:
            self.search.apply(sim, mv, rng)  # a rejected positioning move is skipped, as live
        return self.search.apply(sim, fm.EndPhaseAction(), rng)

    def label(self, action):
        if action.end_phase:
            return f"END ({len(action.moves)} positioning moves)"
        if not action.moves:
            return f"skip {action.to_terr}"
        return f"{action.to_terr} | s={action.strength:.1f}"


def end_action(actions):
    return next(a for a in actions if a.end_phase)



# =========================
# Search
# =========================

class OEP:
    def __init__(self, definition, map_graph_, whoAmI, time_budget, population=POPULATION,
                 survival_rate=SURVIVAL_RATE, mutation_rate=MUTATION_RATE,
                 genome_length=GENOME_LENGTH, rollout_length=ROLLOUT_LENGTH, seed=0,
                 metrics_dir="metrics"):
        self.d = definition
        self.map = map_graph_
        self.whoAmI = whoAmI
        self.me = definition.player_index[whoAmI]
        self.time_budget = time_budget  # seconds per searched decision
        self.population = max(2, population)
        self.survivors = max(1, min(self.population - 1, int(self.population * survival_rate)))
        self.mutation_rate = mutation_rate  # probability that an offspring is mutated
        self.genome_length = max(1, genome_length)
        self.rollout_length = max(0, rollout_length)  # player turns played out after mine
        self.rnd = random.Random(seed)
        self.move_rng = self.new_rng()  # moves and purchases roll no dice
        self.dice_seed = self.rnd.getrandbits(62)  # every evaluation's dice, redrawn per search
        self.phases = {p.step_kind: p for p in (PurchasePhase(self), CombatPhase(self),
                                                NoncombatPhase(self))}
        self.by_name = {p.name: p for p in self.phases.values()}
        self.last_stats = None
        self.last_sequence = []

        def logger(name, header):
            return MetricLogger(os.path.join(metrics_dir, name), header=header)

        # efficiency.csv holds the combat-move searches only, so it compares row for row with the
        # combat-only OEP agents; the other phases have their own files.
        self.efficiency_metric = {
            "combatMove": logger("efficiency.csv", OEP_EFFICIENCY_HEADER),
            "purchase": logger("efficiency_purchase.csv", OEP_EFFICIENCY_HEADER),
            "nonCombatMove": logger("efficiency_noncombat.csv", OEP_EFFICIENCY_HEADER),
        }
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = logger("rollout_stats.csv", OEP_ROLLOUT_STATS_HEADER)

    # ---- forward model -------------------------------------------------------------------

    def apply(self, sim, action, rng):
        return fm.apply_in_place(self.d, sim, action, rng) != fm.REJECTED

    def view(self, sim, player=None):
        return map_graph.GameState.from_sim(self.map, self.d, sim, player or self.whoAmI)

    def new_rng(self):
        return fm.SeededRng(self.rnd.getrandbits(62))

    def finish_turn(self, sim, rng):
        """My remaining phases after the current one, with no further decisions: battles as the
        live adapter fights them, placement at the home factory, every other phase ended."""
        for _ in range(MAX_TURN_STEPS):
            if sim.terminal is not None or sim.decision is None:
                break
            step = self.d.steps[sim.step]
            if step.actor != self.me or step.kind in (END_TURN, END_ROUND):
                break
            before = (sim.step, sim.round)
            if step.kind == BATTLE:
                ok = self.fight_battles(sim, rng)
            elif step.kind == PLACE:
                ok = self.place_at_home(sim, rng)
            else:
                ok = self.apply(sim, fm.EndPhaseAction(), rng)
            if not ok or (sim.step, sim.round) == before:
                break

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

    def place_at_home(self, sim, rng, player=None):
        """Everything bought goes to the home factory (TripleMind's placement policy)."""
        player = player or self.whoAmI
        held = sim.held_units(sim.decision.actor)
        home = ag._home_factory(self.view(sim, player), player)
        if held and home is not None:
            territory = self.d.territory_index[home]
            if not self.apply(sim, fm.PlaceAction(tuple(u.id for u in held), territory), rng):
                for u in held:  # over the production limit: place what fits
                    if not self.apply(sim, fm.PlaceAction((u.id,), territory), rng):
                        break
        return self.apply(sim, fm.EndPhaseAction(), rng)

    # ---- rollout -------------------------------------------------------------------------

    def play_turns(self, sim, rng, turns):
        """The next `turns` player turns after mine (2: the opponent's, then mine in a two-player
        game), every player with TripleMind's deterministic policy (play_step). Returns the turns
        started; fewer when the game ends first."""
        current, played = self.me, 0
        for _ in range(MAX_TURN_STEPS * (turns + 1)):
            if sim.terminal is not None or sim.decision is None:
                break
            step = self.d.steps[sim.step]
            if step.actor >= 0 and step.actor != current:
                if played == turns:
                    break
                current, played = step.actor, played + 1
            before = (sim.step, sim.round)
            if not self.play_step(sim, step, rng) or (sim.step, sim.round) == before:
                break
        return played

    def play_step(self, sim, step, rng):
        """One step of a rollout turn, as TripleMind plays it: the first of purchase_actions;
        attacks with strength > ROLLOUT_STRENGTH_MARGIN x defence in priority order; battles with
        default casualties, no retreat; noncombat_move; placement at the home factory."""
        if step.kind == BATTLE:
            return self.fight_battles(sim, rng)
        player = self.d.player_names[step.actor] if step.actor >= 0 else None
        if player is None:
            pass
        elif step.kind == PURCHASE:
            actions = ag.purchase_actions(self.view(sim, player), self.d, player)
            if actions:
                self.apply(sim, actions[0], rng)
        elif step.kind == COMBAT_MOVE:
            self.play_attacks(sim, player, rng)
        elif step.kind == NONCOMBAT_MOVE:
            for mv in ag.noncombat_move(self.view(sim, player), self.d, player):
                self.apply(sim, mv, rng)  # a rejected move is skipped, as live
        elif step.kind == PLACE:
            return self.place_at_home(sim, rng, player)
        return self.apply(sim, fm.EndPhaseAction(), rng)

    def play_attacks(self, sim, player, rng):
        """The combat move's highest-rated attacks (CombatPhase's rating, for any player): the
        first target in priority order whose reachable units beat its defence by
        ROLLOUT_STRENGTH_MARGIN, until none is left."""
        attacked = set()
        while True:
            game = self.view(sim, player)
            m = game.map
            for target in ag.enemy_territories_by_priority(game, player):
                if target in attacked:
                    continue
                reachable = ag.get_reachable_units(game, target, player)
                if not reachable:
                    continue
                strength = ag.calculate_attack_strength(m, [u for u, _, _ in reachable])
                defence = ag.defender_strength(m, game.territories[target], player)
                if strength > ROLLOUT_STRENGTH_MARGIN * defence:
                    break
            else:
                return
            attacked.add(target)
            for mv in ag._group_moves(self.d, [(u, route) for u, _, route in reachable]):
                self.apply(sim, mv, rng)

    # ---- genomes -------------------------------------------------------------------------

    def build(self, phase, root, choose):
        """A genome built gene by gene on a clone of root: choose(i, offered) returns the action
        for gene i from `offered` ({gene: action} legal here), or None for a random legal one."""
        state = root.clone()
        genes, actions = [], []
        for i in range(self.genome_length):
            offered = phase.actions(state)
            by_gene = {phase.gene(a): a for a in offered}
            action = choose(i, by_gene)
            if action is None:
                action = self.rnd.choice(offered)
            if not action.end_phase and not phase.step(state, action, self.move_rng):
                action = end_action(offered)  # rejected by the forward model: end here
            genes.append(phase.gene(action))
            actions.append(action)
            if action.end_phase:
                return Genome(tuple(genes), actions, state)
        actions.append(end_action(phase.actions(state)))  # actions used up: END
        return Genome(tuple(genes), actions, state)

    def random_genome(self, phase, root):
        """Random legal actions until the genome's actions are used up (or END)."""
        return self.build(phase, root, lambda i, offered: None)

    def crossover(self, phase, root, a, b):
        """Uniform: gene i from a random parent if legal in the child's state, else the other
        parent's, else the parents' next genes, else a random legal action."""
        def choose(i, offered):
            first, second = (a, b) if self.rnd.random() < 0.5 else (b, a)
            for j in range(i, self.genome_length):
                for parent in (first, second):
                    action = offered.get(parent.gene_at(j))
                    if action is not None:
                        return action
            return None
        return self.build(phase, root, choose)

    def mutate(self, phase, root, genome):
        """One random gene replaced by another legal action; once a later gene is illegal, the
        rest is regenerated randomly but legally."""
        k = self.rnd.randrange(len(genome.genes))
        regenerating = False

        def choose(i, offered):
            nonlocal regenerating
            if i < k:
                return offered.get(genome.genes[i])  # same prefix, so still legal
            if i == k:
                others = [act for g, act in offered.items() if g != genome.genes[k]]
                return self.rnd.choice(others) if others else offered.get(genome.genes[k])
            if not regenerating:
                action = offered.get(genome.genes[i]) if i < len(genome.genes) else None
                if action is not None:
                    return action
                regenerating = True
            return None
        return self.build(phase, root, choose)

    def evaluate(self, phase, genome, cache, stats, game_number, round_, generation):
        """The genome's actions (already on genome.state), END, the rest of my turn without
        decisions, then rollout_length player turns (play_turns; 0: static evaluation),
        evaluate_state. Deterministic, so cached per resulting phase state."""
        key = phase.key(genome.state)
        if key in cache:
            genome.fitness = cache[key]
            stats["cache_hits"] += 1
        else:
            state = genome.state
            rng = fm.SeededRng(self.dice_seed)
            turns = 0
            try:
                if phase.finish(state, genome.actions[-1], rng):
                    self.finish_turn(state.sim, rng)
                    if self.rollout_length:
                        turns = self.play_turns(state.sim, rng, self.rollout_length)
                reward, terminal = evaluate_state(self, state.sim, self.whoAmI)
            except Exception:
                reward, terminal = -0.5, False
            cache[key] = genome.fitness = reward
            stats["evaluations"] += 1
            self.rollout_stats_metric.buffer(self.whoAmI, game_number, round_, generation, turns,
                                             reward, 1 if terminal else 0, len(genome.actions) - 1,
                                             phase.name)
        genome.state = None  # no longer needed: offspring are rebuilt from the root

    # ---- search --------------------------------------------------------------------------

    def search(self, phase_name, root_sim):
        """The phase's chosen actions, as forward-model actions for the engine (forced ones
        first)."""
        phase = self.by_name[phase_name]
        started = time.time()
        deadline = started + self.time_budget
        game_number, round_ = root_sim.game_number, root_sim.round
        self.move_rng = self.new_rng()
        self.dice_seed = self.rnd.getrandbits(62)  # the same dice for every evaluation

        root, forced = phase.start(root_sim.clone())
        root_actions = phase.actions(root)
        stats = {"evaluations": 0, "cache_hits": 0}
        best, generations, best_per_generation = None, 0, []
        if len(root_actions) > 1:  # a single action (END) is no choice
            cache = {}
            population = [self.random_genome(phase, root) for _ in range(self.population)]
            out_of_time = False
            while not out_of_time:
                for genome in population:
                    if genome.fitness is None:
                        if best is not None and time.time() >= deadline:
                            out_of_time = True
                            break
                        self.evaluate(phase, genome, cache, stats, game_number, round_,
                                      generations)
                    if best is None or genome.fitness > best.fitness:
                        best = genome
                if out_of_time:
                    break
                generations += 1
                best_per_generation.append(best.fitness)
                # survivors: the best part, kept with their values; the rest: offspring
                population.sort(key=lambda g: g.fitness, reverse=True)
                survivors = population[:self.survivors]
                offspring = []
                while len(survivors) + len(offspring) < self.population:
                    if time.time() >= deadline:
                        out_of_time = True
                        break
                    a, b = (self.rnd.sample(survivors, 2) if len(survivors) > 1
                            else (survivors[0], survivors[0]))
                    child = self.crossover(phase, root, a, b)
                    if self.rnd.random() < self.mutation_rate:
                        child = self.mutate(phase, root, child)
                    offspring.append(child)
                population = survivors + offspring
            sequence = best.actions
        else:
            sequence = root_actions

        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        chosen = [a for a in sequence if not a.end_phase]
        self.last_stats = {"phase": phase.name, "generations": generations,
                           "evaluations": stats["evaluations"], "cache_hits": stats["cache_hits"],
                           "best": round(best.fitness, 4) if best else None,
                           "seconds": round(search_time, 3)}
        self.last_sequence = [phase.label(a) for a in sequence]
        self.efficiency_metric[phase.name].log(
            self.whoAmI,
            game_number,
            round_,
            search_time,
            generations,  # num_iterations: generations
            best.fitness if best else "",  # root_value: the best genome's value
            len(root_actions),  # num_children_at_root: legal first actions, END included
            self.rollout_length,  # max_rollout: player turns played out after mine
            stats["evaluations"],  # num_action_sequences: distinct genomes evaluated
            len(chosen),  # num_of_terr_attacked: actions chosen (attacks, units, reinforcements)
            self.genome_length,  # max_tree_depth: genome length
            stats["evaluations"],
            stats["cache_hits"],
            ";".join(f"{v:.4f}" for v in best_per_generation),
            " / ".join(self.last_sequence),
            phase.name)
        return forced + phase.response_actions(sequence)


# =========================
# State evaluation: TripleMind's evaluate_state (mcts.py), as the other OEP agents use it: on a
# bare forward-model state, without the rollout-depth discount and scaled to [-1, 1]:
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
