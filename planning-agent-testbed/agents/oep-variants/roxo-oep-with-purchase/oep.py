"""Online Evolutionary Planning as adapted to TripleA in Roxo's thesis (MCTS-based Planning for
Grand Strategy Games, IST, Section 4.4), after Justesen et al.'s OEP (IEEE ToG 2018, Algorithm 1).

One search per move phase. A genome is a set of territory-based actions, one per candidate
territory of the phase, each with the units moved there (or a skip):

  * candidate territories: in a combat move TripleMind's targets (action_generators: the land
    territories next to the player's that it does not own, most important first) that some unit
    can reach; in a noncombat move the player's own territories;
  * a territory's actions: in a combat move TripleMind's attack sets for it (combat_actions'
    attack_sets: growing prefixes of the attackers, best donors first, whose strength passes 1.5x
    the defence, the full force if it beats the defence, or the cheapest unit for an undefended
    territory), then skip; in a noncombat move, as by Roxo's forward model (world_model.Game), every
    non-empty combination of the unit groups that can reach it, then skip. Units already in
    another action of the genome are not available.

Genome creation (4.4.1): the territories in random order, a random action for each from the units
still available (a territory served first has more units to choose from, hence the shuffle).
Evaluation (4.4.2): the genome's actions are applied to the current state, the phase is ended
(battles are deterministic, fast_battle), then a playout runs for at most 10 rounds or until the
game ends, and TripleMind's evaluate_state rates the result. The playout moves as TripleMind's
rollouts do
(mcts._rollout_combat and _rollout_noncombat) instead of with Roxo's Weak AI: in a combat
move, a random one of combat_actions' options for the next target until none is left; in a
noncombat move, noncombat_move's plan. Purchase and place phases still buy nothing (Roxo's forward model).
The random choices are reseeded the same way for every evaluation, so a genome's value is fixed
and it is evaluated once. Procreate (4.4.3): territories in random order, each gene taken from a
random one of the two parents; a gene whose units are already used is replaced by a random action
for the same territory from the units still available. Mutation (4.4.4): a new genome, with
probability 0.1, has one gene replaced by a random action for the same territory.

Generations follow OEP's Algorithm 1: evaluate the new genomes, keep the best half (survival rate
0.5), pair each survivor with a random other survivor to procreate back to the population size
(50), repeat until the time budget is used, play the best genome.

With trace on, run() also records the genealogy of the distinct genomes it evaluated (self.tree, one
GenomeNode per phenotype) for oep_tree.save_oep_tree to draw after the search.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import action_generators as ag  # TripleMind's combat action generation
import map_graph
import triplea_fm as fm
from triplea_fm.definition import COMBAT_MOVE, NONCOMBAT_MOVE
from world_model import SKIP, TERRITORY, Game, TerritoryAction, WorldModel


@dataclass
class Genome:
    genes: List[TerritoryAction]  # one per candidate territory, in the order they were built
    fitness: Optional[float] = None
    # Recorded only when tracing: the generation it was made for, its parents' phenotypes, mutation.
    generation: int = 0
    parents: Tuple[Tuple, ...] = ()
    mutated: bool = False

    def key(self) -> Tuple:
        """The phenotype: which units go where (skips and gene order do not change it)."""
        return tuple(sorted((a.territory, a.units) for a in self.genes if a.kind == TERRITORY))

    def actions(self) -> List[TerritoryAction]:
        return [a for a in self.genes if a.kind == TERRITORY]


@dataclass
class GenomeNode:
    """A distinct phenotype of the search's genealogy, as first produced (for oep_tree)."""
    index: int  # order of discovery
    generation: int  # generation it was first evaluated in (0: the random initial population)
    fitness: float
    parents: Tuple[Tuple, ...]  # the phenotypes of the parents of the genome that first produced it
    mutated: bool
    actions: List[TerritoryAction]
    produced: int = 1  # genomes with this phenotype made during the search (duplicates included)


@dataclass
class SearchStats:
    seconds: float = 0.0
    generations: int = 0
    evaluations: int = 0
    duplicates: int = 0  # genomes whose phenotype was already evaluated
    territories: int = 0
    best_value: float = 0.0
    max_rollout: int = 0  # rounds
    best_per_generation: List[float] = field(default_factory=list)
    rollouts: List[Tuple[int, float, bool]] = field(default_factory=list)  # (rounds, reward, terminal)


class RoxoOEP:
    def __init__(self, game: Game, map: map_graph.MapGraph, seconds: float = 1.0,
                 population: int = 50, survival_rate: float = 0.5, mutation_rate: float = 0.1,
                 playout_rounds: int = 10, seed: int = 0, trace: bool = False):
        self.game = game
        self.map = map  # TripleMind's map graph, for its combat action generation
        self.seconds = seconds
        self.population = population
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.playout_rounds = playout_rounds
        self.rnd = random.Random(seed)  # the evolution's own randomness
        self.stats = SearchStats()
        self.trace = trace  # record the genealogy (costs a little search time)
        self.tree: Dict[Tuple, GenomeNode] = {}  # phenotype -> node, of the last run when tracing
        self.best_key: Optional[Tuple] = None

    # ---- the phase's territories and their actions ----------------------------------------

    def _setup(self, s):
        """Candidate territories of the phase and, per territory, what its actions are built from,
        from the root state: in a combat move TripleMind's attackers (best donors first) and the
        defence; in a noncombat move the unit groups that can reach it."""
        g, d = self.game, self.game.d
        self.state = s
        self.combat = g.is_combat(s)
        self.sets_cache = {}
        if self.combat:
            player = d.player_names[g.actor(s)]
            view = map_graph.GameState.from_sim(self.map, d, s, player)
            self.territories, self.options = [], {}
            for name in ag.enemy_territories_by_priority(view, player):
                attackers, defence = ag.target_attackers(view, name, player)
                if attackers:  # combat_actions passes over a target no unit can reach
                    t = d.territory_index[name]
                    self.territories.append(t)
                    self.options[t] = (attackers, defence)
        else:
            world = WorldModel(g, s)
            self.territories = list(world.allied)
            self.options = {t: g.move_option_groups(s, t) for t in self.territories}

    def attack_sets(self, t: int, used=()) -> List[Tuple[int, ...]]:
        """TripleMind's attack sets for territory t (unit ids, strongest first) from the attackers
        not in `used`; cached per set of free attackers."""
        attackers, defence = self.options[t]
        free = [x for x in attackers if x[0].uid not in used]
        key = (t, tuple(x[0].uid for x in free))
        sets = self.sets_cache.get(key)
        if sets is None:
            sets = self.sets_cache[key] = [tuple(u.uid for u, _, _ in infos)
                                           for infos, _ in ag.attack_sets(self.map, free, defence)]
        return sets

    def random_action(self, t: int, used) -> TerritoryAction:
        """A random one of territory t's actions, from the units not in `used`."""
        if self.combat:
            sets = self.attack_sets(t, used)
            choice = self.rnd.randrange(len(sets) + 1)  # the sets, then skip
            return (TerritoryAction(TERRITORY, t, sets[choice])
                    if choice < len(sets) else TerritoryAction(SKIP, t))
        groups = [[u for u in grp if u.id not in used] for grp in self.options[t]]
        groups = [grp for grp in groups if grp]
        # Uniform over the 2^k - 1 non-empty combinations of the k groups and skip (subset 0).
        subset = self.rnd.randrange(1 << len(groups))
        if subset == 0:
            return TerritoryAction(SKIP, t)
        units = tuple(u.id for i, grp in enumerate(groups) if subset >> i & 1 for u in grp)
        return TerritoryAction(TERRITORY, t, units)

    def _has_choice(self) -> bool:
        """True if some territory has more than one action with every unit available."""
        for t in self.territories:
            if self.combat:
                if self.attack_sets(t):
                    return True  # at least one unit set and skip
            elif self.options[t]:
                return True  # at least one group combination and skip
        return False

    # ---- genetic operators ----------------------------------------------------------------

    def shuffled_territories(self) -> List[int]:
        order = list(self.territories)
        self.rnd.shuffle(order)
        return order

    def new_genome(self) -> Genome:
        """4.4.1: territories in random order, a random action for each from the free units."""
        used, genes = set(), []
        for t in self.shuffled_territories():
            a = self.random_action(t, used)
            used.update(a.units)
            genes.append(a)
        return Genome(genes)

    def procreate(self, a: Genome, b: Genome) -> Genome:
        """4.4.3: territories in random order, each gene from a random parent; a gene whose units
        are already used becomes a random action for its territory from the free units."""
        by_territory = ({x.territory: x for x in a.genes}, {x.territory: x for x in b.genes})
        used, genes = set(), []
        for t in self.shuffled_territories():
            gene = by_territory[self.rnd.randrange(2)][t]
            if any(u in used for u in gene.units):
                gene = self.random_action(t, used)
            used.update(gene.units)
            genes.append(gene)
        return Genome(genes)

    def mutate(self, genome: Genome) -> bool:
        """4.4.4: with probability mutation_rate, one gene becomes a random action for the same
        territory, from the units the other genes leave free. True if a gene was replaced."""
        if not genome.genes or self.rnd.random() >= self.mutation_rate:
            return False
        i = self.rnd.randrange(len(genome.genes))
        used = {u for j, x in enumerate(genome.genes) if j != i for u in x.units}
        genome.genes[i] = self.random_action(genome.genes[i].territory, used)
        return True

    # ---- evaluation (4.4.2) -----------------------------------------------------------------

    def playout_moves(self, s, rnd: random.Random):
        """The acting player's moves for this move phase, as in TripleMind's rollouts: in a combat
        move, a random one of combat_actions' options (attack sets, then skip) for each target in
        turn, in a noncombat move noncombat_move's plan. A rejected move is skipped."""
        g, d = self.game, self.game.d
        player = d.player_names[g.actor(s)]
        if not g.is_combat(s):
            view = map_graph.GameState.from_sim(self.map, d, s, player)
            for m in ag.noncombat_move(view, d, player):
                fm.apply_in_place(d, s, m, g.rng)
            return
        excluded = set()
        while True:
            view = map_graph.GameState.from_sim(self.map, d, s, player)
            actions, excluded = ag.combat_actions(view, d, excluded, player)
            if not actions:
                return
            for m in rnd.choice(actions).moves:  # moves roll no dice; battles come with advance
                fm.apply_in_place(d, s, m, g.rng)

    def evaluate(self, genome: Genome, playout_seed: int) -> float:
        g = self.game
        s = self.state.clone()
        for a in genome.actions():
            g.move_units(s, a.territory, a.units)
        start_round = s.round
        g.advance(s)  # end this move phase; its battles are resolved (deterministically)
        rnd = random.Random(playout_seed)  # the rollout's choices: the same for every genome
        while s.terminal is None and s.round - start_round < self.playout_rounds:
            self.playout_moves(s, rnd)
            g.advance(s)
        rounds = s.round - start_round
        reward = evaluate_state(g.d, self.map, s, g.d.player_names[g.player], rounds,
                                self.playout_rounds)
        self.stats.evaluations += 1
        self.stats.max_rollout = max(self.stats.max_rollout, rounds)
        self.stats.rollouts.append((rounds, reward, s.terminal is not None))
        return reward

    # ---- search (OEP, Algorithm 1) ------------------------------------------------------------

    def run(self, root_state) -> List[TerritoryAction]:
        """The best genome's territory actions for root_state's move phase."""
        started = time.time()
        deadline = started + self.seconds
        stats = self.stats = SearchStats()
        self._setup(root_state)
        stats.territories = len(self.territories)
        playout_seed = self.rnd.getrandbits(32)
        fitness: Dict[Tuple, float] = {}  # phenotype -> value (evaluations are deterministic)
        self.tree = {}
        self.best_key = None

        # No territory with a choice (e.g. none can be attacked): one genome is the whole space.
        population = [self.new_genome() for _ in range(self.population if self._has_choice() else 1)]
        best: Optional[Genome] = None
        while time.time() < deadline:
            for genome in population:
                if genome.fitness is not None:
                    continue
                if time.time() >= deadline:
                    break
                key = genome.key()
                if key in fitness:
                    stats.duplicates += 1
                    if self.trace:
                        self.tree[key].produced += 1
                else:
                    fitness[key] = self.evaluate(genome, playout_seed)
                    if self.trace:
                        self.tree[key] = GenomeNode(len(self.tree), genome.generation, fitness[key],
                                                    genome.parents, genome.mutated, genome.actions())
                genome.fitness = fitness[key]
            scored = sorted((x for x in population if x.fitness is not None),
                            key=lambda x: x.fitness, reverse=True)
            if not scored:
                break
            if best is None or scored[0].fitness > best.fitness:
                best = scored[0]
            stats.generations += 1
            stats.best_per_generation.append(best.fitness)
            if time.time() >= deadline or len(population) == 1:
                break
            # 50% elitism, then procreation back to the population size
            survivors = scored[:max(1, int(len(scored) * self.survival_rate))]
            children = []
            while len(survivors) + len(children) < self.population:
                parent = survivors[len(children) % len(survivors)]
                others = [x for x in survivors if x is not parent] or survivors
                other = self.rnd.choice(others)
                child = self.procreate(parent, other)
                mutated = self.mutate(child)
                if self.trace:
                    child.generation = stats.generations
                    child.parents = tuple(dict.fromkeys((parent.key(), other.key())))
                    child.mutated = mutated
                children.append(child)
            population = survivors + children

        stats.seconds = time.time() - started
        stats.best_value = best.fitness if best else 0.0
        self.best_key = best.key() if best else None
        return best.actions() if best else []


def evaluate_state(d, m: map_graph.MapGraph, s, me: str, depth: int, depth_budget: int) -> float:
    """TripleMind's evaluate_state (tripleMind/mcts.py) of forward-model state s for player me,
    after a playout of depth rounds out of depth_budget. A won game scores 0.5 + 0.5 x the share of
    the budget left, a lost one the negative; otherwise 0.5 x my share of the units' cost (TUV,
    infrastructure excluded) + 0.5 x my share of the owned territories' production, minus 0.5."""
    # --- terminal check: the forward model's victory test, whose winners are alliance ids ---
    terminal = s.terminal
    if terminal is not None:
        if d.alliance_of[d.player_index[me]] in terminal.winners:
            return 0.5 + 0.5 * (depth_budget - depth) / depth_budget
        return -0.5 - 0.5 * (depth_budget - depth) / depth_budget

    game = map_graph.GameState.from_sim(m, d, s, me)
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

    score = 0.50 * tuv_term + 0.50 * income_term
    return score - 0.5


def move_phases_per_round(d) -> int:
    """Move phases in one round (combat and noncombat move steps of every player)."""
    return sum(1 for st in d.steps if st.kind in (COMBAT_MOVE, NONCOMBAT_MOVE) and st.actor >= 0)
