"""Online Evolutionary Planning as adapted to TripleA in Roxo's thesis (MCTS-based Planning for
Grand Strategy Games, IST, Section 4.4), after Justesen et al.'s OEP (IEEE ToG 2018, Algorithm 1).

One search per move phase. A genome is a set of territory-based actions, one per candidate
territory of the phase, each with the units moved there (or a skip):

  * candidate territories: in a combat move the enemy territories next to the player's (as
    WorldModel lists them), in a noncombat move the player's own territories;
  * a territory's actions are generated as by the forward model (world_model.Game): in a combat move
    the unit sets of getUnitsUpToStrengthSet when the available units can take it (Weak AI strength
    > 1.1x), then skip; in a noncombat move every non-empty combination of the unit groups that can
    reach it, then skip. Units already in another action of the genome are not available.

Genome creation (4.4.1): the territories in random order, a random action for each from the units
still available (a territory served first has more units to choose from, hence the shuffle).
Evaluation (4.4.2): the genome's actions are applied to the current state, the phase is ended
(battles are deterministic, fast_battle), then a deterministic playout guided by the Weak AI
heuristic runs for at most 10 rounds or until the game ends, and Roxo's reward rates the result.
The Weak AI's shuffle is reseeded the same way for every evaluation, so a genome's value is fixed
and it is evaluated once. Procreate (4.4.3): territories in random order, each gene taken from a
random one of the two parents; a gene whose units are already used is replaced by a random action
for the same territory from the units still available. Mutation (4.4.4): a new genome, with
probability 0.1, has one gene replaced by a random action for the same territory.

Generations follow OEP's Algorithm 1: evaluate the new genomes, keep the best half (survival rate
0.5), pair each survivor with a random other survivor to procreate back to the population size
(50), repeat until the time budget is used, play the best genome.
"""

from __future__ import annotations

import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import triplea_fm as fm
import weak_ai
from triplea_fm.definition import COMBAT_MOVE, NONCOMBAT_MOVE
from world_model import SKIP, TERRITORY, Game, TerritoryAction, WorldModel


@dataclass
class Genome:
    genes: List[TerritoryAction]  # one per candidate territory, in the order they were built
    fitness: Optional[float] = None

    def key(self) -> Tuple:
        """The phenotype: which units go where (skips and gene order do not change it)."""
        return tuple(sorted((a.territory, a.units) for a in self.genes if a.kind == TERRITORY))

    def actions(self) -> List[TerritoryAction]:
        return [a for a in self.genes if a.kind == TERRITORY]


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
    def __init__(self, game: Game, seconds: float = 1.0, population: int = 50,
                 survival_rate: float = 0.5, mutation_rate: float = 0.1, playout_rounds: int = 10,
                 seed: int = 0):
        self.game = game
        self.seconds = seconds
        self.population = population
        self.survival_rate = survival_rate
        self.mutation_rate = mutation_rate
        self.playout_rounds = playout_rounds
        self.rnd = random.Random(seed)  # the evolution's own randomness
        self.stats = SearchStats()

    # ---- the phase's territories and their actions ----------------------------------------

    def _setup(self, s):
        """Candidate territories of the phase and, per territory, the units that can reach it
        (a list in a combat move, the unit groups in a noncombat move), from the root state."""
        g = self.game
        world = WorldModel(g, s)
        self.state = s
        self.combat = g.is_combat(s)
        if self.combat:
            self.territories = list(dict.fromkeys(world.enemy))  # unique, in WorldModel order
            self.options = {t: g.move_options(s, t, combat=True) for t in self.territories}
        else:
            self.territories = list(world.allied)
            self.options = {t: g.move_option_groups(s, t) for t in self.territories}

    def random_action(self, t: int, used) -> TerritoryAction:
        """A random one of territory t's actions, from the units not in `used`."""
        g, d, s = self.game, self.game.d, self.state
        if self.combat:
            units = [u for u in self.options[t] if u.id not in used]
            if not units or not g.can_attack_successfully(s, units, t):
                return TerritoryAction(SKIP, t)
            sets = g.units_up_to_strength_sets(weak_ai.strength(d, s.units_in(t), False), units)
            choice = self.rnd.randrange(len(sets) + 1)  # the sets, then skip
            return (TerritoryAction(TERRITORY, t, tuple(u.id for u in sets[choice]))
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
        g, s = self.game, self.state
        for t in self.territories:
            if self.combat:
                units = self.options[t]
                if units and g.can_attack_successfully(s, units, t):
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

    def mutate(self, genome: Genome):
        """4.4.4: with probability mutation_rate, one gene becomes a random action for the same
        territory, from the units the other genes leave free."""
        if not genome.genes or self.rnd.random() >= self.mutation_rate:
            return
        i = self.rnd.randrange(len(genome.genes))
        used = {u for j, x in enumerate(genome.genes) if j != i for u in x.units}
        genome.genes[i] = self.random_action(genome.genes[i].territory, used)

    # ---- evaluation (4.4.2) -----------------------------------------------------------------

    def evaluate(self, genome: Genome, playout_seed: int) -> float:
        g = self.game
        s = self.state.clone()
        for a in genome.actions():
            g.move_units(s, a.territory, a.units)
        start_round = s.round
        g.advance(s)  # end this move phase; its battles are resolved (deterministically)
        phases = 1
        g.rnd = random.Random(playout_seed)  # the Weak AI's shuffle: the same for every genome
        while s.terminal is None and s.round - start_round < self.playout_rounds:
            phases += 1
            p = g.actor(s)
            moves = (weak_ai.combat_moves(g.d, s, p, g.rnd) if g.is_combat(s)
                     else weak_ai.noncombat_moves(g.d, s, p, g.graph))
            for m in moves:
                fm.apply_in_place(g.d, s, m, g.rng)  # a rejected move is skipped, as in Game.playout
            g.advance(s)
        reward = g.reward(s, phases)
        rounds = s.round - start_round
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
                else:
                    fitness[key] = self.evaluate(genome, playout_seed)
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
                child = self.procreate(parent, self.rnd.choice(others))
                self.mutate(child)
                children.append(child)
            population = survivors + children

        stats.seconds = time.time() - started
        stats.best_value = best.fitness if best else 0.0
        return best.actions() if best else []


def move_phases_per_round(d) -> int:
    """Move phases in one round (combat and noncombat move steps of every player)."""
    return sum(1 for st in d.steps if st.kind in (COMBAT_MOVE, NONCOMBAT_MOVE) and st.actor >= 0)
