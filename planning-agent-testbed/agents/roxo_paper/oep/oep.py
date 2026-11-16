"""Online Evolutionary Planning as M. Roxo adapts it to TripleA ("MCTS-based Planning for Grand
Strategy Games", IST 2021, Sec. 5.6, after Justesen et al., IEEE ToG 2018) for the purchase,
combat move and non-combat move. One evolution per phase request.

  * genome: one gene per item of the phase (phases.py: an enemy territory, one of my territories,
    or a unit rule), the item's option chosen for it (units sent / bought, or skip), in a random
    item order: an item served first has more units (or PUs) to choose from;
  * creation (5.6.1): the items in random order, for each a random one of its options from the
    units still available;
  * evaluation (5.6.2): the genome's actions applied, the phase ended, then a playout of at most
    10 rounds by Roxo's Weak AI heuristic for every player, rated with Roxo's reward (playout.py).
    The playout's random choices (the Weak AI's shuffle) and dice are fixed per search, so a value
    is cached per resulting state and survivors are never re-evaluated (Algorithm 3's visits);
  * procreation (5.6.3, Algorithm 7): each survivor with a random other survivor; the items in
    random order, for each the gene of a random one of the two parents, or a random option of the
    item when that gene's units are already used (or it is no longer affordable);
  * mutation (5.6.4): a new genome, with probability MUTATION_RATE, has one random gene replaced by
    another random option of its item; later genes made illegal are replaced by random options;
  * population 50, survival rate 0.5: the best half is kept each generation, the rest refilled
    with offspring, until the time budget is used; the best genome found is played.

A phase whose only action is END is answered without evolving.
"""

import os
import time

import fast_battle
from helper import EFFICIENCY_HEADER, ROLLOUT_STATS_HEADER, MetricLogger
from playout import PLAYOUT_ROUNDS, SearchBase

# TripleMind's columns first (the plot scripts read them by name), then this agent's own.
OEP_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["evaluations", "cache_hits",
                                             "best_value_per_generation", "best_genome", "phase"]
OEP_ROLLOUT_STATS_HEADER = ROLLOUT_STATS_HEADER + ["tl", "phase"]

POPULATION = 50
SURVIVAL_RATE = 0.5
MUTATION_RATE = 0.1


class Genome:
    """One gene (action) per item, applied in `order`; `state` is the phase state they reach (kept
    until evaluated)."""
    __slots__ = ("order", "genes", "state", "fitness")

    def __init__(self, order, genes, state):
        self.order = order
        self.genes = genes
        self.state = state
        self.fitness = None

    def sequence(self):
        return [self.genes[i] for i in self.order]


class OEP(SearchBase):
    def __init__(self, definition, map_graph_, whoAmI, time_budget, population=POPULATION,
                 survival_rate=SURVIVAL_RATE, mutation_rate=MUTATION_RATE,
                 playout_rounds=PLAYOUT_ROUNDS, dice_seeds=1, seed=0, metrics_dir="metrics"):
        super().__init__(definition, map_graph_, whoAmI, time_budget, seed, playout_rounds)
        self.population = max(2, population)
        self.survivors = max(1, min(self.population - 1, int(self.population * survival_rate)))
        self.mutation_rate = mutation_rate  # probability that an offspring is mutated
        # K fixed dice seeds per search, every evaluation averaged over them (fast battles roll no
        # dice, so one is exact there); one fixed seed for the playout's random choices.
        self.k = 1 if fast_battle.DETERMINISTIC else max(1, dice_seeds)
        self.last_stats = None
        self.last_sequence = []

        def logger(name, header):
            return MetricLogger(os.path.join(metrics_dir, name), header=header)

        # efficiency.csv holds the combat-move searches only, so it compares row for row with the
        # combat-only agents; the other phases have their own files.
        self.efficiency_metric = {
            "combatMove": logger("efficiency.csv", OEP_EFFICIENCY_HEADER),
            "purchase": logger("efficiency_purchase.csv", OEP_EFFICIENCY_HEADER),
            "nonCombatMove": logger("efficiency_noncombat.csv", OEP_EFFICIENCY_HEADER),
        }
        # Buffered during a search and written once at its end, to keep file I/O out of the loop.
        self.rollout_stats_metric = logger("rollout_stats.csv", OEP_ROLLOUT_STATS_HEADER)

    # ---- genomes -------------------------------------------------------------------------

    def build(self, phase, root, order, choose):
        """A genome built item by item on a clone of root: choose(item, state, options) returns the
        gene for item, or None for a random option."""
        state = root.clone()
        genes = {}
        for item in order:
            options = phase.options(state, item)
            action = choose(item, state, options)
            if action is None:
                action = self.rnd.choice(options)
            if not phase.step(state, action, self.move_rng):
                action = options[-1]  # rejected by the forward model: the skip
                phase.step(state, action, self.move_rng)
            genes[item] = action
        return Genome(order, genes, state)

    def shuffled(self, items):
        order = list(items)
        self.rnd.shuffle(order)
        return tuple(order)

    def random_genome(self, phase, root, items):
        return self.build(phase, root, self.shuffled(items), lambda item, state, options: None)

    def crossover(self, phase, root, items, a, b):
        """Items in random order, each the gene of a random parent; a random option when it is no
        longer legal."""
        def choose(item, state, options):
            parent = a if self.rnd.random() < 0.5 else b
            return phase.rebuild(state, parent.genes[item])
        return self.build(phase, root, self.shuffled(items), choose)

    def mutate(self, phase, root, genome):
        """One random gene replaced by another option of its item; later genes made illegal are
        replaced by random options."""
        k = self.rnd.choice(genome.order)
        old = phase.action_key(genome.genes[k])

        def choose(item, state, options):
            if item == k:
                others = [o for o in options if phase.action_key(o) != old]
                return self.rnd.choice(others) if others else options[-1]
            return phase.rebuild(state, genome.genes[item])
        return self.build(phase, root, genome.order, choose)

    def evaluate(self, phase, genome, cache, stats, start_round, game_number, generation):
        """END, then the playout, rated with the reward; once per fixed dice seed, averaged."""
        key = phase.key(genome.state)
        if key in cache:
            genome.fitness = cache[key]
            stats["cache_hits"] += 1
        else:
            total = 0.0
            for i, dice in enumerate(self.dice_seeds):
                last = i == len(self.dice_seeds) - 1
                state = genome.state if last else genome.state.clone()  # the last one in place
                reward, rounds, terminal = self.playout(phase, state, phase.end_of(state),
                                                        start_round, self.policy_seed, dice)
                total += reward
                self.rollout_stats_metric.buffer(self.whoAmI, game_number, start_round,
                                                 generation, rounds, reward,
                                                 1 if terminal else 0, len(genome.order),
                                                 phase.name)
            cache[key] = genome.fitness = total / len(self.dice_seeds)
            stats["evaluations"] += 1
        genome.state = None  # no longer needed: offspring are rebuilt from the root

    # ---- search --------------------------------------------------------------------------

    def search(self, phase_name, root_sim):
        """The phase's chosen actions, as forward-model actions for the engine (forced ones
        first)."""
        phase = self.by_name[phase_name]
        started = time.time()
        deadline = started + self.time_budget
        game_number, start_round = root_sim.game_number, root_sim.round
        self.new_search()
        self.dice_seeds = [self.new_seed() for _ in range(self.k)]
        self.policy_seed = self.new_seed()

        root, forced = phase.start(root_sim.clone())
        root_actions = phase.actions(root)
        stats = {"evaluations": 0, "cache_hits": 0}
        best, generations, best_per_generation = None, 0, []
        sequence = []
        if len(root_actions) > 1:  # a single action (END) is no choice
            items = phase.pending(root)
            cache = {}
            population = []
            while len(population) < self.population and (not population
                                                         or time.time() < deadline):
                population.append(self.random_genome(phase, root, items))
            out_of_time = False
            while not out_of_time:
                for genome in population:
                    if genome.fitness is None:
                        if best is not None and time.time() >= deadline:
                            out_of_time = True
                            break
                        self.evaluate(phase, genome, cache, stats, start_round, game_number,
                                      generations)
                    if best is None or genome.fitness > best.fitness:
                        best = genome
                if out_of_time:
                    break
                generations += 1
                best_per_generation.append(best.fitness)
                # survivors: the best half, kept with their values; each makes one offspring with
                # a random other survivor (Algorithm 7)
                population = [g for g in population if g.fitness is not None]
                population.sort(key=lambda g: g.fitness, reverse=True)
                survivors = population[:self.survivors]
                offspring = []
                for parent in survivors:
                    if len(survivors) + len(offspring) >= self.population:
                        break
                    if time.time() >= deadline:
                        out_of_time = True
                        break
                    others = [g for g in survivors if g is not parent] or survivors
                    child = self.crossover(phase, root, items, parent, self.rnd.choice(others))
                    if self.rnd.random() < self.mutation_rate:
                        child = self.mutate(phase, root, child)
                    offspring.append(child)
                while len(survivors) + len(offspring) < self.population and not out_of_time:
                    if time.time() >= deadline:  # population above 2 x survivors: refill
                        out_of_time = True
                        break
                    a, b = (self.rnd.sample(survivors, 2) if len(survivors) > 1
                            else (survivors[0], survivors[0]))
                    offspring.append(self.crossover(phase, root, items, a, b))
                population = survivors + offspring
            sequence = best.sequence()

        search_time = time.time() - started
        self.rollout_stats_metric.flush()
        chosen = [a for a in sequence if not a.is_skip]
        self.last_stats = {"phase": phase.name, "generations": generations,
                           "evaluations": stats["evaluations"], "cache_hits": stats["cache_hits"],
                           "best": round(best.fitness, 4) if best else None,
                           "seconds": round(search_time, 3)}
        self.last_sequence = [phase.label(a) for a in sequence]
        self.efficiency_metric[phase.name].log(
            self.whoAmI,
            game_number,
            start_round,
            search_time,
            generations,  # num_iterations: generations
            best.fitness if best else "",  # root_value: the best genome's value
            len(root_actions),  # num_children_at_root: options of the first item with a choice
            self.playout_rounds,  # max_rollout: rounds a playout may advance
            stats["evaluations"],  # num_action_sequences: distinct genomes evaluated
            len(chosen),  # num_of_terr_attacked: non-skip genes played
            len(best.order) if best else 0,  # max_tree_depth: genome length (items)
            stats["evaluations"],
            stats["cache_hits"],
            ";".join(f"{v:.4f}" for v in best_per_generation),
            " / ".join(self.last_sequence),
            phase.name)
        return forced + phase.response_actions(sequence)
