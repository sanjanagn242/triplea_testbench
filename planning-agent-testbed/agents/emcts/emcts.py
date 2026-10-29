"""Evolutionary MCTS (Baier and Cowling, "Evolutionary MCTS for Multi-Action Adversarial Games",
IEEE CIG 2018) for TripleA's combat move.

EMCTS searches a tree whose nodes are complete action sequences (genomes, as in OEP) and whose
edges are mutations of one gene: a child is its parent's genome with one atomic action changed.

  * genome: one gene per candidate territory, as in roxo-oep-with-purchase's combat genes:
    TripleMind's targets (action_generators.enemy_territories_by_priority, kept if some unit can
    reach them), each gene one of TripleMind's attack sets for its territory (attack_sets: growing
    prefixes of the attackers, best donors first) or skip. Units in another gene are not
    available to it;
  * root (initialisation strategy phi): the Greedy Action AI's genome: the territories in priority
    order, each gene the option whose genome (genes decided so far, the rest skipped) evaluates
    best;
  * edges: every (gene, other option for that gene) pair, the option built from the units the
    other genes leave free, so every mutation is legal and no repair strategy psi is needed. A
    node's mutations are expanded one per iteration, in random order;
  * no rollouts (rollout_depth 1, the paper's setting): each new node's genome is evaluated once
    with the heuristic evaluation function at the end of the turn (the genome's moves, the phase's
    battles resolved with the deterministic Fast AI estimate of fast_battle, TripleMind's
    noncombat plan, then evaluate_state). With rollout_depth n > 1, a rollout then plays every
    player's moves as TripleMind's rollouts do until the end of this player's n-th turn (n - 1
    more rounds) before evaluate_state; its random choices are seeded once per search, so
    evaluations stay deterministic and are cached per phenotype;
  * selection: UCB1 with exploration constant C (0 in the paper: pure exploitation); values are
    averaged in backpropagation, unchanged from MCTS;
  * bridge burning: the time budget is split into phases (40 in the paper); after each phase the
    most promising mutation at the root (most visits, then best mean value) is made, its child
    becomes the root and keeps its subtree. The root after the last phase is played.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import action_generators as ag  # TripleMind's combat and noncombat action generation
import map_graph
import triplea_fm as fm
from world_model import SKIP, TERRITORY, Game, TerritoryAction

Genes = Tuple[TerritoryAction, ...]  # one per candidate territory, in priority order


@dataclass
class Node:
    genes: Genes
    parent: Optional["Node"] = None
    depth: int = 0  # mutations from the search's first root
    children: List["Node"] = field(default_factory=list)
    untried: Optional[List[Tuple[int, TerritoryAction]]] = None  # mutations not expanded yet
    visits: int = 0
    value: float = 0.0

    def mean(self) -> float:
        return self.value / self.visits if self.visits else 0.0


@dataclass
class SearchStats:
    seconds: float = 0.0
    iterations: int = 0
    evaluations: int = 0  # distinct genomes evaluated
    cache_hits: int = 0
    territories: int = 0
    root_mutations: int = 0  # children the first root can have
    greedy_value: float = 0.0
    best_value: float = 0.0  # the played genome's value
    max_depth: int = 0
    nodes: int = 0
    leaves: int = 0
    phases: int = 0  # bridge-burning phases run
    rollouts: List[Tuple[int, float, bool]] = field(default_factory=list)  # (rounds, reward, terminal)


def key(genes: Genes) -> Tuple:
    """The phenotype: which units go where (skips do not change it)."""
    return tuple(sorted((a.territory, a.units) for a in genes if a.kind == TERRITORY))


class EMCTS:
    def __init__(self, game: Game, map: map_graph.MapGraph, seconds: float = 1.0, phases: int = 40,
                 exploration: float = 0.0, seed: int = 0, rollout_depth: int = 1):
        self.game = game
        self.map = map  # TripleMind's map graph, for its action generation
        self.seconds = seconds
        self.phases = phases
        self.exploration = exploration
        self.rollout_depth = max(1, rollout_depth)  # turns of ours evaluated (1: no rollout)
        self.rollout_rnd = random.Random(seed)  # draws each search's rollout seed
        self.playout_seed = 0
        self.rnd = random.Random(seed)
        self.stats = SearchStats()
        # The last search's tree, for emcts_tree: its first root and the root after each phase.
        self.first_root: Optional[Node] = None
        self.path: List[Node] = []

    # ---- the phase's territories and their options (as roxo-oep-with-purchase) ---------------

    def _setup(self, s):
        g, d = self.game, self.game.d
        self.state = s
        self.player = d.player_names[g.actor(s)]
        self.sets_cache = {}
        self.values: Dict[Tuple, float] = {}  # phenotype -> value
        view = map_graph.GameState.from_sim(self.map, d, s, self.player)
        self.territories, self.attackers = [], {}
        for name in ag.enemy_territories_by_priority(view, self.player):
            attackers, defence = ag.target_attackers(view, name, self.player)
            if attackers:  # combat_actions passes over a target no unit can reach
                t = d.territory_index[name]
                self.territories.append(t)
                self.attackers[t] = (attackers, defence)

    def attack_sets(self, t: int, used=()) -> List[Tuple[int, ...]]:
        """TripleMind's attack sets for territory t (unit ids, strongest first) from the attackers
        not in `used`; cached per set of free attackers."""
        attackers, defence = self.attackers[t]
        free = [x for x in attackers if x[0].uid not in used]
        cache_key = (t, tuple(x[0].uid for x in free))
        sets = self.sets_cache.get(cache_key)
        if sets is None:
            sets = self.sets_cache[cache_key] = [tuple(u.uid for u, _, _ in infos)
                                                 for infos, _ in ag.attack_sets(self.map, free, defence)]
        return sets

    def options(self, genes: Genes, i: int) -> List[TerritoryAction]:
        """Gene i's options given the other genes: its attack sets from the free units, then skip."""
        t = self.territories[i]
        used = {u for j, a in enumerate(genes) if j != i for u in a.units}
        return ([TerritoryAction(TERRITORY, t, units) for units in self.attack_sets(t, used)]
                + [TerritoryAction(SKIP, t)])

    def mutations(self, genes: Genes) -> List[Tuple[int, TerritoryAction]]:
        """Every legal one-gene change of the genome (the node's edges), in random order."""
        out = [(i, a) for i in range(len(genes)) for a in self.options(genes, i) if a != genes[i]]
        self.rnd.shuffle(out)
        return out

    # ---- evaluation (no rollouts) -------------------------------------------------------------

    def playout_moves(self, s, rnd: random.Random):
        """The acting player's moves for this move phase, as in TripleMind's rollouts (and
        roxo-oep-with-purchase's playouts): in a combat move, a random one of combat_actions'
        options for each target in turn, in a noncombat move noncombat_move's plan."""
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
            for m in rnd.choice(actions).moves:
                fm.apply_in_place(d, s, m, g.rng)

    def evaluate(self, genes: Genes) -> float:
        """The heuristic value of the state at the end of the searching player's rollout_depth-th
        turn (1: this turn, without a rollout). The genome's moves, the phase's battles
        (deterministic), TripleMind's noncombat plan; with a deeper rollout, then every player's
        moves as TripleMind's rollouts make them until this player has ended rollout_depth - 1
        more turns. The rollout's random choices are seeded the same way for every genome, so a
        genome's value is fixed."""
        k = key(genes)
        value = self.values.get(k)
        if value is not None:
            self.stats.cache_hits += 1
            return value
        g, d = self.game, self.game.d
        s = self.state.clone()
        for a in genes:
            if a.kind == TERRITORY:
                g.move_units(s, a.territory, a.units)
        start_round = s.round
        g.advance(s)  # end the combat move; its battles are resolved, next is our noncombat move
        rnd = random.Random(self.playout_seed)
        turns = 0  # turns of the searching player ended: when play passes from it to another
        was_ours = True
        while s.terminal is None:
            ours = d.player_names[g.actor(s)] == self.player
            if was_ours and not ours:
                turns += 1
                if turns >= self.rollout_depth:
                    break
            if s.round - start_round >= self.rollout_depth:
                break  # our turns no longer come (e.g. we have no units left)
            was_ours = ours
            self.playout_moves(s, rnd)
            g.advance(s)
        rounds = s.round - start_round
        value = self.values[k] = evaluate_state(d, self.map, s, self.player, max(turns - 1, 0),
                                                self.rollout_depth)
        self.stats.evaluations += 1
        self.stats.rollouts.append((rounds, value, s.terminal is not None))
        return value

    # ---- phi: the Greedy Action AI ------------------------------------------------------------

    def greedy_genome(self, deadline: float) -> Genes:
        """Territories in priority order; each gene the option (attack sets, then skip) whose
        genome, with the genes not decided yet skipped, evaluates best (the first on a tie)."""
        genes = [TerritoryAction(SKIP, t) for t in self.territories]
        for i in range(len(genes)):
            if time.time() >= deadline:
                break
            best, best_value = genes[i], None
            for a in self.options(tuple(genes), i):
                genes[i] = a
                v = self.evaluate(tuple(genes))
                if best_value is None or v > best_value:
                    best, best_value = a, v
            genes[i] = best
        return tuple(genes)

    # ---- tree ---------------------------------------------------------------------------------

    def select(self, node: Node) -> Node:
        """UCB1 child (ties at random)."""
        log_n = math.log(node.visits)
        best, best_score = [], None
        for c in node.children:
            score = c.mean() + self.exploration * math.sqrt(log_n / c.visits)
            if best_score is None or score > best_score:
                best, best_score = [c], score
            elif score == best_score:
                best.append(c)
        return best[0] if len(best) == 1 else self.rnd.choice(best)

    def expand(self, node: Node) -> Node:
        i, a = node.untried.pop()
        genes = node.genes[:i] + (a,) + node.genes[i + 1:]
        child = Node(genes, node, node.depth + 1)
        node.children.append(child)
        return child

    def iterate(self, root: Node):
        node = root
        while True:
            if node.untried is None:
                node.untried = self.mutations(node.genes)
            if node.untried:
                node = self.expand(node)
                break
            if not node.children:
                break  # no mutation at all: evaluate the node again (cached)
            node = self.select(node)
        self.stats.max_depth = max(self.stats.max_depth, node.depth)
        value = self.evaluate(node.genes)
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    @staticmethod
    def most_promising(node: Node) -> Node:
        return max(node.children, key=lambda c: (c.visits, c.mean()))

    # ---- search -------------------------------------------------------------------------------

    def run(self, root_state) -> List[TerritoryAction]:
        """The played genome's territory actions for root_state's combat move."""
        started = time.time()
        deadline = started + self.seconds
        stats = self.stats = SearchStats()
        self._setup(root_state)
        self.playout_seed = self.rollout_rnd.getrandbits(32)  # the rollouts' choices, the same for every genome
        stats.territories = len(self.territories)
        root = Node(self.greedy_genome(deadline))
        root.visits, root.value = 1, self.evaluate(root.genes)
        stats.greedy_value = root.value
        root.untried = self.mutations(root.genes)
        stats.root_mutations = len(root.untried)
        first_root = self.first_root = root
        self.path = [root]

        if root.untried:  # otherwise the greedy genome is the only one
            phase_start = time.time()
            phase_seconds = max(0.0, deadline - phase_start) / self.phases
            for phase in range(self.phases):
                phase_end = phase_start + (phase + 1) * phase_seconds
                while time.time() < phase_end:
                    self.iterate(root)
                    stats.iterations += 1
                stats.phases += 1
                if not root.children:
                    break
                root = self.most_promising(root)  # burn the bridges to the other mutations
                self.path.append(root)

        stats.seconds = time.time() - started
        stats.best_value = self.evaluate(root.genes)
        stats.nodes, stats.leaves = count_nodes(first_root)
        return [a for a in root.genes if a.kind == TERRITORY]


def count_nodes(root: Node) -> Tuple[int, int]:
    """(nodes, leaves) of the tree under root."""
    nodes, leaves, stack = 0, 0, [root]
    while stack:
        n = stack.pop()
        nodes += 1
        if n.children:
            stack.extend(n.children)
        else:
            leaves += 1
    return nodes, leaves


def evaluate_state(d, m: map_graph.MapGraph, s, me: str, depth: int, depth_budget: int) -> float:
    """TripleMind's evaluate_state (tripleMind/mcts.py) of forward-model state s for player me,
    after a playout of depth rounds out of depth_budget. A won game scores 0.5 + 0.5 x the share of
    the budget left, a lost one the negative; otherwise 0.5 x my share of the units' cost (TUV,
    infrastructure excluded) + 0.5 x my share of the owned territories' production, minus 0.5."""
    terminal = s.terminal
    if terminal is not None:
        if d.alliance_of[d.player_index[me]] in terminal.winners:
            return 0.5 + 0.5 * (depth_budget - depth) / depth_budget
        return -0.5 - 0.5 * (depth_budget - depth) / depth_budget

    game = map_graph.GameState.from_sim(m, d, s, me)
    my_tuv = 0.0
    enemy_tuv = 0.0
    income = 0
    income_denom = 0
    nodes = m.G.nodes
    for name, terr in game.territories.items():
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

    tuv_term = my_tuv / max(my_tuv + enemy_tuv, 1e-9)
    income_term = income / income_denom if income_denom > 0 else 0
    return 0.50 * tuv_term + 0.50 * income_term - 0.5
