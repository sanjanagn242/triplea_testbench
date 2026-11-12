"""TBETS: MCTS over complete turn genomes (one node per player turn), with the genome, decoding and
mutation of Evolutionary MCTS (Baier and Cowling, "Evolutionary MCTS for Multi-Action Adversarial
Games", IEEE CIG 2018) for TripleA's purchase, combat move and noncombat move.

  * genome: one flat sequence of genes for one player's turn, not split by phase (each gene
    carries the phase it plays in): the searching player's remaining turn from the current phase,
    or a later player's whole turn;
  * genes and their options (the atomic actions of heterogeneous-mcts):
      - purchase: one gene per unit bought, a production rule id; its options are the player's
        other candidate units (land combat units bought with PUs, HeterogeneousMCTS.unit_rules);
      - combat: as emcts-fh, one gene per TripleMind target (enemy_territories_by_priority, kept
        if some unit can reach it), each one of TripleMind's attack sets or skip; units in another
        combat gene are not available to it;
      - noncombat: one gene per territory short of cover, most important first, as
        noncombat_generators.noncombat_actions offers them: a reinforcement set (strongest first)
        or skip. The forced air landings come before the genes and the positioning moves after
        them; neither is a gene;
  * decoding (repair psi), in play order, in the state the earlier genes reach:
      - purchase: each stored unit is bought in order if it is still affordable (else dropped),
        then the PUs left are spent as TripleMind does (the first purchase_actions option), so a
        purchase decoded from nothing is TripleMind's purchase;
      - combat: as emcts-fh (a stored gene is kept when its territory is still a target and its
        units still reach it unused; every other target takes its strongest attack set, or skip);
      - noncombat: each territory the generator offers takes its stored option if that is still
        offered, else the strongest reinforcement;
    battles are deterministic (fast_battle, default casualties, no retreat; dice with
    TRIPLEA_ROLLOUT_BATTLE=dice) and placement puts everything bought at the home factory;
  * tree: the root is the current state. Its children are genomes of the searching player's
    remaining turn; a node's children are genomes of the next turn in the game (the next
    player's), played from the state its path reaches, and so on. The root's first child is the
    greedy genome (the Greedy Action AI: from the genome decoded from nothing, each gene in play
    order takes the option whose genome evaluates best for the turn's player);
  * selection: from the root, the best UCB1 child (exploration constant C) down to a leaf; each
    node's mean is from its turn's player's point of view, so every level maximises;
  * expansion, on the way down: every node with children that selection passes through gets one
    new child, a sibling of its children: a mutation of its UCB1 best child that still has a
    mutation left, simulated and backpropagated before selection goes on (so one iteration adds a
    node per level);
  * expansion, at the leaf: with probability greedy_pct (or when the leaf has no mutation left)
    a child, the next turn's greedy genome; otherwise one gene of the leaf is mutated (one of that
    gene's other options, not tried from that genome yet), which makes a sibling of it. A genome
    already among the siblings (after repair) is visited again instead of added; a leaf that ends
    the game with no mutation left is evaluated again;
  * simulation: the path's genomes are decoded turn by turn, then every player plays
    TripleMind's greedy policies until `rollout` rounds have started after the round of the last
    genome's turn (0: stop at the end of that turn; 1: at the end of that round), and the state
    is scored with evaluate_state for the searching player. Evaluations are deterministic and
    cached per path;
  * backpropagation: the value (the searching player's) is added to every node of the path,
    sign-flipped at the nodes of players not allied with the searching player;
  * answer: the root's most visited child (best mean on a tie); its genes of the current phase
    are played.
"""

from __future__ import annotations

import math
import random
import time
from collections import Counter
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import action_generators as ag  # TripleMind's combat, noncombat and purchase action generation
import fast_battle
import map_graph
import noncombat_generators as ncg  # heterogeneous-mcts's atomic noncombat options
import triplea_fm as fm
from triplea_fm import toolkit
from triplea_fm.actions import CONTINUE, RetreatAction, SelectBattleAction
from triplea_fm.definition import COMBAT_MOVE
from triplea_fm.state import BATTLE_SELECTION_D, CASUALTIES_D, MOVE_D, PLACE_D, PURCHASE_D, RETREAT_D
from world_model import SKIP, TERRITORY, Game, TerritoryAction

# The phase a gene plays in.
PURCHASE, COMBAT, NONCOMBAT = "purchase", "combat", "noncombat"


@dataclass(frozen=True)
class Gene:
    """One atomic action of the genome: the phase it plays in and its value (purchase: rule id;
    combat and noncombat: TerritoryAction)."""
    phase: str
    value: object


Genome = Tuple[Gene, ...]  # the remaining turn, in play order


@dataclass
class Turn:
    """What a decoded phase's genes' options are built from, and what it played."""
    player: str
    territories: List[int] = field(default_factory=list)  # combat: the targets
    attackers: Dict[int, Tuple[list, float]] = field(default_factory=dict)  # combat: territory -> (unit infos, defence)
    choices: List[list] = field(default_factory=list)  # purchase, noncombat: each gene's options
    actions: list = field(default_factory=list)  # purchase, noncombat: the forward-model actions played


Turns = Dict[str, Turn]  # per decoded phase


@dataclass(eq=False)
class Node:
    """The root (the current state, no genome) or one player's turn genome, played from the state
    its parent's path reaches."""
    player: Optional[str] = None  # the turn's player
    genome: Genome = ()  # decoded
    turns: Turns = field(default_factory=dict)
    parent: Optional["Node"] = None
    depth: int = 0  # turns from the root
    number: int = 0  # creation order: the root is 0, the greedy genome 1
    origin: str = ""  # "greedy" or "mutation"
    source: Optional["Node"] = None  # a mutation's: the genome mutated
    mutation: Optional[Tuple[int, object, object]] = None  # (gene, old value, new value)
    next_player: Optional[str] = None  # who plays the turn after this one (None: the game ends)
    children: List["Node"] = field(default_factory=list)
    index: Dict[Tuple, "Node"] = field(default_factory=dict)  # children by (player, genome key)
    untried: Optional[List[Tuple[int, object]]] = None  # this genome's mutations not made yet
    visits: int = 0
    value: float = 0.0  # the turn's player's (the root: the searching player's)

    def mean(self) -> float:
        return self.value / self.visits if self.visits else 0.0

    def path(self) -> "Path":
        """(player, genome) of every turn from the root's first child to this node."""
        out, n = [], self
        while n.parent is not None:
            out.append((n.player, n.genome))
            n = n.parent
        return tuple(reversed(out))


Path = Tuple[Tuple[str, Genome], ...]  # one (player, genome) per turn, from the current one
# (value, decoded genome per turn, decoded phases per turn, who plays the turn after the last)
Result = Tuple[float, List[Genome], List[Turns], Optional[str]]


@dataclass
class SearchStats:
    seconds: float = 0.0
    iterations: int = 0
    evaluations: int = 0  # distinct genomes evaluated (as stored, before repair)
    unique_plans: int = 0  # distinct legal plans: decoded (repaired) genomes evaluated
    cache_hits: int = 0
    territories: int = 0  # the greedy genome's combat genes
    genes: int = 0  # the greedy genome's
    root_mutations: int = 0  # mutations the greedy genome offers
    greedy_genomes: int = 0  # greedy genomes created (the first included): children
    mutations: int = 0  # mutations made: siblings
    greedy_value: float = 0.0
    best_value: float = 0.0  # the played genome's mean
    max_depth: int = 0  # deepest node: turns from the root
    nodes: int = 0
    leaves: int = 0
    rollouts: List[Tuple[int, float, bool]] = field(default_factory=list)  # (rounds, reward, terminal)


def key(genome: Genome) -> Tuple:
    """A genome's cache key: its purchase and noncombat genes in order, and its combat genes as
    which units go where and which targets are skipped."""
    rest = tuple((g.phase, g.value) for g in genome if g.phase != COMBAT)
    combat = tuple(sorted((g.value.territory, g.value.kind, g.value.units) for g in genome if g.phase == COMBAT))
    return rest, combat


def with_gene(genome: Genome, i: int, value) -> Genome:
    return genome[:i] + (Gene(genome[i].phase, value),) + genome[i + 1:]


class TBETS:
    def __init__(self, game: Game, map: map_graph.MapGraph, seconds: float = 1.0,
                 exploration: float = 0.0, rollout: int = 2, greedy_pct: float = 0.2, seed: int = 0):
        self.game = game
        self.map = map  # TripleMind's map graph, for its action generation
        self.seconds = seconds
        self.exploration = exploration
        self.rollout = max(0, rollout)  # rounds started after the root's when a rollout stops
        self.rnd = random.Random(seed)
        self.stats = SearchStats()
        self._unit_rules: Dict[str, List[int]] = {}
        self.greedy_pct = greedy_pct  # share of expansions that add the next turn's greedy genome
        # The last search's tree, for tbets_tree: its root and the played genome's node.
        self.root: Optional[Node] = None
        self.best: Optional[Node] = None

    def _setup(self, s):
        d = self.game.d
        self.state = s
        self.player = d.player_names[self.game.actor(s)]
        self.sets_cache = {}
        self.values: Dict[Tuple, Result] = {}  # path key -> result
        self.plans = set()  # keys of the decoded genomes evaluated

    def phase_of(self, s) -> Optional[str]:
        """The phase s's decision plays genes in, or None for a decision played between phases."""
        kind = s.decision.kind
        if kind == PURCHASE_D:
            return PURCHASE
        if kind == MOVE_D:
            return COMBAT if self.game.d.steps[s.step].kind == COMBAT_MOVE else NONCOMBAT
        return None

    def apply(self, s, action) -> bool:
        """Apply an action with the forward model; False if the model rejects it."""
        return fm.apply_in_place(self.game.d, s, action, self.game.rng) != fm.REJECTED

    def view(self, s, player: str):
        return map_graph.GameState.from_sim(self.map, self.game.d, s, player)

    # ---- purchase genes (heterogeneous-mcts's one unit per action) -----------------------------

    def unit_rules(self, player: str) -> List[int]:
        """The player's candidate rules (HeterogeneousMCTS.unit_rules): land combat units bought
        with PUs only, one unit each, since placement puts everything at the home factory."""
        if player not in self._unit_rules:
            d, m = self.game.d, self.map
            pus = d.resource_names.index("PUs")
            rules = []
            for rid in d.frontier[d.player_index[player]]:
                rule = d.production_rules[rid]
                if len(rule.results) != 1 or [r for r, _ in rule.costs] != [pus]:
                    continue
                type_id, qty = rule.results[0]
                ut = m.unit_types.get(d.unit_types[type_id].name)
                if (qty == 1 and rule.costs[0][1] > 0 and ut and ut.domain == "land"
                        and not ut.is_infrastructure and not ut.is_aa):
                    rules.append(rid)
            self._unit_rules[player] = rules
        return self._unit_rules[player]

    def decode_purchase(self, s, player: str, stored: tuple) -> Tuple[tuple, Turn]:
        """Buys the stored units in order, dropping those no longer affordable, then spends the
        PUs left on TripleMind's purchase (the first purchase_actions option)."""
        genes = [rid for rid in stored if self.apply(s, fm.PurchaseAction(((rid, 1),)))]
        actions = ag.purchase_actions(self.view(s, player), self.game.d, player)
        for rid, count in actions[0].quantities if actions else ():
            genes.extend(rid for _ in range(count) if self.apply(s, fm.PurchaseAction(((rid, 1),))))
        options = self.unit_rules(player)
        bought = Counter(genes)
        played = [fm.PurchaseAction(tuple(sorted(bought.items())))] if bought else []
        return tuple(genes), Turn(player, choices=[options] * len(genes), actions=played)

    # ---- combat genes (as emcts-fh) ------------------------------------------------------------

    def turn(self, s) -> Turn:
        """The acting player's targets in s, a combat move: TripleMind's, in priority order, kept
        if some unit can reach them."""
        d = self.game.d
        player = d.player_names[self.game.actor(s)]
        view = map_graph.GameState.from_sim(self.map, d, s, player)
        territories, attackers = [], {}
        for name in ag.enemy_territories_by_priority(view, player):
            found, defence = ag.target_attackers(view, name, player)
            if found:  # combat_actions passes over a target no unit can reach
                t = d.territory_index[name]
                territories.append(t)
                attackers[t] = (found, defence)
        return Turn(player, territories, attackers)

    def attack_sets(self, turn: Turn, t: int, used=()) -> List[Tuple[int, ...]]:
        """TripleMind's attack sets for territory t (unit ids, strongest first) from the attackers
        not in `used`; cached per set of free attackers and defence."""
        attackers, defence = turn.attackers[t]
        free = [x for x in attackers if x[0].uid not in used]
        cache_key = (t, defence, tuple((x[0].uid, x[1]) for x in free))
        sets = self.sets_cache.get(cache_key)
        if sets is None:
            sets = self.sets_cache[cache_key] = [tuple(u.uid for u, _, _ in infos)
                                                 for infos, _ in ag.attack_sets(self.map, free, defence)]
        return sets

    def options(self, turn: Turn, genes: tuple, i: int) -> List[TerritoryAction]:
        """Combat gene i's options given the other combat genes: its attack sets from the free
        units, then skip."""
        t = turn.territories[i]
        used = {u for j, a in enumerate(genes) if j != i for u in a.units}
        return ([TerritoryAction(TERRITORY, t, units) for units in self.attack_sets(turn, t, used)]
                + [TerritoryAction(SKIP, t)])

    def decode_combat(self, turn: Turn, stored: Dict[int, TerritoryAction]) -> tuple:
        """The turn's genes from the stored ones (by territory): a stored skip is kept, a stored
        attack is kept if its units still reach the target and are not used by a gene kept before
        it; every other target takes its strongest attack set from the units left, or skip."""
        kept, used = {}, set()
        for t in turn.territories:
            a = stored.get(t)
            if a is None:
                continue
            if a.kind == TERRITORY:
                reach = {x[0].uid for x in turn.attackers[t][0]}
                if not set(a.units) <= reach or used & set(a.units):
                    continue
                used.update(a.units)
            kept[t] = a
        seg = []
        for t in turn.territories:
            a = kept.get(t)
            if a is None:
                sets = self.attack_sets(turn, t, used)
                a = TerritoryAction(TERRITORY, t, sets[0]) if sets else TerritoryAction(SKIP, t)
                used.update(a.units)
            seg.append(a)
        return tuple(seg)

    def play_combat(self, s, stored: tuple) -> Tuple[tuple, Turn]:
        turn = self.turn(s)
        genes = self.decode_combat(turn, {a.territory: a for a in stored})
        for a in genes:
            if a.kind == TERRITORY:
                self.game.move_units(s, a.territory, a.units)
        return genes, turn

    # ---- noncombat genes (heterogeneous-mcts's one territory per action) ----------------------

    def decode_noncombat(self, s, player: str, stored: tuple) -> Tuple[tuple, Turn]:
        """The forced air landings, then one gene per territory noncombat_actions offers (its
        stored option if still offered, else the strongest reinforcement), then the positioning
        moves."""
        d = self.game.d
        wanted = {a.territory: a for a in stored}
        turn = Turn(player)
        moved, excluded, genes = set(), set(), []

        def play(moves):
            for mv in moves:
                if self.apply(s, mv):
                    moved.update(mv.units)
                    turn.actions.append(mv)

        play(ncg.land_air_moves(self.view(s, player), d, player))
        for _ in range(len(d.territory_names) + 1):
            actions, excluded = ncg.noncombat_actions(self.view(s, player), d, excluded, moved, player)
            if actions[0].end_phase:
                play(actions[0].moves)
                break
            t = d.territory_index[actions[0].to_terr]
            options = [TerritoryAction(TERRITORY, t, tuple(sorted(u for mv in a.moves for u in mv.units)))
                       if a.moves else TerritoryAction(SKIP, t) for a in actions]
            j = options.index(wanted[t]) if wanted.get(t) in options else 0
            play(actions[j].moves)
            genes.append(options[j])
            turn.choices.append(options)
        return tuple(genes), turn

    # ---- the rollout's greedy play and what lies between phases -----------------------------

    def play_greedy(self, s, phase: str, player: str):
        """A rollout decision with TripleMind's greedy policies: the first purchase_actions
        option, the strongest attack set per target, noncombat_move's plan."""
        if phase == PURCHASE:
            actions = ag.purchase_actions(self.view(s, player), self.game.d, player)
            if actions:
                self.apply(s, actions[0])
        elif phase == COMBAT:
            self.play_combat(s, ())
        else:
            for mv in ag.noncombat_move(self.view(s, player), self.game.d, player):
                self.apply(s, mv)

    def _place_units(self, s, player: str) -> bool:
        """Everything bought at the home factory, as tripleMind-with-opp-nodes' rollouts."""
        held = s.held_units(s.decision.actor)
        home = ag._home_factory(self.view(s, player), player)
        if held and home is not None:
            territory = self.game.d.territory_index[home]
            if not self.apply(s, fm.PlaceAction(tuple(u.id for u in held), territory)):
                for u in held:  # over the production limit: place what fits
                    if not self.apply(s, fm.PlaceAction((u.id,), territory)):
                        break
        return self.apply(s, fm.EndPhaseAction())

    def _between_phases(self, s) -> bool:
        """A decision that plays no gene: placement, or battles as the live adapter answers
        them (fight every battle in order, the most likely outcome with fast_battle or dice, default
        casualties, never retreat)."""
        kind = s.decision.kind
        if kind == PLACE_D:
            return self._place_units(s, self.game.d.player_names[s.decision.actor])
        if kind == BATTLE_SELECTION_D:
            pending = [b.id for b in s.battles.values()
                       if b.attacker == s.decision.actor and b.status == "PENDING"]
            return self.apply(s, SelectBattleAction(min(pending))) if pending else self.apply(s, fm.EndPhaseAction())
        if kind == CASUALTIES_D:
            return self.apply(s, toolkit.default_casualties(self.game.d, s))
        if kind == RETREAT_D:
            return self.apply(s, RetreatAction(CONTINUE))
        return self.apply(s, fm.EndPhaseAction())

    # ---- mutations ----------------------------------------------------------------------------

    def gene_options(self, genome: Genome, turns: Turns, i: int) -> list:
        """Gene i's options in its decoded genome (a combat gene's from the units the other combat
        genes leave free)."""
        phase = genome[i].phase
        values = tuple(g.value for g in genome if g.phase == phase)
        j = sum(1 for g in genome[:i] if g.phase == phase)  # its place among its phase's genes
        turn = turns[phase]
        return self.options(turn, values, j) if phase == COMBAT else turn.choices[j]

    def mutations(self, node: Node) -> List[Tuple[int, object]]:
        """Every one-gene change of the node's genome, (gene, new value), in random order."""
        out = [(i, a) for i, g in enumerate(node.genome)
               for a in self.gene_options(node.genome, node.turns, i) if a != g.value]
        self.rnd.shuffle(out)
        return out

    def ally(self, player: str) -> bool:
        """Whether player's turns maximise the searching player's value (the player or an ally)."""
        d = self.game.d
        return d.alliance_of[d.player_index[player]] == d.alliance_of[d.player_index[self.player]]

    # ---- evaluation (simulation) --------------------------------------------------------------

    def evaluate(self, path: Path) -> Result:
        """Decodes the path's genomes turn by turn from the current state (each turn's stored genes
        found by their phase), then plays every decision greedily until `rollout` rounds have
        started after the round of the last genome's turn (0: none), and scores that state with
        evaluate_state for the searching player. Returns (value, decoded genomes, their turns,
        who plays the turn after the last genome's); cached per path and per decoded path."""
        k = tuple((p, key(g)) for p, g in path)
        # A cached value is read back only with fast battles: with dice every evaluation rolls again.
        cached = self.values.get(k) if fast_battle.DETERMINISTIC else None
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        d = self.game.d
        stored: List[Dict[str, list]] = []
        for _, genome in path:
            by_phase: Dict[str, list] = {}
            for g in genome:
                by_phase.setdefault(g.phase, []).append(g.value)
            stored.append(by_phase)
        s = self.state.clone()
        start_round = last_round = s.round
        decoded = [[] for _ in path]
        turns: List[Turns] = [{} for _ in path]
        t, turn_player, next_player = 0, self.player, None
        ok, steps = True, 0
        try:
            while ok and s.terminal is None and steps < 5000:
                steps += 1
                phase = self.phase_of(s)
                if phase is None:
                    ok = self._between_phases(s)
                    continue
                player = d.player_names[s.decision.actor]
                if player != turn_player:  # the next turn
                    turn_player, t = player, t + 1
                    if t < len(path) and path[t][0] != player:
                        t = len(path)  # not the path's player: roll out from here
                    if t == len(path):
                        next_player = player
                if t >= len(path):
                    if s.round >= last_round + self.rollout:
                        break
                    self.play_greedy(s, phase, player)  # the rollout
                    ok = self.apply(s, fm.EndPhaseAction())
                    continue
                last_round = s.round
                mine = tuple(stored[t].get(phase, ()))
                if phase == PURCHASE:
                    values, turn = self.decode_purchase(s, player, mine)
                elif phase == COMBAT:
                    values, turn = self.play_combat(s, mine)
                else:
                    values, turn = self.decode_noncombat(s, player, mine)
                decoded[t].extend(Gene(phase, v) for v in values)
                turns[t][phase] = turn
                ok = self.apply(s, fm.EndPhaseAction())
        except Exception:
            pass  # score the state reached
        value = evaluate_state(d, self.map, s, self.player)
        result = (value, [tuple(g) for g in decoded], turns, next_player)
        self.values[k] = result
        self.values.setdefault(tuple((p, key(g)) for (p, _), g in zip(path, result[1])), result)
        self.plans.add(tuple((p, key(g)) for (p, _), g in zip(path, result[1])))
        self.stats.evaluations += 1
        self.stats.rollouts.append((s.round - start_round, value, s.terminal is not None))
        return result

    # ---- greedy genomes: the Greedy Action AI -------------------------------------------------

    def greedy_genome(self, prefix: Path, player: str, deadline: float) -> Result:
        """The player's turn genome after the prefix's turns: from the genome decoded from
        nothing, each gene in play order takes the option whose path evaluates best for the
        player (highest value for the searching player and allies, lowest for the others; the
        current option on a tie). Stops at the deadline."""
        sign = 1.0 if self.ally(player) else -1.0
        best = self.evaluate(prefix + ((player, ()),))
        i = 0
        while i < len(best[1][-1]) and time.time() < deadline:
            genome, turns = best[1][-1], best[2][-1]
            for a in self.gene_options(genome, turns, i):
                if a != genome[i].value:
                    result = self.evaluate(prefix + ((player, with_gene(genome, i, a)),))
                    if sign * result[0] > sign * best[0]:
                        best = result
            i += 1
        return best

    # ---- tree ---------------------------------------------------------------------------------

    def add(self, parent: Node, player: str, result: Result, origin: str,
            source: Optional[Node] = None, mutation=None) -> Node:
        """The turn genome's node under parent (new unless it is among parent's children already),
        with the value backpropagated from it to the root
        (negated at the nodes of players not allied with the searching player)."""
        value, genomes, turns, next_player = result
        genome = genomes[-1]
        k = (player, key(genome))
        node = parent.index.get(k)
        if node is None:
            self.number += 1
            node = Node(player, genome, turns[-1], parent, parent.depth + 1, self.number, origin,
                        source, mutation, next_player)
            node.untried = self.mutations(node)
            parent.children.append(node)
            parent.index[k] = node
            self.stats.max_depth = max(self.stats.max_depth, node.depth)
        n = node
        while n is not None:
            n.visits += 1
            n.value += value if n.player is None or self.ally(n.player) else -value
            n = n.parent
        return node

    def select(self, node: Node, children: Optional[List[Node]] = None) -> Node:
        """The UCB1 best of node's children (or of `children`, some of them); ties at random."""
        log_n = math.log(node.visits)
        best, best_score = [], None
        for c in node.children if children is None else children:
            score = c.mean() + self.exploration * math.sqrt(log_n / c.visits)
            if best_score is None or score > best_score:
                best, best_score = [c], score
            elif score == best_score:
                best.append(c)
        return best[0] if len(best) == 1 else self.rnd.choice(best)

    def widen(self, node: Node):
        """A node with children gets one new child, a sibling of the others: a mutation of its UCB1
        best child that has a mutation left (none if no child has), simulated and backpropagated."""
        parents = [c for c in node.children if c.untried]
        if not parents:
            return
        c = self.select(node, parents)
        i, a = c.untried.pop()
        self.stats.mutations += 1
        self.add(node, c.player, self.evaluate(node.path() + ((c.player, with_gene(c.genome, i, a)),)),
                 "mutation", c, (i, c.genome[i].value, a))

    def iterate(self, deadline: float):
        """Selection down to a leaf, every node on the way that has children getting one new
        sibling of its children (a mutation); expansion at the leaf (a child: the next turn's
        greedy genome, or a sibling: a mutation of the leaf); simulation; backpropagation."""
        leaf = self.root
        while leaf.children:
            self.widen(leaf)
            leaf = self.select(leaf)
        greedy = self.rnd.random() < self.greedy_pct or not leaf.untried
        if greedy and leaf.next_player is not None:
            self.stats.greedy_genomes += 1
            self.add(leaf, leaf.next_player,
                     self.greedy_genome(leaf.path(), leaf.next_player, deadline), "greedy")
        elif leaf.untried:
            i, a = leaf.untried.pop()
            self.stats.mutations += 1
            prefix = leaf.parent.path()
            self.add(leaf.parent, leaf.player,
                     self.evaluate(prefix + ((leaf.player, with_gene(leaf.genome, i, a)),)),
                     "mutation", leaf, (i, leaf.genome[i].value, a))
        else:  # the game ends after the leaf, which has no mutation left: evaluate it again
            self.add(leaf.parent, leaf.player, self.evaluate(leaf.path()), leaf.origin)

    # ---- search -------------------------------------------------------------------------------

    def run(self, root_state) -> Tuple[Optional[str], tuple, Optional[Turn]]:
        """(phase, values, Turn) of the played genome's genes in root_state's phase (purchase,
        combat move or noncombat move): the Turn's actions are what a purchase or noncombat move
        plays; (None, (), None) if the genome does not reach that phase (the game is over)."""
        started = time.time()
        deadline = started + self.seconds
        stats = self.stats = SearchStats()
        self._setup(root_state)
        self.root = Node()
        self.number = 0
        greedy = self.add(self.root, self.player, self.greedy_genome((), self.player, deadline), "greedy")
        stats.greedy_genomes = 1
        stats.greedy_value = greedy.mean()
        stats.genes = len(greedy.genome)
        stats.territories = sum(1 for g in greedy.genome if g.phase == COMBAT)
        stats.root_mutations = len(greedy.untried)
        if greedy.genome:  # otherwise there is nothing to choose
            while time.time() < deadline:
                self.iterate(deadline)
                stats.iterations += 1

        best = self.best = max(self.root.children, key=lambda c: (c.visits, c.mean()))
        stats.seconds = time.time() - started
        stats.best_value = best.mean()
        stats.nodes, stats.leaves = count_nodes(self.root)
        stats.unique_plans = len(self.plans)
        phase = self.phase_of(root_state)
        if phase not in best.turns:
            return None, (), None
        return phase, tuple(g.value for g in best.genome if g.phase == phase), best.turns[phase]


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


def evaluate_state(d, m: map_graph.MapGraph, s, me: str) -> float:
    """TripleMind's evaluate_state (tripleMind/mcts.py) of forward-model state s for player me at
    the end of this turn. A won game scores 1, a lost one -1; otherwise 0.5 x my share of the
    units' cost (TUV, infrastructure excluded) + 0.5 x my share of the owned territories'
    production, minus 0.5."""
    terminal = s.terminal
    if terminal is not None:
        return 1.0 if d.alliance_of[d.player_index[me]] in terminal.winners else -1.0

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
