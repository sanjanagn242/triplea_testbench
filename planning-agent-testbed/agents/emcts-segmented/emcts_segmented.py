"""Segmented Evolutionary MCTS: Evolutionary MCTS (Baier and Cowling, "Evolutionary MCTS for
Multi-Action Adversarial Games", IEEE CIG 2018) for TripleA's purchase, combat move and noncombat
move. The genome holds the searching player's phases from the current one to the end of the turn
`horizon` turns after this one. Every other player's turn in between is played greedily and is
only a marker in the genome: EMCTS looks for the player's actions against greedy opponents.

EMCTS searches a tree whose nodes are complete action sequences (genomes) and whose edges are
mutations of one gene: a child is its parent's genome with one atomic action changed.

  * genome: a sequence of segments. Each starts with its marker (Segment.phase, .turn: turns after
    the current one, .player). The searching player's purchase, combat move and noncombat move
    are segments with genes. Another player's turn is a single OTHER_TURN segment with no genes,
    which marks where that turn is played (horizon 1 with two players at my purchase:
    [P C N]mine [other]opp). A phase the game skips (nothing to buy, nothing to move) has no
    segment. Decoding finds each stored segment by its marker (turn, phase);
  * genes and their options (the atomic actions of heterogeneous-mcts):
      - purchase: one gene per unit bought, a production rule id; its options are the player's
        other candidate units (land combat units bought with PUs, HeterogeneousMCTS.unit_rules);
      - combat: as emcts-fh, one gene per TripleMind target (enemy_territories_by_priority, kept
        if some unit can reach it), each one of TripleMind's attack sets or skip; units in another
        gene of the segment are not available to it;
      - noncombat: one gene per territory short of cover, most important first, as
        noncombat_generators.noncombat_actions offers them: a reinforcement set (strongest first)
        or skip. The forced air landings come before the genes and the positioning moves after
        them; neither is a gene;
  * decoding (repair psi), segment by segment, in the state the earlier segments reach:
      - purchase: each stored unit is bought in order if it is still affordable (else dropped),
        then the PUs left are spent as TripleMind does (the first purchase_actions option), so a
        segment decoded from nothing is TripleMind's purchase;
      - combat: as emcts-fh (a stored gene is kept when its territory is still a target and its
        units still reach it unused; every other target takes its strongest attack set, or skip);
      - noncombat: each territory the generator offers takes its stored option if that is still
        offered, else the strongest reinforcement;
      - another player's turn: TripleMind's greedy policies, in the state reached (the first
        purchase_actions option, the strongest attack set per target, noncombat_move's plan);
    between segments battles are deterministic (fast_battle, default casualties, no retreat; dice
    with TRIPLEA_ROLLOUT_BATTLE=dice) and placement puts everything bought at the home factory;
  * root (initialisation strategy phi): this turn's combat segment, if the genome has one, is the
    Greedy Action AI's (the territories in priority order, each gene the option whose genome
    evaluates best), as emcts-fh's; every other segment is decoded from nothing;
  * edges: every (segment, gene, other option for that gene) of the player's segments. Changing a
    gene can change what the later segments offer, which decoding repairs. A node's mutations are
    expanded one per iteration, in random order;
  * evaluation: the decoded genome is played to the first decision of the turn after the last one
    (so the last turn's battles and placement are in) and scored with evaluate_state for the
    searching player. Evaluations are deterministic and cached per genome;
  * selection: UCB1 with exploration constant C (0 in the paper); every edge is the searching
    player's, so means are maximised;
  * bridge burning: the time budget is split into phases (40 in the paper); after each phase the
    most promising mutation at the root is made and its child becomes the root. The first segment
    of the root after the last phase (the current phase's) is played.
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

# Segment markers: the phase a segment's genes play, or another player's whole (greedy) turn.
PURCHASE, COMBAT, NONCOMBAT, OTHER_TURN = "purchase", "combat", "noncombat", "other"


@dataclass(frozen=True)
class Segment:
    """One phase of the searching player's turn, or another player's whole turn: its marker
    (phase, turn: turns after the current one, player), then its genes (purchase: rule ids;
    combat and noncombat: TerritoryAction; another player's turn: none)."""
    phase: str
    turn: int
    player: str
    genes: tuple = ()

    def with_gene(self, i: int, gene) -> "Segment":
        return Segment(self.phase, self.turn, self.player, self.genes[:i] + (gene,) + self.genes[i + 1:])


Genes = Tuple[Segment, ...]  # from the current phase on, in play order


@dataclass
class Turn:
    """What a decoded segment's genes' options are built from, and what it played."""
    player: str
    territories: List[int] = field(default_factory=list)  # combat: the targets
    attackers: Dict[int, Tuple[list, float]] = field(default_factory=dict)  # combat: territory -> (unit infos, defence)
    choices: List[list] = field(default_factory=list)  # purchase, noncombat: each gene's options
    actions: list = field(default_factory=list)  # purchase, noncombat: the forward-model actions played


@dataclass
class Node:
    genes: Genes  # decoded
    parent: Optional["Node"] = None
    depth: int = 0  # mutations from the search's first root
    mutation: Optional[Tuple[int, int, object, object]] = None  # (segment, gene, old, new)
    turns: List[Turn] = field(default_factory=list)  # one per decoded segment
    children: List["Node"] = field(default_factory=list)
    untried: Optional[List[Tuple[int, int, object]]] = None  # mutations not expanded yet
    visits: int = 0
    value: float = 0.0

    def mean(self) -> float:
        return self.value / self.visits if self.visits else 0.0


@dataclass
class SearchStats:
    seconds: float = 0.0
    iterations: int = 0
    evaluations: int = 0  # distinct genomes evaluated (as stored, before repair)
    unique_plans: int = 0  # distinct legal plans: decoded (repaired) genomes evaluated
    cache_hits: int = 0
    territories: int = 0  # this turn's combat segment's
    segments: int = 0  # the first root's
    root_mutations: int = 0  # children the first root can have
    greedy_value: float = 0.0
    best_value: float = 0.0  # the played genome's value
    max_depth: int = 0
    nodes: int = 0
    leaves: int = 0
    phases: int = 0  # bridge-burning phases run
    rollouts: List[Tuple[int, float, bool]] = field(default_factory=list)  # (rounds, reward, terminal)


def key(genes: Genes) -> Tuple:
    """A genome's cache key: per segment, its marker and its genes (a combat segment's as which
    units go where and which targets are skipped)."""
    out = []
    for seg in genes:
        if seg.phase == COMBAT:
            body = tuple(sorted((a.territory, a.kind, a.units) for a in seg.genes))
        else:
            body = seg.genes
        out.append((seg.turn, seg.phase, body))
    return tuple(out)


class EMCTSSegmented:
    def __init__(self, game: Game, map: map_graph.MapGraph, seconds: float = 1.0, phases: int = 40,
                 exploration: float = 0.0, seed: int = 0, horizon: int = 1):
        self.game = game
        self.map = map  # TripleMind's map graph, for its action generation
        self.seconds = seconds
        self.phases = phases
        self.exploration = exploration
        self.horizon = max(0, horizon)  # turns searched after this one
        self.rnd = random.Random(seed)
        self.stats = SearchStats()
        self._unit_rules: Dict[str, List[int]] = {}
        # The last search's tree, for emcts_segmented_tree: its first root and the root after each phase.
        self.first_root: Optional[Node] = None
        self.path: List[Node] = []

    def _setup(self, s):
        d = self.game.d
        self.state = s
        self.player = d.player_names[self.game.actor(s)]
        self.sets_cache = {}
        self.values: Dict[Tuple, Tuple[float, Genes, List[Turn]]] = {}  # key -> (value, decoded, turns)
        self.plans = set()  # keys of the decoded genomes evaluated

    def phase_of(self, s) -> Optional[str]:
        """The segment kind s's decision plays, or None for a decision played between segments."""
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

    # ---- purchase segments (heterogeneous-mcts's one unit per action) --------------------------

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

    # ---- combat segments (as emcts-fh) ---------------------------------------------------------

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
        """Combat gene i's options given the segment's other genes: its attack sets from the free
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

    # ---- noncombat segments (heterogeneous-mcts's one territory per action) -------------------

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

    # ---- another player's turn (greedy) and what lies between segments -----------------------

    def play_greedy(self, s, phase: str, player: str):
        """Another player's decision with TripleMind's greedy policies: the first purchase_actions
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

    def _between_segments(self, s) -> bool:
        """A decision that is not a segment's: placement, or battles as the live adapter answers
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

    def mutations(self, node: Node) -> List[Tuple[int, int, object]]:
        """Every one-gene change of the decoded genome's segments (the node's edges), in random
        order. Another player's turn has no genes, so it is never mutated."""
        out = []
        for k, (turn, seg) in enumerate(zip(node.turns, node.genes)):
            if seg.phase == COMBAT:
                out.extend((k, i, a) for i in range(len(seg.genes))
                           for a in self.options(turn, seg.genes, i) if a != seg.genes[i])
            else:
                out.extend((k, i, a) for i, options in enumerate(turn.choices)
                           for a in options if a != seg.genes[i])
        self.rnd.shuffle(out)
        return out

    # ---- evaluation ---------------------------------------------------------------------------

    def evaluate(self, genes: Genes) -> Tuple[float, Genes, List[Turn]]:
        """Decodes the genome segment by segment, each stored segment found by its marker, from the
        current phase to the first decision of turn horizon + 1, and scores that state with
        evaluate_state for the searching player. Returns (value, decoded genome, its turns);
        cached per genome and per decoded genome."""
        k = key(genes)
        # A cached value is read back only with fast battles: with dice every evaluation rolls again.
        cached = self.values.get(k) if fast_battle.DETERMINISTIC else None
        if cached is not None:
            self.stats.cache_hits += 1
            return cached
        d = self.game.d
        stored = {(seg.turn, seg.phase): seg.genes for seg in genes}
        s = self.state.clone()
        start_round = s.round
        decoded, turns = [], []
        turn_no, last_player, ok, steps = 0, None, True, 0
        try:
            while ok and s.terminal is None and steps < 5000:
                steps += 1
                phase = self.phase_of(s)
                if phase is None:
                    ok = self._between_segments(s)
                    continue
                player = d.player_names[s.decision.actor]
                if player != last_player:
                    if last_player is not None:
                        turn_no += 1
                    if turn_no > self.horizon:
                        break
                    last_player = player
                    if player != self.player:  # the marker of another player's greedy turn
                        decoded.append(Segment(OTHER_TURN, turn_no, player))
                        turns.append(Turn(player))
                if player != self.player:
                    self.play_greedy(s, phase, player)
                else:
                    mine = stored.get((turn_no, phase), ())
                    if phase == PURCHASE:
                        seg_genes, turn = self.decode_purchase(s, player, mine)
                    elif phase == COMBAT:
                        seg_genes, turn = self.play_combat(s, mine)
                    else:
                        seg_genes, turn = self.decode_noncombat(s, player, mine)
                    decoded.append(Segment(phase, turn_no, player, seg_genes))
                    turns.append(turn)
                ok = self.apply(s, fm.EndPhaseAction())
        except Exception:
            pass  # score the state reached
        depth = decoded[-1].turn if decoded else 0
        value = evaluate_state(d, self.map, s, self.player, depth, self.horizon + 1)
        result = (value, tuple(decoded), turns)
        self.values[k] = result
        self.values.setdefault(key(result[1]), result)
        self.plans.add(key(result[1]))
        self.stats.evaluations += 1
        self.stats.rollouts.append((s.round - start_round, value, s.terminal is not None))
        return result

    # ---- phi: the Greedy Action AI for this turn's combat move --------------------------------

    def greedy_genome(self, deadline: float) -> Genes:
        """This turn's combat segment, if the genome has one (searching from a purchase or combat
        move): its territories in priority order, each gene the option (attack sets, then skip)
        whose genome, with the genes not decided yet skipped and every other segment decoded from
        nothing, evaluates best (the first on a tie). Otherwise the empty genome (all decoded from
        nothing)."""
        _, decoded, turns = self.evaluate(())
        k = next((k for k, seg in enumerate(decoded) if seg.turn == 0 and seg.phase == COMBAT), None)
        if k is None:
            return ()
        turn, seg = turns[k], decoded[k]
        self.stats.territories = len(turn.territories)
        genes = [TerritoryAction(SKIP, t) for t in turn.territories]
        for i in range(len(genes)):
            if time.time() >= deadline:
                break
            best, best_value = genes[i], None
            for a in self.options(turn, tuple(genes), i):
                genes[i] = a
                v = self.evaluate((Segment(COMBAT, 0, seg.player, tuple(genes)),))[0]
                if best_value is None or v > best_value:
                    best, best_value = a, v
            genes[i] = best
        return (Segment(COMBAT, 0, seg.player, tuple(genes)),)

    # ---- tree ---------------------------------------------------------------------------------

    def make_node(self, genes: Genes, parent: Optional[Node] = None, mutation=None):
        value, decoded, turns = self.evaluate(genes)
        node = Node(decoded, parent, parent.depth + 1 if parent else 0, mutation, turns)
        return node, value

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

    def expand(self, node: Node) -> Tuple[Node, float]:
        k, i, a = node.untried.pop()
        seg = node.genes[k]
        genes = node.genes[:k] + (seg.with_gene(i, a),) + node.genes[k + 1:]
        child, value = self.make_node(genes, node, (k, i, seg.genes[i], a))
        node.children.append(child)
        return child, value

    def iterate(self, root: Node):
        node = root
        while True:
            if node.untried is None:
                node.untried = self.mutations(node)
            if node.untried:
                node, value = self.expand(node)
                break
            if not node.children:
                value = self.evaluate(node.genes)[0]  # no mutation at all: evaluate again (cached)
                break
            node = self.select(node)
        self.stats.max_depth = max(self.stats.max_depth, node.depth)
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    @staticmethod
    def most_promising(node: Node) -> Node:
        return max(node.children, key=lambda c: (c.visits, c.mean()))

    # ---- search -------------------------------------------------------------------------------

    def run(self, root_state) -> Tuple[Optional[Segment], Optional[Turn]]:
        """The played genome's first segment, root_state's phase (purchase, combat move or
        noncombat move), and its Turn (whose actions a purchase or noncombat move plays); (None,
        None) if the genome is empty (the game is over)."""
        started = time.time()
        deadline = started + self.seconds
        stats = self.stats = SearchStats()
        self._setup(root_state)
        root, value = self.make_node(self.greedy_genome(deadline))
        root.visits, root.value = 1, value
        stats.greedy_value = value
        stats.segments = len(root.genes)
        root.untried = self.mutations(root)
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
        stats.best_value = self.evaluate(root.genes)[0]
        stats.nodes, stats.leaves = count_nodes(first_root)
        stats.unique_plans = len(self.plans)
        return (root.genes[0], root.turns[0]) if root.genes else (None, None)


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
    after depth turns past the first out of depth_budget. A won game scores 0.5 + 0.5 x the
    share of the budget left, a lost one the negative; otherwise 0.5 x my share of the units' cost
    (TUV, infrastructure excluded) + 0.5 x my share of the owned territories' production, minus 0.5."""
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
