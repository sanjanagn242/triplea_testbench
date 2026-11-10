"""TBETS-seg-hybrid: tbets-segmented's genomes, greedy genomes and mutations, with heterogeneous-mcts's
statistics per gene.

The tree is HeterogeneousMCTS's: a node is a forward-model state at a searched decision, an edge one
atomic action of it (one unit bought, one attack or skip of a target territory, one reinforcement
or skip of a territory short of cover), its statistics kept from the chooser's point of view.

A phase genome is the path of edges of one phase of one player's turn, from the node the phase
starts at (its phase start) to the node its last edge leads to (its phase end, at the next
searched decision). Genomes that share their first genes share those edges, so the tree is a trie
of genomes and every edge's statistics cover all the genomes through it. The tree only ever grows
by whole genomes, so every leaf is a phase end:

  * gene: an edge's action, compared across states by its key (gene_key): a purchase by its
    production rule, a move by its territory and the units it moves (a skip moves none);
  * decoding (repair, tbets-segmented's), from a phase start: each decision takes its stored gene
    if that is still offered, else its strongest action. A purchase buys the stored units in order,
    dropping those no longer offered, then the strongest; a move phase matches stored genes by
    territory;
  * greedy genome (the Greedy Action AI): the genome decoded from nothing (every gene strongest),
    then for each gene in order, every other option of that gene is tried (the genome decoded
    again with that gene changed, played off the tree and rolled out) and the best for the phase's
    player is kept; stops at the deadline;
  * mutation of a genome: one gene changed to another option of its decision (not tried from that
    genome yet, drawn at random), and the genome decoded again from its phase start. A mutation
    that decodes to a genome already in the tree is a duplicate: the next mutation is tried
    instead, and if every one is a duplicate, nothing is added;
  * adding a genome: decoded from its phase start along the tree, following the edges that are
    already there and making the rest (make_edge); simulated once at its phase end and
    backpropagated from there to the root;
  * selection: the best UCB1 edge from the root down to a leaf, gene by gene. With `widen` (as
    tbets-segmented), every phase end with children that selection passes through gets one new
    sibling genome: a mutation of its genome;
  * expansion at the leaf: with probability greedy_pct % (or when the leaf's genome has no mutation
    left) a child, the greedy genome of the next phase; otherwise a mutation of the leaf's genome
    (a sibling). A leaf with no decision after it and no new mutation is simulated again;
  * simulation and backpropagation: HeterogeneousMCTS's;
  * answer: the most visited edge at each decision from the root to the end of the phase.
"""

import random
import time

from hmcts import (COMBAT_PHASE, NONCOMBAT_PHASE, PURCHASE_PHASE, Edge, HeterogeneousMCTS, MCTSNode,
                   PurchaseUnit, no_target)


def gene_key(action):
    """An action's gene, comparable across states."""
    if isinstance(action, PurchaseUnit):
        return action.rule_id
    return action.to_terr, tuple(sorted(u for mv in action.moves for u in mv.units))


def options(state):
    """The state's decision's actions (END is never a gene)."""
    return [a for a in state.actions if not a.end_phase]


class TBETSSegHybrid(HeterogeneousMCTS):
    def __init__(self, *args, greedy_pct=5.0, widen=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.greedy_pct = greedy_pct / 100.0  # share of leaf expansions that add a greedy genome
        self.widen = widen  # mutate every phase end selection passes through, as tbets-segmented
        self.greedy_genomes = 0
        self.mutations = 0
        self.duplicates = 0

    # ---- decoding ------------------------------------------------------------------------

    @staticmethod
    def stored(phase, keys):
        """Genes to decode from: a purchase's in order, a move phase's by territory."""
        return list(keys) if phase == PURCHASE_PHASE else {k[0]: k for k in keys}

    @staticmethod
    def pick(state, stored):
        """The action state's decision takes: its stored gene if still offered, else the strongest
        (the first of the strongest on a tie)."""
        offered = options(state)
        if state.phase == PURCHASE_PHASE:
            by_rule = {a.rule_id: a for a in offered}
            while stored and stored[0] not in by_rule:
                stored.pop(0)  # no longer offered (unaffordable, or before the last unit bought)
            if stored:
                return by_rule[stored.pop(0)]
        else:
            want = stored.get(offered[0].to_terr)
            for a in offered:
                if gene_key(a) == want:
                    return a
        return max(offered, key=lambda a: a.strength)

    def trial(self, state, keys):
        """Decodes a genome from state (a phase start) off the tree: (its genes, each gene's
        options, the state at its phase end)."""
        state = state.clone()
        stored = self.stored(state.phase, keys)
        genes, choices = [], []
        while options(state):
            action = self.pick(state, stored)
            genes.append(gene_key(action))
            choices.append([gene_key(a) for a in options(state)])
            state.played_by = state.next_moves_by
            state.apply_move(action)
            state.legal_moves()
            if no_target(state):
                self.pass_phase(state)
                break
        return genes, choices, state

    # ---- tree ----------------------------------------------------------------------------

    def make_edge(self, node, action):
        """An edge for `action` from node, to the next decision; if `action` ends its chooser's
        phase, the child is the next searched decision (pass_phase)."""
        new_state = node.state.clone()
        new_state.played_by = node.state.next_moves_by
        new_state.apply_move(action)
        new_state.legal_moves()
        passes_phase = no_target(new_state)
        if passes_phase:
            self.pass_phase(new_state)
        edge = Edge(action, new_node(new_state, node.tree_level + 1), self.sign_of(new_state.played_by),
                    passes_phase)
        edge.child.parent = node
        edge.child.in_edge = edge
        return edge

    @staticmethod
    def path_to(node):
        """The edges from the root to node."""
        path = []
        while node.in_edge is not None:
            path.append(node.in_edge)
            node = node.parent
        return path[::-1]

    @staticmethod
    def genome_of(end):
        """(phase start, [(node, edge) per gene]) of the genome a phase end closes."""
        genes, node = [], end
        while True:
            genes.append((node.parent, node.in_edge))
            node = node.parent
            if node.in_edge is None or node.in_edge.passes_phase:
                return node, genes[::-1]

    def place(self, start, keys):
        """Adds the genome decoded from `keys` at phase start `start`, following the edges already
        in the tree: (its phase end, the path to it from the root, whether an edge was made)."""
        stored = self.stored(start.state.phase, keys)
        path, node, made = self.path_to(start), start, False
        while options(node.state):
            action = self.pick(node.state, stored)
            key = gene_key(action)
            edge = node.index.get(key)
            if edge is None:
                edge = self.make_edge(node, action)
                node.edges.append(edge)
                node.index[key] = edge
                made = True
            path.append(edge)
            node = edge.child
            if edge.passes_phase:
                break
        return node, path, made

    def untried_mutations(self, end):
        """The genome's one-gene changes not made yet, (gene, new gene), in random order."""
        if end.mutations is None:
            _, genes = self.genome_of(end)
            end.mutations = [(i, k) for i, (node, edge) in enumerate(genes)
                             for k in (gene_key(a) for a in options(node.state))
                             if k != gene_key(edge.action)]
            random.shuffle(end.mutations)
        return end.mutations

    def rollout(self, state):
        value, depth = self.simulate(state)
        self.max_depth = max(self.max_depth, depth)
        return value

    # ---- greedy genomes and mutations ----------------------------------------------------

    def greedy_genome(self, start, deadline):
        """Adds the greedy genome of start's phase: decoded from nothing, then each gene in order
        takes the option whose genome rolls out best for the phase's player (the current one on a
        tie). At least the genome decoded from nothing is tried, the rest before the deadline."""
        self.greedy_genomes += 1
        sign = self.sign_of(start.state.next_moves_by)
        best, choices, end = self.trial(start.state, ())
        best_value = self.rollout(end)
        i = 0
        while i < len(best) and self.clock() < deadline:
            current = best
            for k in choices[i]:
                if k == current[i]:
                    continue
                genes, genes_choices, end = self.trial(start.state, current[:i] + [k] + current[i + 1:])
                value = self.rollout(end)
                if sign * value > sign * best_value:
                    best, best_value, choices = genes, value, genes_choices
            i += 1
        _, path, _ = self.place(start, best)
        self.backpropagate(path, best_value)

    def mutate(self, end):
        """Adds a sibling genome of the one `end` closes: its next untried mutation that decodes to
        a genome not in the tree yet. False if every one left is a duplicate."""
        start, genes = self.genome_of(end)
        keys = [gene_key(edge.action) for _, edge in genes]
        untried = self.untried_mutations(end)
        while untried:
            i, k = untried.pop()
            new_end, path, made = self.place(start, keys[:i] + [k] + keys[i + 1:])
            if made:
                self.mutations += 1
                self.backpropagate(path, self.rollout(new_end.state))
                return True
            self.duplicates += 1
        return False

    # ---- search --------------------------------------------------------------------------

    def grow(self, root, start_time):
        deadline = start_time + self.time_budget
        self.greedy_genomes = self.mutations = self.duplicates = 0
        root.parent, root.in_edge, root.index, root.mutations = None, None, {}, None
        if no_target(root.state):  # nothing to decide: no search
            return
        self.greedy_genome(root, deadline)
        while self.clock() < deadline:
            self.iteration += 1
            self.iterate(root, deadline)

    def iterate(self, root, deadline):
        node, path = root, []
        while node.edges:
            edge = node.best_edge_uct(self.exploration_constant)
            path.append(edge)
            node = edge.child
            if self.widen and edge.passes_phase and node.edges:
                self.mutate(node)
        can_grow = not node.is_terminal() and not no_target(node.state)
        if can_grow and (random.random() < self.greedy_pct or not self.untried_mutations(node)):
            self.greedy_genome(node, deadline)
        elif self.mutate(node):
            pass
        elif can_grow:  # every mutation left was a duplicate
            self.greedy_genome(node, deadline)
        else:  # nothing after the leaf and no new mutation: simulate it again
            self.backpropagate(path, self.rollout(node.state))

    def play_phase(self, initial_state, chosen):
        """HeterogeneousMCTS's, with the decisions of a combat or non-combat move the chosen path
        stops short of taking their strongest action."""
        state = initial_state.clone()
        for action in chosen:
            state.apply_move(action)
        rest = []
        while state.phase in (COMBAT_PHASE, NONCOMBAT_PHASE):
            state.legal_moves()
            actions = options(state)
            if not actions:
                break
            best = max(actions, key=lambda a: a.strength)
            rest.append(best)
            state.apply_move(best)
        self.last_choice = list(chosen) + rest
        return super().play_phase(initial_state, self.last_choice)


def new_node(state, tree_level):
    """An MCTSNode with this search's bookkeeping: its parent and in-edge (set by make_edge), its
    edges by gene and, at a phase end, its genome's untried mutations."""
    node = MCTSNode(state, tree_level)
    node.parent, node.in_edge, node.index, node.mutations = None, None, {}, None
    return node
