"""TBETS-M-UCT: tbets-segmented's greedy seeding and mutation, on heterogeneous-mcts's tree of atomic
actions.

The tree is HeterogeneousMCTS's: a node is a forward-model state at a searched decision, an edge one
atomic action of it (one unit bought, one attack or skip of a target territory, one reinforcement
or skip of a territory short of cover), its statistics kept from the chooser's point of view. What
changes is how the tree grows: a node's actions are not expanded one after the other. Instead, as
in tbets-segmented, where a gene is an atomic action and a mutation changes one gene:

  * greedy child: the first child of a node is the greedy action of its decision: every action is
    tried (applied, then rolled out as simulate() does) and the one whose rollout is best for the
    chooser is kept (strongest first, so the strongest wins a tie; stops at the deadline);
  * mutation: a mutation of a node is another action of its parent's decision (not an edge of the
    parent yet, drawn at random), so it makes a sibling of the node;
  * selection: from the root, the best UCB1 edge down to a leaf. With `widen`, every node with
    edges that selection passes through first gets one new edge, a mutation of its children (while
    it has an action left), simulated and backpropagated before selection goes on;
  * expansion, at the leaf: with probability greedy_pct % (or when its parent has no action left)
    a child, the greedy action of the leaf's decision; otherwise a mutation of the leaf (a
    sibling). A leaf with no decision after it (game over, or nothing left to decide) and no
    mutation left is simulated again;
  * simulation and backpropagation: HeterogeneousMCTS's;
  * answer: the most visited path from the root to the end of the phase, as HeterogeneousMCTS;
    where the tree stops short of the end, every remaining decision of the phase takes its
    strongest action (tbets's decoding from nothing).
"""

import random
import time

from hmcts import (COMBAT_PHASE, NONCOMBAT_PHASE, Edge, HeterogeneousMCTS, MCTSNode, no_target)


class TBETSMUCT(HeterogeneousMCTS):
    def __init__(self, *args, greedy_pct=5.0, widen=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.greedy_pct = greedy_pct / 100.0  # share of leaf expansions that add a greedy child
        self.widen = widen  # mutate every node selection passes through, as tbets-segmented
        self.greedy_children = 0
        self.mutations = 0

    # ---- edges ---------------------------------------------------------------------------

    @staticmethod
    def untried(node):
        """node's actions that are not edges yet, in random order (END is never an edge)."""
        if node.untried_actions is None:
            node.untried_actions = [a for a in node.state.actions if not a.end_phase]
            random.shuffle(node.untried_actions)
        return node.untried_actions

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
        return Edge(action, MCTSNode(new_state, node.tree_level + 1),
                    self.sign_of(new_state.played_by), passes_phase)

    def add(self, node, edge, value, path):
        """Adds edge to node and backpropagates value along path + edge."""
        node.edges.append(edge)
        untried = self.untried(node)
        if edge.action in untried:
            untried.remove(edge.action)
        self.backpropagate(path + [edge], value)

    def rollout(self, state):
        value, depth = self.simulate(state)
        self.max_depth = max(self.max_depth, depth)
        return value

    # ---- greedy children and mutations -----------------------------------------------------

    def greedy_child(self, node, path, deadline):
        """Adds node's greedy child: of its decision's actions (strongest first), the one whose
        rollout is best for the chooser, keeping the earlier on a tie; at least one is tried,
        the rest only before the deadline."""
        self.greedy_children += 1
        actions = sorted((a for a in node.state.actions if not a.end_phase),
                         key=lambda a: a.strength, reverse=True)
        best, best_value = None, None
        for action in actions:
            if best is not None and self.clock() >= deadline:
                break
            edge = self.make_edge(node, action)
            value = self.rollout(edge.child.state)
            if best is None or edge.sign * value > edge.sign * best_value:
                best, best_value = edge, value
        self.add(node, best, best_value, path)

    def mutate(self, parent, path):
        """Adds a sibling of parent's children: another of parent's actions, at random."""
        self.mutations += 1
        edge = self.make_edge(parent, self.untried(parent).pop())
        self.add(parent, edge, self.rollout(edge.child.state), path)

    # ---- search --------------------------------------------------------------------------

    def grow(self, root, start_time):
        deadline = start_time + self.time_budget
        self.greedy_children = self.mutations = 0
        if no_target(root.state):  # nothing to decide: no search
            return
        self.greedy_child(root, [], deadline)
        while self.clock() < deadline:
            self.iteration += 1
            self.iterate(root, deadline)

    def iterate(self, root, deadline):
        node, path = root, []
        while node.edges:
            if self.widen and self.untried(node):
                self.mutate(node, path)
            edge = node.best_edge_uct(self.exploration_constant)
            path.append(edge)
            node = edge.child
        parent = path[-2].child if len(path) > 1 else root
        can_mutate = bool(self.untried(parent))
        can_grow = not node.is_terminal() and not no_target(node.state)
        if can_grow and (random.random() < self.greedy_pct or not can_mutate):
            self.greedy_child(node, path, deadline)
        elif can_mutate:
            self.mutate(parent, path[:-1])
        else:  # nothing after the leaf and no mutation left: simulate it again
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
            actions = [a for a in state.actions if not a.end_phase]
            if not actions:
                break
            best = max(actions, key=lambda a: a.strength)
            rest.append(best)
            state.apply_move(best)
        self.last_choice = list(chosen) + rest
        return super().play_phase(initial_state, self.last_choice)
