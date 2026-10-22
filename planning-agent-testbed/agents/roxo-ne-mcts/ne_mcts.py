"""Non-Exploring MCTS as implemented in Roxo's MCTSGrandStrategy fork
(mctsclean/nonExploringMcts/NonExploringMcts.java).

  * Selection: from the root, while the node is not terminal (at most 100 levels): if the node has
    an action it has not expanded yet, expand it; otherwise go to the child with the best mean
    value Q / N (no exploration term).
  * Expansion: copy the node's world, apply the action, generate the child's actions.
  * Playout: the Weak AI plays up to 40 move phases (Game.playout).
  * Backpropagation: N += 1 and Q += reward on every node of the path; the reward is always the
    searching player's, also in the opponent's levels.

The tree does not stop at the end of the phase: Advance leads into the battles and the next move
phases, ours and the opponent's. A search runs for 1 second or 500 iterations. The plan follows the
best mean child from the root until an Advance, keeping the non-skip actions.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from world_model import ADVANCE, SKIP, Game, TerritoryAction, WorldModel

MAX_SELECTION_DEPTH = 100


class MCTSNode:
    __slots__ = ("state", "parent", "action", "children", "n", "q")

    def __init__(self, state: WorldModel, parent: Optional["MCTSNode"], action: Optional[TerritoryAction]):
        self.state = state
        self.parent = parent
        self.action = action
        self.children: List[MCTSNode] = []
        self.n = 0
        self.q = 0.0


@dataclass
class SearchStats:
    seconds: float = 0.0
    iterations: int = 0
    root_value: float = 0.0
    root_children: int = 0
    max_selection_depth: int = 0
    max_rollout: int = 0  # rounds
    rollouts: List[Tuple[int, float, bool]] = field(default_factory=list)  # (rounds, reward, terminal)


class NonExploringMcts:
    def __init__(self, game: Game, seconds: float = 1.0, max_iterations: int = 500):
        self.game = game
        self.seconds = seconds
        self.max_iterations = max_iterations
        self.stats = SearchStats()
        self.root: Optional[MCTSNode] = None

    def run(self, root_state) -> List[TerritoryAction]:
        """Searches from root_state (a kernel state at our move decision); returns the plan."""
        started = time.time()
        world = WorldModel(self.game, root_state)
        world.generate_actions()
        self.root = MCTSNode(world, None, None)
        stats = self.stats = SearchStats()
        while time.time() - started < self.seconds and (
                not self.max_iterations or stats.iterations < self.max_iterations):
            node, depth = self._selection(self.root)
            stats.max_selection_depth = max(stats.max_selection_depth, depth)
            result = self.game.playout(node.state.state.clone())
            stats.rollouts.append((result.rounds, result.reward, result.terminal))
            stats.max_rollout = max(stats.max_rollout, result.rounds)
            self._backpropagate(node, result.reward)
            stats.iterations += 1
        plan = self._best_final_actions(self.root)
        stats.seconds = time.time() - started
        stats.root_value = self.root.q / self.root.n if self.root.n else 0.0
        stats.root_children = len(self.root.children)
        return plan

    def _selection(self, node: MCTSNode) -> Tuple[MCTSNode, int]:
        depth = 0
        while not node.state.terminal and depth < MAX_SELECTION_DEPTH:
            depth += 1
            action = node.state.next_action()
            if action is not None:
                return self._expand(node, action), depth
            best = self._best_child(node)
            if best is None:
                return node, depth
            node = best
        return node, depth

    @staticmethod
    def _expand(parent: MCTSNode, action: TerritoryAction) -> MCTSNode:
        world = parent.state.child()
        world.apply(action)
        world.generate_actions()
        child = MCTSNode(world, parent, action)
        parent.children.append(child)
        return child

    @staticmethod
    def _backpropagate(node: Optional[MCTSNode], reward: float):
        while node is not None:
            node.n += 1
            node.q += reward
            node = node.parent

    @staticmethod
    def _best_child(node: MCTSNode) -> Optional[MCTSNode]:
        """bestUCTChild / BestChild: the first child with the highest Q / N."""
        best, best_value = None, float("-inf")
        for c in node.children:
            value = c.q / c.n
            if value > best_value:
                best, best_value = c, value
        return best

    def _best_final_actions(self, node: MCTSNode) -> List[TerritoryAction]:
        """BestFinalActions: follow the best child until an Advance (or a leaf); skips are dropped."""
        plan = []
        while True:
            best = self._best_child(node)
            if best is None or best.action.kind == ADVANCE:
                return plan
            if best.action.kind != SKIP:
                plan.append(best.action)
            node = best

    def leaves(self, node: Optional[MCTSNode] = None) -> int:
        """Number of root-to-leaf paths in the tree."""
        node = node or self.root
        if node is None:
            return 0
        if not node.children:
            return 1
        return sum(self.leaves(c) for c in node.children)
