#!/usr/bin/env python3
"""Classical MCTS agent: UCT over this player's decisions in the current phase.

Every phase (purchase, combatMove, nonCombatMove, place) gets its own search with the same time
limit, SECONDS. Each search builds a tree rooted at this player's decision; the tree stops at the
end of the phase (leaves past it are played out, not expanded), so every tree node is this
player's choice and maximising the value is correct. Actions are always the exhaustive legal set
(no candidate filtering). Iterations: UCB1 selection, expansion of one random untried action, a
uniformly random playout (all players) to the end of the game, scored 1 for a win and 0 otherwise
(a playout still unfinished after ROLLOUT_MAX_ROUNDS rounds or at the deadline scores 0), and
backpropagation of that value. The returned actions follow the most-visited children.

All tunable parameters are in the PARAMETERS block of ClassicalMctsAgent. Change a default there, or
override it for one run with its environment variable, e.g.
  TRIPLEA_CLASSIC_MCTS_EXPLORATION=1.4 TRIPLEA_AGENT_SECONDS=1.0 bash run-agent.sh
"""

from __future__ import annotations

import csv
import os
import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))  # search_agent.py next to this file

import triplea_fm as fm  # noqa: E402

from search_agent import SearchAgent  # noqa: E402
from mctsNode import MCTSNode, save_mcts_tree  # noqa: E402


def param(env_name: str, default, cast=float):
    """A parameter's value: the environment variable if it is set, otherwise the default."""
    value = os.environ.get(env_name)
    return default if value is None or value == "" else cast(value)

class MetricLogger:
    def __init__(self, filename, header=None):
        self.filename = filename
        self.header = header
        self._buffer = []
        
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        if header and not os.path.exists(filename):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def log(self, *values):
        """Append one row immediately to CSV."""
        with open(self.filename, "a", newline="") as f:
            csv.writer(f).writerow(values)

    def buffer(self, *values):
        """Stage a row in memory without touching disk."""
        self._buffer.append(values)

    def flush(self):
        """Write all buffered rows to CSV in one shot, then clear buffer."""
        if not self._buffer:
            return
        with open(self.filename, "a", newline="") as f:
            csv.writer(f).writerows(self._buffer)
        self._buffer.clear()


class ClassicalMctsAgent(SearchAgent):
    SECONDS = param("TRIPLEA_AGENT_SECONDS", 300.0)            # search time per phase, same for every phase (s)
    SEED = param("TRIPLEA_AGENT_SEED", 0, int)               # agent randomness + simulation dice

    # Search
    EXPLORATION = param("TRIPLEA_CLASSIC_MCTS_EXPLORATION", 1.414)        # UCT constant C (rewards are 0 or 1)
    MAX_ITERATIONS = param("TRIPLEA_CLASSIC_MCTS_MAX_ITERATIONS", 100_000, int)  # cap besides the time
    ROLLOUT_MAX_ROUNDS = param("TRIPLEA_CLASSIC_MCTS_ROLLOUT", 100, int)  # playout cap (rounds); unfinished = loss

    # Debugging: dump each search tree to logs/trees/ (Graphviz .dot, optionally .png).
    SAVE_TREE = param("TRIPLEA_CLASSIC_MCTS_SAVE_TREE", 0, int)            # 1 = save after every search
    TREE_PNG = param("TRIPLEA_CLASSIC_MCTS_TREE_PNG", 0, int)              # 1 = also render .png (needs `dot`)
    TREE_MAX_NODES = param("TRIPLEA_CLASSIC_MCTS_TREE_MAX_NODES", 500, int)  # BFS cap on dumped nodes

    # Efficiency metrics: one row per search (game, round, phase) in metrics/efficiency.csv; in a
    # tournament under its folder (TRIPLEA_TESTBENCH_OUTPUT_DIR) instead of this agent's.
    METRICS_FILE = (Path(os.environ.get("TRIPLEA_TESTBENCH_OUTPUT_DIR") or Path(__file__).resolve().parent)
                    / "metrics" / "efficiency.csv")
    # ====================================================================

    def __init__(self):
        super().__init__(seconds=self.SECONDS, seed=self.SEED)
        self.iteration = 0
        self.max_depth = 0
        self.efficiency_metric = MetricLogger(
            str(self.METRICS_FILE),
            header=["game", "round", "player", "phase", "num_iterations", "avg_rollout_length",
                    "max_rollout_round", "max_rollout_rounds_ahead", "max_depth", "root_value",
                    "root_actions", "time_taken"])

    def setup(self):
        self.log(f"parameters: seconds={self.SECONDS} seed={self.SEED} C={self.EXPLORATION} "
                 f"max_iterations={self.MAX_ITERATIONS} rollout_max_rounds={self.ROLLOUT_MAX_ROUNDS} "
                 f"save_tree={self.SAVE_TREE} tree_png={self.TREE_PNG}")

    def time_budget(self, phase: str) -> float:
        return self.SECONDS  # same limit for every phase

    def is_leaf(self, root: fm.SimulationState, node: MCTSNode) -> bool:
        """Nodes past this player's phase (or game over) are played out but never expanded."""
        return not self.same_phase(root, node.state)

    def outcome(self, state: fm.SimulationState) -> float:
        """1 if this player's alliance has won, 0 otherwise (lost or unfinished)."""
        if state.terminal is None:
            return 0.0
        return 1.0 if self.d.alliance_of[self.me] in state.terminal.winners else 0.0

    def playout(self, state: fm.SimulationState, deadline: float) -> tuple:
        """Uniformly random playout (all players, exhaustive actions) to the end of the game, in
        place; stops unfinished after ROLLOUT_MAX_ROUNDS rounds or at the deadline.
        Returns (outcome() of the final state, playout length in actions)."""
        rng = self.new_rng()
        horizon = state.round + self.ROLLOUT_MAX_ROUNDS
        steps = 0
        while (state.terminal is None and state.decision is not None and state.round < horizon
               and time.time() < deadline):
            actions = self.legal_actions(state, exhaustive=True)
            action = self.rnd.choice(actions) if actions else fm.EndPhaseAction()
            if not self.apply(state, action, rng) and not self.apply(state, fm.EndPhaseAction(), rng):
                break
            steps += 1
        return self.outcome(state), steps

    def expand(self, node: MCTSNode) -> MCTSNode:
        """Add one child for a random untried action; returns it (or node if nothing applies)."""
        while node.untried_actions:
            action = node.untried_actions.pop(self.rnd.randrange(len(node.untried_actions)))
            new_state = node.state.clone()
            if self.apply(new_state, action, self.new_rng()):
                child = MCTSNode(new_state, parent=node, action=action)
                node.children.append(child)
                return child
        return node  # every remaining action was rejected

    def backpropagate(self, node: MCTSNode, value: float):
        """Add the value and a visit to every node from `node` up to the root."""
        while node is not None:
            node.visits += 1
            node.value += value
            node = node.parent

    def search(self, root: fm.SimulationState, phase: str, deadline: float) -> List[object]:
        started = time.time()
        root_node = MCTSNode(root.clone())
        root_node.untried_actions = self.legal_actions(root_node.state, exhaustive=True)
        root_actions = len(root_node.untried_actions)
        self.iteration = 0
        self.max_depth = 0
        wins = finished = rollout_steps = 0
        max_rollout_round = root.round  # furthest round any playout reached
        max_rounds_ahead = 0  # most rounds one playout advanced past the round it started in

        while time.time() < deadline and self.iteration < self.MAX_ITERATIONS:
            self.iteration += 1
            node, depth = root_node, 0

            # 1. Selection: descend through fully expanded in-phase nodes by UCB1.
            while not self.is_leaf(root, node):
                if node.untried_actions is None:
                    node.untried_actions = self.legal_actions(node.state, exhaustive=True)
                if not node.is_fully_expanded():
                    break
                best = node.best_child_uct(self.EXPLORATION)
                if best is None:
                    break
                node, depth = best, depth + 1

            # 2. Expansion: one new child, unless the node is past the phase.
            if not self.is_leaf(root, node) and not node.is_fully_expanded():
                child = self.expand(node)
                if child is not node:
                    node, depth = child, depth + 1
            self.max_depth = max(self.max_depth, depth)

            # 3. Simulation
            final = node.state.clone()
            value, steps = self.playout(final, deadline)
            rollout_steps += steps
            max_rollout_round = max(max_rollout_round, final.round)
            max_rounds_ahead = max(max_rounds_ahead, final.round - node.state.round)
            finished += final.terminal is not None
            wins += value == 1.0

            # 4. Backpropagation
            self.backpropagate(node, value)

        # Recommended line: most-visited child at each level, until the phase ends.
        action_seq = []
        node = root_node
        while node.children:
            node = max(node.children, key=lambda c: (c.visits, c.value))
            if node.action.kind == "end_phase":
                break
            action_seq.append(node.action)
            if self.is_leaf(root, node):
                break
        self.log(f"  {phase}: {self.iteration} iterations, played max until {max_rollout_round}, {finished} playouts finished ({wins} wins), "
                 f"tree depth {self.max_depth}, root children {len(root_node.children)}, {len(action_seq)} actions")
        
        self.efficiency_metric.log(
            root.game_number, root.round, self.player_name, phase, self.iteration,
            rollout_steps / self.iteration if self.iteration else 0.0, max_rollout_round,
            max_rounds_ahead,
            self.max_depth,
            root_node.value / root_node.visits if root_node.visits else 0.0,
            root_actions, time.time() - started)

        if self.SAVE_TREE:
            self.save_tree(root_node, root, phase, finished, wins)
        
        return action_seq

    def save_tree(self, root_node: MCTSNode, root: fm.SimulationState, phase: str, finished: int, wins: int):
        """Write this search's tree to logs/trees/tree-<player>-game-<n>-round-<r>-<phase>.dot/.png."""
        tree_dir = self.log_dir / "trees"
        tree_dir.mkdir(parents=True, exist_ok=True)
        prefix = tree_dir / f"tree-{self.player_name}-game-{root.game_number}-round-{root.round}-{phase}"
        title = (f"{self.player_name} game {root.game_number} round {root.round} {phase}: "
                 f"{self.iteration} iterations, {finished} playouts finished ({wins} wins)")
        dot_path, png_path = save_mcts_tree(root_node, self.d, str(prefix), title,
                                            max_nodes=self.TREE_MAX_NODES, render_img=bool(self.TREE_PNG))
        self.log(f"  tree saved: {png_path or dot_path}")

    # Optional hooks (see SearchAgent): battle(observation), time_budget(phase). Helpers:
    # legal_actions, simulate, apply, same_phase, new_rng, rollout, evaluate, log; attributes
    # d (GameDefinition), me (player index), rnd.


if __name__ == "__main__":
    ClassicalMctsAgent().run()
