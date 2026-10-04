"""Open-loop MCTS over one phase of the agent's own decisions, using triplea_fm as the simulator.

Tree edges are the agent's actions within the current phase (candidates from the ActionToolkit,
ordered by a heuristic prior, with progressive widening). Each iteration replays the edge sequence
from the root with fresh dice, so chance is sampled rather than branched (open loop). When the
phase ends, a rollout plays every player with the heuristic policy for a fixed horizon and the
state is scored by `evaluate`.
"""

from __future__ import annotations

import math
import random
import time
from typing import List, Optional

import triplea_fm as fm
from triplea_fm import toolkit

from triplea_fm.baselines import HeuristicPolicy, Precomp, evaluate


class Node:
    __slots__ = ("action", "children", "untried", "n", "w")

    def __init__(self, action=None):
        self.action = action
        self.children: List["Node"] = []
        self.untried: Optional[list] = None
        self.n = 0
        self.w = 0.0


class PhaseSearch:
    def __init__(self, d, pc: Optional[Precomp] = None, *, time_budget: float = 2.0,
                 max_iterations: int = 100000, rollout_rounds: int = 2, max_children: int = 24,
                 exploration: float = 0.7, widening: float = 2.0, seed: int = 0):
        self.d = d
        self.pc = pc or Precomp(d)
        self.policy = HeuristicPolicy(d, self.pc)
        self.time_budget = time_budget
        self.max_iterations = max_iterations
        self.rollout_rounds = rollout_rounds
        self.max_children = max_children
        self.c = exploration
        self.widening = widening
        self.rnd = random.Random(seed)
        self.iterations = 0

    # ---- phase boundary ----
    @staticmethod
    def _same_phase(root, s) -> bool:
        dec = s.decision
        return (dec is not None and s.terminal is None and dec.actor == root.decision.actor
                and dec.step_id == root.decision.step_id and dec.kind == root.decision.kind
                and s.round == root.round)

    def _expand(self, s, node: Node):
        cands = toolkit.candidates(self.d, s)
        cands.sort(key=lambda a: -self.policy.prior(s, a))
        node.untried = cands[: self.max_children]

    def search(self, root_state) -> List[object]:
        """Returns the action sequence for the rest of the phase (ends with EndPhase when the
        phase is ended explicitly)."""
        me = root_state.decision.actor
        root = Node()
        self._expand(root_state, root)
        deadline = time.time() + self.time_budget
        self.iterations = 0
        while self.iterations < self.max_iterations and time.time() < deadline:
            self.iterations += 1
            s = root_state.clone()
            rng = fm.SeededRng(self.rnd.getrandbits(62))
            node, path = root, [root]
            while self._same_phase(root_state, s):
                if node.untried is None:
                    self._expand(s, node)
                allowed = max(1, int(self.widening * math.sqrt(node.n + 1)))
                if node.untried and len(node.children) < allowed:
                    a = node.untried.pop(0)
                    child = Node(a)
                    node.children.append(child)
                    if fm.apply_in_place(self.d, s, a, rng) == fm.REJECTED:
                        node.children.remove(child)
                        break
                    path.append(child)
                    node = child
                    break
                child = self._select(node, s)
                if child is None:
                    break
                fm.apply_in_place(self.d, s, child.action, rng)
                path.append(child)
                node = child
            value = self._rollout(s, rng, me, root_state.round)
            for n in path:
                n.n += 1
                n.w += value
        return self._best_sequence(root_state, root)

    def _select(self, node: Node, s) -> Optional[Node]:
        best, best_score = None, -1e9
        log_n = math.log(node.n + 1)
        for ch in node.children:
            if not fm.validate(self.d, s, ch.action).ok:  # open loop: dice may invalidate an edge
                continue
            score = ch.w / (ch.n + 1e-9) + self.c * math.sqrt(log_n / (ch.n + 1e-9))
            if score > best_score:
                best, best_score = ch, score
        return best

    def _rollout(self, s, rng, me, start_round) -> float:
        horizon = start_round + self.rollout_rounds
        steps = 0
        while s.terminal is None and s.round <= horizon and steps < 5000:
            a = self.policy.act(s, self.rnd)
            if fm.apply_in_place(self.d, s, a, rng) == fm.REJECTED:
                a = fm.EndPhaseAction()
                if fm.apply_in_place(self.d, s, a, rng) == fm.REJECTED:
                    break
            steps += 1
        return evaluate(self.d, self.pc, s, me)

    def _best_sequence(self, root_state, root: Node) -> List[object]:
        seq = []
        s = root_state.clone()
        rng = fm.SeededRng(12345)
        node = root
        while self._same_phase(root_state, s):
            visited = [c for c in node.children if c.n > 0 and fm.validate(self.d, s, c.action).ok]
            if visited:
                node = max(visited, key=lambda c: (c.n, c.w))
                a = node.action
            else:
                a = self.policy.act(s, self.rnd)  # beyond the tree: greedy completion
                node = Node()
            if fm.apply_in_place(self.d, s, a, rng) == fm.REJECTED:
                break
            seq.append(a)
            if a.kind == "end_phase":
                break
        return seq
