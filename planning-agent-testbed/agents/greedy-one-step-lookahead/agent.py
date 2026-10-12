#!/usr/bin/env python3
"""Search-agent template: implement MyAgent.search(); SearchAgent (search_agent.py) does the rest.

The example search is a greedy one-step lookahead: at each decision of the phase it tries every
toolkit candidate on a clone, plays the rest of the round with the library's heuristic policy,
scores the result with evaluate(), and keeps the best action. Replace it with your own search
(MCTS, beam search, rolling-horizon evolution, ...).

Environment (read by SearchAgent):
  TRIPLEA_AGENT_SECONDS  base search time per decision (default 2.0, scaled per phase)
  TRIPLEA_AGENT_SEED     seed of the agent's randomness and simulation dice (default 0)
  TRIPLEA_AGENT_LOG      1 enables logging, anything else disables it (set in run-agent.sh)
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).resolve().parent))  # search_agent.py next to this file

import triplea_fm as fm  # noqa: E402

from search_agent import SearchAgent  # noqa: E402
from triplea_fm.definition import COMBAT_MOVE, NULL_PLAYER

class MyAgent(SearchAgent):
    # def evaluate(self, s: fm.SimulationState) -> float:
    #     """Return a score for the given state. Higher is better."""
    #     if s.terminal is not None:
    #         return 1.0 if self.d.alliance_of[self.me] in s.terminal.winners else 0.0
    #     n = len(self.d.player_names)
    #     score = [0.0] * n
    #     for u in s.units.values():
    #         if u.owner != NULL_PLAYER and not self.d.unit_types[u.type].is_infrastructure:
    #             score[u.owner] += self.precomp.cost[u.type]
    #     total = sum(score)
    #     return score[self.me] / total

    def search(self, root: fm.SimulationState, phase: str, deadline: float) -> List[object]:
        # TODO: replace this example with your search. Return the phase's actions in order.
        s = root.clone()
        rng = self.new_rng()
        plan = []
        while self.same_phase(root, s) and time.time() < deadline:
            best, best_value = None, -1.0
            # self.log(f"In round {s.round} num_legal_actions={len(self.legal_actions(s))}")
            for a in self.legal_actions(s):
                if time.time() >= deadline:
                    break
                after = self.simulate(s, a)
                if after is None:
                    continue
                value = self.evaluate(self.rollout(after, self.new_rng(), rounds=0))
                if value > best_value:
                    best, best_value = a, value
            if best is None or best.kind == "end_phase":
                break
            self.apply(s, best, rng)
            plan.append(best)
        return plan

    # Optional hooks (see SearchAgent): setup() once per game, battle(observation),
    # time_budget(phase). Helpers: legal_actions, simulate, apply, same_phase, new_rng,
    # rollout, evaluate, log; attributes d (GameDefinition), me (player index), rnd.


if __name__ == "__main__":
    MyAgent().run()
