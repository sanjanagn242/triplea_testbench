#!/usr/bin/env python3
"""TripleMind with a behaviour-cloning policy in place of MCTS (SearchAgent in search_agent.py does
the protocol, state building, action checking and encoding).

Every phase is played as by TripleMind (agents/tripleMind/agent.py), with TripleMind's own map
graph and action generators:
  purchase       the first purchase the generator proposes
  combatMove     targets come from TripleMind's heuristic (generators.ag.combat_actions, most
                 important first); for each target the BC policy picks one of the generator's
                 formations (an attack set or the skip) where TripleMind runs MCTS. The chosen
                 moves are applied with the forward model and the next target is generated from
                 the new state, as MCTS walks down its tree, until no target is left
  nonCombatMove  the generator's noncombat moves
  place          {} : the engine's built-in placement of all purchased units
  battle         fight all (protocol v1)

Environment:
  TRIPLEMIND_BC_CHECKPOINT  policy checkpoint (default behaviour-cloning/runs/mlp-baseline/best.pt)
  TRIPLEMIND_BC_THREADS     torch threads (default 1)
  TRIPLEMIND_VIEW           1 shows TripleMind's live map window, 0 hides it (set in run-agent.sh)
  TRIPLEA_AGENT_SEED        seed of the agent's randomness (default 0)
  TRIPLEA_AGENT_LOG         1 enables logging, anything else disables it (set in run-agent.sh)
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path
from typing import Dict, List

HERE = Path(__file__).resolve().parent
TRIPLEMIND = HERE.parents[1] / "tripleMind"
sys.path.insert(0, str(TRIPLEMIND))  # TripleMind's map graph and action generators
sys.path.insert(0, str(HERE))  # search_agent.py and bc_policy.py next to this file

import triplea_fm as fm  # noqa: E402

import generators  # noqa: E402  (TripleMind)
import map_graph  # noqa: E402  (TripleMind)
from generators import ag  # noqa: E402  (Cython build when available)
from search_agent import SearchAgent  # noqa: E402


class TripleMindBC(SearchAgent):
    def __init__(self):
        super().__init__()
        self.checkpoint = os.environ.get("TRIPLEMIND_BC_CHECKPOINT")
        self.game_xml = None
        self.map = None
        self.view = None
        self.bc = None

    # ---- setup ----------------------------------------------------------------------------

    def game_start(self, request: dict) -> dict:
        self.game_xml = request.get("gameXml")
        return super().game_start(request)

    def setup(self):
        random.seed(self.seed)
        # Any map: territories + <connection>s from the rules XML become the graph.
        self.map = map_graph.MapGraph.from_xml(self.game_xml)
        self.view = (map_graph.MapView(self.map, self.player_name)
                     if os.environ.get("TRIPLEMIND_VIEW", "1") == "1" else None)
        self.log(f"map graph: {self.map}")
        self.log(f"action generators: {generators.IMPLEMENTATION}")
        try:
            import torch
            from bc_policy import DEFAULT_CHECKPOINT, BCPolicy
            torch.set_num_threads(int(os.environ.get("TRIPLEMIND_BC_THREADS", "1")))
            self.bc = BCPolicy(self.checkpoint or DEFAULT_CHECKPOINT, self.d, self.map,
                               self.player_name, self.bridge)
            self.log(f"BC policy: {self.bc.checkpoint}")
        except Exception as e:  # keep playing: fall back to the generator's first formation
            self.bc = None
            self.log(f"BC policy unavailable ({e!r}); combat moves use the generator's first "
                     f"(strongest) formation")

    # ---- phases ---------------------------------------------------------------------------

    def turn(self, request: dict) -> dict:
        if self.view is not None and self.map is not None and request.get("state"):
            self.view.draw(map_graph.GameState(self.map, request["state"], self.player_name))
        if request.get("phase") == "place":
            # As TripleMind: no placements field, so the engine places all purchased units
            # ({"placements": []} would place none and the units would be destroyed).
            self.log(f"round {request['state'].get('round')}: {self.player_name} placing all "
                     f"purchased units")
            return {}
        return super().turn(request)

    def search(self, root: fm.SimulationState, phase: str, deadline: float) -> List[object]:
        view = map_graph.GameState.from_sim(self.map, self.d, root, self.player_name)
        if phase == "purchase":
            legal = ag.purchase_actions(view, self.d)
            return [legal[0]] if legal else []
        if phase == "combatMove":
            return self.combat_move(root)
        if phase == "nonCombatMove":
            return list(ag.noncombat_move(view, self.d))
        return []

    def combat_move(self, root: fm.SimulationState) -> List[object]:
        """One formation per heuristic target, chosen by the BC policy, until no target is left."""
        s, rng = root.clone(), self.new_rng()
        excluded = set()
        committed: Dict[str, str] = {}  # engine unit id -> territory it attacks
        plan = []
        for _ in range(len(self.d.territory_names) + 1):  # each target is excluded once handled
            view = map_graph.GameState.from_sim(self.map, self.d, s, self.player_name)
            actions, excluded = ag.combat_actions(view, self.d, excluded, self.player_name)
            if not actions:
                break  # no target left: end of the phase
            index = 0
            if self.bc is not None:
                try:
                    index, _, info = self.bc.choose(s, actions, committed)
                    self.log(f"  {info['target']}: {info['kind']} {index} of {len(actions)} "
                             f"(p={max(info['probs']):.2f}) {actions[index]}")
                except Exception as e:  # e.g. a map the policy was not trained on
                    self.bc = None
                    self.log(f"BC policy disabled for this game ({e!r}); combat moves use the "
                             f"generator's first (strongest) formation")
            for mv in actions[index].moves:
                if self.apply(s, mv, rng):
                    plan.append(mv)
                    dest = self.d.territory_names[mv.route[-1]]
                    for uid in mv.units:
                        committed[self.bridge.uuid_of[uid]] = dest
                else:
                    self.log(f"  rejected {fm.action_to_json(self.d, mv)}")
        return plan


if __name__ == "__main__":
    TripleMindBC().run()
