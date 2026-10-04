#!/usr/bin/env python3
"""Python MCTS agent for the planning-agent testbench (newline-delimited JSON over stdin/stdout).

No allies - only the player who is running this agent is considered.

Each decision request is answered by building a forward-model state from the observation
(triplea_fm.testbench.LiveBridge), running PhaseSearch for that phase, and translating the chosen
action sequence into the testbench action format. Diagnostics go to stderr and to
planning-agent-testbed/logs/agent-<faction>-game-<n>.txt; stdout carries protocol lines only.

Environment:
  TRIPLEA_MCTS_SECONDS     search time per combat-move decision (default 2.0)
  TRIPLEA_MCTS_ITERS       iteration budget (default 100000; unused, the search is time-bounded)
  TRIPLEA_MCTS_ROLLOUT     rollout horizon in rounds (default 2)
  TRIPLEA_MCTS_EXP_CONST   UCB1 exploration constant (default 1.414)
  TRIPLEA_MCTS_SEED        search seed (default 0)
  TRIPLEMIND_VIEW          1 (default) shows the live map window, 0 hides it
"""

# from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this agent's own modules (mcts.py)

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import PHASE_KINDS, LiveBridge  # noqa: E402

from mcts import MCTS, MCTSGameState  # noqa: E402

import map_graph  # noqa: E402
import action_generators

PROTOCOL_VERSION = 1
PHASE_TIME_SHARE = {"combatMove": 1.0}
LOG_DIR = HERE.parent.parent / "logs"


class TripleMindAgent:
    def __init__(self, time_budget, iter_budget, depth_budget, exploration_constant, seed=0):
        self.whoAmI = None
        self.log_path = None
        self.definition = None
        self.bridge = None
        

        # MCTS parameters
        self.time_budget = time_budget
        self.iter_budget = iter_budget
        self.depth_budget = depth_budget
        self.exploration_constant = exploration_constant
        self.seed = seed
        self.search = None
    
    def log(self, message: str):
        line = f"{message}"
        print(line, file=sys.stderr, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def handle_request(self, request):
        kind = request.get("type")
        if kind == "game_start":
            return self.game_start(request)
        elif kind == "turn_request":
            return self.get_move(request)
        else:
            return {"error": f"Unknown request type: {kind}"}

    def game_start(self, request):
        self.whoAmI = request["playerName"]
        LOG_DIR.mkdir(parents=True, exist_ok=True)
        self.log_path = LOG_DIR / f"agent-{self.whoAmI}-game-{request.get('gameNumber', 1)}.txt"
        self.log_path.write_text(f"Game {request.get('gameNumber', 1)}\n", encoding="utf-8")
        # self.log(f"request: {json.dumps(request)}")
        self.build_map(request)
        definition, report = fm.load_definition(request["gameXml"], strict=False)
        if not report.supported:
            for f in report.features:
                self.log(f"unsupported feature: {f}")
            self.log("forward model unavailable for this game; answering with empty actions")
            self.definition = None
            return {"ready": True, "protocolVersion": PROTOCOL_VERSION}
        self.definition = definition
        self.bridge = LiveBridge(definition, self.whoAmI, request.get("gameNumber", 1))
        random.seed(self.seed)
        self.search = MCTS(definition, self.map, self.whoAmI, self.time_budget,
                           self.depth_budget, self.exploration_constant)
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def build_map(self, request):
        # Any map: territories + <connection>s from the rules XML become the graph.
        self.map = map_graph.MapGraph.from_xml(request["gameXml"])
        self.view = (map_graph.MapView(self.map, self.whoAmI)
                     if os.environ.get("TRIPLEMIND_VIEW", "1") == "1" else None)
        self.log(f"map graph: {self.map}")

    def get_move(self, request):
        phase = request.get("phase")
        game_state = map_graph.GameState(self.map, request["state"], self.whoAmI)
        if self.view is not None:
            self.view.draw(game_state)
        if phase == "battle":
            self.log(f"turn {request.get('turn', 1)}: {self.whoAmI} fighting all")
            return {"fightAll": True}
        if self.bridge is None or phase not in PHASE_KINDS:
            return {}
        # Also registers this turn's unit ids with the bridge, which encode() needs for moves and
        # placements; the returned SimulationState is what search will run on.
        fm_state = self.bridge.build_state(request["state"], phase)
        # Generators work on the forward-model view so units carry the ids the bridge encodes.
        sim_game = map_graph.GameState.from_sim(self.map, self.definition, fm_state, self.whoAmI)
        response = {}
        if phase == "purchase":
            legal_moves = action_generators.purchase_actions(sim_game, self.definition)
            if legal_moves:
                move = legal_moves[0]
                response = self.encode("purchase", [move])
            self.log(f"round {game_state.round}: {self.whoAmI} purchase -> {response}")
        elif phase == "combatMove":
            attacks = self.search.mcts_search(MCTSGameState(self.search, fm_state, self.whoAmI))
            moves = [mv for attack in attacks for mv in attack.moves]
            if moves:
                response = self.encode("combatMove", moves)
            self.log(f"round {game_state.round}: {self.whoAmI} combat move "
                     f"({self.search.iteration} iterations) {attacks}")
            self.log(f"round {game_state.round}: {self.whoAmI} combat move -> {response}")
        elif phase == "nonCombatMove":
            moves = action_generators.noncombat_move(sim_game, self.definition)
            if moves:
                response = self.encode("nonCombatMove", moves)
            self.log(f"round {game_state.round}: {self.whoAmI} non-combat move -> {response}") 
        elif phase == "place":
            self.log(f"round {game_state.round}: {self.whoAmI} placing all purchased units")
        return response

    def encode(self, phase: str, sequence) -> dict:
        """Translate a sequence of triplea_fm actions into the engine's action object."""
        b = self.bridge
        if phase == "purchase":
            merged = {}
            for a in sequence:
                if a.kind == "purchase":
                    for rid, count in a.quantities:
                        merged[rid] = merged.get(rid, 0) + count
            out = b.purchase_to_json(fm.PurchaseAction(tuple(sorted(merged.items()))))
        elif phase in ("combatMove", "nonCombatMove"):
            out = {"moves": [b.move_to_json(a) for a in sequence if a.kind == "move"]}
        else:
            out = {"placements": [b.place_to_json(a) for a in sequence if a.kind == "place"]}
        # self.log(f"  action: {json.dumps(out)}")
        return out

def main():
    agent = TripleMindAgent(float(os.environ.get("TRIPLEA_MCTS_SECONDS", "2.0")),
                  int(os.environ.get("TRIPLEA_MCTS_ITERS", "100000")),
                  int(os.environ.get("TRIPLEA_MCTS_ROLLOUT", "10")),
                  float(os.environ.get("TRIPLEA_MCTS_EXP_CONST", "1.414")),
                  int(os.environ.get("TRIPLEA_MCTS_SEED", "0")))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = agent.handle_request(json.loads(line))
        except Exception as e:  # report, never crash the protocol loop
            traceback.print_exc(file=sys.stderr)
            agent.log(f"error: {e!r}")
            response = {"error": repr(e)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
