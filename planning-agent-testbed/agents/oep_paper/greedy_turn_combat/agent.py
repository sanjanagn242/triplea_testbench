#!/usr/bin/env python3
"""Greedy Turn agent (OEP paper baseline), combat move only, for the planning-agent testbench
(newline-delimited JSON over stdin/stdout).

The combat move is searched by GreedyTurnSearch (greedy_turn.py): a depth-first search over the
whole turn's attack sequences (one attack = all reachable units -> one target), every sequence
scored with the forward model, the best one played. Every other phase is TripleMind's
deterministic greedy policy: the first purchase_actions option, noncombat_move, the engine's
placement and fightAll. Diagnostics go to stderr and to
planning-agent-testbed/logs/agent-<faction>-game-<n>.txt; stdout carries protocol lines only.

Environment:
  TRIPLEA_GREEDY_SECONDS   search time per combat-move decision (default 1.0)
  TRIPLEA_GREEDY_SEED      search seed (default 0)
  TRIPLEA_AGENT_LOG        1 enables logging, anything else disables it (set in run-agent.sh)
"""

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this agent's own modules (greedy_turn.py, helper.py, ...)

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import PHASE_KINDS, LiveBridge  # noqa: E402

from greedy_turn import GreedyTurnSearch  # noqa: E402
from helper import (GAME_RESULTS_HEADER, MetricLogger, log_game_result, metrics_dir,  # noqa: E402
                    output_folder, turn_position)
import map_graph  # noqa: E402
from generators import ag as action_generators  # noqa: E402

PROTOCOL_VERSION = 1
LOG_DIR = HERE.parent.parent.parent / "logs"  # planning-agent-testbed/logs
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh
# Metrics go in a subfolder per map and opponent, e.g. metrics/capture-the-flag-diag/vs-tripleMind-mcts/
# (helper.output_folder); in a tournament under TRIPLEA_TESTBENCH_OUTPUT_DIR (helper.metrics_dir).
METRICS_DIR = metrics_dir(HERE)
GAME_RESULTS_CSV = "game_results.csv"  # one row per game this agent played
# game_results.csv also names the winners (TripleMind's columns first, so its scripts still read it).
GREEDY_GAME_RESULTS_HEADER = GAME_RESULTS_HEADER + ["winners"]


class Agent:
    def __init__(self, time_budget, seed=0):
        self.player = None
        self.log_path = None
        self.definition = None
        self.bridge = None
        self.map = None
        self.search = None
        self.game_number = None
        self.turn_position = None
        self.game_started = None
        self.output_folder = "unknown-game/vs-unknown"
        self.time_budget = time_budget
        self.seed = seed
        self.phase_handlers = {
            "purchase": self.purchase,
            "combatMove": self.combat_move,
            "nonCombatMove": self.non_combat_move,
            "place": self.place,
        }

    def log(self, message: str):
        if not LOGGING:
            return
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, file=sys.stderr, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def handle(self, request: dict) -> dict:
        kind = request.get("type")
        if kind == "game_start":
            return self.game_start(request)
        if kind == "turn_request":
            return self.turn(request)
        if kind == "game_end":
            return self.game_end(request)
        return {"error": f"unknown request type {kind!r}"}

    def game_start(self, request: dict) -> dict:
        self.player = request["playerName"]
        self.game_number = request.get("gameNumber", 1)
        self.game_started = time.time()
        if LOGGING:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LOG_DIR / f"agent-{self.player}-game-{self.game_number}.txt"
            self.log_path.write_text(f"{self.game_number}\n", encoding="utf-8")
        self.map = map_graph.MapGraph.from_xml(request["gameXml"])
        definition, report = fm.load_definition(request["gameXml"], strict=False)
        self.turn_position = turn_position(definition, self.player)
        self.output_folder = output_folder(definition, self.player)
        if not report.supported:
            for f in report.features:
                self.log(f"unsupported feature: {f}")
            self.log("forward model unavailable for this game; answering with empty actions")
            self.definition = None
            return {"ready": True, "protocolVersion": PROTOCOL_VERSION}
        self.definition = definition
        self.bridge = LiveBridge(definition, self.player, self.game_number)
        self.search = GreedyTurnSearch(definition, self.map, self.player, self.time_budget,
                                       self.seed, metrics_dir=METRICS_DIR / self.output_folder)
        self.log(f"ready as {self.player}")
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def game_end(self, request: dict) -> dict:
        result = request.get("result") or {}
        logger = MetricLogger(str(METRICS_DIR / self.output_folder / GAME_RESULTS_CSV),
                              header=GREEDY_GAME_RESULTS_HEADER)
        seconds = time.time() - self.game_started if self.game_started else None
        logger.log(self.game_number, self.player, result.get("roundsPlayed"),
                   1 if result.get("won") else 0, self.turn_position,
                   round(seconds, 1) if seconds is not None else "",
                   ";".join(result.get("winners") or []))
        self.log(f"game over: winners {result.get('winners')}, rounds {result.get('roundsPlayed')}")
        return {"ok": True}

    def turn(self, request: dict) -> dict:
        phase = request.get("phase")
        if phase == "battle":
            return {"fightAll": True}
        if self.bridge is None or phase not in PHASE_KINDS or phase not in self.phase_handlers:
            return {}
        # Also registers this turn's unit ids with the bridge, which encode() needs.
        sim = self.bridge.build_state(request["state"], phase)
        response = self.phase_handlers[phase](sim)
        self.log(f"round {sim.round} {phase}: action: {json.dumps(response)}")
        return response

    def view(self, sim):
        return map_graph.GameState.from_sim(self.map, self.definition, sim, self.player)

    # --- Per-phase decisions. `sim` is the forward-model state of the observation. ---

    def purchase(self, sim) -> dict:
        options = action_generators.purchase_actions(self.view(sim), self.definition)
        return self.encode("purchase", options[:1]) if options else {}

    def combat_move(self, sim) -> dict:
        attacks = self.search.search(sim)
        moves = [mv for attack in attacks for mv in attack.moves]
        self.log(f"round {sim.round} combat move: {self.search.last_stats} {attacks}")
        return self.encode("combatMove", moves) if moves else {}

    def non_combat_move(self, sim) -> dict:
        moves = action_generators.noncombat_move(self.view(sim), self.definition)
        return self.encode("nonCombatMove", moves) if moves else {}

    def place(self, sim) -> dict:
        # As TripleMind: no placements field, so the engine places the purchased units.
        return {}

    def encode(self, phase: str, sequence) -> dict:
        """Translate a sequence of triplea_fm actions into the engine's action object."""
        b = self.bridge
        if phase == "purchase":
            merged = {}
            for a in sequence:
                if a.kind == "purchase":
                    for rid, count in a.quantities:
                        merged[rid] = merged.get(rid, 0) + count
            return b.purchase_to_json(fm.PurchaseAction(tuple(sorted(merged.items()))))
        if phase in ("combatMove", "nonCombatMove"):
            return {"moves": [b.move_to_json(a) for a in sequence if a.kind == "move"]}
        return {"placements": [b.place_to_json(a) for a in sequence if a.kind == "place"]}


def main():
    agent = Agent(float(os.environ.get("TRIPLEA_GREEDY_SECONDS", "1.0")),
                  int(os.environ.get("TRIPLEA_GREEDY_SEED", "0")))
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = agent.handle(json.loads(line))
        except Exception as e:  # report, never crash the protocol loop
            traceback.print_exc(file=sys.stderr)
            agent.log(f"error: {e!r}")
            response = {"error": repr(e)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
