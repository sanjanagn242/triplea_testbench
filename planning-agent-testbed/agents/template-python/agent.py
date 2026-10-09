#!/usr/bin/env python3
"""Template Python agent for the planning-agent testbench (newline-delimited JSON over stdin/stdout).

Copy this folder to start a new agent and fill in the per-phase methods of Agent. Each method
receives the turn request and returns the action object for that phase. Until a method is filled
in, it answers with the empty action ({}), or {"fightAll": true} for battles. When
TRIPLEA_AGENT_LOG is 1 (set in run-agent.sh), diagnostics go to stderr and to
planning-agent-testbed/logs/agent-<faction>-game-<n>.txt; stdout carries protocol lines only.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent

PROTOCOL_VERSION = 1
LOG_DIR = HERE.parent.parent / "logs"
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh


class Agent:
    def __init__(self):
        self.player = None
        self.log_path = None
        self.phase_handlers = {
            "purchase": self.purchase,
            "combatMove": self.combat_move,
            "battle": self.battle,
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
        return {"error": f"unknown request type {kind!r}"}

    def game_start(self, request: dict) -> dict:
        self.player = request["playerName"]
        if LOGGING:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LOG_DIR / f"agent-{self.player}-game-{request.get('gameNumber', 1)}.txt"
            self.log_path.write_text(f"{request.get('gameNumber', 1)}\n", encoding="utf-8")
        # TODO: per-game setup. request["gameXml"] holds the map XML and request["initialState"]
        # the starting observation.
        self.log(f"ready as {self.player}")
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def turn(self, request: dict) -> dict:
        phase = request.get("phase")
        handler = self.phase_handlers.get(phase)
        response = handler(request["state"]) if handler else {}
        self.log(f"{phase}: action: {json.dumps(response)}")
        return response

    # --- Per-phase decisions. `state` is the current observation (request["state"]). ---

    def purchase(self, state: dict) -> dict:
        # TODO: return {"purchases": [{"rule": "buyInfantry", "count": 4}, ...]}
        return {}

    def combat_move(self, state: dict) -> dict:
        # TODO: return {"moves": [{"from": "A", "to": "B", "unitIds": ["..."]}, ...]}
        # Optional per move: "route": ["A", "B", "C"], "intent": "RAID" or "NORMAL".
        return {}

    def battle(self, state: dict) -> dict:
        # Asks TripleA to resolve every pending battle.
        return {"fightAll": True}

    def non_combat_move(self, state: dict) -> dict:
        # TODO: return {"moves": [...]} in the same format as combat_move.
        return {}

    def place(self, state: dict) -> dict:
        # TODO: return {"placements": [{"territory": "X", "unitIds": ["..."]}, ...]}
        # Units left unplaced are destroyed at the end of the step.
        return {}


def main():
    agent = Agent()
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
