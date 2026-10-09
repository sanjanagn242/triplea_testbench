#!/usr/bin/env python3
"""Do-nothing Python agent for the planning-agent testbench (newline-delimited JSON over stdin/stdout).

Every turn request, in every phase, is answered with an empty action object ({}). When
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
        self.log(f"ready as {self.player}")
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def turn(self, request: dict) -> dict:
        self.log(f"{request.get('phase')}: action: {{}}")
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
