#!/usr/bin/env python3
"""Hard AI agent for the planning-agent testbench (newline-delimited JSON over stdin/stdout).

A Python port of the engine's Hard AI (ProAi) for the forward model's land/air V1 profile. Each
turn request is turned into a forward-model state (triplea_fm.testbench.LiveBridge) and answered
by hardai.HardAi; every simulation it runs (battle odds, the turn simulated during purchase,
checking moves and placements) uses the forward model. When TRIPLEA_AGENT_LOG is 1 (set in
run-agent.sh), diagnostics go to stderr and to planning-agent-testbed/logs/agent-<faction>-game-<n>.txt;
stdout carries protocol lines only.

Environment:
  TRIPLEA_HARD_AI_SEED            seed of the AI's own randomness and simulation dice (default 0)
  TRIPLEA_HARD_AI_RUN_SCALE       scales the battle simulations per estimate (default 1.0, as ProAi)
  TRIPLEA_HARD_AI_BATTLE_SAMPLES  battle-phase samples in the purchase simulation (default 5)
  TRIPLEA_AGENT_LOG               1 enables logging, anything else disables it (set in run-agent.sh)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this agent's own package (hardai)

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import LiveBridge  # noqa: E402

from hardai import HardAi  # noqa: E402

PROTOCOL_VERSION = 1
LOG_DIR = HERE.parent.parent / "logs"
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh


class Agent:
    def __init__(self, seed: int, run_scale: float, battle_samples: int):
        self.seed = seed
        self.run_scale = run_scale
        self.battle_samples = battle_samples
        self.player = None
        self.log_path = None
        self.definition = None
        self.bridge = None
        self.ai = None
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
        game_number = request.get("gameNumber", 1)
        if LOGGING:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LOG_DIR / f"agent-{self.player}-game-{game_number}.txt"
            self.log_path.write_text(f"{game_number}\n", encoding="utf-8")
        definition, report = fm.load_definition(request["gameXml"], strict=False)
        if not report.supported:
            for f in report.features:
                self.log(f"profile refusal: {f.code} {f.feature_id}={f.effective_value}: {f.explanation}")
            self.log("forward model unavailable for this game; answering with empty actions")
            self.definition = None
            return {"ready": True, "protocolVersion": PROTOCOL_VERSION}
        self.definition = definition
        self.bridge = LiveBridge(definition, self.player, game_number)
        self.ai = HardAi(definition, definition.player_index[self.player], self.seed,
                         self.battle_samples, self.run_scale, self.log)
        self.log(f"ready as {self.player}; definition {definition.definition_digest[:12]}")
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def turn(self, request: dict) -> dict:
        phase = request.get("phase")
        handler = self.phase_handlers.get(phase)
        if handler is None:
            return {}
        if phase != "battle" and self.definition is None:
            return {}
        started = time.time()
        response = handler(request)
        self.log(f"{phase}: {time.time() - started:.2f}s action: {json.dumps(response)}")
        return response

    def _state(self, request: dict, phase: str):
        state = self.bridge.build_state(request["state"], phase)
        try:
            fm.create_state(self.definition, state)
        except fm.StateError as e:
            self.log(f"round {state.round} {phase}: state rejected by forward model: {e}")
            return None
        return state

    # --- Per-phase decisions ---

    def purchase(self, request: dict) -> dict:
        state = self._state(request, "purchase")
        action = self.ai.purchase(state) if state is not None else None
        return self.bridge.purchase_to_json(action) if action is not None else {}

    def combat_move(self, request: dict) -> dict:
        state = self._state(request, "combatMove")
        if state is None:
            return {}
        return {"moves": [self.bridge.move_to_json(a) for a in self.ai.combat_move(state)]}

    def battle(self, request: dict) -> dict:
        # In protocol v1 the engine resolves battles and picks this agent's casualties/retreats.
        return {"fightAll": True}

    def non_combat_move(self, request: dict) -> dict:
        state = self._state(request, "nonCombatMove")
        if state is None:
            return {}
        return {"moves": [self.bridge.move_to_json(a) for a in self.ai.noncombat_move(state)]}

    def place(self, request: dict) -> dict:
        state = self._state(request, "place")
        if state is None:
            return {}
        return {"placements": [self.bridge.place_to_json(a) for a in self.ai.place(state)]}


def main():
    agent = Agent(int(os.environ.get("TRIPLEA_HARD_AI_SEED", "0")),
                  float(os.environ.get("TRIPLEA_HARD_AI_RUN_SCALE", "1.0")),
                  int(os.environ.get("TRIPLEA_HARD_AI_BATTLE_SAMPLES", "5")))
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
