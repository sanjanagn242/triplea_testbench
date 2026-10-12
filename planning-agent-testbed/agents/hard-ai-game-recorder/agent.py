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
  TRIPLEA_HARD_AI_RECORD          1 saves a game record per game in records/<faction>-game-<n>.jsonl
                                  (default 0)
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
RECORD_DIR = HERE / "records"
RECORDING = os.environ.get("TRIPLEA_HARD_AI_RECORD", "0") == "1"


class GameRecorder:
    """One game's record for this faction, as JSON Lines: a "header" line (metadata and the initial
    observation), one "step" line per decision (the raw observation and the action answered to it),
    and a "result" line (winners and the final observation) when the engine sends game_end.

    The process may be stopped at any time, so the agent flushes after every step; a file without
    a result line is a game that was stopped before it ended."""

    def __init__(self, filename, meta, initial_state):
        self.filename = filename
        self.meta = meta
        self.initial_state = initial_state
        self.num_steps = 0
        self._buffer = []
        self._header_written = False

        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def buffer(self, phase, state, action, seconds):
        """Stage one decision in memory without touching disk."""
        self._buffer.append({
            "type": "step",
            "index": self.num_steps,
            "round": state.get("round"),
            "step": state.get("step"),
            "phase": phase,
            "state": state,          # raw observation, as received
            "action": action,        # response, as sent
            "policy_info": None,     # search statistics; Hard AI is heuristic and has none
            "seconds": seconds,
        })
        self.num_steps += 1

    def finish(self, result, final_state):
        """Stage the game's result, as sent by the engine in game_end."""
        self._buffer.append({
            "type": "result",
            "winners": result.get("winners", []),
            "won": result.get("won", False),
            "rounds_played": result.get("roundsPlayed"),
            "status": result.get("status"),
            "num_steps": self.num_steps,
            "final_state": final_state,
        })

    def flush(self):
        """Write all buffered lines (after the header, on the first flush), then clear buffer."""
        if not self._buffer:
            return
        if not self._header_written:
            # A new record replaces an earlier one for the same faction and game number.
            with open(self.filename, "w", encoding="utf-8") as f:
                f.write(json.dumps({"type": "header", "meta": self.meta,
                                    "initial_state": self.initial_state}) + "\n")
            self._header_written = True
        with open(self.filename, "a", encoding="utf-8") as f:
            f.writelines(json.dumps(line) + "\n" for line in self._buffer)
        self._buffer.clear()


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
        self.recorder = None
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
        if kind == "game_end":
            return self.game_end(request)
        return {"error": f"unknown request type {kind!r}"}

    def game_start(self, request: dict) -> dict:
        self.player = request["playerName"]
        game_number = request.get("gameNumber", 1)
        if LOGGING:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LOG_DIR / f"agent-{self.player}-game-{game_number}.txt"
            self.log_path.write_text(f"{game_number}\n", encoding="utf-8")
        definition, report = fm.load_definition(request["gameXml"], strict=False)
        if RECORDING:
            state = request.get("initialState") or {}
            self.recorder = GameRecorder(str(RECORD_DIR / f"{self.player}-game-{game_number}.jsonl"), {
                "agent": HERE.name,
                "player": self.player,
                "game_number": game_number,
                "game_name": state.get("gameName"),
                "map_name": state.get("mapName"),
                "definition_digest": definition.definition_digest if report.supported else None,
                "schema_version": request.get("schemaVersion"),
                "seed": self.seed,
                "run_scale": self.run_scale,
                "battle_samples": self.battle_samples,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, state)
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

    def game_end(self, request: dict) -> dict:
        result = request.get("result") or {}
        self.log(f"game over: {result.get('status')} winners {result.get('winners')}")
        if self.recorder:
            self.recorder.finish(result, request.get("state"))
            self.recorder.flush()
        return {"ok": True}

    def turn(self, request: dict) -> dict:
        phase = request.get("phase")
        handler = self.phase_handlers.get(phase)
        if handler is None:
            return {}
        if phase != "battle" and self.definition is None:
            return {}
        started = time.time()
        response = handler(request)
        seconds = time.time() - started
        self.log(f"{phase}: {seconds:.2f}s action: {json.dumps(response)}")
        if self.recorder:
            self.recorder.buffer(phase, request["state"], response, seconds)
            self.recorder.flush()
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
