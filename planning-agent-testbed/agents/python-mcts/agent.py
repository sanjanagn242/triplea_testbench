#!/usr/bin/env python3
"""Python MCTS agent for the planning-agent testbench (newline-delimited JSON over stdin/stdout).

Each decision request is answered by building a forward-model state from the observation
(triplea_fm.testbench.LiveBridge), running PhaseSearch for that phase, and translating the chosen
action sequence into the testbench action format. Diagnostics go to stderr and to
planning-agent-testbed/logs/agent-<faction>-game-<n>.txt; stdout carries protocol lines only.

Environment:
  TRIPLEA_MCTS_SECONDS   base search time per decision (default 2.0)
  TRIPLEA_MCTS_ROLLOUT   rollout horizon in rounds (default 2)
  TRIPLEA_MCTS_SEED      search seed (default 0)
  TRIPLEA_AGENT_LOG      1 enables logging, anything else disables it (set in run-agent.sh)
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this agent's own modules (mcts.py)

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import LiveBridge  # noqa: E402

from mcts import PhaseSearch  # noqa: E402
from triplea_fm.baselines import Precomp  # noqa: E402

PROTOCOL_VERSION = 1
PHASE_TIME_SHARE = {"purchase": 0.6, "combatMove": 1.5, "nonCombatMove": 0.8, "place": 0.3}
LOG_DIR = HERE.parent.parent / "logs"
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh


class Agent:
    def __init__(self, base_seconds: float, rollout_rounds: int, seed: int):
        self.base_seconds = base_seconds
        self.rollout_rounds = rollout_rounds
        self.seed = seed
        self.definition = None
        self.bridge = None
        self.search = None
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
        definition, report = fm.load_definition(request["gameXml"], strict=False)
        if not report.supported:
            for f in report.features:
                self.log(f"profile refusal: {f.code} {f.feature_id}={f.effective_value}: {f.explanation}")
            self.log("forward model unavailable for this game; answering with empty actions")
            self.definition = None
            return {"ready": True, "protocolVersion": PROTOCOL_VERSION}
        self.definition = definition
        self.bridge = LiveBridge(definition, self.player)
        self.search = PhaseSearch(definition, Precomp(definition), rollout_rounds=self.rollout_rounds,
                                  seed=self.seed)
        self.log(f"ready as {self.player}; definition {definition.definition_digest[:12]}")
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def turn(self, request: dict) -> dict:
        phase = request.get("phase")
        if phase == "battle":
            return {"fightAll": True}
        if phase not in PHASE_TIME_SHARE or self.definition is None:
            return {}
        state = self.bridge.build_state(request["state"], phase)
        try:
            fm.create_state(self.definition, state)
        except fm.StateError as e:
            self.log(f"round {state.round} {phase}: state rejected by forward model: {e}")
            return {}
        self.search.time_budget = self.base_seconds * PHASE_TIME_SHARE[phase]
        started = time.time()
        sequence = self.search.search(state)
        self.log(f"round {state.round} {phase}: {self.search.iterations} iterations in "
                 f"{time.time() - started:.2f}s -> {len(sequence)} actions")
        return self.encode(phase, sequence)

    def encode(self, phase: str, sequence) -> dict:
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
        self.log(f"  action: {json.dumps(out)}")
        return out


def main():
    agent = Agent(float(os.environ.get("TRIPLEA_MCTS_SECONDS", "2.0")),
                  int(os.environ.get("TRIPLEA_MCTS_ROLLOUT", "2")),
                  int(os.environ.get("TRIPLEA_MCTS_SEED", "0")))
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
