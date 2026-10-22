#!/usr/bin/env python3
"""Roxo NE-MCTS agent for the planning-agent testbench (newline-delimited JSON over stdin/stdout).

Non-Exploring MCTS as implemented by M. Roxo in his MCTSGrandStrategy fork of TripleA
(ai/mctsclean/nonExploringMcts, for "MCTS-based Planning for Grand Strategy Games", IST thesis):
one search per move phase over territory-based actions, Weak-AI playouts and Fast-AI battles in
the forward model. Built on the template-python agent. Purchase and placement are not planned
(the thesis setting has no production), and TripleA resolves the real battles with dice. When TRIPLEA_AGENT_LOG is 1 (set in run-agent.sh), diagnostics go to
stderr and to planning-agent-testbed/logs/agent-<faction>-game-<n>.txt; stdout carries protocol
lines only. Metrics go to metrics/<game>/vs-<opponent>/ as for TripleMind agents.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "tripleMind"))  # helper.py: shared metric files and folders
sys.path.insert(0, str(HERE))  # this agent's modules

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import LiveBridge  # noqa: E402

from helper import (EFFICIENCY_HEADER, GAME_RESULTS_HEADER, ROLLOUT_STATS_HEADER,  # noqa: E402
                    MetricLogger, log_game_result, metrics_dir, output_folder, turn_position)
from ne_mcts import NonExploringMcts  # noqa: E402
from world_model import Game  # noqa: E402

PROTOCOL_VERSION = 1
LOG_DIR = HERE.parent.parent / "logs"
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh
SECONDS = float(os.environ.get("ROXO_SECONDS", "1.0"))
PLAYOUT_PHASES = int(os.environ.get("ROXO_PLAYOUT_PHASES", "40"))
MAX_ITERATIONS = int(os.environ.get("ROXO_MAX_ITERATIONS", "500"))
SEED = int(os.environ.get("ROXO_SEED", "0"))
METRICS_DIR = metrics_dir(HERE)


class Agent:
    def __init__(self):
        self.player = None
        self.log_path = None
        self.definition = None
        self.bridge = None
        self.search = None
        self.game_number = 1
        self.game_started = None
        self.turn_position = None
        self.output_folder = "unknown-game/vs-unknown"
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
        self.game_number = request.get("gameNumber", 1)
        self.game_started = time.time()
        if LOGGING:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LOG_DIR / f"agent-{self.player}-game-{self.game_number}.txt"
            self.log_path.write_text(f"{self.game_number}\n", encoding="utf-8")
        definition, report = fm.load_definition(request["gameXml"], strict=False)
        self.turn_position = turn_position(definition, self.player)
        self.output_folder = output_folder(definition, self.player)
        if not report.supported:
            for f in report.features:
                self.log(f"profile refusal: {f.code} {f.feature_id}={f.effective_value}: {f.explanation}")
            self.log("forward model unavailable for this game; answering with empty actions")
            return {"ready": True, "protocolVersion": PROTOCOL_VERSION}
        self.definition = definition
        self.bridge = LiveBridge(definition, self.player, self.game_number)
        self.game = Game(definition, definition.player_index[self.player], PLAYOUT_PHASES, SEED)
        self.search = NonExploringMcts(self.game, SECONDS, MAX_ITERATIONS)
        folder = METRICS_DIR / self.output_folder
        self.efficiency = MetricLogger(str(folder / "efficiency.csv"), header=EFFICIENCY_HEADER)
        self.rollout_stats = MetricLogger(str(folder / "rollout_stats.csv"), header=ROLLOUT_STATS_HEADER)
        self.log(f"ready as {self.player}; {SECONDS}s / {MAX_ITERATIONS} iterations per move phase, "
                 f"{PLAYOUT_PHASES}-phase playouts")
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def game_end(self, request: dict) -> dict:
        result = request.get("result") or {}
        log_game_result(MetricLogger(str(METRICS_DIR / self.output_folder / "game_results.csv"),
                                     header=GAME_RESULTS_HEADER),
                        self.game_number, self.player, result, self.turn_position,
                        time.time() - self.game_started if self.game_started else None)
        self.log(f"game over: {result.get('status')} winners {result.get('winners')}")
        return {"ok": True}

    def turn(self, request: dict) -> dict:
        phase = request.get("phase")
        handler = self.phase_handlers.get(phase)
        response = handler(request["state"]) if handler else {}
        self.log(f"{phase}: action: {json.dumps(response)}")
        return response

    # --- Per-phase decisions. `state` is the current observation (request["state"]). ---

    def purchase(self, state: dict) -> dict:
        # The thesis setting has no production; nothing is bought.
        return {}

    def combat_move(self, state: dict) -> dict:
        return self.plan_moves(state, "combatMove")

    def battle(self, state: dict) -> dict:
        # Asks TripleA to resolve every pending battle (with dice, unlike the agent's simulations).
        return {"fightAll": True}

    def non_combat_move(self, state: dict) -> dict:
        return self.plan_moves(state, "nonCombatMove")

    def place(self, state: dict) -> dict:
        return {}

    def plan_moves(self, observation: dict, phase: str) -> dict:
        if self.search is None:
            return {}
        state = self.bridge.build_state(observation, phase)
        try:
            state = fm.create_state(self.definition, state)
        except fm.StateError as e:
            self.log(f"round {state.round} {phase}: state rejected by forward model: {e}")
            return {}
        plan = self.search.run(state.clone())
        self.record(state, phase)
        return {"moves": [self.bridge.move_to_json(m) for m in self.routes(state, plan)]}

    def routes(self, state, plan):
        """ForwardModel.calculateMoveRoutes + doMove: every unit of the plan routed from where it
        stands now, then moves with the same route merged (into the later one)."""
        moves = [m for a in plan for m in self.game.routed_moves(state, a.territory, a.units)]
        merged = []
        for i, m in enumerate(moves):
            later = next((j for j in range(i + 1, len(moves)) if moves[j].route == m.route), None)
            if later is None:
                merged.append(m)
            else:
                moves[later] = fm.MoveAction(moves[later].units + m.units, m.route)
        return merged

    def record(self, state, phase: str):
        st = self.search.stats
        self.log(f"round {state.round} {phase}: {st.iterations} iterations in {st.seconds:.2f}s, "
                 f"{st.root_children} root children, selection depth {st.max_selection_depth}, "
                 f"root value {st.root_value:.3f}")
        self.efficiency.log(self.player, self.game_number, state.round, round(st.seconds, 3),
                            st.iterations, round(st.root_value, 4), st.root_children,
                            st.max_rollout, self.search.leaves())
        for i, (rounds, reward, terminal) in enumerate(st.rollouts):
            self.rollout_stats.buffer(self.player, self.game_number, state.round, i, rounds,
                                      round(reward, 4), 1 if terminal else 0)
        self.rollout_stats.flush()


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
