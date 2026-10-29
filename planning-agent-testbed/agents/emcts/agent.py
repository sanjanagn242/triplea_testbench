#!/usr/bin/env python3
"""Evolutionary MCTS agent for the planning-agent testbench (newline-delimited JSON over
stdin/stdout).

The combat move is searched with Evolutionary MCTS (Baier and Cowling, IEEE CIG 2018; emcts.py)
over roxo-oep-with-purchase's combat genomes (one gene per TripleMind target: an attack set or
skip). Purchase, noncombat move and place are TripleMind's: the first purchase_actions option,
noncombat_move's plan, and {} for place (the engine places the units bought). Battles are fought
by TripleA (fightAll). When TRIPLEA_AGENT_LOG is 1 (set in run-agent.sh), diagnostics go to stderr
and to planning-agent-testbed/logs/agent-<faction>-game-<n>.txt; stdout carries protocol lines
only. Metrics go to metrics/<game>/vs-<opponent>/ as for TripleMind agents.
"""

from __future__ import annotations

import json
import os
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this agent's modules (helper.py: a copy of TripleMind's)

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import LiveBridge  # noqa: E402

from helper import (EFFICIENCY_HEADER, GAME_RESULTS_HEADER, ROLLOUT_STATS_HEADER,  # noqa: E402
                    MetricLogger, log_game_result, metrics_dir, output_folder, turn_position)
from emcts import EMCTS  # noqa: E402
from emcts_tree import save_emcts_tree  # noqa: E402
import map_graph  # noqa: E402
from generators import ag as action_generators  # TripleMind's purchase and noncombat policies  # noqa: E402
from world_model import Game  # noqa: E402

PROTOCOL_VERSION = 1
LOG_DIR = HERE.parent.parent / "logs"  # planning-agent-testbed/logs
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh
SECONDS = float(os.environ.get("EMCTS_SECONDS", "1.0"))
PHASES = int(os.environ.get("EMCTS_PHASES", "40"))
EXP_CONST = float(os.environ.get("EMCTS_EXP_CONST", "0"))
SEED = int(os.environ.get("EMCTS_SEED", "0"))
ROLLOUT = int(os.environ.get("EMCTS_ROLLOUT", "1"))  # turns of ours evaluated; 1: no rollout
# Debugging: after every search, draw its tree in logs/trees/ (emcts_tree.py).
SAVE_TREE = os.environ.get("EMCTS_SAVE_TREE", "0") == "1"
TREE_PNG = os.environ.get("EMCTS_TREE_PNG", "1") == "1"  # also render .png (needs Graphviz `dot`)
TREE_MAX_NODES = int(os.environ.get("EMCTS_TREE_MAX_NODES", "300"))
TREE_MAX_DEPTH = int(os.environ.get("EMCTS_TREE_MAX_DEPTH", "3"))  # levels drawn below the path
TREE_RANKDIR = os.environ.get("EMCTS_TREE_RANKDIR", "LR")  # LR: horizontal, TB: vertical
METRICS_DIR = metrics_dir(HERE)
# TripleMind's columns first (its plot scripts read them by name), then this agent's own.
EMCTS_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["evaluations", "cache_hits", "greedy_value",
                                               "best_value", "phases", "tree_nodes"]


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
        self.map = map_graph.MapGraph.from_xml(request["gameXml"])
        # Roxo's forward model: routes the genes' units and resolves the battles deterministically.
        self.game = Game(definition, definition.player_index[self.player], seed=SEED)
        self.search = EMCTS(self.game, self.map, SECONDS, PHASES, EXP_CONST, SEED, ROLLOUT)
        folder = METRICS_DIR / self.output_folder
        self.efficiency = MetricLogger(str(folder / "efficiency.csv"), header=EMCTS_EFFICIENCY_HEADER)
        self.rollout_stats = MetricLogger(str(folder / "rollout_stats.csv"), header=ROLLOUT_STATS_HEADER)
        self.log(f"ready as {self.player}; {SECONDS}s per combat move, {PHASES} phases, C={EXP_CONST}, "
                 f"rollout {ROLLOUT} turn(s)")
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

    def view(self, state: dict, phase: str):
        """TripleMind's view of the forward-model state (units carry the ids the bridge encodes)."""
        sim = self.bridge.build_state(state, phase)
        return map_graph.GameState.from_sim(self.map, self.definition, sim, self.player)

    def purchase(self, state: dict) -> dict:
        """TripleMind's greedy purchase: the first purchase_actions option, if any."""
        if self.bridge is None:
            return {}
        options = action_generators.purchase_actions(self.view(state, "purchase"), self.definition)
        return self.bridge.purchase_to_json(options[0]) if options else {}

    def combat_move(self, observation: dict) -> dict:
        if self.search is None:
            return {}
        state = self.bridge.build_state(observation, "combatMove")
        try:
            state = fm.create_state(self.definition, state)
        except fm.StateError as e:
            self.log(f"round {state.round} combatMove: state rejected by forward model: {e}")
            return {}
        plan = self.search.run(state.clone())
        self.record(state, plan)
        if SAVE_TREE:
            self.save_tree(state)
        return {"moves": [self.bridge.move_to_json(m) for m in self.routes(state, plan)]}

    def battle(self, state: dict) -> dict:
        # Asks TripleA to resolve every pending battle (with dice, unlike the agent's simulations).
        return {"fightAll": True}

    def non_combat_move(self, state: dict) -> dict:
        """TripleMind's noncombat plan."""
        if self.bridge is None:
            return {}
        moves = action_generators.noncombat_move(self.view(state, "nonCombatMove"), self.definition)
        return {"moves": [self.bridge.move_to_json(m) for m in moves]} if moves else {}

    def place(self, state: dict) -> dict:
        return {}

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

    def save_tree(self, state):
        """Draw this search's tree to
        logs/trees/emcts-<player>-game-<n>-round-<r>.dot/.png."""
        tree_dir = LOG_DIR / "trees"
        tree_dir.mkdir(parents=True, exist_ok=True)
        prefix = tree_dir / f"emcts-{self.player}-game-{self.game_number}-round-{state.round}"
        st = self.search.stats
        title = (f"{self.player} game {self.game_number} round {state.round} combatMove: "
                 f"{st.iterations} iterations, {st.nodes} nodes, {st.evaluations} distinct genomes "
                 f"evaluated, {st.phases} phases, {st.seconds:.2f}s, greedy {st.greedy_value:.3f}, "
                 f"played {st.best_value:.3f}")
        try:
            dot_path, png_path = save_emcts_tree(self.search, self.definition, state, str(prefix),
                                                 title, TREE_MAX_NODES, TREE_MAX_DEPTH, TREE_RANKDIR,
                                                 TREE_PNG)
            self.log(f"  tree saved: {png_path or dot_path}")
        except Exception as e:  # a drawing is never worth losing the turn over
            self.log(f"tree not saved: {e!r}")

    def record(self, state, plan):
        st = self.search.stats
        self.log(f"round {state.round} combatMove: {st.iterations} iterations, {st.evaluations} "
                 f"evaluations ({st.cache_hits} cached) in {st.seconds:.2f}s, {st.territories} "
                 f"territories, {st.root_mutations} root mutations, greedy {st.greedy_value:.3f}, "
                 f"played {st.best_value:.3f}")
        self.efficiency.log(self.player, self.game_number, state.round, round(st.seconds, 3),
                            st.iterations,
                            round(st.best_value, 4),  # root_value: the played genome's value
                            st.root_mutations,  # num_children_at_root: the first root's mutations
                            max((r for r, _, _ in st.rollouts), default=0),  # rounds
                            st.leaves,  # num_action_sequences: leaves of the tree
                            len(plan),  # num_of_terr_attacked
                            st.max_depth,  # max_tree_depth: mutations from the first root
                            st.evaluations, st.cache_hits, round(st.greedy_value, 4),
                            round(st.best_value, 4), st.phases, st.nodes)
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
