#!/usr/bin/env python3
"""Roxo OEP with greedy purchase, for the planning-agent testbench (newline-delimited JSON over
stdin/stdout). The roxo-oep agent (agents/roxo-oep), except that the purchase phase buys with
TripleMind's deterministic greedy policy (the first purchase_actions option) instead of nothing;
the engine places the units bought (place answers {}).

Online Evolutionary Planning as adapted to TripleA by M. Roxo ("MCTS-based Planning for Grand
Strategy Games", IST thesis, Section 4.4): one evolution per move phase over genomes of
territory-based actions (one per territory), evaluated with playouts of up to 10 rounds and
Fast-AI battles in the forward model (oep.py). The combat genes' actions, the playouts' moves and
the reward are TripleMind's; the noncombat genes' actions and the battles those of the
roxo-ne-mcts agent. Built on the template-python agent. The search's
playouts still buy nothing (Roxo's forward model has no production), and TripleA resolves the real
battles with dice. When TRIPLEA_AGENT_LOG is 1 (set in run-agent.sh), diagnostics go to stderr and
to planning-agent-testbed/logs/agent-<faction>-game-<n>.txt; stdout carries protocol lines only.
Metrics go to metrics/<game>/vs-<opponent>/ as for TripleMind agents.
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
from oep import RoxoOEP, move_phases_per_round  # noqa: E402
from oep_tree import save_oep_tree  # noqa: E402
import map_graph  # noqa: E402
from generators import ag as action_generators  # TripleMind's purchase policy  # noqa: E402
from world_model import Game  # noqa: E402

PROTOCOL_VERSION = 1
LOG_DIR = HERE.parent.parent.parent / "logs"  # planning-agent-testbed/logs
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh
SECONDS = float(os.environ.get("ROXO_SECONDS", "1.0"))
POPULATION = int(os.environ.get("ROXO_OEP_POPULATION", "50"))
SURVIVAL_RATE = float(os.environ.get("ROXO_OEP_SURVIVAL", "0.5"))
MUTATION_RATE = float(os.environ.get("ROXO_OEP_MUTATION", "0.1"))
PLAYOUT_ROUNDS = int(os.environ.get("ROXO_OEP_PLAYOUT_ROUNDS", "10"))
SEED = int(os.environ.get("ROXO_SEED", "0"))
# Debugging: after every search, draw the genealogy of its genomes in logs/trees/ (oep_tree.py).
SAVE_TREE = os.environ.get("ROXO_OEP_SAVE_TREE", "0") == "1"
TREE_PNG = os.environ.get("ROXO_OEP_TREE_PNG", "1") == "1"  # also render .png (needs Graphviz `dot`)
TREE_MAX_NODES = int(os.environ.get("ROXO_OEP_TREE_MAX_NODES", "300"))
TREE_RANKDIR = os.environ.get("ROXO_OEP_TREE_RANKDIR", "LR")  # LR: horizontal, TB: vertical
METRICS_DIR = metrics_dir(HERE)
# TripleMind's columns first (its plot scripts read them by name), then this agent's own.
BEST_PER_GENERATION_SAVED = 100  # best_value_per_generation keeps the first 100 generations
OEP_EFFICIENCY_HEADER = EFFICIENCY_HEADER +["phase", "evaluations", "duplicates",
                                             "best_value_per_generation"]


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
        # TripleMind's map graph: the purchase policy and the combat genes' action generation.
        self.map = map_graph.MapGraph.from_xml(request["gameXml"])
        # Game's playout length only sets Roxo's reward (Game.reward), which OEP no longer uses.
        self.game = Game(definition, definition.player_index[self.player],
                         PLAYOUT_ROUNDS * move_phases_per_round(definition), SEED)
        self.search = RoxoOEP(self.game, self.map, SECONDS, POPULATION, SURVIVAL_RATE,
                              MUTATION_RATE, PLAYOUT_ROUNDS, SEED, trace=SAVE_TREE)
        folder = METRICS_DIR / self.output_folder
        self.efficiency = MetricLogger(str(folder / "efficiency.csv"), header=OEP_EFFICIENCY_HEADER)
        self.rollout_stats = MetricLogger(str(folder / "rollout_stats.csv"), header=ROLLOUT_STATS_HEADER)
        self.log(f"ready as {self.player}; {SECONDS}s per move phase, population {POPULATION}, "
                 f"survival {SURVIVAL_RATE}, mutation {MUTATION_RATE}, {PLAYOUT_ROUNDS}-round playouts")
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
        """TripleMind's greedy purchase: the first purchase_actions option, if any."""
        if self.bridge is None:
            return {}
        sim = self.bridge.build_state(state, "purchase")
        game = map_graph.GameState.from_sim(self.map, self.definition, sim, self.player)
        options = action_generators.purchase_actions(game, self.definition)
        return self.bridge.purchase_to_json(options[0]) if options else {}

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
        self.record(state, phase, plan)
        if SAVE_TREE:
            self.save_tree(state, phase)
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

    def save_tree(self, state, phase: str):
        """Draw this search's genealogy to
        logs/trees/oep-<player>-game-<n>-round-<r>-<phase>.dot/.png."""
        tree_dir = LOG_DIR / "trees"
        tree_dir.mkdir(parents=True, exist_ok=True)
        prefix = tree_dir / f"oep-{self.player}-game-{self.game_number}-round-{state.round}-{phase}"
        st = self.search.stats
        title = (f"{self.player} game {self.game_number} round {state.round} {phase}: "
                 f"{st.generations} generations, {st.evaluations} distinct genomes evaluated, "
                 f"{st.duplicates} duplicates, {st.seconds:.2f}s, best {st.best_value:.3f}")
        dot_path, png_path = save_oep_tree(self.search.tree, self.search.best_key, self.definition,
                                           state, str(prefix), title, TREE_MAX_NODES, TREE_RANKDIR,
                                           TREE_PNG)
        self.log(f"  genealogy saved: {png_path or dot_path}")

    def record(self, state, phase: str, plan):
        st = self.search.stats
        self.log(f"round {state.round} {phase}: {st.generations} generations, {st.evaluations} "
                 f"evaluations ({st.duplicates} duplicates) in {st.seconds:.2f}s, "
                 f"{st.territories} territories, best value {st.best_value:.3f}")
        self.efficiency.log(self.player, self.game_number, state.round, round(st.seconds, 3),
                            st.generations,  # num_iterations: generations
                            round(st.best_value, 4),  # root_value: the best genome's value
                            st.territories,  # num_children_at_root: genes (candidate territories)
                            st.max_rollout,  # rounds
                            st.evaluations,  # num_action_sequences: distinct genomes evaluated
                            len(plan),  # num_of_terr_attacked: actions of the best genome
                            "",  # max_tree_depth: no tree
                            phase, st.evaluations, st.duplicates,
                            ";".join(f"{v:.4f}" for v in
                                     st.best_per_generation[:BEST_PER_GENERATION_SAVED]))
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
