#!/usr/bin/env python3
"""TBETS (modified) agent for the planning-agent testbench (newline-delimited JSON
over stdin/stdout).

The purchase, combat move and noncombat move are each searched with TBETS (tbets.py): MCTS over
phase genomes (one gene sequence per searched phase). The root is the current state, its
children genomes of the current phase, and every node's children genomes of the next searched
phase (purchase -> combat move -> noncombat move -> the next player's purchase ...); nodes are made greedily (the Greedy Action AI) or by mutating the selected genome
into a sibling, as in Evolutionary MCTS (Baier and Cowling, IEEE CIG 2018). Genes are heterogeneous-mcts's atomic actions (one unit bought, one attack set or
skip per target, one reinforcement or skip per territory short of cover). Place is {} (the
engine places the units bought). Battles are fought by TripleA (fightAll). When TRIPLEA_AGENT_LOG is 1 (set in run-agent.sh), diagnostics go to stderr
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

from helper import (EFFICIENCY_HEADER, GAME_RESULTS_HEADER, RolloutStats,  # noqa: E402
                    MetricLogger, log_game_result, metrics_dir, output_folder, turn_position)
from tbets import COMBAT, NONCOMBAT, PURCHASE, TBETS  # noqa: E402
from tbets_tree import save_tbets_tree  # noqa: E402
import map_graph  # noqa: E402
from generators import ag as action_generators  # TripleMind's purchase and noncombat policies  # noqa: E402
from world_model import TERRITORY, Game  # noqa: E402

PROTOCOL_VERSION = 1
LOG_DIR = HERE.parent.parent / "logs"  # planning-agent-testbed/logs
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh
SECONDS = float(os.environ.get("TBETS_SECONDS", "1.0"))
EXP_CONST = float(os.environ.get("TBETS_EXP_CONST", "0"))
SEED = int(os.environ.get("TBETS_SEED", "0"))
ROLLOUT = int(os.environ.get("TBETS_ROLLOUT", "2"))  # rounds started after the root's when a rollout stops
GREEDY_PCT = float(os.environ.get("TBETS_GREEDY_PCT", "5"))  # % of expansions adding the next phase's greedy genome
# Debugging: after every search, draw its tree in logs/trees/ (tbets_tree.py).
SAVE_TREE = os.environ.get("TBETS_SAVE_TREE", "0") == "1"
TREE_PNG = os.environ.get("TBETS_TREE_PNG", "0") == "1"  # also render .png (needs Graphviz `dot`)
TREE_MAX_NODES = int(os.environ.get("TBETS_TREE_MAX_NODES", "30"))  # genomes drawn in the .dot/.png; the .html has all
TREE_RANKDIR = os.environ.get("TBETS_TREE_RANKDIR", "LR")  # LR: genomes stacked, TB: side by side
METRICS_DIR = metrics_dir(HERE)
# TripleMind's columns first (its plot scripts read them by name), then this agent's own.
TBETS_EFFICIENCY_HEADER = EFFICIENCY_HEADER + ["evaluations", "cache_hits", "greedy_value",
                                               "best_value", "phases", "tree_nodes", "horizon",
                                               "phase", "unique_plans"]
# The genome phase each searched testbench phase plays.
GENE_PHASE = {"purchase": PURCHASE, "combatMove": COMBAT, "nonCombatMove": NONCOMBAT}


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
        self.game = Game(definition)
        self.search = TBETS(self.game, self.map, SECONDS, EXP_CONST, ROLLOUT, GREEDY_PCT / 100, SEED)
        folder = METRICS_DIR / self.output_folder
        self.efficiency = MetricLogger(str(folder / "efficiency.csv"), header=TBETS_EFFICIENCY_HEADER)
        self.rollout_stats = RolloutStats(str(folder / "rollout_stats.csv"))
        self.log(f"ready as {self.player}; {SECONDS}s per searched phase, C={EXP_CONST}, "
                 f"rollout {ROLLOUT} round(s), {GREEDY_PCT}% greedy children (next turn), else mutated siblings")
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

    def search_genome(self, observation: dict, phase: str):
        """(forward-model state, the played genome's genes in this phase, their Turn), or (state,
        None, None) if the state is rejected or the genome does not reach this phase."""
        state = self.bridge.build_state(observation, phase)
        try:
            state = fm.create_state(self.definition, state)
        except fm.StateError as e:
            self.log(f"round {state.round} {phase}: state rejected by forward model: {e}")
            return state, None, None
        played, genes, turn = self.search.run(state.clone())
        if played != GENE_PHASE[phase]:
            self.log(f"round {state.round} {phase}: search returned {played}; "
                     f"playing TripleMind's greedy {phase}")
            return state, None, None
        self.record(state, phase, genes)
        if SAVE_TREE:
            self.save_tree(state, phase)
        return state, genes, turn

    def purchase(self, observation: dict) -> dict:
        if self.search is None:
            return {}
        _, genes, turn = self.search_genome(observation, "purchase")
        if genes is None:  # TripleMind's greedy purchase: the first purchase_actions option, if any
            options = action_generators.purchase_actions(self.view(observation, "purchase"), self.definition)
            return self.bridge.purchase_to_json(options[0]) if options else {}
        # The units bought, one PurchaseAction with their counts.
        return self.bridge.purchase_to_json(turn.actions[0]) if turn.actions else {}

    def combat_move(self, observation: dict) -> dict:
        if self.search is None:
            return {}
        state, genes, _ = self.search_genome(observation, "combatMove")
        if genes is None:
            return {}
        plan = [a for a in genes if a.kind == TERRITORY]
        return {"moves": [self.bridge.move_to_json(m) for m in self.routes(state, plan)]}

    def battle(self, state: dict) -> dict:
        # Asks TripleA to resolve every pending battle (with dice, unlike the agent's simulations).
        return {"fightAll": True}

    def non_combat_move(self, observation: dict) -> dict:
        if self.search is None:
            return {}
        _, genes, turn = self.search_genome(observation, "nonCombatMove")
        if genes is None:  # TripleMind's noncombat plan
            moves = action_generators.noncombat_move(self.view(observation, "nonCombatMove"), self.definition)
        else:  # the air landings, the reinforcements the genes chose, then the positioning moves
            moves = turn.actions
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

    def save_tree(self, state, phase: str):
        """Draw this search's tree to
        logs/trees/tbets-modified-<player>-game-<n>-round-<r>-<phase>.dot/.png/.html."""
        tree_dir = LOG_DIR / "trees"
        tree_dir.mkdir(parents=True, exist_ok=True)
        prefix = tree_dir / f"tbets-modified-{self.player}-game-{self.game_number}-round-{state.round}-{phase}"
        st = self.search.stats
        title = (f"{self.player} game {self.game_number} round {state.round} {phase}: "
                 f"{st.iterations} iterations, {st.nodes} nodes, {st.evaluations} distinct genomes "
                 f"evaluated, {st.seconds:.2f}s, greedy {st.greedy_value:.3f}, "
                 f"played {st.best_value:.3f}")
        try:
            dot_path, png_path = save_tbets_tree(self.search, self.definition, state, str(prefix),
                                                 title, TREE_MAX_NODES, TREE_RANKDIR,
                                                 TREE_PNG)
            self.log(f"  tree saved: {png_path or dot_path}")
        except Exception as e:  # a drawing is never worth losing the turn over
            self.log(f"tree not saved: {e!r}")

    def record(self, state, phase: str, genes):
        st = self.search.stats
        attacked = sum(1 for a in genes if a.kind == TERRITORY) if phase == "combatMove" else 0
        self.log(f"round {state.round} {phase}: {st.iterations} iterations, {st.evaluations} "
                 f"evaluations ({st.cache_hits} cached) in {st.seconds:.2f}s, {st.genes} genes, "
                 f"{st.territories} territories, {st.nodes} nodes, depth {st.max_depth} phases ({st.greedy_genomes} greedy "
                 f"children, {st.mutations} mutated siblings), greedy {st.greedy_value:.3f}, "
                 f"played {st.best_value:.3f}")
        self.efficiency.log(self.player, self.game_number, state.round, round(st.seconds, 3),
                            st.iterations,
                            round(st.best_value, 4),  # root_value: the played genome's value
                            len(self.search.root.children),  # num_children_at_root: my genomes
                            max((r for r, _, _ in st.rollouts), default=0),  # rounds advanced
                            st.leaves,  # num_action_sequences: leaves of the tree
                            attacked,  # num_of_terr_attacked (0 outside the combat move)
                            st.max_depth,  # max_tree_depth: phases from the root
                            st.evaluations, st.cache_hits, round(st.greedy_value, 4),
                            round(st.best_value, 4),
                            0,  # phases: no bridge burning
                            st.nodes,
                            ROLLOUT,  # horizon: rollout length in rounds
                            phase,
                            st.unique_plans)
        for rounds, reward, terminal in st.rollouts:
            self.rollout_stats.add(reward, rounds, terminal)
        self.rollout_stats.write(self.player, self.game_number, state.round, GENE_PHASE[phase])


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
