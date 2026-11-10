#!/usr/bin/env python3
"""TBETS-seg-hybrid agent (tbets-segmented's genomes and mutations, with heterogeneous MCTS's per-gene statistics) for the planning-agent testbench (newline-delimited JSON over
stdin/stdout).

No allies - only the player who is running this agent is considered.

Each purchase, combat-move and non-combat-move request is answered by building a forward-model
state from the observation (triplea_fm.testbench.LiveBridge) and running the one search,
tbets_seg_hybrid.TBETSSegHybrid (hmcts.HeterogeneousMCTS's tree, grown by tbets-segmented's phase genomes), from that decision: its tree goes on through this player's later phases
and the opponents' (opponent nodes), every node valued by the same reward. Place is not searched
(the engine places the purchased units) and battles are fought with fightAll. Diagnostics go to
stderr and to planning-agent-testbed/logs/agent-<faction>-game-<n>.txt; stdout carries protocol
lines only.

Environment:
  TRIPLEA_MCTS_SECONDS     search time per decision (default 1.0)
  TRIPLEA_MCTS_ITERS       iteration budget (default 100000; unused, the search is time-bounded)
  TRIPLEA_MCTS_ROLLOUT     rollout horizon in rounds; 1 ends a rollout when the round the leaf is
                           in is over (default 2)
  TRIPLEA_MCTS_EXP_CONST   UCB1 exploration constant (default 1.414)
  TRIPLEA_MCTS_EPSILON     probability of a random move / unit in the epsilon-greedy combat-move
                           and purchase rollouts, else the strongest (default 0.1)
  TRIPLEA_MCTS_SEED        search seed (default 0)
  TBETS_GREEDY_PCT         % of leaf expansions that add a child, the greedy genome of the next
                           phase, instead of a sibling, a mutation of the leaf's genome (default 5)
  TBETS_WIDEN              1 (default): every phase end selection passes through first gets one
                           mutated sibling genome, as tbets-segmented; 0: only the leaf is expanded
  TRIPLEMIND_VIEW          1 (default) shows the live map window, 0 hides it
  TRIPLEA_AGENT_LOG        1 enables logging, anything else disables it (set in run-agent.sh)
  TRIPLEMIND_SAVE_TREE     1 draws each search tree in trees/<game>/vs-<opponent>/, as tree_*
                           (combat move), tree_noncombat_* (non-combat move) and tree_purchase_*
                           (purchase) .dot, .html and .png (default 0)
  TRIPLEMIND_RECORD        1 saves the search tree of each searched decision of each game in
                           records/<faction>-game-<n>.jsonl (default 0)
"""

# from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))  # this agent's own modules (hmcts.py)

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import PHASE_KINDS, LiveBridge, observation_from_state  # noqa: E402

from hmcts import SEARCHED_PHASES, HMCTSGameState  # noqa: E402
from tbets_seg_hybrid import TBETSSegHybrid  # noqa: E402
from helper import (GAME_RESULTS_HEADER, MetricLogger, log_game_result, metrics_dir,  # noqa: E402
                    output_folder, turn_position)

import map_graph  # noqa: E402
import generators  # noqa: E402

PROTOCOL_VERSION = 1
LOG_DIR = HERE.parent.parent.parent / "logs"
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh
# Metrics and records go in a subfolder per map and opponent, e.g.
# metrics/capture-the-flag-diag/vs-Hard-AI/ (helper.output_folder). In a tournament the metrics go
# to the tournament's folder instead (TRIPLEA_TESTBENCH_OUTPUT_DIR, helper.metrics_dir).
METRICS_DIR = metrics_dir(HERE)
RECORD_DIR = HERE / "records"
GAME_RESULTS_CSV = "game_results.csv"  # one row per game this agent played
RECORDING = os.environ.get("TRIPLEMIND_RECORD", "0") == "1"
TREE_DIR = HERE / "trees"
SAVING_TREES = os.environ.get("TRIPLEMIND_SAVE_TREE", "0") == "1"


class GameRecorder:
    """One game's record for this faction, as JSON Lines: a "header" line (metadata and the initial
    observation), one "step" line per searched decision (the raw observation, the action answered
    to it and the search tree's chosen path), and a "result" line (winners and the final
    observation) when the engine sends game_end.

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

    def buffer(self, phase, state, action, seconds, search=None, path=None):
        """Stage one decision in memory without touching disk."""
        self._buffer.append({
            "type": "step",
            "index": self.num_steps,
            "round": state.get("round"),
            "step": state.get("step"),
            "phase": phase,
            "state": state,          # raw observation, as received
            "action": action,        # response, as sent
            "search": search,        # search statistics (iterations, max depth, seconds)
            "path": path,            # chosen path through the search tree, root first
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



class TripleMindAgent:
    def __init__(self, time_budget, iter_budget, depth_budget, exploration_constant, seed=0,
                 rollout_epsilon=0.1, combat_rollout="epsilon_greedy", greedy_pct=5.0,
                 widen=True):
        self.whoAmI = None
        self.log_path = None
        self.definition = None
        self.bridge = None
        self.recorder = None
        self.game_number = None
        self.turn_position = None
        self.game_started = None
        self.output_folder = "unknown-game/vs-unknown"

        # MCTS parameters
        self.time_budget = time_budget
        self.iter_budget = iter_budget
        self.depth_budget = depth_budget
        self.exploration_constant = exploration_constant
        self.rollout_epsilon = rollout_epsilon
        self.combat_rollout = combat_rollout
        self.greedy_pct = greedy_pct
        self.widen = widen
        self.seed = seed
        self.search = None

    def log(self, message: str):
        if not LOGGING:
            return
        line = f"{message}"
        print(line, file=sys.stderr, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def handle_request(self, request):
        kind = request.get("type")
        if kind == "game_start":
            return self.game_start(request)
        elif kind == "turn_request":
            return self.turn(request)
        elif kind == "game_end":
            return self.game_end(request)
        else:
            return {"error": f"Unknown request type: {kind}"}

    def game_start(self, request):
        self.whoAmI = request["playerName"]
        self.game_started = time.time()
        if LOGGING:
            LOG_DIR.mkdir(parents=True, exist_ok=True)
            self.log_path = LOG_DIR / f"agent-{self.whoAmI}-game-{request.get('gameNumber', 1)}.txt"
            self.log_path.write_text(f"Game {request.get('gameNumber', 1)}\n", encoding="utf-8")
        # self.log(f"request: {json.dumps(request)}")
        self.build_map(request)
        definition, report = fm.load_definition(request["gameXml"], strict=False)
        self.game_number = request.get("gameNumber", 1)
        self.turn_position = turn_position(definition, self.whoAmI)
        self.output_folder = output_folder(definition, self.whoAmI)
        if RECORDING:
            game_number = request.get("gameNumber", 1)
            state = request.get("initialState") or {}
            self.recorder = GameRecorder(str(RECORD_DIR / self.output_folder / f"{self.whoAmI}-game-{game_number}.jsonl"), {
                "agent": HERE.name,
                "player": self.whoAmI,
                "opponents": self.output_folder,
                "game_number": game_number,
                "game_name": state.get("gameName"),
                "map_name": state.get("mapName"),
                "definition_digest": definition.definition_digest if report.supported else None,
                "schema_version": request.get("schemaVersion"),
                "seed": self.seed,
                "time_budget": self.time_budget,
                "iter_budget": self.iter_budget,
                "depth_budget": self.depth_budget,
                "exploration_constant": self.exploration_constant,
                "rollout_epsilon": self.rollout_epsilon,
                "combat_rollout": self.combat_rollout,
                "greedy_pct": self.greedy_pct,
                "widen": self.widen,
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, state)
        if not report.supported:
            for f in report.features:
                self.log(f"unsupported feature: {f}")
            self.log("forward model unavailable for this game; answering with empty actions")
            self.definition = None
            return {"ready": True, "protocolVersion": PROTOCOL_VERSION}
        self.definition = definition
        self.bridge = LiveBridge(definition, self.whoAmI, request.get("gameNumber", 1))
        random.seed(self.seed)
        trees_dir = str(TREE_DIR / self.output_folder) if SAVING_TREES else None
        self.search = TBETSSegHybrid(definition, self.map, self.whoAmI, self.time_budget,
                                     self.depth_budget, self.exploration_constant,
                                     metrics_dir=METRICS_DIR / self.output_folder,
                                     trees_dir=trees_dir,
                                     rollout_epsilon=self.rollout_epsilon,
                                     combat_rollout=self.combat_rollout,
                                     greedy_pct=self.greedy_pct, widen=self.widen)
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def game_end(self, request):
        result = request.get("result") or {}
        log_game_result(MetricLogger(str(METRICS_DIR / self.output_folder / GAME_RESULTS_CSV),
                                     header=GAME_RESULTS_HEADER),
                        self.game_number, self.whoAmI, result, self.turn_position,
                        time.time() - self.game_started if self.game_started else None)
        if self.recorder:
            self.recorder.finish(result, request.get("state"))
            self.recorder.flush()
        return {"ok": True}

    def turn(self, request):
        started = time.time()
        if self.search is not None:
            self.search.last_root = None
        response = self.get_move(request)
        seconds = time.time() - started
        # Only the searched phases are recorded.
        phase = request.get("phase")
        if self.recorder and phase in SEARCHED_PHASES:
            search = self.search
            root = search.last_root if search is not None else None
            action = self.with_unit_types(response, root.state.sim) if root is not None else response
            self.recorder.buffer(phase, request["state"], action, seconds,
                                 search.last_stats if root is not None else None,
                                 self.tree_path(root) if root is not None else None)
            self.recorder.flush()
        return response

    def tree_path(self, root):
        """The nodes the search chose, root first: at each tree level, the child with the most
        visits (as mcts_search picks its actions). Each entry has the node's forward-model state,
        all its children with their states, and the chosen child's index; the path stops at a
        node with no children or at the node that passes the phase on (its state is the next
        searched decision, ours or an opponent's; its children are that decision's options)."""
        path = []
        node, edge = root, None
        while node is not None:
            entry = self.node_json(node, edge)
            entry["children"] = [self.node_json(e.child, e) for e in node.edges]
            best = max(node.edges, key=lambda e: e.visits) if node.edges else None
            entry["chosen"] = node.edges.index(best) if best is not None else None
            path.append(entry)
            if best is None or (edge is not None and edge.passes_phase):
                break
            node, edge = best.child, best
        return path

    def node_json(self, node, edge):
        """node as a record entry, with the statistics of the edge it was reached by on this
        path (the root's: its edges' totals)."""
        action = edge.action if edge is not None else None
        visits = edge.visits if edge is not None else node.visits()
        value = (edge.mean() if edge is not None else
                 sum(e.value for e in node.edges) / visits if visits else 0)
        return {
            "level": node.tree_level,
            "played_by": node.state.played_by,  # whose move led here; value is from their view
            "next_moves_by": node.state.next_moves_by,  # whose moves the node's children are
            "phase": node.state.phase,  # the decision the node's children decide
            "action": (None if action is None else
                       {"buy": action.unit_type, "end_phase": action.end_phase}
                       if hasattr(action, "unit_type") else
                       self.combat_action_json(action, node.state.sim)),
            "visits": visits,
            "value": value,
            "state": self.observation(node.state.sim),
            "terminal_reason": None if node.edges else self.terminal_reason(node),
        }

    @staticmethod
    def terminal_reason(node):
        """Why a tree node has no children: "no_actions" when the game is over there,
        "no_target" when its decision had nothing left to decide, "timeout" when
        it still had untried actions as the search time ran out."""
        state = node.state
        if state.is_terminal():
            return "no_actions"
        if state.actions and all(a.end_phase for a in state.actions):
            return "no_target"
        return "timeout" if state.actions else "no_actions"

    def observation(self, sim):
        """The forward-model state as an engine-style observation (the shape of a step's "state"),
        with the engine's unit ids; a unit the engine never sent keeps its model id."""
        obs = observation_from_state(self.definition, sim, self.whoAmI)
        uuid_of = self.bridge.uuid_of
        units = [u for t in obs["territories"] for u in t["units"]]
        units += [u for p in obs["players"] for u in p["unitsInReserve"]]
        for u in units:
            u["id"] = uuid_of.get(int(u["id"]), u["id"])
        return obs

    def combat_action_json(self, action, sim):
        moves = []
        for mv in action.moves:
            out = self.bridge.move_to_json(mv)
            out["unitTypes"] = self.unit_type_names(sim, mv.units)
            moves.append(out)
        return {
            "to": action.to_terr,
            "end_phase": action.end_phase,
            "strength": action.strength,
            "moves": moves,
        }

    def with_unit_types(self, response, sim):
        """A copy of a move response whose moves also list unitTypes, in unitIds order."""
        if "moves" not in response:
            return response
        index_of = {uuid: uid for uid, uuid in self.bridge.uuid_of.items()}
        return {**response, "moves": [
            {**mv, "unitTypes": self.unit_type_names(sim, [index_of[u] for u in mv["unitIds"]])}
            for mv in response.get("moves", [])]}

    def unit_type_names(self, sim, unit_ids):
        """Unit type names of forward-model unit ids (a unit's type never changes)."""
        return [self.definition.unit_types[sim.units[u].type].name for u in unit_ids]

    def build_map(self, request):
        # Any map: territories + <connection>s from the rules XML become the graph.
        self.map = map_graph.MapGraph.from_xml(request["gameXml"])
        self.view = (map_graph.MapView(self.map, self.whoAmI)
                     if os.environ.get("TRIPLEMIND_VIEW", "1") == "1" else None)
        self.log(f"map graph: {self.map}")
        self.log(f"action generators: {generators.IMPLEMENTATION}")

    def get_move(self, request):
        phase = request.get("phase")
        game_state = map_graph.GameState(self.map, request["state"], self.whoAmI)
        if self.view is not None:
            self.view.draw(game_state)
        if phase == "battle":
            self.log(f"turn {request.get('turn', 1)}: {self.whoAmI} fighting all")
            return {"fightAll": True}
        if self.bridge is None or phase not in PHASE_KINDS:
            return {}
        # Also registers this turn's unit ids with the bridge, which encode() needs for moves and
        # placements; the returned SimulationState is what search will run on.
        fm_state = self.bridge.build_state(request["state"], phase)
        response = {}
        if phase in SEARCHED_PHASES:
            state = HMCTSGameState(self.search, fm_state, self.whoAmI)
            if state.phase != phase:
                self.log(f"round {game_state.round}: {self.whoAmI} {phase} not at that decision "
                         f"in the forward model ({state.phase}); answering with no action")
                return response
            actions = self.search.mcts_search(state)
            if actions:
                response = self.encode(phase, actions)
            self.log(f"round {game_state.round}: {self.whoAmI} {phase} "
                     f"({self.search.iteration} iterations) {self.search.last_choice}")
            self.log(f"round {game_state.round}: {self.whoAmI} {phase} -> {response}")
        elif phase == "place":
            self.log(f"round {game_state.round}: {self.whoAmI} placing all purchased units")
        return response

    def encode(self, phase: str, sequence) -> dict:
        """Translate a sequence of triplea_fm actions into the engine's action object."""
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
        # self.log(f"  action: {json.dumps(out)}")
        return out

def main():
    agent = TripleMindAgent(float(os.environ.get("TRIPLEA_MCTS_SECONDS", "1.0")),
                            int(os.environ.get("TRIPLEA_MCTS_ITERS", "100000")),
                            int(os.environ.get("TRIPLEA_MCTS_ROLLOUT", "2")),
                            float(os.environ.get("TRIPLEA_MCTS_EXP_CONST", "1.414")),
                            int(os.environ.get("TRIPLEA_MCTS_SEED", "0")),
                            float(os.environ.get("TRIPLEA_MCTS_EPSILON", "0.1")),
                            os.environ.get("TRIPLEA_MCTS_COMBAT_ROLLOUT", "epsilon_greedy"),
                            float(os.environ.get("TBETS_GREEDY_PCT", "5")),
                            os.environ.get("TBETS_WIDEN", "1") == "1")
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            response = agent.handle_request(json.loads(line))
        except Exception as e:  # report, never crash the protocol loop
            traceback.print_exc(file=sys.stderr)
            agent.log(f"error: {e!r}")
            response = {"error": repr(e)}
        sys.stdout.write(json.dumps(response) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
