#!/usr/bin/env python3
"""Offline arena: plays full games with the forward model standing in for the TripleA engine.

A testbench tool, independent of any agent and of the Java engine. Agent-controlled factions run
as real subprocesses speaking the testbench protocol (exactly what ExternalAgentPlayer sends:
game_start, then turn_request per phase with a v1 observation). The arena emulates the Java
adapter: moves are applied one by one and rejected ones are skipped; "battle" fights every
battle; casualties and retreats for agent factions use the library's default (standing in for
WeakAi, as in protocol v1). Other factions play the library's baseline HeuristicPolicy.

Usage (from planning-agent-testbed/):
  python3 tools/arena.py --agent Russians=agents/python-mcts/run-agent.sh --games 2 --seconds 0.5
  python3 tools/arena.py --agent Russians=agents/cpp-mcts/run-agent.sh \
      --agent Italians=agents/python-mcts/run-agent.sh --games 2   # head-to-head
"""

from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parent
sys.path.insert(0, str(TESTBED / "forward-model" / "python"))

import triplea_fm as fm  # noqa: E402
from triplea_fm.actions import action_from_json  # noqa: E402
from triplea_fm.definition import COMBAT_MOVE, NONCOMBAT_MOVE  # noqa: E402
from triplea_fm.state import (BATTLE_SELECTION_D, CASUALTIES_D, MOVE_D, PLACE_D,  # noqa: E402
                              PURCHASE_D, RETREAT_D)
from triplea_fm.testbench import observation_from_state  # noqa: E402

from triplea_fm.baselines import HeuristicPolicy  # noqa: E402

DEFAULT_XML = TESTBED / "games/capture-the-flag/map/capture_the_flag/map/games/capture_the_flag.xml"


class AgentProcess:
    def __init__(self, command, env):
        self.proc = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                     stderr=subprocess.DEVNULL, text=True, env=env)
        self.request_id = 0

    def ask(self, message: dict) -> dict:
        self.request_id += 1
        message = dict(message, schemaVersion=1, requestId=self.request_id)
        self.proc.stdin.write(json.dumps(message) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("agent closed its output")
        return json.loads(line)

    def close(self):
        try:
            self.proc.stdin.close()
            self.proc.wait(timeout=5)
        except Exception:
            self.proc.kill()


def play_game(d, xml_text, agent_cmds, game_number, seconds, seed, max_rounds, verbose):
    env = dict(os.environ, TRIPLEA_MCTS_SECONDS=str(seconds), TRIPLEA_MCTS_SEED=str(seed),
               TRIPLEA_AGENT_SECONDS=str(seconds), TRIPLEA_AGENT_SEED=str(seed))
    rnd = random.Random(seed)
    heuristic = HeuristicPolicy(d)
    s = fm.create_initial_state(d)
    rng = fm.SeededRng(10_000 + seed)
    agents = {}
    for name, cmd in agent_cmds.items():
        agents[d.player_index[name]] = AgentProcess(cmd, env)
        reply = agents[d.player_index[name]].ask({
            "type": "game_start", "playerName": name, "gameXml": xml_text,
            "initialState": observation_from_state(d, s, name), "gameNumber": game_number})
        assert reply.get("ready"), reply
    stats = {"rejected": 0, "applied": 0}

    def step(action):
        nonlocal s, rng
        r = fm.transition(d, s, action, rng)
        if r.status in ("DECISION", "TERMINAL"):
            s, rng = r.next_state, r.next_rng
            stats["applied"] += 1
            return True
        stats["rejected"] += 1
        if verbose:
            print("  rejected:", fm.action_to_json(d, action), r.validation.code, r.validation.message)
        return False

    while not fm.is_terminal(s) and s.round <= max_rounds:
        dec = s.decision
        actor = dec.actor
        agent = agents.get(actor)
        if agent is None or dec.kind in (CASUALTIES_D, RETREAT_D, BATTLE_SELECTION_D):
            step(heuristic.act(s, rnd))
            continue
        name = d.player_names[actor]
        kind = d.steps[s.step].kind
        phase = {PURCHASE_D: "purchase", PLACE_D: "place"}.get(dec.kind) or \
            ("combatMove" if kind == COMBAT_MOVE else "nonCombatMove")
        reply = agent.ask({"type": "turn_request", "playerName": name, "phase": phase,
                           "state": observation_from_state(d, s, name), "nonCombat": kind == NONCOMBAT_MOVE,
                           "gameNumber": game_number})
        if "error" in reply:
            print(f"  agent error: {reply['error']}")
        if phase == "purchase" and reply.get("purchases"):
            q = {p["rule"]: p["count"] for p in reply["purchases"]}
            step(action_from_json(d, {"kind": "purchase", "quantities": q}))
        elif phase in ("combatMove", "nonCombatMove"):
            for m in reply.get("moves", []):
                route = m.get("route") or [m["from"], m["to"]]
                ids = [int(u) for u in m["unitIds"] if int(u) in s.units]
                if ids and s.decision is not None and s.decision.kind == MOVE_D:
                    step(fm.MoveAction(tuple(ids), tuple(d.territory_index[t] for t in route),
                                       m.get("intent", "NORMAL")))
        elif phase == "place":
            for p in reply.get("placements", []):
                ids = [int(u) for u in p["unitIds"] if int(u) in s.units]
                if ids and s.decision is not None and s.decision.kind == PLACE_D:
                    step(fm.PlaceAction(tuple(ids), d.territory_index[p["territory"]]))
        if s.decision is not None and s.decision.actor == actor and s.decision.kind == dec.kind:
            step(fm.EndPhaseAction())
    # As the engine does: game_end carries the result (winners are players) and the final observation.
    winners = [n for pid, n in enumerate(d.player_names)
               if s.terminal and d.alliance_of[pid] in s.terminal.winners]
    status = f"{s.terminal.rule_id} victory" if s.terminal else "Arena round limit reached."
    for actor, a in agents.items():
        name = d.player_names[actor]
        a.ask({"type": "game_end", "playerName": name, "state": observation_from_state(d, s, name),
               "result": {"winners": winners, "won": name in winners, "roundsPlayed": s.round,
                          "status": status}, "gameNumber": game_number})
    for a in agents.values():
        a.close()
    winner = d.alliance_names[s.terminal.winners[0]] if s.terminal else None
    return winner, s.round, stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--xml", default=str(DEFAULT_XML))
    ap.add_argument("--agent", action="append", default=[],
                    help="Faction=command (command run via bash); repeatable")
    ap.add_argument("--games", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=0.5)
    ap.add_argument("--max-rounds", type=int, default=100)
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()
    xml_text = Path(args.xml).read_text()
    d, _ = fm.load_definition(xml_text)
    cmds = {}
    for spec in args.agent or [f"Russians={TESTBED / 'agents/python-mcts/run-agent.sh'}"]:
        faction, cmd = spec.split("=", 1)
        cmds[faction] = ["bash", cmd]
    results = {}
    for g in range(1, args.games + 1):
        t0 = time.time()
        winner, rounds, stats = play_game(d, xml_text, cmds, g, args.seconds, g, args.max_rounds, args.verbose)
        results[winner] = results.get(winner, 0) + 1
        print(f"game {g}: winner={winner} rounds={rounds} applied={stats['applied']} "
              f"rejected={stats['rejected']} time={time.time() - t0:.0f}s", flush=True)
    print("results:", results)


if __name__ == "__main__":
    main()
