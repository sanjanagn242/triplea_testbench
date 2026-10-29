#!/usr/bin/env python3
"""Records the states an agent is shown, and the game's outcome, without changing the agent.

The agents' run-agent.sh ends with `exec "${TRIPLEA_TESTBENCH_PYTHON:-python3}" -u agent.py`.
Pointing TRIPLEA_TESTBENCH_PYTHON at this script puts it between the engine and the agent: it starts
the real interpreter with the same arguments and relays every protocol line unchanged, both ways.
On the way it keeps each turn_request's observation, and when the engine sends game_end it labels
them with the game's reward for this agent's faction and writes one JSON file for the game.

Environment:
  EVAL_REWARD_OUT      folder for the game files (default evaluate-reward/records)
  EVAL_REWARD_PHASES   comma-separated phases to keep, or "all" (default combatMove: the state
                       before the agent's combat moves, the root of its search tree)
  EVAL_REWARD_PYTHON   the real interpreter (default python3)

A game that is stopped before game_end has no reward, and one without a winner (a draw, the round
limit) is discarded: neither writes anything.
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
OUT = Path(os.environ.get("EVAL_REWARD_OUT") or HERE / "records")
PHASES = {p.strip() for p in os.environ.get("EVAL_REWARD_PHASES", "combatMove").split(",") if p.strip()}
PYTHON = os.environ.get("EVAL_REWARD_PYTHON") or "python3"


def reward(result):
    """The environment's reward for this faction: 1 won, -1 lost, 0 no winner (discarded)."""
    if result.get("won"):
        return 1
    return -1 if result.get("winners") else 0


class Recorder:
    def __init__(self):
        self.games = {}  # (player, game number) -> {"meta": ..., "records": [...]}

    def see(self, line):
        if '"type"' not in line:
            return
        message = json.loads(line)
        kind = message.get("type")
        if kind == "game_start":
            self.game(message)  # the state an agent is shown first, before any turn_request
        elif kind == "turn_request" and ("all" in PHASES or message.get("phase") in PHASES):
            state = message.get("state") or {}
            self.game(message)["records"].append({
                "round": state.get("round"),
                "step": state.get("step"),
                "phase": message.get("phase"),
                "state": state,
            })
        elif kind == "game_end":
            self.finish(message)

    def game(self, message):
        key = (message.get("playerName"), message.get("gameNumber", 1))
        if key not in self.games:
            state = message.get("initialState") or message.get("state") or {}
            self.games[key] = {"meta": {
                "agent": Path(os.environ.get("TRIPLEA_TESTBENCH_OUTPUT_DIR", "")).name or None,
                "player": key[0],
                "game_number": key[1],
                "game_folder": os.environ.get("TRIPLEA_TESTBENCH_GAME"),
                "agents": os.environ.get("TRIPLEA_TESTBENCH_AGENTS"),
                "game_name": state.get("gameName"),
                "map_name": state.get("mapName"),
                "run_started_at_millis": message.get("runStartedAtMillis"),
                "started": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }, "records": []}
        return self.games[key]

    def finish(self, message):
        game = self.game(message)
        result = message.get("result") or {}
        value = reward(result)
        meta = game["meta"]
        if value == 0:  # a draw is discarded
            del self.games[(meta["player"], meta["game_number"])]
            return
        rounds_played = result.get("roundsPlayed")
        for record in game["records"]:
            record["reward"] = value
            if rounds_played is not None and record["round"] is not None:
                record["rounds_to_end"] = rounds_played - record["round"]
        game["result"] = {**result, "reward": value}
        game["final_state"] = message.get("state")
        OUT.mkdir(parents=True, exist_ok=True)
        target = OUT / f"{meta['player']}-game-{meta['game_number']}.json"
        partial = target.with_suffix(".json.tmp")
        partial.write_text(json.dumps(game), encoding="utf-8")
        partial.replace(target)
        del self.games[(meta["player"], meta["game_number"])]


def relay(source, sink):
    for line in source:
        sink.write(line)
        sink.flush()


def main():
    child = subprocess.Popen([PYTHON, *sys.argv[1:]], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                             text=True, bufsize=1)
    signal.signal(signal.SIGTERM, lambda *_: (child.terminate(), sys.exit(143)))
    replies = threading.Thread(target=relay, args=(child.stdout, sys.stdout), daemon=True)
    replies.start()
    recorder = Recorder()
    for line in sys.stdin:
        try:
            child.stdin.write(line)
            child.stdin.flush()
        except BrokenPipeError:
            break
        # Recorded after forwarding, so the agent's clock is not charged for it; a recording
        # failure never reaches the game.
        try:
            recorder.see(line)
        except Exception as e:  # noqa: BLE001
            print(f"evaluate-reward tap: {e!r}", file=sys.stderr)
    try:
        child.stdin.close()
    except BrokenPipeError:
        pass
    code = child.wait()
    replies.join(timeout=5)
    sys.exit(code)


if __name__ == "__main__":
    main()
