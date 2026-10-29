#!/usr/bin/env python3
"""Plays an agent against itself through tools/tournament.py, records every game with tap.py and
merges them into the state-round-reward database (build_database.py; database.json by default,
shared by all runs, so every run adds to the running statistics).

The agent is played as a tournament variant whose only settings are TRIPLEA_TESTBENCH_PYTHON (the
tap) and the tap's own variables, so the agent's code and run script are unchanged. A run's own files
go in runs/<timestamp>/: the spec, records/ (one file per faction and won/lost game) and tournament/
(the tournament's results and metrics).

Usage (from anywhere):
  python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py --games 20
  python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py --games 100 --num-batches 4 \\
      --env TRIPLEA_MCTS_SECONDS=1.0 --phases combatMove,purchase
"""

import argparse
import os
import json
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parent
TOURNAMENT = TESTBED / "tools" / "tournament.py"
AGENTS = ["tripleMind-mcts"]
MAP = {"name": "Capture The Flag", "gameName": "Capture The Flag", "aliases": ["capture_the_flag", "CTF"]}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--games", type=int, default=10, help="self-play games per agent (default 10)")
    ap.add_argument("--agents", nargs="+", default=AGENTS, help=f"agent IDs (default {' '.join(AGENTS)})")
    ap.add_argument("--game-folder", default="capture-the-flag-diag",
                    help="games/<folder> to play on (default capture-the-flag-diag)")
    ap.add_argument("--round-limit", type=int, default=100)
    ap.add_argument("--phases", default="combatMove",
                    help="phases whose states are kept, comma-separated, or \"all\" (default combatMove)")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra agent setting, e.g. TRIPLEA_MCTS_SECONDS=1.0 (repeatable)")
    ap.add_argument("--num-batches", type=int, help="games played in parallel (tournament.py --num-batches)")
    ap.add_argument("--out", type=Path, help="run folder (default evaluate-reward/runs/<timestamp>)")
    ap.add_argument("--database", type=Path, default=HERE / "database.json",
                    help="database the run's games are merged into (default evaluate-reward/database.json)")
    args = ap.parse_args()

    run = (args.out or HERE / "runs" / time.strftime("%Y-%m-%d_%H-%M-%S")).resolve()
    records = run / "records"
    records.mkdir(parents=True, exist_ok=True)
    env = {
        "TRIPLEA_TESTBENCH_PYTHON": str(HERE / "tap.py"),
        # the interpreter the tap starts: whatever the agent would have used without it
        "EVAL_REWARD_PYTHON": os.environ.get("TRIPLEA_TESTBENCH_PYTHON", "python3"),
        "EVAL_REWARD_OUT": str(records),
        "EVAL_REWARD_PHASES": args.phases,
        **dict(e.split("=", 1) for e in args.env),
    }
    spec = {
        "games": args.games,
        "selfPlay": True,
        "agents": [{"agent": a, "id": f"{a}-evalrec", "env": env} for a in args.agents],
        "roundLimit": args.round_limit,
        "aiMovePauseMs": 0,
        "aiCombatStepPauseMs": 0,
        "maps": [{"gameFolder": args.game_folder, "map": MAP}],
    }
    spec_path = run / "evaluate-reward-selfplay.json"
    spec_path.write_text(json.dumps(spec, indent=2) + "\n")

    command = [sys.executable, str(TOURNAMENT), "--spec", str(spec_path), "--out", str(run / "tournament")]
    if args.num_batches:
        command += ["--num-batches", str(args.num_batches)]
    print(" ".join(command), flush=True)
    code = subprocess.call(command)

    subprocess.check_call([sys.executable, str(HERE / "build_database.py"), str(records),
                           "--out", str(args.database.resolve())])
    sys.exit(code)


if __name__ == "__main__":
    main()
