#!/usr/bin/env python3
"""Plays each agent of selfplay.json against itself through tools/tournament.py, records every game
with tap.py and merges them into the state-round-reward database (build_database.py; database.json
by default, shared by all runs, so every run adds to the running statistics).

selfplay.json (or --config) says what to play; the options below override it:
  "agents"      agent IDs, each played only against itself. An entry may also be an object that
                adds settings for that agent: {"agent": "tripleMind-mcts", "env": {"TRIPLEA_MCTS_SECONDS": "1.0"}};
                give it an "id" (default <agent>-evalrec) when the same agent is listed twice.
  "games"       self-play games per agent
  "gameFolder", "map"   the map to play on (games/<gameFolder>)
  "roundLimit"  rounds before a game ends without a winner (such games are discarded)
  "phases"      phases whose states are kept, comma-separated, or "all"
  "env"         settings for every agent
  "gui"         true shows the game window (default false: no window)

Each agent is played as a tournament variant whose only settings are TRIPLEA_TESTBENCH_PYTHON (the
tap), the tap's own variables and the settings above, so the agent's code and run script are
unchanged. A run's own files go in runs/<timestamp>/: the config, records/ (one file per agent,
faction and won/lost game) and tournament/<agent>/ (that agent's spec and tournament results; the
agent's metric CSVs are removed unless --keep-metrics).

Usage (from anywhere):
  python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py
  python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py --games 100 --num-batches 4
  python3 planning-agent-testbed/evaluate-reward/collect_selfplay.py --config my-agents.json
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parent
TOURNAMENT = TESTBED / "tools" / "tournament.py"
CONFIG = HERE / "selfplay.json"


def agent_entry(entry, env):
    """A config "agents" entry -> the tournament's variant object for it."""
    if isinstance(entry, str):
        entry = {"agent": entry}
    return {"agent": entry["agent"], "id": entry.get("id") or f"{entry['agent']}-evalrec",
            "env": {**env, **{k: str(v) for k, v in (entry.get("env") or {}).items()}}}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", type=Path, default=CONFIG,
                    help="agents and settings to play (default evaluate-reward/selfplay.json)")
    ap.add_argument("--agents", nargs="+", help="agent IDs, instead of the config's")
    ap.add_argument("--games", type=int, help="self-play games per agent")
    ap.add_argument("--round-limit", type=int)
    ap.add_argument("--phases", help="phases whose states are kept, comma-separated, or \"all\"")
    ap.add_argument("--env", action="append", default=[], metavar="KEY=VALUE",
                    help="extra setting for every agent, e.g. TRIPLEA_MCTS_SECONDS=1.0 (repeatable)")
    ap.add_argument("--keep-metrics", action="store_true",
                    help="keep the agents' metric CSVs (tournament/<agent>/agents/); removed by "
                         "default, the database only needs the records")
    ap.add_argument("--num-batches", type=int, help="games played in parallel (tournament.py --num-batches)")
    ap.add_argument("--out", type=Path, help="run folder (default evaluate-reward/runs/<timestamp>)")
    ap.add_argument("--database", type=Path, default=HERE / "database.json",
                    help="database the run's games are merged into (default evaluate-reward/database.json)")
    args = ap.parse_args()

    config = json.loads(args.config.read_text())
    agents = args.agents or config.get("agents") or []
    if not agents:
        ap.error(f"no agents: list them in {args.config} or give --agents")
    games = args.games or config.get("games", 10)
    phases = args.phases or config.get("phases", "combatMove")

    run = (args.out or HERE / "runs" / time.strftime("%Y-%m-%d_%H-%M-%S")).resolve()
    records = run / "records"
    records.mkdir(parents=True, exist_ok=True)
    (run / "selfplay.json").write_text(json.dumps(config, indent=2) + "\n")
    env = {
        "TRIPLEA_TESTBENCH_PYTHON": str(HERE / "tap.py"),
        # the interpreter the tap starts: whatever the agent would have used without it
        "EVAL_REWARD_PYTHON": os.environ.get("TRIPLEA_TESTBENCH_PYTHON", "python3"),
        "EVAL_REWARD_OUT": str(records),
        "EVAL_REWARD_PHASES": phases,
        **{k: str(v) for k, v in (config.get("env") or {}).items()},
        **dict(e.split("=", 1) for e in args.env),
    }

    # One tournament per agent: with several agents in one spec, tournament.py would also play them
    # against each other.
    failed = []
    for i, entry in enumerate(agents, 1):
        agent = agent_entry(entry, env)
        out = run / "tournament" / agent["id"]
        out.mkdir(parents=True, exist_ok=True)
        spec = {
            "games": games,
            "selfPlay": True,
            "agents": [agent],
            "roundLimit": args.round_limit or config.get("roundLimit", 100),
            "aiMovePauseMs": 0,
            "aiCombatStepPauseMs": 0,
            "maps": [{"gameFolder": config.get("gameFolder", "capture-the-flag-diag"), "map": config["map"]}],
        }
        spec_path = out / "evaluate-reward-selfplay.json"
        spec_path.write_text(json.dumps(spec, indent=2) + "\n")
        command = [sys.executable, str(TOURNAMENT), "--spec", str(spec_path), "--out", str(out),
                   "--gui" if config.get("gui", False) else "--no-gui"]
        if args.num_batches:
            command += ["--num-batches", str(args.num_batches)]
        print(f"\n[{i}/{len(agents)}] {agent['id']} vs itself, {games} games\n{' '.join(command)}", flush=True)
        if subprocess.call(command) != 0:
            failed.append(agent["id"])
        if not args.keep_metrics:
            shutil.rmtree(out / "agents", ignore_errors=True)

    subprocess.check_call([sys.executable, str(HERE / "build_database.py"), str(records),
                           "--out", str(args.database.resolve())])
    if failed:
        print(f"tournament failed for: {', '.join(failed)}", file=sys.stderr)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
