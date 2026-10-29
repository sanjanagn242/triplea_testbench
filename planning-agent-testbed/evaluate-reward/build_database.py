#!/usr/bin/env python3
"""Merges the tap's game files (tap.py) into the state-round-reward database (JSON).

One entry per distinct (faction, round, state): when the same state comes up again in the same round,
its running statistics are updated instead of adding a record. The reward is the environment's, for
that faction at the end of the game: 1 won, -1 lost. Games without a winner (draws, the round limit)
are discarded.

Each observation is reduced to its abstract state (abstract_state.py: owners, unit counts per
owner and type, the players' resources; no unit IDs or static map data), and two observations are
the same state when their abstract states are equal. Entries store the abstract state; --raw also
keeps the first raw observation seen of each (the records folders keep all of them).

The database is updated in place: games already counted (by game_id) are skipped, so a records
folder can be merged again after more games were played.

Usage:
  python3 build_database.py <records folder> [<records folder> ...] [--out database.json]
"""

import argparse
import json
import time
from pathlib import Path

from abstract_state import abstract, state_key

HERE = Path(__file__).resolve().parent
DATABASE = HERE / "database.json"


def add(stats, value, rounds_to_end):
    """Welford's running mean and variance of the reward, plus the mean rounds left."""
    stats["count"] += 1
    stats["wins" if value > 0 else "losses"] += 1
    delta = value - stats["reward_mean"]
    stats["reward_mean"] += delta / stats["count"]
    stats["reward_m2"] += delta * (value - stats["reward_mean"])
    stats["reward_var"] = stats["reward_m2"] / stats["count"]
    if rounds_to_end is not None:
        stats["rounds_to_end_mean"] += (rounds_to_end - stats["rounds_to_end_mean"]) / stats["count"]


def merge(db, path: Path, raw=False):
    """Add one game file; returns "added", "counted" (already in the database), "draw" or "unfinished"."""
    game = json.loads(path.read_text(encoding="utf-8"))
    meta, result = game["meta"], game.get("result")
    game_id = f"{path.parent.parent.name}/{path.stem}"
    if result is None:
        return "unfinished"
    if not result.get("winners"):
        return "draw"
    if game_id in db["counted_games"]:
        return "counted"
    for record in game["records"]:
        state = abstract(record["state"])
        key = state_key(meta["player"], state, meta.get("game_name"))
        entry = db["entries"].get(key)
        if entry is None:
            entry = db["entries"][key] = {
                "player": meta["player"],
                "game_name": meta.get("game_name"),
                "round": record["round"],
                "step": record["step"],
                "phase": record["phase"],
                "count": 0, "wins": 0, "losses": 0,
                "reward_mean": 0.0, "reward_m2": 0.0, "reward_var": 0.0,
                "rounds_to_end_mean": 0.0,
                "games": [],
                "state": state,
            }
            if raw:
                entry["raw_state"] = record["state"]  # the first observation seen of this state
        add(entry, record["reward"], record.get("rounds_to_end"))
        entry["games"].append(game_id)
    db["counted_games"].append(game_id)
    return "added"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("records", nargs="+", type=Path, help="folders of the tap's <player>-game-<n>.json files")
    ap.add_argument("--out", type=Path, default=DATABASE,
                    help=f"database file, updated in place (default {DATABASE.relative_to(HERE.parent)})")
    ap.add_argument("--raw", action="store_true",
                    help="also keep the first raw observation of each new state (about 15 KB each)")
    args = ap.parse_args()

    db = (json.loads(args.out.read_text(encoding="utf-8")) if args.out.exists() else
          {"reward": {"won": 1, "lost": -1}, "counted_games": [], "entries": {}})
    outcomes = {}
    for path in sorted(f for folder in args.records for f in folder.glob("*-game-*.json")):
        outcome = merge(db, path, args.raw)
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
    db["updated"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    db["games"] = len(db["counted_games"])
    db["states"] = len(db["entries"])
    db["records"] = sum(e["count"] for e in db["entries"].values())
    partial = args.out.with_suffix(".json.tmp")
    partial.write_text(json.dumps(db), encoding="utf-8")
    partial.replace(args.out)
    print(f"game files: {', '.join(f'{n} {k}' for k, n in sorted(outcomes.items())) or 'none'}; "
          f"{args.out}: {db['games']} games, {db['records']} records in {db['states']} distinct states")


if __name__ == "__main__":
    main()
