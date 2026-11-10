import csv
import fcntl
import itertools
import json
import math
import os
import re
from collections import defaultdict
from contextlib import contextmanager
from pathlib import Path


@contextmanager
def locked(filename, mode="a"):
    """Open filename holding an exclusive lock on it, so parallel games (separate agent processes)
    writing the same file do not interleave or truncate each other's lines."""
    with open(filename, mode, newline="") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        try:
            yield f
        finally:
            f.flush()
            fcntl.flock(f, fcntl.LOCK_UN)


class MetricLogger:
    def __init__(self, filename, header=None):
        self.filename = filename
        self.header = header

        folder = os.path.dirname(filename)
        if folder:
            os.makedirs(folder, exist_ok=True)
        if header:
            self._ensure_header()

    def _ensure_header(self):
        """Write the header to a new or empty file. A file whose header is a prefix of this one
        (columns added since it was written) gets the new header, its rows padded with blanks. A file
        with any other header (columns renamed or reordered) is moved aside to <name>.old-<n>.csv and
        a new file is started, so rows of the two layouts are never mixed."""
        with locked(self.filename, "a+") as f:
            f.seek(0)
            rows = list(csv.reader(f))
            if not rows:
                csv.writer(f).writerow(self.header)
                return
            old = rows[0]
            if old == list(self.header):
                return
            if old != list(self.header[:len(old)]):
                if os.fstat(f.fileno()).st_ino != os.stat(self.filename).st_ino:
                    return  # another process moved it aside while we waited for the lock
                stem, ext = os.path.splitext(self.filename)
                moved = next(f"{stem}.old-{n}{ext}" for n in itertools.count(1)
                             if not os.path.exists(f"{stem}.old-{n}{ext}"))
                os.rename(self.filename, moved)
                with open(self.filename, "a", newline="") as new:
                    csv.writer(new).writerow(self.header)
                return
            pad = [""] * (len(self.header) - len(old))
            f.seek(0)
            f.truncate()
            writer = csv.writer(f)
            writer.writerow(self.header)
            writer.writerows(row + pad for row in rows[1:])

    def log(self, *values):
        """Append one row immediately to CSV."""
        with locked(self.filename) as f:
            csv.writer(f).writerow(values)

GAME_RESULTS_HEADER = ["game", "player", "rounds_played", "won", "turn_position", "game_time"]

# One row per search (per combat move) in efficiency.csv: time_per_turn is the search's wall-clock
# seconds, root_value the mean backed-up value at the root, max_rollout the most rounds a rollout
# advanced and num_action_sequences the number of root-to-leaf paths (leaves) in the search tree.
EFFICIENCY_HEADER = ["player", "game", "round", "time_per_turn", "num_iterations", "root_value",
                     "num_children_at_root", "max_rollout", "num_action_sequences",
                     "num_of_terr_attacked", "max_tree_depth"]

# One row per search in rollout_stats.csv (one per rollout filled tens of GB per tournament), from
# RolloutStats: phase (purchase, combat or noncombat), the number of rollouts, the mean of their
# rewards and reward_m2, the sum of the rewards' squared deviations from that mean (the reward
# variance within the search is reward_m2 / (rollouts - 1), and searches combine exactly), the
# reward range, the rollouts that stopped in a terminal win (reward > 0) or loss, and depth_counts,
# "depth:count" pairs joined by ";" of the rounds the rollouts advanced.
ROLLOUT_STATS_HEADER = ["player", "game", "round", "phase", "rollouts", "reward_mean", "reward_m2",
                        "reward_min", "reward_max", "terminal_wins", "terminal_losses",
                        "depth_counts"]


class RolloutStats:
    """Running totals of one search's rollouts, written as one rollout_stats.csv row at its end (see
    ROLLOUT_STATS_HEADER). add() is a few additions per rollout and keeps nothing per rollout, so
    the search keeps its time budget; write(), after the search, is the only file access."""

    def __init__(self, filename, phase="combat"):
        self.logger = MetricLogger(filename, header=ROLLOUT_STATS_HEADER)
        self.phase = phase
        self.reset()

    def reset(self):
        self.n = 0
        self.mean = self.m2 = 0.0
        self.lo, self.hi = math.inf, -math.inf
        self.wins = self.losses = 0
        self.depths = defaultdict(int)

    def add(self, reward, depth, terminal):
        """One rollout: its reward, the rounds it advanced and whether it reached a terminal state.
        Mean and m2 are updated as in Welford's algorithm, stable for any number of rollouts."""
        self.n += 1
        delta = reward - self.mean
        self.mean += delta / self.n
        self.m2 += delta * (reward - self.mean)
        if reward < self.lo:
            self.lo = reward
        if reward > self.hi:
            self.hi = reward
        if terminal:
            if reward > 0:
                self.wins += 1
            else:
                self.losses += 1
        self.depths[depth] += 1

    def write(self, player, game, round_, phase=None):
        """Log the search's row (none when it ran no rollout) and start over for the next one."""
        if self.n:
            self.logger.log(player, game, round_, phase or self.phase, self.n, f"{self.mean:.6g}",
                            f"{self.m2:.6g}", f"{self.lo:.6g}", f"{self.hi:.6g}", self.wins,
                            self.losses, ";".join(f"{d}:{c}" for d, c in sorted(self.depths.items())))
        self.reset()

def turn_position(definition, player_name):
    """1-based place of player_name in the round's turn order (order of the combat-move steps)."""
    order = []
    for step in definition.steps:
        if step.kind == "COMBAT_MOVE" and step.actor >= 0 and step.actor not in order:
            order.append(step.actor)
    me = definition.player_index.get(player_name)
    return order.index(me) + 1 if me in order else None


def metrics_dir(agent_dir):
    """Root of this agent's metrics: TRIPLEA_TESTBENCH_OUTPUT_DIR/metrics when the testbench sets it
    (a tournament keeps every agent's metrics in its own folder), otherwise <agent_dir>/metrics."""
    out = os.environ.get("TRIPLEA_TESTBENCH_OUTPUT_DIR")
    return Path(out) / "metrics" if out else Path(agent_dir) / "metrics"


def opponent_folder(definition, player_name):
    """Name of the per-opponent output folder for this game, e.g. "vs-Hard-AI" or
    "vs-tripleMind-mcts": the agent IDs of the factions not allied with player_name, from
    TRIPLEA_TESTBENCH_AGENTS (faction -> agent ID, JSON, set by the testbench for every agent
    process). Several opponents are joined with "_"; "vs-unknown" when the testbench did not say
    (e.g. the offline arena)."""
    try:
        agents = json.loads(os.environ.get("TRIPLEA_TESTBENCH_AGENTS") or "{}")
    except ValueError:
        agents = {}
    me = definition.player_index.get(player_name) if definition is not None else None
    opponents = set()
    for faction, agent_id in agents.items():
        if faction == player_name or not agent_id:
            continue
        other = definition.player_index.get(faction) if definition is not None else None
        if me is not None and other is not None and definition.is_allied(me, other):
            continue
        opponents.add(re.sub(r"[^A-Za-z0-9._-]+", "-", agent_id).strip("-") or "unnamed")
    return "vs-" + "_".join(sorted(opponents)) if opponents else "vs-unknown"


def output_folder(definition, player_name):
    """Relative folder for this game's metrics and records: "<game>/vs-<opponent>", e.g.
    "capture-the-flag-diag/vs-Hard-AI". <game> is the testbench's games/<game> folder, from
    TRIPLEA_TESTBENCH_GAME (set by the testbench for every agent process; "unknown-game" when
    absent, e.g. in the offline arena)."""
    game = re.sub(r"[^A-Za-z0-9._-]+", "-", os.environ.get("TRIPLEA_TESTBENCH_GAME") or "").strip("-")
    return os.path.join(game or "unknown-game", opponent_folder(definition, player_name))


def log_game_result(logger, game_number, player_name, result, position, seconds):
    """One row of metrics/<game>/vs-<opponent>/game_results.csv from the engine's game_end result; seconds is the game's
    wall-clock time from game_start to game_end as seen by the agent (game_time)."""
    logger.log(game_number, player_name, result.get("roundsPlayed"),
               1 if result.get("won") else 0, position,
               round(seconds, 1) if seconds is not None else "")
