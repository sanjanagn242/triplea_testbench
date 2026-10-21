import csv
import fcntl
import itertools
import json
import os
import re
import subprocess
from collections import defaultdict, deque
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
        self._buffer = []

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

    def buffer(self, *values):
        """Stage a row in memory without touching disk."""
        self._buffer.append(values)

    def flush(self):
        """Write all buffered rows to CSV in one shot, then clear buffer."""
        if not self._buffer:
            return
        with locked(self.filename) as f:
            csv.writer(f).writerows(self._buffer)
        self._buffer.clear()


GAME_RESULTS_HEADER = ["game", "player", "rounds_played", "won", "turn_position", "game_time"]

# One row per search (per combat move) in efficiency.csv: time_per_turn is the search's wall-clock
# seconds, root_value the mean backed-up value at the root, max_rollout the most rounds a rollout
# advanced and num_action_sequences the number of root-to-leaf paths (leaves) in the search tree.
EFFICIENCY_HEADER = ["player", "game", "round", "time_per_turn", "num_iterations", "root_value",
                     "num_children_at_root", "max_rollout", "num_action_sequences"]

# One row per rollout in rollout_stats.csv: the rounds it advanced, the reward it backed up and
# whether it stopped in a terminal (game over) state.
ROLLOUT_STATS_HEADER = ["player", "game", "round", "iteration", "rollout_depth_reached", "reward",
                        "is_terminal"]


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


class DebugLogger:
    def __init__(self, filename, is_debug=True):
        self.filename = filename
        self._buffer = []
        self.is_debug = is_debug

        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)

    def log(self, message):
        """Append one message immediately to log file."""
        if self.is_debug:
            with locked(self.filename) as f:
                f.write(message + "\n")


def _safe(s: str, maxlen=60):
    s = s.replace('"', "'").replace("\n", " ")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")

def attack_summary(action, state):
    """
    Returns string like: '2x infantry, 1x armour' for a CombatAction.
    `state` is the MCTSGameState the action was generated in: CombatAction.moves are
    fm.MoveActions holding forward-model unit ids, whose types are looked up in state.sim.
    """
    if not getattr(action, "moves", None):
        return ""
    d, sim = state.search.d, state.sim
    counts = defaultdict(int)
    for mv in action.moves:
        for uid in mv.units:
            u = sim.units.get(uid)
            counts[d.unit_types[u.type].name if u else f"unit#{uid}"] += 1
    return ", ".join(f"{qty}x {ut}" for ut, qty in counts.items())

def save_mcts_tree_png(root, out_prefix, max_nodes=500, render_img=False):
    """
    Saves:
      - out_prefix + ".dot"
      - out_prefix + ".png" (if graphviz 'dot' is available)
    """
    # BFS traversal
    q = deque([root])
    node_id = {id(root): 0}
    parent_id = {id(root): -1}
    nodes = [root]
    edges = []

    while q and len(nodes) < max_nodes:
        n = q.popleft()
        for ch in n.children:
            if id(ch) not in node_id:
                node_id[id(ch)] = len(nodes)
                parent_id[id(ch)] = node_id[id(n)]
                nodes.append(ch)
                q.append(ch)
            edges.append((node_id[id(n)], node_id[id(ch)], ch.action, n.state))

    dot_path = out_prefix + ".dot"
    png_path = out_prefix + ".png"

    # game = root.state.game_num
    # round_num = root.state.round
    with open(dot_path, "w") as f:
        f.write("digraph MCTS {\n")
        f.write('  rankdir=TB;\n')
        f.write('  node [shape=box, fontsize=10];\n')

        for n in nodes:
            nid = node_id[id(n)]
            pid = parent_id[id(n)]
            avg = (n.value / n.visits) if n.visits else 0.0
            label = f"#{nid}\\nvis={n.visits}\\navg={avg:.3f}\\ntl={n.tree_level}"
            f.write(f'  n{nid} [label="{label}"];\n')
            # logger.buffer(game, round_num, nid, pid, len(n.state.actions), n.tree_level, n.value, n.visits, avg)

        for a, b, act, parent_state in edges:
            if act is None:
                el = ""
            elif getattr(act, "end_phase", False):
                el = "END"
            elif not act.moves:
                el = f"{act.to_terr} | skip"
            else:
                el = f"{act.to_terr} | s={act.strength} | {attack_summary(act, parent_state)}"
            el = _safe(el, 60)

            f.write(f'  n{a} -> n{b} [label="{el}"];\n')

        f.write("}\n")

    # Try render to PNG
    if render_img:
        try:
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True)
            return dot_path, png_path
        except Exception:
            return dot_path, None
    else:
        return dot_path, None
