import csv
import fcntl
import html
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
                     "num_children_at_root", "max_rollout", "num_action_sequences",
                     "num_of_terr_attacked", "max_tree_depth"]

# One row per rollout in rollout_stats.csv: the rounds it advanced, the reward it backed up and
# whether it stopped in a terminal (game over) state.
ROLLOUT_STATS_HEADER = ["player", "game", "round", "iteration", "rollout_depth_reached", "reward",
                        "is_terminal"]

# One row per tree level of the chosen action sequence in action_seq_stats.csv: how the chosen
# (most-visited) child compared to its siblings. visit_count / var / std_dev are over the visit
# counts of all children at that level, avg_* over their average values (both from the chooser's
# point of view); *_z is the chosen child's z-score among them (blank when they are all equal).
ACTION_SEQ_HEADER = ["player", "game", "round", "tl", "chooser", "action", "num_children",
                     "visit_count", "visit_share", "second_visit_count", "mean_visits", "var",
                     "std_dev", "visit_z", "avg_value", "second_avg_value", "mean_avg_value",
                     "avg_value_std_dev", "avg_value_z"]


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

_TREE_HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MCTS tree</title><style>
body { font: 13px monospace; margin: 1em; }
details { margin-left: 1.4em; border-left: 1px dotted #bbb; padding-left: 4px; }
summary { cursor: pointer; white-space: nowrap; }
.me { background: lightblue; } .opp { background: lightpink; }
.node { padding: 0 4px; border-radius: 3px; }
.best > summary { font-weight: bold; }
.act { color: #555; } .unexp { margin-left: 2.8em; color: gray; }
#bar { position: sticky; top: 0; background: white; padding: 4px 0; border-bottom: 1px solid #ccc; }
</style></head><body>
<div id="bar">
<button onclick="setAll(true)">Expand all</button>
<button onclick="setAll(false)">Collapse all</button>
<button onclick="bestPath()">Best path only</button>
min visits <input id="minv" type="number" value="0" min="0" style="width:5em" oninput="filter()">
&nbsp; bold = most-visited child; blue = played by me, pink = played by the opponent (the value is that player's)
</div>
<script>
function setAll(o) { document.querySelectorAll('details').forEach(d => d.open = o); }
function bestPath() { document.querySelectorAll('details').forEach(d =>
  d.open = d.classList.contains('best') || d.parentElement.id === 'tree'); }
function filter() { const m = +document.getElementById('minv').value;
  document.querySelectorAll('details').forEach(d => d.style.display = +d.dataset.v < m ? 'none' : ''); }
</script>
<div id="tree">
"""


def unplayed_actions(n):
    """The node's generated actions the search never expanded into a child."""
    return list(n.state.actions[n.state.actionIndex:])


def save_mcts_tree_html(root, whoAmI, html_path, edge_label):
    """Writes the whole tree as a collapsible HTML outline (no node limit): children sorted by
    visits, the most-visited path opened by default, unplayed actions listed in gray."""
    out = [_TREE_HTML_HEAD]
    counter = itertools.count()
    written = {}  # id(node) -> its number; a DAG's shared node (purchase_mcts) is written once

    def write(n, act_label, best, parent_open):
        if id(n) in written:
            out.append(f'<div class="unexp">{html.escape(act_label)} -> #{written[id(n)]} '
                       f'(transposition)</div>\n')
            return
        nid = next(counter)
        written[id(n)] = nid
        avg = (n.value / n.visits) if n.visits else 0.0
        cls = "me" if n.state.played_by == whoAmI else "opp"
        is_open = parent_open and (best or nid == 0)
        out.append(f'<details data-v="{n.visits}" class="{"best" if best else ""}"'
                   f'{" open" if is_open else ""}><summary>'
                   f'<span class="act">{html.escape(act_label)}</span> '
                   f'<span class="node {cls}">#{nid} vis={n.visits} avg={avg:.3f} '
                   f'tl={n.tree_level} played_by={html.escape(n.state.played_by)} '
                   f'next_moves_by={html.escape(n.state.next_moves_by)} '
                   f'actions={n.state.num_actions}</span></summary>\n')
        kids = sorted(n.children, key=lambda c: c.visits, reverse=True)
        for i, ch in enumerate(kids):
            write(ch, edge_label(n.action_to(ch), n.state) or "(root)", i == 0, is_open)
        for act in unplayed_actions(n):
            out.append(f'<div class="unexp">- {html.escape(edge_label(act, n.state))}</div>\n')
        out.append("</details>\n")

    write(root, "(root)", True, True)
    out.append("</div></body></html>\n")
    with open(html_path, "w") as f:
        f.write("".join(out))
    return html_path


def save_mcts_tree_png(root, whoAmI, out_prefix, max_nodes=500, render_img=False):
    """
    Saves:
      - out_prefix + ".dot"
      - out_prefix + ".html" (collapsible outline of the full tree)
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
            edges.append((node_id[id(n)], node_id[id(ch)], n.action_to(ch), n.state))

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
            label = f"#{nid}\\nvis={n.visits}\\navg={avg:.3f}\\ntl={n.tree_level}\\nplayed_by={n.state.played_by}\\nnext_moves_by={n.state.next_moves_by}\\nactions={n.state.num_actions}"
            # fill by the player whose move led to the node (its value's point of view): whoAmI
            # blue, any other player red
            fill = "lightblue" if n.state.played_by == whoAmI else "lightpink"
            f.write(f'  n{nid} [label="{label}", style=filled, fillcolor={fill}];\n')
            # logger.buffer(game, round_num, nid, pid, len(n.state.actions), n.tree_level, n.value, n.visits, avg)

        def edge_label(act, parent_state):
            if act is None:
                el = ""
            elif getattr(act, "end_phase", False):
                el = "END"
            elif hasattr(act, "unit_type"):  # purchase_mcts.PurchaseUnit
                el = f"buy {act.unit_type}"
            elif not act.moves:
                el = f"{act.to_terr} | skip"
            else:
                el = f"{act.to_terr} | s={act.strength:g} | {attack_summary(act, parent_state)}"
            return _safe(el, 60)

        for a, b, act, parent_state in edges:
            f.write(f'  n{a} -> n{b} [label="{edge_label(act, parent_state)}"];\n')

        # Actions the search never expanded: an arrowless dashed edge to a small point each, so
        # the unexplored options are still visible.
        for n in nodes:
            nid = node_id[id(n)]
            for i, act in enumerate(unplayed_actions(n)):
                f.write(f'  u{nid}_{i} [shape=point, width=0.05];\n')
                f.write(f'  n{nid} -> u{nid}_{i} [label="{edge_label(act, n.state)}", '
                        f'arrowhead=none, style=dashed, color=gray, fontcolor=gray];\n')

        f.write("}\n")

    save_mcts_tree_html(root, whoAmI, out_prefix + ".html", edge_label)

    # Try render to PNG
    if render_img:
        try:
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True)
            return dot_path, png_path
        except Exception:
            return dot_path, None
    else:
        return dot_path, None
