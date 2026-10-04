import csv
import os
import subprocess
from collections import defaultdict, deque

class MetricLogger:
    def __init__(self, filename, header=None):
        self.filename = filename
        self.header = header
        self._buffer = []
        
        folder = os.path.dirname(filename)
        if folder and not os.path.exists(folder):
            os.makedirs(folder, exist_ok=True)
        if header and not os.path.exists(filename):
            with open(filename, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(header)

    def log(self, *values):
        """Append one row immediately to CSV."""
        with open(self.filename, "a", newline="") as f:
            csv.writer(f).writerow(values)

    def buffer(self, *values):
        """Stage a row in memory without touching disk."""
        self._buffer.append(values)

    def flush(self):
        """Write all buffered rows to CSV in one shot, then clear buffer."""
        if not self._buffer:
            return
        with open(self.filename, "a", newline="") as f:
            csv.writer(f).writerows(self._buffer)
        self._buffer.clear()

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
            with open(self.filename, "a") as f:
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
