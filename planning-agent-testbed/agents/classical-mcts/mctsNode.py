import math
import subprocess
from collections import Counter, deque
from typing import List, Optional

import triplea_fm as fm  # noqa: E402


class MCTSNode:
    """A search-tree node: the state reached by `action` from `parent` (closed-loop: one sampled state)."""

    def __init__(self, state: fm.SimulationState, parent: Optional["MCTSNode"] = None, action=None):
        self.state = state
        self.parent = parent
        self.action = action                # the action that led here from parent (None at the root)
        self.children: List["MCTSNode"] = []
        self.visits = 0
        self.value = 0.0                    # sum of backed-up values, from the searching player's view
        self.untried_actions: Optional[list] = None  # set by the agent on first visit

    def is_fully_expanded(self):
        return self.untried_actions is not None and len(self.untried_actions) == 0

    def is_terminal(self):
        return self.state.terminal is not None

    def best_child_uct(self, exploration_constant):
        """UCB1: mean value + C * sqrt(ln N / n)."""
        best_value = float('-inf')
        best_child = None
        log_n = math.log(self.visits + 1)
        for child in self.children:
            uct_value = (child.value / (child.visits + 1e-6)) + exploration_constant * math.sqrt(log_n / (child.visits + 1e-6))
            if uct_value > best_value:
                best_value = uct_value
                best_child = child
        return best_child


# ---- debugging: dump a search tree as Graphviz ----

def _safe(s: str, maxlen=60):
    s = s.replace('"', "'").replace("\n", " ")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def action_summary(d: fm.GameDefinition, state: fm.SimulationState, action) -> str:
    """Short edge label, e.g. 'A -> B | 2x infantry, 1x armour'. `state` is the parent's state."""
    if action is None:
        return ""
    tn = d.territory_names
    if action.kind == "end_phase":
        return "END"
    if action.kind == "purchase":
        return "buy " + ", ".join(f"{c}x {d.production_rules[r].name}" for r, c in action.quantities)
    if action.kind in ("move", "place"):
        counts = Counter(d.unit_types[state.units[u].type].name for u in action.units if u in state.units)
        units = ", ".join(f"{q}x {t}" for t, q in counts.items())
        where = (f"{tn[action.route[0]]} -> {tn[action.route[-1]]}" if action.kind == "move"
                 else f"place @ {tn[action.territory]}")
        return f"{where} | {units}"
    return action.kind


def save_mcts_tree(root: MCTSNode, d: fm.GameDefinition, out_prefix: str, title: str = "",
                   max_nodes=500, render_img=False):
    """Writes out_prefix + ".dot" (BFS, at most max_nodes nodes) and, with render_img and Graphviz's
    `dot` installed, out_prefix + ".png". Returns (dot_path, png_path or None)."""
    q = deque([root])
    node_id = {id(root): 0}
    depth = {id(root): 0}
    nodes = [root]
    edges = []

    while q and len(nodes) < max_nodes:
        n = q.popleft()
        for ch in n.children:
            if len(nodes) >= max_nodes:
                break
            if id(ch) not in node_id:
                node_id[id(ch)] = len(nodes)
                depth[id(ch)] = depth[id(n)] + 1
                nodes.append(ch)
                q.append(ch)
            edges.append((node_id[id(n)], node_id[id(ch)], action_summary(d, n.state, ch.action)))

    dot_path = out_prefix + ".dot"
    png_path = out_prefix + ".png"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("digraph MCTS {\n")
        f.write('  rankdir=TB;\n')
        f.write('  node [shape=box, fontsize=10];\n')
        if title:
            f.write(f'  labelloc=t; label="{_safe(title, 200)}";\n')
        for n in nodes:
            nid = node_id[id(n)]
            avg = (n.value / n.visits) if n.visits else 0.0
            untried = "?" if n.untried_actions is None else len(n.untried_actions)
            label = f"#{nid}\\nvis={n.visits}\\navg={avg:.3f}\\ndepth={depth[id(n)]}\\nuntried={untried}"
            f.write(f'  n{nid} [label="{label}"];\n')
        for a, b, el in edges:
            f.write(f'  n{a} -> n{b} [label="{_safe(el, 60)}"];\n')
        f.write("}\n")

    if render_img:
        try:
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True, capture_output=True)
            return dot_path, png_path
        except Exception:
            return dot_path, None
    return dot_path, None
