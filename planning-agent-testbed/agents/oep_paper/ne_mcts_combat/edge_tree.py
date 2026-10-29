"""Debugging: draw one combat-move search of the edge-based MCTS agents (NE-MCTS, BB-MCTS and their
all-attacks variants) as a Graphviz graph and a collapsible HTML outline, the counterpart of
TripleMind's tree dump (helper.save_mcts_tree_png), which expects TripleMind's node objects.

These searches keep visits and values on edges, and the transposition table shares a plan's node
between every order of the same attacks, so the search is a DAG:

  * one box per node (plan), drawn once, with an incoming edge from every parent that reached it;
    it shows the node's number, its attacks so far and the visits through it;
  * one arrow per edge, labelled with the action (target, strength and the units sent), its visits
    N and mean value Q, coloured from red (worst Q of the search) to green (best); an END edge
    ends in a small terminal oval;
  * actions a node generated but never expanded are grey dashed stubs;
  * the chosen sequence is drawn bold blue;
  * BB-MCTS: at the node where a phase committed its action, the edges not committed (burnt, never
    searched again) are dashed red, and the node is marked "phase k".

The HTML outline lists the same DAG as nested <details> (children by visits, the chosen path open);
a node already listed is referred to by its number instead of being repeated.
"""

from __future__ import annotations

import html
import subprocess
from collections import Counter, deque
from typing import Dict, List, Sequence

MAX_STUBS = 12  # unexpanded actions drawn per node; more are summarised


def _safe(s: str, maxlen=80):
    s = s.replace('"', "'").replace("\n", " ")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def action_label(d, sim, action) -> str:
    """'END' or 'Germany s=5: 2x infantry, 1x armour' (unit types read from the parent's state)."""
    if action.end_phase:
        return "END"
    counts = Counter(d.unit_types[sim.units[u].type].name
                     for mv in action.moves for u in mv.units if u in sim.units)
    units = ", ".join(f"{q}x {t}" for t, q in counts.items())
    return f"{action.to_terr} s={action.strength}: {units}"


def _walk(root, max_nodes: int, keep: set) -> List:
    """Nodes reachable from root, breadth first, at most max_nodes plus those in `keep`."""
    order, seen, queue = [], {id(root)}, deque([root])
    while queue:
        node = queue.popleft()
        if len(order) < max_nodes or id(node) in keep:
            order.append(node)
        for e in node.edges:
            c = e.child
            if c is not None and id(c) not in seen:
                seen.add(id(c))
                queue.append(c)
    return order


def save_edge_tree(root, d, out_prefix: str, chosen: Sequence = (), phases: Sequence = (),
                   title: str = "", max_nodes: int = 300, rankdir: str = "TB",
                   render_img: bool = False):
    """Writes out_prefix + ".dot" and ".html" (and ".png" with render_img and Graphviz's `dot`).
    `chosen` is the chosen sequence's edges, root first; `phases` the nodes where BB-MCTS
    committed an action, in phase order. Returns (dot_path, png_path or None, html_path)."""
    chosen_ids = {id(e) for e in chosen}
    path_nodes = {id(root)} | {id(e.child) for e in chosen if e.child is not None}
    phase_of = {id(n): k for k, n in enumerate(phases, 1)}
    nodes = _walk(root, max_nodes, path_nodes)
    num = {id(n): i for i, n in enumerate(nodes)}
    edges = [e for n in nodes for e in n.edges if e.visits]
    means = [e.mean for e in edges]
    lo, hi = (min(means), max(means)) if means else (0.0, 0.0)

    def hue(q):
        return 0.33 * ((q - lo) / (hi - lo) if hi > lo else 1.0)

    lines = []
    for n in nodes:
        i = num[id(n)]
        visits = sum(e.visits for e in n.edges)
        label = f"#{i}  depth {n.depth}\\nN={visits}"
        if id(n) in phase_of:
            label += f"\\nphase {phase_of[id(n)]}"
        if n.attacked:
            label += "\\n" + _safe(", ".join(sorted(n.attacked)), 60)
        style = ', penwidth=3, color="blue"' if id(n) in path_nodes else ""
        lines.append(f'  n{i} [label="{label}"{style}];')
        for k, e in enumerate(n.edges):
            text = f"{_safe(action_label(d, n.sim, e.action), 50)}\\nN={e.visits} Q={e.mean:.3f}"
            if id(e) in chosen_ids:
                attrs = 'color="blue", penwidth=3, fontcolor="blue"'
            elif id(n) in phase_of:
                attrs = 'color="red", style=dashed'  # burnt by the phase's commitment
            else:
                attrs = f'color="{hue(e.mean):.3f} 0.9 0.75"' if e.visits else 'color="gray"'
            if e.child is None:
                lines.append(f'  e{i}_{k} [label="END\\nQ={e.mean:.3f}", shape=oval, fontsize=8];')
                lines.append(f'  n{i} -> e{i}_{k} [label="{text}", {attrs}];')
            elif id(e.child) in num:
                lines.append(f'  n{i} -> n{num[id(e.child)]} [label="{text}", {attrs}];')
        untried = list(n.actions[len(n.edges):]) if n.actions is not None else []
        for k, a in enumerate(untried[:MAX_STUBS]):
            lines.append(f'  u{i}_{k} [shape=point, width=0.05];')
            lines.append(f'  n{i} -> u{i}_{k} [label="{_safe(action_label(d, n.sim, a), 40)}", '
                         f'arrowhead=none, style=dashed, color=gray, fontcolor=gray];')
        if len(untried) > MAX_STUBS:
            lines.append(f'  u{i}_more [label="+{len(untried) - MAX_STUBS} untried", shape=plaintext, '
                         f'fontcolor=gray, fontsize=8];')
            lines.append(f'  n{i} -> u{i}_more [arrowhead=none, style=dashed, color=gray];')

    dot_path, png_path = out_prefix + ".dot", out_prefix + ".png"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("digraph MCTS {\n")
        f.write(f"  rankdir={rankdir};\n")
        f.write("  node [shape=box, fontsize=10]; edge [fontsize=8];\n")
        if title:
            f.write(f'  labelloc=t; label="{_safe(title, 300)}";\n')
        f.write("\n".join(lines) + "\n}\n")
    html_path = _save_html(root, d, out_prefix + ".html", num, chosen_ids, phase_of, title)

    if render_img:
        try:
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True, capture_output=True)
            return dot_path, png_path, html_path
        except Exception:
            return dot_path, None, html_path
    return dot_path, None, html_path


_HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MCTS tree</title><style>
body { font: 13px monospace; margin: 1em; }
details { margin-left: 1.4em; border-left: 1px dotted #bbb; padding-left: 4px; }
summary { cursor: pointer; white-space: nowrap; }
.node { padding: 0 4px; border-radius: 3px; background: lightblue; }
.chosen > summary { font-weight: bold; color: blue; }
.burnt > summary { color: #b00; }
.act { color: #555; } .muted { margin-left: 2.8em; color: gray; }
#bar { position: sticky; top: 0; background: white; padding: 4px 0; border-bottom: 1px solid #ccc; }
</style></head><body>
<div id="bar">
<button onclick="setAll(true)">Expand all</button>
<button onclick="setAll(false)">Collapse all</button>
&nbsp; bold blue = chosen sequence; red = burnt by a BB-MCTS phase; gray = never expanded;
"see #k" = a node shared with another order of the same attacks
</div>
<script>function setAll(o) { document.querySelectorAll('details').forEach(d => d.open = o); }</script>
"""


def _save_html(root, d, path: str, num: Dict[int, int], chosen_ids: set, phase_of: Dict[int, int],
               title: str) -> str:
    out = [_HTML_HEAD, f"<h3>{html.escape(title)}</h3>\n" if title else ""]
    listed = set()

    def node_summary(n):
        i = num.get(id(n), "?")
        phase = f" phase {phase_of[id(n)]}" if id(n) in phase_of else ""
        visits = sum(e.visits for e in n.edges)
        return f'<span class="node">#{i} depth {n.depth} N={visits}{phase}</span>'

    def write(n, open_):
        listed.add(id(n))
        for e in sorted(n.edges, key=lambda e: e.visits, reverse=True):
            label = html.escape(action_label(d, n.sim, e.action))
            stats = f"N={e.visits} Q={e.mean:.3f}"
            cls = "chosen" if id(e) in chosen_ids else ("burnt" if id(n) in phase_of else "")
            c = e.child
            if c is None:
                out.append(f'<div class="{cls or "muted"}" style="margin-left:2.8em">'
                           f'<span class="act">{label}</span> {stats} (end of phase)</div>\n')
                continue
            if id(c) in listed:
                out.append(f'<div class="muted"><span class="act">{label}</span> {stats} → see '
                           f'#{num.get(id(c), "?")}</div>\n')
                continue
            is_open = open_ and id(e) in chosen_ids
            out.append(f'<details class="{cls}"{" open" if is_open else ""}><summary>'
                       f'<span class="act">{label}</span> {stats} {node_summary(c)}</summary>\n')
            write(c, is_open)
            out.append("</details>\n")
        for a in (list(n.actions[len(n.edges):]) if n.actions is not None else []):
            out.append(f'<div class="muted">- {html.escape(action_label(d, n.sim, a))} (never expanded)</div>\n')

    out.append(f"<details open><summary>(root) {node_summary(root)}</summary>\n")
    write(root, True)
    out.append("</details></body></html>\n")
    with open(path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    return path
