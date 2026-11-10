"""Debugging: draw one TBETS search tree in TripleMind's style (tripleMind/helper.py
save_mcts_tree_png and save_mcts_tree_html).

The root is the current state; its children are genomes of the current phase, and every node's
children are genomes of the next searched phase in the game (purchase -> combat move -> noncombat
move -> the next player's purchase ...).

  * .dot / .png: the root and the max_nodes most visited nodes (best first by visits, a node only
    after its parent), plus the played path (the most visited child at each level), as boxes
    "#n <player> <phase> vis= avg=" (n: creation order, the greedy genome being #1; avg: the mean from
    the node's player's point of view) listing the genome's genes in play order, each with its phase first
    ("P: buyArmour", "C: TopBar | 2x infantry", "N: Germany | skip"). Edges say how the genome
    was made: "greedy" (a child: the next phase's greedy genome) or "mutation of #m: gene i: old ->
    new" (a sibling of #m). The played path is bold; a node notes how many children are not drawn;
  * .html: the whole tree as a collapsible outline (no node limit), children sorted by visits,
    the played path opened, each node's genes listed.

Unit names are looked up in the root state, so units bought this turn are named "unit".
"""

from __future__ import annotations

import heapq
import html
import subprocess
from collections import Counter
from typing import List

from tbets import COMBAT, PURCHASE, Node
from world_model import TERRITORY

PHASE_TAG = {PURCHASE: "P", COMBAT: "C"}  # noncombat: "N"


def _safe(s: str, maxlen=60):
    s = s.replace('"', "'").replace("\n", " ")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def _units(d, state, unit_ids) -> str:
    """'2x infantry, 1x armour'; units not in the root state (bought later) count as 'unit'."""
    counts = Counter(d.unit_types[state.units[u].type].name if u in state.units else "unit"
                     for u in unit_ids)
    return ", ".join(f"{q}x {t}" for t, q in counts.items())


def _value(d, state, phase: str, value) -> str:
    """'buyArmour', 'TopBar | 2x infantry' or 'Germany | skip'."""
    if phase == PURCHASE:
        rule = next((r for r in d.production_rules if r.id == value), None)
        return rule.name if rule else str(value)
    terr = d.territory_names[value.territory]
    return f"{terr} | skip" if value.kind != TERRITORY else f"{terr} | {_units(d, state, value.units)}"


def gene_label(d, state, gene) -> str:
    """'P: buyArmour', 'C: TopBar | 2x infantry', 'N: Germany | skip'."""
    return _safe(f"{PHASE_TAG.get(gene.phase, 'N')}: {_value(d, state, gene.phase, gene.value)}")


def origin_label(d, state, n: Node) -> str:
    """How the genome was made: 'greedy' or 'mutation of #m: gene i: old -> new'."""
    if n.origin != "mutation":
        return n.origin
    i, old, new = n.mutation
    phase = n.source.genome[i].phase
    return _safe(f"mutation of #{n.source.number}: gene {i}: {_value(d, state, phase, old)} -> "
                 f"{_value(d, state, phase, new)}", 90)


_TREE_HTML_HEAD = """<!DOCTYPE html>
<html><head><meta charset="utf-8"><title>MCTS tree</title><style>
body { font: 13px monospace; margin: 1em; }
details { margin-left: 1.4em; border-left: 1px dotted #bbb; padding-left: 4px; }
summary { cursor: pointer; white-space: nowrap; }
.node { padding: 0 4px; border-radius: 3px; background: lightblue; }
.best > summary { font-weight: bold; }
.act { color: #555; } .unexp { margin-left: 2.8em; color: gray; }
#bar { position: sticky; top: 0; background: white; padding: 4px 0; border-bottom: 1px solid #ccc; }
</style></head><body>
<div id="bar">
<button onclick="setAll(true)">Expand all</button>
<button onclick="setAll(false)">Collapse all</button>
<button onclick="bestPath()">Best path only</button>
min visits <input id="minv" type="number" value="0" min="0" style="width:5em" oninput="filter()">
&nbsp; bold = most-visited child; gray = actions not played (never expanded)
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


def played_path(root: Node) -> List[Node]:
    """The most visited child at each level, from the root (the root's is the played genome)."""
    path, n = [root], root
    while n.children:
        n = max(n.children, key=lambda c: (c.visits, c.mean()))
        path.append(n)
    return path


def _header(n: Node) -> str:
    return f"#{n.number} {n.player} {n.phase} vis={n.visits} avg={n.mean():.3f}"


def save_tbets_tree_html(search, d, state, html_path: str):
    """Writes the whole tree as a collapsible HTML outline: children sorted by visits, the played
    path opened, each node's genes listed."""
    on_path = {id(n) for n in played_path(search.root)}
    out = [_TREE_HTML_HEAD]

    def write(n: Node, label: str):
        best = id(n) in on_path
        summary = (f"#0 (root) vis={n.visits} avg={n.mean():.3f} children={len(n.children)}"
                   if n.parent is None else _header(n) + f" genes={len(n.genome)} children={len(n.children)}")
        out.append(f'<details data-v="{n.visits}" class="{"best" if best else ""}"'
                   f'{" open" if best else ""}><summary>'
                   f'<span class="act">{html.escape(label)}</span> '
                   f'<span class="node">{html.escape(summary)}</span></summary>\n')
        for g in n.genome:
            out.append(f'<div class="unexp">{html.escape(gene_label(d, state, g))}</div>\n')
        for c in sorted(n.children, key=lambda c: (c.visits, c.mean()), reverse=True):
            write(c, origin_label(d, state, c))
        out.append("</details>\n")

    write(search.root, "(root)")
    out.append("</div></body></html>\n")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("".join(out))
    return html_path


def save_tbets_tree(search, d, state, out_prefix: str, title: str = "", max_nodes=30,
                    rankdir="LR", render_img=False):
    """Saves out_prefix + ".dot" (the root, the played path and the max_nodes most visited nodes),
    out_prefix + ".html" (the whole tree, collapsible) and, with render_img and Graphviz's `dot`
    installed, out_prefix + ".png". Returns (dot_path, png_path or None)."""
    root: Node = search.root
    path = played_path(root)
    on_path = {id(n) for n in path}
    shown = {id(n): n for n in path}
    order = 0
    heap = [(-root.visits, order, root)]
    while heap and len(shown) < max(max_nodes + 1, len(path)):
        _, _, n = heapq.heappop(heap)
        shown[id(n)] = n
        for c in n.children:
            order += 1
            heapq.heappush(heap, (-c.visits, order, c))
    for n in list(shown.values()):  # every drawn node's most visited line, so later phases show
        while n.children:
            n = max(n.children, key=lambda c: (c.visits, c.mean()))
            shown[id(n)] = n
    nodes = sorted(shown.values(), key=lambda n: (n.depth, n.number))

    dot_path = out_prefix + ".dot"
    png_path = out_prefix + ".png"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("digraph MCTS {\n")
        f.write(f"  rankdir={rankdir};\n")
        f.write("  node [shape=box, fontsize=10];\n")
        if title:
            f.write(f'  labelloc=t; label="{_safe(title, 300)}";\n')
        for n in nodes:
            hidden = sum(1 for c in n.children if id(c) not in shown)
            more = f"\\n+{hidden} children not drawn" if hidden else ""
            bold = ", penwidth=3" if id(n) in on_path else ""
            if n.parent is None:
                f.write(f'  n0 [label="#0 (root)\\nvis={n.visits}\\navg={n.mean():.3f}{more}"{bold}];\n')
                continue
            genes = "".join(f"{gene_label(d, state, g)}\\l" for g in n.genome) or "(no genes)\\l"
            label = f"#{n.number} {n.player} {n.phase}\\nvis={n.visits}\\navg={n.mean():.3f}{more}\\n{genes}"
            f.write(f'  n{n.number} [label="{label}"{bold}];\n')
            f.write(f'  n{n.parent.number} -> n{n.number} [label="{origin_label(d, state, n)}"{bold}];\n')
        f.write("}\n")

    save_tbets_tree_html(search, d, state, out_prefix + ".html")

    if render_img:
        try:
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True, capture_output=True)
            return dot_path, png_path
        except Exception:
            return dot_path, None
    return dot_path, None
