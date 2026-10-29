"""Debugging: draw one EMCTS search tree as a Graphviz graph (the EMCTS counterpart of
roxo-oep-with-purchase's oep_tree and classical-mcts's tree dump).

A search grows tens of thousands of nodes, so only the most visited are drawn:

  * nodes are genomes: each shows its depth (mutations from the first root), visits, mean backed-up
    value and its own evaluation, coloured red (lowest mean of the drawn nodes) to green
    (highest); the first root (the greedy genome) and the played genome also list their moves
    (territory <- units);
  * edges are mutations, labelled with the gene changed (territory: old -> new);
  * the bridge-burning path (the root after each phase, from the greedy genome to the played one)
    is drawn in blue, the played genome bold;
  * the drawn nodes are the path plus the most visited nodes reachable from the first root (best
    first by visits) at most max_depth levels below the path, up to max_nodes. With C = 0 a small
    search dives hundreds of mutations deep, mostly revisiting a few genomes; the depth limit keeps
    the drawing to the choices near the path. A node notes how many of its children were left out.
"""

from __future__ import annotations

import heapq
import subprocess
from collections import Counter
from typing import List

from emcts import Node, key
from world_model import TERRITORY

MAX_GENES_SHOWN = 6  # move lines per genome; more are summarised as "+n more"


def _safe(s: str, maxlen=60):
    s = s.replace('"', "'").replace("\n", " ")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def gene_summary(d, state, action) -> str:
    """'Germany <- 2x infantry, 1x armour' (or 'Germany: skip') for a gene of the root state."""
    if action.kind != TERRITORY:
        return f"{d.territory_names[action.territory]}: skip"
    counts = Counter(d.unit_types[state.units[u].type].name for u in action.units if u in state.units)
    return _safe(f"{d.territory_names[action.territory]} <- "
                 + ", ".join(f"{q}x {t}" for t, q in counts.items()))


def _option(d, state, action) -> str:
    """A gene's option without its territory: 'skip' or '2x infantry, 1x armour'."""
    if action.kind != TERRITORY:
        return "skip"
    counts = Counter(d.unit_types[state.units[u].type].name for u in action.units if u in state.units)
    return ", ".join(f"{q}x {t}" for t, q in counts.items())


def _mutation_label(d, state, node: Node) -> str:
    """'TopBar: skip -> 2x infantry' for the gene in which node differs from its parent."""
    i = next((j for j, (a, b) in enumerate(zip(node.parent.genes, node.genes)) if a != b), None)
    if i is None:
        return ""
    old, new = node.parent.genes[i], node.genes[i]
    return _safe(f"{d.territory_names[new.territory]}: {_option(d, state, old)} -> "
                 f"{_option(d, state, new)}", 70)


def _moves(d, state, node: Node) -> str:
    genes = [a for a in node.genes if a.kind == TERRITORY]
    lines = [gene_summary(d, state, a) for a in genes[:MAX_GENES_SHOWN]]
    if len(genes) > MAX_GENES_SHOWN:
        lines.append(f"+{len(genes) - MAX_GENES_SHOWN} more")
    return "".join(f"{m}\\l" for m in lines or ["(no moves)"])


def save_emcts_tree(search, d, state, out_prefix: str, title: str = "", max_nodes=300,
                    max_depth=3, rankdir="LR", render_img=False):
    """Writes out_prefix + ".dot" (the burning path plus the most visited nodes at most max_depth
    levels below it, up to max_nodes) and, with render_img and Graphviz's `dot` installed, out_prefix + ".png".
    Returns (dot_path, png_path or None)."""
    root: Node = search.first_root
    path: List[Node] = search.path
    played = path[-1]

    on_path = {id(n) for n in path}
    shown = {id(n): n for n in path}
    order = 0
    heap = [(-root.visits, order, root, 0)]  # (visits, tie-break, node, levels below the path)
    while heap and len(shown) < max(max_nodes, len(path)):
        _, _, n, below = heapq.heappop(heap)
        shown[id(n)] = n
        for c in n.children:
            c_below = 0 if id(c) in on_path else below + 1
            if c_below <= max_depth:
                order += 1
                heapq.heappush(heap, (-c.visits, order, c, c_below))
    nodes = list(shown.values())
    name = {id(n): f"n{i}" for i, n in enumerate(nodes)}
    means = [n.mean() for n in nodes]
    lo, hi = (min(means), max(means)) if means else (0.0, 0.0)

    lines = []
    for n in nodes:
        value = search.values.get(key(n.genes))
        hidden = sum(1 for c in n.children if id(c) not in shown)
        label = (f"depth {n.depth}  visits {n.visits}\\nmean {n.mean():.3f}"
                 + (f"  eval {value:.3f}" if value is not None else "")
                 + (f"\\n+{hidden} children not drawn" if hidden else ""))
        if n is root:
            label = "greedy root\\n" + label + "\\n" + _moves(d, state, n)
        elif n is played:
            label = "played\\n" + label + "\\n" + _moves(d, state, n)
        hue = 0.33 * ((n.mean() - lo) / (hi - lo) if hi > lo else 1.0)
        attrs = f'label="{label}", fillcolor="{hue:.3f} 0.35 1.0"'
        if n is played:
            attrs += ', penwidth=3, color="blue", peripheries=2'
        elif id(n) in on_path:
            attrs += ', penwidth=2, color="blue"'
        lines.append(f"  {name[id(n)]} [{attrs}];")
        if n.parent is not None and id(n.parent) in name:
            edge = f'label="{_mutation_label(d, state, n)}", fontsize=8'
            if id(n) in on_path and id(n.parent) in on_path:
                edge += ', color="blue", fontcolor="blue", penwidth=2.5'
            lines.append(f"  {name[id(n.parent)]} -> {name[id(n)]} [{edge}];")

    dot_path = out_prefix + ".dot"
    png_path = out_prefix + ".png"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("digraph EMCTS {\n")
        f.write(f"  rankdir={rankdir};\n")
        f.write("  node [shape=box, style=filled, fontsize=10];\n")
        if title:
            f.write(f'  labelloc=t; label="{_safe(title, 300)}";\n')
        f.write("\n".join(lines) + "\n")
        f.write("}\n")

    if render_img:
        try:
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True, capture_output=True)
            return dot_path, png_path
        except Exception:
            return dot_path, None
    return dot_path, None
