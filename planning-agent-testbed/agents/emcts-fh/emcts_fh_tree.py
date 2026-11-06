"""Debugging: draw one fixed-horizon EMCTS search tree as a Graphviz graph (the EMCTS counterpart of
roxo-oep-with-purchase's oep_tree and classical-mcts's tree dump).

A search grows tens of thousands of nodes, so only the most visited are drawn:

  * nodes are genomes: each shows its depth (mutations from the first root), visits, mean backed-up
    value and its own evaluation, coloured red (lowest mean of the drawn nodes) to green
    (highest); the first root (the greedy genome) and the played genome also list their moves
    by segment marker (T<turn> player phase: ..., turn 0 being this turn): a combat segment's
    attacks, and "greedy" for another player's turn;
  * nodes in the first genome_levels levels (depth < genome_levels) list their whole genome
    instead, uncut: every segment in order, each combat gene (skips too);
  * edges are mutations, labelled with the gene changed (T<turn> player phase: territory: old -> new);
    unit names are looked up in the root state, so units bought later in the horizon are not named;
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

from emcts_fh import OTHER_TURN, Node, key
from world_model import TERRITORY

MAX_GENES_SHOWN = 6  # move lines per genome; more are summarised as "+n more"


def _safe(s: str, maxlen=60):
    s = s.replace('"', "'").replace("\n", " ")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def _units(d, state, unit_ids) -> str:
    """'2x infantry, 1x armour'; units not in the root state (bought later) count as 'unit'."""
    counts = Counter(d.unit_types[state.units[u].type].name if u in state.units else "unit"
                     for u in unit_ids)
    return ", ".join(f"{q}x {t}" for t, q in counts.items())


def gene_summary(d, state, action) -> str:
    """'Germany <- 2x infantry, 1x armour' (or 'Germany: skip') for a gene."""
    if action.kind != TERRITORY:
        return f"{d.territory_names[action.territory]}: skip"
    return _safe(f"{d.territory_names[action.territory]} <- {_units(d, state, action.units)}")


def _option(d, state, action) -> str:
    """A gene's option without its territory: 'skip' or '2x infantry, 1x armour'."""
    if action.kind != TERRITORY:
        return "skip"
    return _units(d, state, action.units)



def _marker(seg) -> str:
    return f"T{seg.turn} {seg.player} {seg.phase}"


def _mutation_label(d, state, node: Node) -> str:
    """'T1 Germans combat: TopBar: skip -> 2x infantry' for the gene node's mutation changed."""
    if node.mutation is None:
        return ""
    k, _, old, new = node.mutation
    seg = node.parent.genes[k]
    return _safe(f"{_marker(seg)}: {d.territory_names[new.territory]}: "
                 f"{_option(d, state, old)} -> {_option(d, state, new)}", 80)


def _segment_lines(d, state, seg, full: bool = False) -> List[str]:
    """The segment's moves; with full, every gene uncut (skips too)."""
    if seg.phase == OTHER_TURN:
        return [f"T{seg.turn} {seg.player}: greedy turn"]
    if full:
        return [f"{_marker(seg)}: {d.territory_names[a.territory]}: {_option(d, state, a)}"
                for a in seg.genes] or [f"{_marker(seg)}: (no targets)"]
    return [f"{_marker(seg)}: {gene_summary(d, state, a)}" for a in seg.genes if a.kind == TERRITORY]


def _moves(d, state, node: Node, full: bool = False) -> str:
    genes = [line for seg in node.genes for line in _segment_lines(d, state, seg, full)]
    lines = genes if full else genes[:MAX_GENES_SHOWN]
    if len(genes) > len(lines):
        lines.append(f"+{len(genes) - MAX_GENES_SHOWN} more")
    return "".join(f"{m.replace(chr(34), chr(39))}\\l" for m in lines or ["(no moves)"])


def save_emcts_tree(search, d, state, out_prefix: str, title: str = "", max_nodes=300,
                    max_depth=3, rankdir="LR", render_img=False, genome_levels=5):
    """Writes out_prefix + ".dot" (the burning path plus the most visited nodes at most max_depth
    levels below it, up to max_nodes; nodes with depth < genome_levels list their whole genome)
    and, with render_img and Graphviz's `dot` installed, out_prefix + ".png".
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
        value = search.values.get(key(n.genes), (None,))[0]
        hidden = sum(1 for c in n.children if id(c) not in shown)
        label = (f"depth {n.depth}  visits {n.visits}\\nmean {n.mean():.3f}"
                 + (f"  eval {value:.3f}" if value is not None else "")
                 + (f"\\n+{hidden} children not drawn" if hidden else ""))
        full = n.depth < genome_levels
        if n is root:
            label = "greedy root\\n" + label + "\\n" + _moves(d, state, n, full)
        elif n is played:
            label = "played\\n" + label + "\\n" + _moves(d, state, n, full)
        elif full:
            label += "\\n" + _moves(d, state, n, full)
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
        f.write("digraph EMCTSFH {\n")
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
