"""Debugging: draw one OEP search as a Graphviz graph, the OEP counterpart of classical-mcts's
search-tree dump (mctsNode.save_mcts_tree).

OEP has no search tree; what it builds is a genealogy. A move phase's search runs thousands of
generations but evaluates only the distinct phenotypes (most children repeat a genome already
seen), so the graph has one node per distinct genome, as first produced:

  * the root is the phase's state; the random initial population (generation 0) hangs from it;
  * every later genome has an edge from each of its two parents (one if both were the same
    phenotype); dashed edges mark a child that was also mutated;
  * genomes born in the same generation share a rank: a column with rankdir LR (generations left to
    right, genomes stacked vertically) or a row with rankdir TB (generations top to bottom);
  * a node shows its discovery number, generation, fitness, how many genomes of the search had that
    phenotype, and its moves (territory <- units); its colour runs from red (worst fitness of the
    search) to green (best);
  * the played genome is drawn bold, with the edges of its ancestry in blue.
"""

from __future__ import annotations

import subprocess
from collections import Counter
from typing import Dict, List, Optional, Tuple

from oep import GenomeNode

MAX_GENES_SHOWN = 6  # move lines per node; more are summarised as "+n more"


def _safe(s: str, maxlen=60):
    s = s.replace('"', "'").replace("\n", " ")
    return s[:maxlen] + ("…" if len(s) > maxlen else "")


def gene_summary(d, state, action) -> str:
    """'Germany <- 2x infantry, 1x armour' for a territory action of the root state."""
    counts = Counter(d.unit_types[state.units[u].type].name for u in action.units if u in state.units)
    return _safe(f"{d.territory_names[action.territory]} <- "
                 + ", ".join(f"{q}x {t}" for t, q in counts.items()))


def _ancestry(nodes: Dict[Tuple, GenomeNode], key: Optional[Tuple]) -> set:
    seen, stack = set(), [key] if key in nodes else []
    while stack:
        k = stack.pop()
        if k in seen:
            continue
        seen.add(k)
        stack.extend(p for p in nodes[k].parents if p in nodes)
    return seen


def save_oep_tree(nodes: Dict[Tuple, GenomeNode], best_key: Optional[Tuple], d, state,
                  out_prefix: str, title: str = "", max_nodes=300, rankdir="LR", render_img=False):
    """Writes out_prefix + ".dot" (the first max_nodes genomes discovered, plus the played genome's
    ancestry) and, with render_img and Graphviz's `dot` installed, out_prefix + ".png".
    Returns (dot_path, png_path or None)."""
    lineage = _ancestry(nodes, best_key)
    shown = [k for k, n in nodes.items() if n.index < max_nodes or k in lineage]
    shown.sort(key=lambda k: nodes[k].index)
    name = {k: f"g{nodes[k].index}" for k in shown}
    values = [nodes[k].fitness for k in shown]
    lo, hi = (min(values), max(values)) if values else (0.0, 0.0)

    by_generation: Dict[int, List[str]] = {}
    lines = []
    for k in shown:
        n = nodes[k]
        by_generation.setdefault(n.generation, []).append(name[k])
        moves = [gene_summary(d, state, a) for a in n.actions[:MAX_GENES_SHOWN]]
        if len(n.actions) > MAX_GENES_SHOWN:
            moves.append(f"+{len(n.actions) - MAX_GENES_SHOWN} more")
        label = (f"#{n.index}  gen {n.generation}\\nfitness {n.fitness:.3f}  x{n.produced}"
                 + ("  mutated" if n.mutated else "") + "\\n"
                 + "".join(f"{m}\\l" for m in moves or ["(no moves)"]))
        hue = 0.33 * ((n.fitness - lo) / (hi - lo) if hi > lo else 1.0)
        best = ', penwidth=3, color="blue", peripheries=2' if k == best_key else ""
        lines.append(f'  {name[k]} [label="{label}", fillcolor="{hue:.3f} 0.35 1.0"{best}];')
        parents = [p for p in n.parents if p in name] if n.parents else ["root"]
        for p in parents:
            on_lineage = k in lineage and (p == "root" or p in lineage)
            style = ["dashed"] if n.mutated else []
            attrs = f'style="{",".join(style)}"' if style else ""
            if on_lineage:
                attrs += (", " if attrs else "") + 'color="blue", penwidth=2.5'
            src = "root" if p == "root" else name[p]
            lines.append(f"  {src} -> {name[k]}" + (f" [{attrs}]" if attrs else "") + ";")

    dot_path = out_prefix + ".dot"
    png_path = out_prefix + ".png"
    with open(dot_path, "w", encoding="utf-8") as f:
        f.write("digraph OEP {\n")
        f.write(f"  rankdir={rankdir};\n")
        f.write("  node [shape=box, style=filled, fontsize=10];\n")
        if title:
            f.write(f'  labelloc=t; label="{_safe(title, 300)}";\n')
        root_label = f"root\\nround {state.round}\\n{len(nodes)} distinct genomes"
        if len(shown) < len(nodes):
            root_label += f" ({len(shown)} drawn)"
        f.write(f'  root [label="{root_label}", shape=ellipse, fillcolor="white"];\n')
        f.write("\n".join(lines) + "\n")
        for members in by_generation.values():
            f.write(f"  {{rank=same; {'; '.join(members)};}}\n")
        f.write("}\n")

    if render_img:
        try:
            subprocess.run(["dot", "-Tpng", dot_path, "-o", png_path], check=True, capture_output=True)
            return dot_path, png_path
        except Exception:
            return dot_path, None
    return dot_path, None
