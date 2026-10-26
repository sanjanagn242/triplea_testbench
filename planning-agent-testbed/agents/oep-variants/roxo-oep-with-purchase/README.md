# Roxo OEP with greedy purchase

Agent ID `roxo-oep-with-purchase`. [Roxo OEP](../../roxo-oep/README.md) (Online Evolutionary
Planning after Roxo's thesis, Section 4.4), with four changes:

- the **purchase** phase buys units with TripleMind's deterministic greedy policy (the first
  `purchase_actions` option, as the TripleMind and `oep_paper` agents do) instead of buying nothing.
  The **place** phase still answers `{}`, so the engine places the units bought with its built-in
  placement, as for TripleMind;
- the **combat-move genes** use TripleMind's combat action generation instead of the Weak AI's:
  one gene per TripleMind target (`enemy_territories_by_priority`: land next to the player's that it
  does not own, kept if some unit can reach it), and each gene picks one of TripleMind's attack sets
  for that territory (`combat_actions`' buildup: growing prefixes of the attackers, adjacent and
  strongest first, above 1.5x the defence and below 3.5x + 4, the full force if it beats the
  defence, or the cheapest unit for an undefended territory) or skip. Units in another gene are not
  available, as before. Moves are still routed by Roxo's forward model;
- the **playouts** that evaluate a genome move as TripleMind's rollouts do instead of with the Weak
  AI: in a combat move a random one of `combat_actions`' options for each target in turn, in a
  noncombat move `noncombat_move`'s plan. The random choices are seeded the same way for every
  genome, so a genome's value stays fixed and it is still evaluated once. Battles stay Roxo's
  deterministic Fast AI battles, and purchase and place phases in the playout still buy nothing;
- the playout's end state is rated with TripleMind's **`evaluate_state`** instead of Roxo's reward:
  a win scores 0.5 + 0.5 x the share of the playout's rounds left (a loss the negative); otherwise
  0.5 x my share of the units' cost + 0.5 x my share of the owned territories' production, minus
  0.5 (so in [-0.5, 0.5]).

Non-combat genes are unchanged (every combination of the unit groups that can reach a territory, as
Roxo's forward model makes them): TripleMind has no per-territory non-combat options, only one
fixed plan. Everything else is Roxo OEP unchanged: one evolution per move phase over
territory-based genomes, playouts of up to 10 rounds, and the same
settings (`ROXO_SECONDS`, `ROXO_OEP_POPULATION`, `ROXO_OEP_SURVIVAL`, `ROXO_OEP_MUTATION`,
`ROXO_OEP_PLAYOUT_ROUNDS`, `ROXO_SEED`) and metric files. See its README for the algorithm.

## What it is for

Roxo's setting has no production, so Roxo OEP never buys. On maps with production (e.g.
Capture The Flag) that leaves it behind any agent that buys; this variant removes that handicap so
the search itself can be compared with TripleMind and the `oep_paper` agents.

The search does not model production: Roxo's forward model ends purchase and place phases without
buying, so the playouts assume neither side buys. Only the live game buys.

## Files

A self-contained copy of `agents/roxo-oep` (`agent.py`, `oep.py`, `world_model.py`, `weak_ai.py`,
`fast_battle.py`, `land_graph.py`, `run-agent.sh`) plus copies of TripleMind's `helper.py`,
`map_graph.py`, `action_generators.py` and `generators.py` (pure Python) for the metrics and the
purchase policy. Differences from roxo-oep's files:

| File | Change |
|---|---|
| `agent.py` | `purchase()` plays TripleMind's greedy purchase; builds TripleMind's `MapGraph` at game start; `helper.py` is the local copy; log folder one level up (this agent is one folder deeper) |
| `action_generators.py` | `combat_actions` split into `target_attackers` and `attack_sets` (same results) so the genes can use them |
| `run-agent.sh` | forward-model path one level up; `TRIPLEMIND_PURE_PYTHON=1`; genealogy settings |
| `oep.py` | combat genes from TripleMind's generator (`attack_sets`, cached per set of free units); playouts with TripleMind's rollout policy (`playout_moves`) rated by its `evaluate_state`; with `trace`, records the genealogy of the distinct genomes evaluated (`tree`, `best_key`) |
| `oep_tree.py` | new: draws that genealogy as Graphviz `.dot`/`.png` |

## Visualising a search

OEP's counterpart of classical-mcts's tree dump. With `ROXO_OEP_SAVE_TREE=1`, every move phase's
search is drawn to `planning-agent-testbed/logs/trees/oep-<player>-game-<n>-round-<r>-<phase>.dot`
(and `.png` with Graphviz's `dot` installed) once its time budget is used. A search runs thousands of
generations but evaluates only a few distinct genomes (the rest are duplicates), so the graph has one
node per distinct genome, as first produced: the random initial population hangs from the root, each
later genome has an edge from each parent (dashed: also mutated), and genomes born in the same
generation share a column (`LR`) or row (`TB`). A node shows its generation, fitness, how many genomes
of the search had that phenotype (`xN`) and its moves, coloured red (worst) to green (best); the
played genome is drawn bold with its ancestry in blue.

| Setting | Default | Meaning |
|---|---|---|
| `ROXO_OEP_SAVE_TREE` | 0 | 1 draws every search (tracing costs a little search time) |
| `ROXO_OEP_TREE_PNG` | 1 | also render a `.png` (needs `dot`) |
| `ROXO_OEP_TREE_MAX_NODES` | 300 | genomes drawn (the first discovered, plus the played genome's ancestry) |
| `ROXO_OEP_TREE_RANKDIR` | LR | `LR`: generations left to right; `TB`: top to bottom |

## Testing note

The offline arena (`tools/arena.py`) does not place units for an agent that answers `{}` to the
place phase, so purchases are lost there (as for TripleMind). Compare it in the engine, e.g. with
`tools/tournament.py`, where the engine places them.
