# tbets-seg-hybrid

[`tbets-segmented`](../../tbets-segmented/README.md)'s search (phase genomes, greedy genomes,
one-gene mutations with repair) stored in [`heterogeneous-mcts`](../heterogeneous-mcts/README.md)'s
tree of atomic actions. The search is `tbets_seg_hybrid.TBETSSegHybrid`, a subclass of
`hmcts.HeterogeneousMCTS`.

## How it differs from its parents

| | tbets-segmented | heterogeneous-mcts | tbets-seg-hybrid |
|---|---|---|---|
| tree node | one whole phase genome | one decision | one decision |
| tree edge | parent → genome | one atomic action | one atomic action (a gene) |
| how a phase's choices are made | greedy genome, then mutations of whole genomes | every action enumerated, one per visit | greedy genome, then mutations of whole genomes |
| statistics | per genome | per action (edge) | per gene (edge): genomes that share their first genes share those edges |
| UCB1 | once per phase, among genomes | at every decision | at every decision |
| answer | the most visited genome | the most visited edge per decision | the most visited edge per decision |

So it asks whether scoring tbets-segmented's genomes gene by gene (credit shared by every genome
through an edge) helps or hurts compared with scoring each genome on its own.

## The search (`tbets_seg_hybrid.py`)

- **Gene**: an edge's action, compared across states by its key: a purchase by its production
  rule; a combat or non-combat move by its territory and the units it moves (a skip moves none).
- **Genome**: the path of edges of one phase of one player's turn, from the node the phase starts
  at (its *phase start*) to the node its last edge leads to (its *phase end*, at the next searched
  decision). The tree grows only by whole genomes, so every leaf is a phase end.
- **Decoding (repair)**, from a phase start: each decision takes its stored gene if it is still
  offered, else its strongest action. A purchase buys the stored units in order, dropping those no
  longer offered, then the strongest (most attack + defense per PU) until nothing is affordable; a
  move phase matches stored genes by territory.
- **Greedy genome** (Greedy Action AI): the genome decoded from nothing (every gene strongest),
  then, for each gene in order, every other option of that gene is tried (the genome decoded again
  with that gene changed, played off the tree and rolled out); the best for the phase's player is
  kept. Stops at the deadline (the genome decoded from nothing is always tried).
- **Mutation**: the genome with one gene changed to another option of its decision (not tried from
  that genome yet, drawn at random), decoded again from its phase start. A mutation that decodes to
  a genome already in the tree is a **duplicate**: the next mutation is tried instead; if all are
  duplicates, nothing is added.
- **Adding a genome**: decoded along the tree from its phase start, following the edges already
  there and making the rest; simulated once at its phase end and backpropagated to the root.
- **Selection**: the best UCB1 edge from the root down to a leaf, gene by gene. With
  `TBETS_WIDEN=1`, every phase end with children that selection passes through first gets one
  sibling genome: a mutation of its genome.
- **Expansion at the leaf**: with probability `TBETS_GREEDY_PCT` % (or when the leaf's genome has no
  mutation left) a child: the greedy genome of the next phase. Otherwise a mutation of the leaf's
  genome (a sibling). If every mutation left is a duplicate, the greedy genome; a leaf with no
  decision after it and no new mutation is simulated again.
- **Simulation, backpropagation, rewards, deterministic battles**: heterogeneous-mcts's (below).
- **Answer**: the most visited edge at each decision from the root to the end of the phase; where
  that path stops short of the end of a combat or non-combat move, the rest takes the strongest
  actions.

Duplicates are common (often more than new genomes): a one-gene change frequently repairs back to
a genome already in the tree, e.g. a purchase unit that drops the later stored units.

## The tree (from heterogeneous-mcts)

- **One tree across phases and players**: when a phase has nothing left to decide, the next node
  is the next searched decision, whoever's it is (this player's next phase or an opponent's).
  Battles, place and end of turn in between are played by the rollout policies.
- **One reward**: every node is valued by `evaluate_state` (TUV share and income share, or the
  win/loss value), kept from the view of the player who chose its edge.
- **Deterministic battles**: every battle the search fights is resolved by the Fast AI's most
  likely outcome (`fast_battle.py`). The real game's battles are fought by the engine with dice.
- **Rollout horizon** (`TRIPLEA_MCTS_ROLLOUT`) is in rounds: 1 ends a rollout when the round the
  leaf is in is over.
- Purchase units are offered in candidate order from the last unit bought, so each set of units
  has one path in the tree.

## Settings (`run-agent.sh`)

`TBETS_GREEDY_PCT` (default 5) and `TBETS_WIDEN` (default 1) are this agent's; the `TRIPLEA_MCTS_*`
ones are heterogeneous-mcts's. Metrics and trees keep heterogeneous-mcts's names, split by the
root's phase (`efficiency.csv` / `_noncombat` / `_purchase`, `tree_*` / `tree_noncombat_*` /
`tree_purchase_*`); `num_action_sequences` counts the tree's leaves, i.e. the distinct genomes
placed in the tree.

## Docs

The `.drawio` files in `docs/` are heterogeneous-mcts's and show its expansion, not this search.
