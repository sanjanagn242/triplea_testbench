# TBETS (complete turn)

Agent ID `tbets-complete-turn`. Started as a copy of [`emcts-segmented`](../emcts-segmented/README.md)
(itself a variant of [`emcts-fh`](../emcts-fh/README.md)). It searches with MCTS over **turn genomes**: the
root is the current state, its children are genomes of the player's remaining turn, and every node's
children are genomes of the next player's turn. Genomes are made greedily or by mutation as in
Evolutionary MCTS, with
[`heterogeneous-mcts`](../mcts-variants/heterogeneous-mcts/README.md)'s atomic actions as genes.

| Phase | Policy |
|---|---|
| purchase | TBETS (`tbets.py`): plays the best genome's purchase genes |
| combat move | TBETS: plays the best genome's combat genes |
| battle | `fightAll` (TripleA rolls the dice) |
| noncombat move | TBETS: plays the best genome's noncombat genes |
| place | `{}`: the engine places the units bought |

Each phase is searched again, so later phases are planned from the real state after the dice.

## The genome

A genome is **one flat sequence of genes for one player's turn**, not split by phase: the searching
player's remaining turn (from the current phase to the end of the turn) at the root's children, or
a later player's whole turn deeper in the tree. Each gene carries the phase it plays in
(`Gene.phase`, `Gene.value`), and the genes are in play order: purchase, then combat, then
noncombat. A phase with nothing to decide has no genes.

| Phase | Genes | Mutation: change one gene to | Decoding (repair) |
|---|---|---|---|
| purchase | one per unit bought: a production rule id | another candidate unit (land combat units bought with PUs, as `HeterogeneousMCTS.unit_rules`) | buy the stored units in order, dropping those no longer affordable, then spend the PUs left on TripleMind's purchase (the first `purchase_actions` option) |
| combat | one per TripleMind target: an attack set or skip, as in `emcts-fh` | another attack set from the units the other combat genes leave free, or skip | `emcts-fh`'s: an unusable stored gene takes the strongest attack set |
| noncombat | one per territory short of cover, most important first (`noncombat_generators.noncombat_actions`): a reinforcement set or skip | another reinforcement set offered for that territory, or skip | each territory offered takes its stored option if that is still offered, else the strongest reinforcement |

The noncombat move also plays heterogeneous-mcts's forced moves, which are not genes: before the
genes, the air landings (`land_air_moves`); after them, the positioning moves. Between phases,
battles use the deterministic Fast AI estimate (`fast_battle`), default casualties and no retreat,
and placement puts everything bought at the home factory.

## The search

The root is the current state. Its children are genomes of the searching player's remaining turn;
a node's children are genomes of the **next turn in the game** (the next player's), played from the
state its path reaches; and so on. With two players, the levels alternate: my turn, the opponent's,
mine, ...

- **Start**: the root's first child is the greedy genome: the Greedy Action AI, which starts from the
  genome decoded from nothing and, for each gene in play order, keeps the option whose genome
  evaluates best for the turn's player.
- **Selection**: from the root, the best UCB1 child down to a leaf (plain UCT: every level
  maximises, since each node's mean is from its turn's player's point of view).
- **Expansion**, on the way down: every node with children that selection passes through gets
  **one new sibling** of its children per visit: a mutation of its UCB1-best child that still has
  a mutation left, simulated and backpropagated before selection goes on. One iteration therefore
  adds a node at each level of its path.
- **Expansion**, at the leaf: with probability `TBETS_GREEDY_PCT` % (or when the leaf has no mutation
  left), a **child**: the next turn's greedy genome (the Greedy Action AI for the next player).
  Otherwise one gene of the leaf is mutated to another of its options, which makes a **sibling** of
  it. A genome already among the siblings (after repair) is visited again instead of added; a leaf
  after which the game ends, with no mutation left, is evaluated again.
- **Simulation**: the path's genomes are decoded turn by turn, then every player plays TripleMind's
  greedy policies until `TBETS_ROLLOUT` rounds have started after the round of the last genome's
  turn. The state is scored with TripleMind's `evaluate_state` for the searching player (a win 1, a
  loss -1). Evaluations are deterministic and cached per path.
- **Backpropagation**: the value (the searching player's `evaluate_state`) is added to every
  node of the path, up to the root, negated at the nodes of players not allied with the searching
  player.
- **Answer**: the root's most visited child (best mean on a tie); its genes of the current phase
  are played.

A greedy child runs the Greedy Action AI over a whole turn, so a search
makes far fewer iterations than with genomes of the current turn only (tens per second rather
than hundreds).

Live, a purchase plays the units its genes bought as one purchase. A noncombat move plays the air
landings, the chosen reinforcements and the positioning moves. A combat move plays the attacks,
routed as in `emcts-fh`. If the forward model rejects the state, or the genome does not reach this
phase, purchase and noncombat fall back to TripleMind's greedy policy, and the combat move plays
no moves (as in `emcts-fh`).

## Settings (`run-agent.sh`)

| Setting | Default | Meaning |
|---|---|---|
| `TBETS_SECONDS` | 1.0 | search time per searched phase (three per turn) |
| `TBETS_EXP_CONST` | 0 | UCB1 exploration constant |
| `TBETS_SEED` | 0 | seed for the mutation order, the child-or-sibling draw and UCB tie-breaks |
| `TBETS_ROLLOUT` | 2 | rollout length in rounds: greedy play until that many rounds have started after the round of the last genome's turn (0: stop at the end of that turn; 1: at the end of that round) |
| `TBETS_GREEDY_PCT` | 5 | % of expansions that add a child (the next turn's greedy genome) instead of a sibling (a mutation of the selected leaf) |
| `TRIPLEA_ROLLOUT_BATTLE` | fast | battles in the search's simulations: `fast` (the Fast AI's most likely outcome, no dice; evaluations cached) or `dice` (dice rolled, default casualties, no retreat; every evaluation rolls again) |
| `TRIPLEA_AGENT_LOG` | 0 | 1 logs to stderr and `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt` |

## Visualising a search

Set `TBETS_SAVE_TREE=1` to draw every search to
`planning-agent-testbed/logs/trees/tbets-complete-turn-<player>-game-<n>-round-<r>-<phase>`, in
TripleMind's style (`tripleMind/helper.py`'s `save_mcts_tree_png` and `save_mcts_tree_html`):

- `.dot` (and `.png` with `TBETS_TREE_PNG=1`): the root, the `TBETS_TREE_MAX_NODES` most visited
  nodes (default 30) and each drawn node's most visited line below it, stacked left to right
  (`TBETS_TREE_RANKDIR=LR`). Each box is `#n <player> vis= avg=` (`n`: creation order, the greedy
  genome being #1; `avg`: the mean from the turn's player's point of view) and lists the genome's genes with their phase
  first: `P: buyArmour`, `C: TopBar | 2x infantry`, `N: Germany | skip`. Each edge says how the
  genome was made: `greedy` (a child: the next turn's greedy genome) or
  `mutation of #m: gene i: old -> new` (a sibling of #m). The played path (the most visited child
  at each level) is bold; a node notes how many of its children are not drawn;
- `.html`: the whole tree as a collapsible outline, children sorted by visits, the played path
  open, each node's genes listed.

## Metrics

Metrics are written as for `emcts-fh`, with one row per search (three per turn). `efficiency.csv`
has one more column, `phase`, before `unique_plans`. `num_children_at_root` is the number of
genomes of my remaining turn, `num_action_sequences` the tree's leaves, `max_tree_depth` the
deepest node in turns from the root, `phases` is 0 (no bridge burning) and `horizon` holds `TBETS_ROLLOUT`. `num_of_terr_attacked` is 0 outside the
combat move.

## Files

| File | Source |
|---|---|
| `agent.py` | `emcts-fh`'s, searching the purchase and noncombat move too |
| `tbets.py` | the search: `emcts-fh`'s decoding and mutation over flat turn genomes, in an MCTS tree with one level per player turn |
| `noncombat_generators.py` | unchanged copy from `heterogeneous-mcts` (atomic noncombat options) |
| `tbets_tree.py` | the tree drawing, in TripleMind's style |
| `docs/tbets-uml.drawio` | UML of the earlier EMCTS search (out of date) |
| the rest | copies from `emcts-fh` with unused functions removed (its `weak_ai.py` and `land_graph.py` dropped) |
