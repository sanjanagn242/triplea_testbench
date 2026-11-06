# Segmented Evolutionary MCTS (purchase, combat, noncombat)

Agent ID `emcts-segmented`. A variant of [`emcts-fh`](../emcts-fh/README.md). Its genome holds a
segment for each of the player's **purchase, combat move and noncombat move**, with
[`heterogeneous-mcts`](../mcts-variants/heterogeneous-mcts/README.md)'s atomic actions as genes.
The agent searches all three phases:

| Phase | Policy |
|---|---|
| purchase | segmented EMCTS (`emcts_segmented.py`): plays the genome's first segment |
| combat move | segmented EMCTS |
| battle | `fightAll` (TripleA rolls the dice) |
| noncombat move | segmented EMCTS |
| place | `{}`: the engine places the units bought |

## The genome

EMCTS here looks for **the player's own actions against greedy opponents**. A genome is a
sequence of segments. It starts at the current phase and ends with the turn that is
`EMCTS_HORIZON` turns after this one, whoever plays that turn. Each segment starts with a
**marker** (`Segment.phase`, `.turn`, `.player`), followed by its genes:

- Each of the searching player's purchase, combat move and noncombat move is a segment with genes.
- Every other player's turn is a single `other` segment with **no genes**. It marks where that
  turn is played, and the turn is played with TripleMind's greedy policies in the state the genome
  has reached. Opponents therefore react to the player's genes without being part of the search.
  Allies count as other players too.

With two players:

| Searching at | `EMCTS_HORIZON=1` | `EMCTS_HORIZON=2` |
|---|---|---|
| my purchase | `[P C N]mine [other]opp` | `[P C N]mine [other]opp [P C N]mine` |
| my combat move | `[C N]mine [other]opp` | `[C N]mine [other]opp [P C N]mine` |
| my noncombat move | `[N]mine [other]opp` | `[N]mine [other]opp [P C N]mine` |

`turn` counts player turns after the current one, with 0 for this turn. A phase the game skips
has no segment: for example, a noncombat move where no unit has movement left, or a purchase with
nothing affordable. Decoding finds each stored segment by its marker `(turn, phase)`. That keeps
segments lined up when a mutation changes what later turns contain.

The genes are heterogeneous-mcts's atomic actions:

| Segment | Genes | Mutation: change one gene to | Decoding (repair) |
|---|---|---|---|
| purchase | one per unit bought: a production rule id | another candidate unit (land combat units bought with PUs, as `HeterogeneousMCTS.unit_rules`) | buy the stored units in order, dropping those no longer affordable, then spend the PUs left on TripleMind's purchase (the first `purchase_actions` option) |
| combat | one per TripleMind target: an attack set or skip, as in `emcts-fh` | another attack set from the units the other genes leave free, or skip | `emcts-fh`'s: an unusable stored gene takes the strongest attack set |
| noncombat | one per territory short of cover, most important first (`noncombat_generators.noncombat_actions`): a reinforcement set or skip | another reinforcement set offered for that territory, or skip | each territory offered takes its stored option if that is still offered, else the strongest reinforcement |
| other | none | never mutated | TripleMind's greedy purchase, strongest attack sets, `noncombat_move` plan |

A noncombat segment also plays heterogeneous-mcts's forced moves, which are not genes: before the
genes, the air landings (`land_air_moves`); after them, the positioning moves. A segment decoded
from nothing is therefore TripleMind's purchase, the strongest attack sets, or the strongest
reinforcement per territory. The last is heterogeneous-mcts's split of `noncombat_move`, not
`noncombat_move` itself.

## The search

UML of the iteration: [docs/emcts-segmented-uml.drawio](docs/emcts-segmented-uml.drawio).

The search is `emcts-fh`'s, applied to segments:

- **Between segments**: battles use the deterministic Fast AI estimate (`fast_battle`), default
  casualties and no retreat. Placement puts everything bought at the home factory.
- **Root (φ)**: when the genome has this turn's combat segment (searching from a purchase or a
  combat move), that segment is the Greedy Action AI's genome, as in `emcts-fh`. Every other
  segment is decoded from nothing.
- **Evaluation**: the decoded genome is played to the first decision of the turn after the last
  one, so the last turn's battles and placement are included. The state is then scored with
  TripleMind's `evaluate_state`. A win scores more the earlier it comes: the depth is the turn it
  came in, out of `horizon + 1`. Evaluations are cached per genome.
- **Selection, backpropagation and bridge burning**: as in `emcts-fh`. Every mutation is the
  searching player's, so UCB1 maximises the mean everywhere (no sign flip for opponents).

Live, a purchase plays the units its first segment bought as one purchase. A noncombat move plays
the air landings, the chosen reinforcements and the positioning moves. A combat move plays the
attacks, routed as in `emcts-fh`.

If the forward model rejects the state, or the search's first segment is not this phase's,
purchase and noncombat fall back to TripleMind's greedy policy, and the combat move plays no moves
(as in `emcts-fh`).

## Settings (`run-agent.sh`)

| Setting | Default | Meaning |
|---|---|---|
| `EMCTS_SECONDS` | 1.0 | search time per searched phase (three per turn) |
| `EMCTS_PHASES` | 40 | bridge-burning phases |
| `EMCTS_EXP_CONST` | 0 | UCB1 exploration constant |
| `EMCTS_SEED` | 0 | seed for the order in which mutations are tried and for UCB tie-breaks |
| `EMCTS_HORIZON` | 1 | turns after this one in the genome, any player's (0: the rest of this turn only; with two players, 2 reaches my next turn) |
| `TRIPLEA_AGENT_LOG` | 0 | 1 logs to stderr and `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt` |

## Visualising a search

Set `EMCTS_SAVE_TREE=1` to draw every search to
`planning-agent-testbed/logs/trees/emcts-segmented-<player>-game-<n>-round-<r>-<phase>.dot`. The
settings are `emcts-fh`'s. Genome lines and mutation labels are prefixed with the segment's marker,
`T<turn> <player> <phase>`.

Nodes in the first `EMCTS_TREE_GENOME_LEVELS` levels (default 5: depths 0 to 4) list their whole
genome, uncut. That is every segment in order: the purchase, each combat and noncombat gene (skips
too), and `greedy turn` for another player's turn. Deeper nodes list nothing, except the played
genome, which keeps the short summary.

## Metrics

Metrics are written as for `emcts-fh`, with one row per search (three per turn). `efficiency.csv`
has one more column, `phase`, before `unique_plans`. `num_of_terr_attacked` is 0 outside the combat move.

## Files

| File | Source |
|---|---|
| `agent.py` | `emcts-fh`'s, searching the purchase and noncombat move too |
| `emcts_segmented.py` | `emcts-fh`'s `emcts_fh.py` with marked purchase, combat and noncombat segments and greedy other-player turns |
| `noncombat_generators.py` | unchanged copy from `heterogeneous-mcts` (atomic noncombat options) |
| `emcts_segmented_tree.py` | `emcts-fh`'s `emcts_fh_tree.py`, labelling genes by segment marker |
| `docs/emcts-segmented-uml.drawio` | UML of the search (open in draw.io / diagrams.net): activity of `run()`, of one `iterate()` and of `evaluate()`'s decoding, and the sequence of one iteration that expands |
| the rest | unchanged copies from `emcts-fh` |
