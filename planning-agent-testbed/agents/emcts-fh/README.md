# Fixed-horizon Evolutionary MCTS (combat)

Agent ID `emcts-fh`. A copy of [`emcts`](../emcts/README.md) whose genome also covers the player's
combat moves in the next `EMCTS_HORIZON` turns. Like `emcts`, it follows Baier and Cowling,
"Evolutionary MCTS for Multi-Action Adversarial Games" (IEEE CIG 2018,
https://ieeexplore.ieee.org/document/8490403), and searches the **combat move only**. Every other
phase plays as TripleMind does:

| Phase | Policy |
|---|---|
| purchase | TripleMind's greedy purchase (the first `purchase_actions` option) |
| combat move | fixed-horizon EMCTS (`emcts_fh.py`) |
| battle | `fightAll` (TripleA rolls the dice) |
| noncombat move | TripleMind's `noncombat_move` plan |
| place | `{}`: the engine places the units bought |

[`emcts-segmented`](../emcts-segmented/README.md) is the same search with the purchase and the
noncombat move as genes too, searched at all three phases. That is the only difference between
the two agents.

## The genome

EMCTS here looks for **the player's combat moves against greedy opponents**. A genome is a
sequence of segments. It starts at this combat move and ends with the turn that is
`EMCTS_HORIZON` turns after this one, whoever plays that turn. Each segment starts with a
**marker** (`Segment.phase`, `.turn`, `.player`), followed by its genes:

- Each of the searching player's combat moves is a segment with genes.
- Every other player's turn is a single `other` segment with **no genes**. It marks where that
  turn is played, and the turn is played with TripleMind's greedy policies in the state the genome
  has reached. Allies count as other players too.

With two players: `EMCTS_HORIZON=1` gives `[C]mine [other]opp`, and `EMCTS_HORIZON=2` gives
`[C]mine [other]opp [C]mine`. `turn` counts player turns after the current one, with 0 for this
turn. A combat move the game skips (no unit can move) has no segment.

- **Genes**: each combat segment's genes are the same as roxo-oep-with-purchase's combat genes.
  There is one gene per TripleMind target: land next to the player's territory that it does not
  own (`enemy_territories_by_priority`), kept if some unit can reach it. Each gene is one of
  TripleMind's attack sets for that territory (`attack_sets`) or skip. A gene can't use units that
  another gene of its segment uses.
- **Decoding (repair ψ)**: a later segment's targets and units depend on the earlier segments. A
  genome is therefore decoded in play order, with each stored segment found by its marker
  `(turn, phase)`:
  - In a combat segment, a stored gene is kept when its territory is still a target and its units
    still reach it without clashing with a gene kept before it. A stored skip is always kept.
    Every other target takes its strongest attack set from the units left, or skip if there is
    none.
  - The player's purchase and noncombat move are played with TripleMind's greedy policies, as the
    live agent plays them.
  - Another player's turn is played with TripleMind's greedy purchase, the strongest attack sets
    and `noncombat_move`.
  - Between these, battles use the deterministic Fast AI estimate (`fast_battle`), default
    casualties and no retreat. Placement puts everything bought at the home factory.

## The search

- **Root (φ)**: this combat move is the Greedy Action AI's genome. It goes through the territories
  in priority order and gives each gene the option whose genome scores best. Genes not decided yet
  are skipped during this scoring. The later segments are decoded from nothing.
- **Edges**: every combination of one of the player's combat segments, a gene in it, and another
  option for that gene. Each option is built from the units the segment's other genes leave free.
  Opponents are never mutated. Changing a segment can change the targets of the later ones, and
  decoding repairs them. Each iteration expands one untried mutation, picked in random order.
- **Evaluation**: the decoded genome is played to the first decision of the turn after the last
  one, so the last turn's battles and placement are included. The resulting state is scored with
  TripleMind's `evaluate_state` for the searching player. A win scores more the earlier it comes:
  the depth is the turn it came in, out of `horizon + 1`. Evaluations are deterministic, so they
  are cached per genome.
- **Selection and backpropagation**: UCB1 with exploration constant C, which is 0 in the paper.
  Every edge is the searching player's, so means are maximised. Backpropagation averages the
  values.
- **Bridge burning**: the time budget is split into phases (40 in the paper). After each phase the
  search makes the most promising mutation at the root: the most visits, then the best mean. That
  child becomes the new root and keeps its subtree. The first segment of the root after the last
  phase is the move that is played.

`emcts`' `EMCTS_ROLLOUT` is gone. The horizon replaces it.

## Settings (`run-agent.sh`)

| Setting | Default | Meaning |
|---|---|---|
| `EMCTS_SECONDS` | 1.0 | search time per combat move |
| `EMCTS_PHASES` | 40 | bridge-burning phases |
| `EMCTS_EXP_CONST` | 0 | UCB1 exploration constant |
| `EMCTS_SEED` | 0 | seed for the order in which mutations are tried and for UCB tie-breaks |
| `EMCTS_HORIZON` | 1 | turns after this one in the genome, any player's (0: the rest of this turn only; with two players, 2 reaches my next combat move) |
| `TRIPLEA_AGENT_LOG` | 0 | 1 logs to stderr and `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt` |

## Visualising a search

Set `EMCTS_SAVE_TREE=1` to draw every combat-move search to
`planning-agent-testbed/logs/trees/emcts-fh-<player>-game-<n>-round-<r>.dot`. The settings are
the same as for `emcts`: `EMCTS_TREE_PNG`, `EMCTS_TREE_MAX_NODES`, `EMCTS_TREE_MAX_DEPTH` and
`EMCTS_TREE_RANKDIR`. Moves and mutation labels are prefixed with the segment's marker,
`T<turn> <player> <phase>`, where turn 0 is this turn. Another player's turn shows as
`T<turn> <player>: greedy turn`. Unit names are looked up in the root state, so units bought later
in the horizon are not named.

Nodes in the first `EMCTS_TREE_GENOME_LEVELS` levels (default 5: depths 0 to 4) list their whole
genome, uncut, with skips too. Deeper nodes list nothing, except the played genome, which keeps
the short summary.

## Metrics

Metrics are written as for `emcts`, to `metrics/<game>/vs-<opponent>/`, or under
`$TRIPLEA_TESTBENCH_OUTPUT_DIR/metrics/` in a tournament. `efficiency.csv` has two extra last
columns, `horizon` and `unique_plans`: the distinct legal plans evaluated in the search, as
decoded (repaired) genomes, so stored genomes that decode to the same plan count once (unlike
`evaluations`). In `rollout_stats.csv`, each row totals one search's distinct genomes evaluated,
and its depths (`depth_counts`) are the rounds those genomes' horizons advanced.

## Files

| File | Source |
|---|---|
| `agent.py` | `emcts`' with `EMCTS_HORIZON` in place of `EMCTS_ROLLOUT` |
| `emcts_fh.py` | `emcts`' `emcts.py` with marked segment genomes, decoding and greedy other-player turns (`emcts-segmented`'s search with combat genes only) |
| `emcts_fh_tree.py` | `emcts-segmented`'s tree drawing, without purchase and noncombat genes |
| the rest | unchanged copies from `emcts` |
