# Evolutionary MCTS (combat)

Agent ID `emcts`. Evolutionary MCTS after Baier and Cowling, "Evolutionary MCTS for Multi-Action
Adversarial Games" (IEEE CIG 2018, https://ieeexplore.ieee.org/document/8490403), used for the
**combat move only**. Every other phase plays as TripleMind does:

| Phase | Policy |
|---|---|
| purchase | TripleMind's greedy purchase (the first `purchase_actions` option) |
| combat move | EMCTS (`emcts.py`) |
| battle | `fightAll` (TripleA rolls the dice) |
| noncombat move | TripleMind's `noncombat_move` plan |
| place | `{}`: the engine places the units bought |

## The search

The tree's nodes are complete combat moves (genomes) and its edges are mutations of one gene.

- **Genome and actions**: the same as roxo-oep-with-purchase's combat genes. There is one gene per
  TripleMind target: land next to the player's territory that it does not own
  (`enemy_territories_by_priority`), kept if some unit can reach it. Each gene is one of
  TripleMind's attack sets for that territory (`attack_sets`) or skip. A gene can't use units
  that another gene uses. Moves are routed by Roxo's forward model (`world_model.Game`).
- **Root (φ)**: the Greedy Action AI's genome. It goes through the territories in priority order
  and gives each gene the option whose genome scores best. Genes not decided yet are skipped
  during this scoring.
- **Edges**: every pair of a gene and another option for that gene, where the option is built from
  the units the other genes leave free. Every mutation is therefore legal, so no repair (ψ) is
  needed. Each iteration expands one untried mutation, picked in random order.
- **No rollouts** (`EMCTS_ROLLOUT=1`, the paper's setting): each new node is evaluated once, at
  the end of the turn. The search plays the genome's moves, then resolves the battles with the
  deterministic Fast AI estimate (`fast_battle`), then plays TripleMind's noncombat plan. The
  resulting state is scored with TripleMind's `evaluate_state`. Because evaluations are
  deterministic, they are cached by phenotype.
- **Optional rollouts** (`EMCTS_ROLLOUT=n` > 1): after that turn, a rollout plays n − 1 more rounds
  before the evaluation, ending at the end of the player's n-th turn. Every player moves as in
  TripleMind's rollouts: in a combat move, a random one of `combat_actions`' options for each
  target in turn; in a noncombat move, `noncombat_move`'s plan. Battles stay deterministic, and
  the rollout buys nothing. Its random choices are seeded once per search and are the same for
  every genome, so evaluations stay deterministic and the cache still holds. A win scores more
  the earlier it comes (`evaluate_state`'s depth is the number of turns played after this one).
- **Selection and backpropagation**: UCB1 with exploration constant C, which is 0 in the paper.
  Backpropagation averages the values.
- **Bridge burning**: the time budget is split into phases (40 in the paper). After each phase the
  search makes the most promising mutation at the root (most visits, then best mean value). That
  child becomes the new root and keeps its subtree. The root after the last phase is the move that
  is played.

The search does not model production: Roxo's forward model ends the purchase and place phases
without buying anything.

## Settings (`run-agent.sh`)

| Setting | Default | Meaning |
|---|---|---|
| `EMCTS_SECONDS` | 1.0 | search time per combat move |
| `EMCTS_PHASES` | 40 | bridge-burning phases |
| `EMCTS_EXP_CONST` | 0 | UCB1 exploration constant |
| `EMCTS_SEED` | 0 | seed for the order in which mutations are tried, for UCB tie-breaks and for the rollouts |
| `EMCTS_ROLLOUT` | 1 | rollout depth in the player's own turns (1: no rollout) |
| `TRIPLEA_AGENT_LOG` | 0 | 1 logs to stderr and `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt` |

## Visualising a search

Set `EMCTS_SAVE_TREE=1` to draw every combat-move search to
`planning-agent-testbed/logs/trees/emcts-<player>-game-<n>-round-<r>.dot`. A `.png` is also
rendered when Graphviz's `dot` is installed.

- **Nodes** are genomes. Each shows its depth (mutations from the greedy root), visits, mean value
  and its own evaluation, and is coloured from red (worst mean) to green (best mean). The greedy
  root and the played genome also list their moves.
- **Edges** are mutations, labelled with the gene that changed (`territory: old -> new`).
- **Blue** marks the bridge-burning path, from the greedy root to the played genome, which is drawn
  bold.

With C = 0, a search over a small position dives hundreds of mutations deep while revisiting only a
few genomes. To keep the drawing readable, it shows the path plus the most visited nodes at most
`EMCTS_TREE_MAX_DEPTH` levels below it. A node notes how many of its children were left out.

| Setting | Default | Meaning |
|---|---|---|
| `EMCTS_SAVE_TREE` | 0 | 1 draws every search |
| `EMCTS_TREE_PNG` | 1 | also render a `.png` (needs `dot`) |
| `EMCTS_TREE_MAX_NODES` | 300 | nodes drawn (the path always is) |
| `EMCTS_TREE_MAX_DEPTH` | 3 | levels drawn below the path |
| `EMCTS_TREE_RANKDIR` | LR | `LR`: depth left to right; `TB`: top to bottom |

## Metrics

Metrics are written to `metrics/<game>/vs-<opponent>/`, or under `$TRIPLEA_TESTBENCH_OUTPUT_DIR/metrics/`
in a tournament, as for TripleMind agents:

- `game_results.csv`
- `rollout_stats.csv`: one row per distinct genome evaluated (rounds advanced, value, whether the
  game ended)
- `efficiency.csv`: TripleMind's columns, then `evaluations`, `cache_hits`, `greedy_value`,
  `best_value`, `phases` and `tree_nodes`

In `efficiency.csv`, `num_children_at_root` is the first root's mutations,
`num_action_sequences` is the tree's leaves, and `max_tree_depth` is the most mutations of any node
from the first root.

## Files

| File | Source |
|---|---|
| `agent.py` | new: the testbench protocol, with TripleMind's purchase, noncombat and place |
| `emcts.py` | new: the search, plus a copy of `evaluate_state` from roxo-oep-with-purchase's `oep.py` |
| `emcts_tree.py` | new: draws a search tree as a Graphviz `.dot`/`.png` |
| `action_generators.py`, `generators.py`, `helper.py`, `map_graph.py` | copies from roxo-oep-with-purchase (TripleMind's code, with `combat_actions` split into `target_attackers`/`attack_sets`) |
| `world_model.py`, `fast_battle.py`, `weak_ai.py`, `land_graph.py` | copies from roxo-oep-with-purchase (Roxo's forward model and deterministic battles) |
