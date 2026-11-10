# tbets-m-uct

A variation of [`tbets-segmented`](../../tbets-segmented/README.md) on
[`heterogeneous-mcts`](../heterogeneous-mcts/README.md)'s tree: an edge is one **atomic action**
(not a whole phase genome) and a **mutation** of a node is another action of its parent's
decision, i.e. a **sibling**. The search is `tbets_muct.TBETSMUCT`, a subclass of
`hmcts.HeterogeneousMCTS` that only changes how the tree grows (`grow`) and how the answer is
completed (`play_phase`).

## How the tree grows (`tbets_muct.py`)

A node's actions are no longer expanded one after another. Instead:

- **Greedy child**: a node's first child is the greedy action of its decision: every action is
  applied and rolled out (as `simulate`), and the one whose rollout is best for the chooser is kept
  (strongest first, so the strongest wins a tie; at least one action is tried, the rest only before
  the deadline). The root starts with its greedy child.
- **Mutation**: another action of the parent's decision, not yet an edge, drawn at random. It
  makes a sibling of the node.
- **Selection**: the best UCB1 edge from the root down to a leaf. With `TBETS_WIDEN=1` (default,
  as tbets-segmented) every node selection passes through first gets one new child, a mutation of
  its children, while it has an action left; it is simulated and backpropagated before selection
  goes on.
- **Expansion at the leaf**: with probability `TBETS_GREEDY_PCT` % (or when the leaf's parent has no
  action left) a child: the greedy action of the leaf's decision. Otherwise a mutation of the
  leaf (a sibling). A leaf with no decision after it and no mutation left is simulated again.
- **Simulation, backpropagation, rewards**: heterogeneous-mcts's (below).
- **Answer**: the most-visited path, as below; where the tree stops short of the end of a combat
  or non-combat move, each remaining decision takes its strongest action (tbets's decoding from
  nothing).

Since decisions have few actions (a handful of units, attack sets or reinforcements), a parent
often runs out of mutations, and the leaf then gets a greedy child whatever `TBETS_GREEDY_PCT` is.

## The tree (from heterogeneous-mcts)

One MCTS (`hmcts.HeterogeneousMCTS`) for the purchase, combat move and non-combat move, in place
of all-phases-mcts's three separate searches. Place is never searched (the engine places the
purchased units); battles are fought with `fightAll`.

- **Root** is the requested decision. A node's actions are generated from the decision its state
  is at: single units to buy (purchase), one attack or skip per target (combat move), one
  reinforcement or skip per territory short of cover (non-combat move).
- **One tree across phases and players**: when a phase has nothing left to decide, the next node
  is the next searched decision, whoever's it is (this player's next phase or an opponent's).
  Battles, place and end of turn in between are played by the rollout policies.
- **One reward**: every node is valued by `evaluate_state` (TUV share and income share, or the
  win/loss value), kept from the view of the player who chose its edge.
- **Statistics on edges**: visits and value live on the `Edge` (action → child), the same for every
  phase; a node holds only its state and its edges. UCT and the final choice read the edges.
- **Deterministic battles**: every battle the search fights, in the tree and in rollouts, is
  resolved by the Fast AI's most likely outcome (`fast_battle.py`, copied from emcts; `hmcts.py`
  installs it on import) instead of dice. Fly-over AA and bombing raids still roll dice. The real
  game's battles are fought by the engine with dice.
- **Rollout horizon** (`TRIPLEA_MCTS_ROLLOUT`) is in rounds: 1 ends a rollout when the round the
  leaf is in is over.
- Purchase units are offered in candidate order from the last unit bought, so each set of units
  has one path in the tree.
- The answer is the most-visited path from the root up to the node that passes the phase on. A
  purchase the search stopped short of spends the rest greedily; a non-combat move adds its
  forced air landings and positioning moves.

Settings are in `run-agent.sh`; `TBETS_GREEDY_PCT` (default 5) and `TBETS_WIDEN` (default 1) are
this agent's, the `TRIPLEA_MCTS_*` ones heterogeneous-mcts's. Metrics and trees keep all-phases-mcts's names, split by the
root's phase (`efficiency.csv` / `_noncombat` / `_purchase`, `tree_*` / `tree_noncombat_*` /
`tree_purchase_*`).

## Docs

These are heterogeneous-mcts's and show its expansion (one untried action at a time), not
`tbets_muct.py`'s.

Open the `.drawio` files with [diagrams.net](https://app.diagrams.net) or the VS Code Draw.io extension.

- `docs/heterogeneous-mcts-uml.drawio`: class diagram, modules and dependencies.
- `docs/heterogeneous-mcts-flowchart.drawio`: turn request, `mcts_search`, `expand` / `pass_phase`,
  `simulate`, `legal_moves` / `play_phase`, `evaluate_state`.
- `docs/heterogeneous-mcts-sample-tree.drawio`: the most-visited path of a real 1 s search from
  Russians' round-1 purchase on capture-the-flag-diag. It runs through Russians' purchase and
  combat move, then Germans' purchase and combat move, into round 2. `docs/sample-tree.html` is
  the same tree as a collapsible outline and `docs/sample-tree.dot` is its first 300 nodes.
