# heterogeneous-mcts

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
- **Rollout horizon** (`TRIPLEA_MCTS_ROLLOUT`) is in rounds: 1 ends a rollout when the round the
  leaf is in is over.
- Purchase units are offered in candidate order from the last unit bought, so each set of units
  has one path in the tree.
- The answer is the most-visited path from the root up to the node that passes the phase on. A
  purchase the search stopped short of spends the rest greedily; a non-combat move adds its
  forced air landings and positioning moves.

Settings are in `run-agent.sh`. Metrics and trees keep all-phases-mcts's names, split by the
root's phase (`efficiency.csv` / `_noncombat` / `_purchase`, `tree_*` / `tree_noncombat_*` /
`tree_purchase_*`).

## Docs

Open the `.drawio` files with [diagrams.net](https://app.diagrams.net) or the VS Code Draw.io extension.

- `docs/heterogeneous-mcts-uml.drawio`: class diagram, modules and dependencies.
- `docs/heterogeneous-mcts-flowchart.drawio`: turn request, `mcts_search`, `expand` / `pass_phase`,
  `simulate`, `legal_moves` / `play_phase`, `evaluate_state`.
- `docs/heterogeneous-mcts-sample-tree.drawio`: the most-visited path of a real 1 s search from
  Russians' round-1 purchase on capture-the-flag-diag. It runs through Russians' purchase and
  combat move, then Germans' purchase and combat move, into round 2. `docs/sample-tree.html` is
  the same tree as a collapsible outline and `docs/sample-tree.dot` is its first 300 nodes.
