# hmcts-no-action-sorting

Ablation of `agents/mcts-variants/heterogeneous-mcts`: the tree's actions carry no ordering
heuristic, and the search gives their order no weight. Rollouts, reward and deterministic battles
are unchanged.

- **Combat move**: a node offers the attacks and the skip of *every* reachable target not yet
  decided (`combat_actions_unsorted`), in map order, instead of only the most important target
  (`enemy_territories_by_priority`). Attack sets are listed weakest first. The target of the edge
  played is excluded from its child.
- **Non-combat move**: a node offers the reinforcements and the skip of every territory short of
  cover not yet decided (`noncombat_actions_unsorted`), in map order, instead of the territory with
  the highest priority x shortfall first. Reinforcement sets are listed weakest first.
- **Purchase**: unchanged (candidates were already in production-frontier order).
- **No order bias in the search**: a node expands its untried actions in random order (the base
  expands them in generated order), and UCT ties and the final most-visited choice are broken at
  random (the base: highest strength / first generated).
- Kept: the donor order inside one attack / reinforcement set (it decides which units make up
  the set), the undefended-target cheapest-unit fallback, the purchase candidate order from the
  last unit bought (no duplicate purchase paths), the rollout policies (they still use the
  ordered `combat_actions` / `noncombat_actions`) and the greedy purchase completion.
- **Transposition table**: with every target offered at once, the same decisions can be taken in
  any order (attack A then B = B then A). `expand()` looks the new state up in a per-search table
  (`hmcts.state_key`: the forward-model state with unit ids, without battle ids, plus the
  decided targets, moved units and purchase index) and links to the existing node, so the tree
  is a DAG. Statistics follow Monte-Carlo graph search (Czech et al. 2021): edges keep their own
  visits and value (exploration term, final choice, metrics), and each node sums the rewards of
  every iteration entering it by any edge, which UCT uses as the edge's value. A merged state
  keeps the first state's battle order.
- Metrics on the graph: `num_action_sequences` counts distinct leaf nodes and `max_tree_depth` is
  the longest path. The game records' `last_stats` add `nodes` and `transpositions`, and tree
  drawings show a shared node once (the HTML outline links later parents to it).

See the base README for the rest.

The base agent's description follows.


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

Settings are in `run-agent.sh`. Metrics and trees keep all-phases-mcts's names, split by the
root's phase (`efficiency.csv` / `_noncombat` / `_purchase`, `tree_*` / `tree_noncombat_*` /
`tree_purchase_*`).
