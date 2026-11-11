# hmcts-no-eval-function

A copy of heterogeneous-mcts with no evaluation function: rollouts play to the end of the game and
the reward is only the outcome.

- **Rollouts to terminal**: a rollout finishes the leaf's phase, then plays every decision with the
  rollout policies until the game is over, capped at `TRIPLEA_MCTS_ROLLOUT` rounds (100) and at the
  search's time budget (`TRIPLEA_MCTS_SECONDS`).
- **Reward**: +1 if this player's alliance won, -1 if it lost, 0 for an error (an exception in
  the rollout) or a rollout that reached the round cap with no winner. A rollout the time budget
  cut short has no result and is not backpropagated.
- The `evaluate` combat-move rollout policy (`TRIPLEA_MCTS_COMBAT_ROLLOUT`) is gone, since it
  scored moves with the evaluation function; combat-move rollouts are epsilon-greedy.

Everything else is heterogeneous-mcts's:

- **Root** is the requested decision. A node's actions are generated from the decision its state
  is at: single units to buy (purchase), one attack or skip per target (combat move), one
  reinforcement or skip per territory short of cover (non-combat move).
- **One tree across phases and players**: when a phase has nothing left to decide, the next node
  is the next searched decision, whoever's it is (this player's next phase or an opponent's).
  Battles, place and end of turn in between are played by the rollout policies.
- **Statistics on edges**: visits and value live on the `Edge` (action → child), kept from the
  view of the player who chose it. UCT and the final choice read the edges.
- **Deterministic battles**: every battle the search fights, in the tree and in rollouts, is
  resolved by the Fast AI's most likely outcome (`fast_battle.py`) instead of dice. Fly-over AA
  and bombing raids still roll dice. The real game's battles are fought by the engine with dice.
- Purchase units are offered in candidate order from the last unit bought, so each set of units
  has one path in the tree.
- The answer is the most-visited path from the root up to the node that passes the phase on. A
  purchase the search stopped short of spends the rest greedily; a non-combat move adds its
  forced air landings and positioning moves.

Settings are in `run-agent.sh`. Metrics and trees keep all-phases-mcts's names, split by the
root's phase (`efficiency.csv` / `_noncombat` / `_purchase`, `tree_*` / `tree_noncombat_*` /
`tree_purchase_*`). The design docs are in `../../mcts-variants/heterogeneous-mcts/docs`.
