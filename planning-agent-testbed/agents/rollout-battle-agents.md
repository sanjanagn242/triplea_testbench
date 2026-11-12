# Agents with the rollout battle setting

These agents search the purchase, combat move and noncombat move. Each takes
`TRIPLEA_ROLLOUT_BATTLE`, which sets how battles are resolved in the search's simulations:

- `fast` (default): the Fast AI's most likely outcome, no dice.
- `dice`: the forward model's dice-rolling battles, default casualties, never retreat.

| Family | Agent id | Folder | Search time setting |
|---|---|---|---|
| hmcts | `heterogeneous-mcts` | `mcts-variants/heterogeneous-mcts` | `TRIPLEA_MCTS_SECONDS` |
| hmcts | `hmcts-no-action-sorting` | `hmcts-ablations/no-action-sorting` | `TRIPLEA_MCTS_SECONDS` |
| hmcts | `hmcts-no-eval-function` | `hmcts-ablations/no-eval-function` | `TRIPLEA_MCTS_SECONDS` |
| tbets | `tbets-complete-turn` | `tbets-complete-turn` | `TBETS_SECONDS` |
| tbets | `tbets-modified` | `tbets-modified` | `TBETS_SECONDS` |
| tbets | `tbets-segmented` | `tbets-segmented` | `TBETS_SECONDS` |
| tbets | `tbets-m-uct` | `mcts-variants/tbets-m-uct` | `TRIPLEA_MCTS_SECONDS` |
| tbets | `tbets-seg-hybrid` | `mcts-variants/tbets-seg-hybrid` | `TRIPLEA_MCTS_SECONDS` |
| emcts | `emcts-segmented` | `emcts-segmented` | `EMCTS_SECONDS` |

In a tournament spec, for example:

```json
"tbets-segmented TBETS_SECONDS=1.0 TRIPLEA_ROLLOUT_BATTLE=dice",
"tbets-segmented TBETS_SECONDS=1.0 TRIPLEA_ROLLOUT_BATTLE=fast"
```

With `dice`, `tbets-complete-turn`, `tbets-modified`, `tbets-segmented` and `emcts-segmented`
skip their per-genome value cache, so every evaluation rolls new dice. The five hmcts-based agents
(the three hmcts ones, `tbets-m-uct` and `tbets-seg-hybrid`) play the battle phase when a tree node
is created, so a node keeps the outcome rolled then.
