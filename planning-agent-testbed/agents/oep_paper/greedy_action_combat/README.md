# OEP paper: Greedy Action (combat only)

Agent ID `oep-greedy-action-combat`. The Greedy Action baseline of Justesen, Mahlmann, Risi and
Togelius, *Playing Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search*
(IEEE Transactions on Games, 2018, Section V), applied to the TripleA combat move. Built from
`template-python`; the action generators, map graph and metric helpers are copies of TripleMind's
(`action_generators.py`, `generators.py`, `map_graph.py`, `helper.py`, pure Python).

UML (open in draw.io): [docs/greedy-action-combat-uml.drawio](docs/greedy-action-combat-uml.drawio):
1 class diagram, 2 sequence of one combat-move decision, 3 activity of the greedy loop.

## Algorithm

One action is one attack: **(all my unmoved units that can reach the target, target territory)**,
for each enemy land territory bordering mine that has not been attacked yet this turn, plus
**END** (stop attacking).

```
plan = root state (combat move), attacked = {}
repeat:
    attacks = one action per un-attacked target with reachable units
    if none: stop                                    (reached_end_of_turn = 1)
    candidates = [END] + attacks
    until this level's time share is used (at least one full sweep):
        for each candidate (same dice seed for all candidates in a sweep):
            s = plan + candidate, end combat move, fight the battles (default casualties,
                no retreat), value = evaluate_state(s) in [-1, 1]
                (1 / -1 won / lost, else 2 * (0.5 TUV share + 0.5 income share) - 1)
    best = candidate with the highest mean value (END wins ties)
    if best is END: stop                             (reached_end_of_turn = 1)
    apply best to plan, add its target to attacked
```

- **Time budget**: `TRIPLEA_GREEDY_SECONDS` per combat move (default 1.0). Each level gets
  `remaining time / number of attacks at that level` (an upper bound on the levels left). If the
  budget runs out before a level is scored, the plan stops there (`reached_end_of_turn = 0`).
- **Dice**: battles are random, so candidates are sampled round-robin. When two sweeps give every
  candidate the same value (no battle involves dice), the level stops early.
- **Evaluation point**: after this turn's battles. Non-combat moves and placement change neither
  the TUV nor territory ownership that `evaluate_state` counts, so scoring later would not change
  the ranking.
- **Other phases** are TripleMind's deterministic policy: first `purchase_actions` option,
  `noncombat_move`, engine placement (`{}`), `fightAll`.

## Metrics

Under `metrics/<game>/vs-<opponent>/` (or `$TRIPLEA_TESTBENCH_OUTPUT_DIR/metrics/...` in a
tournament). TripleMind's columns come first, so `agents/tripleMind/scripts/generate_plots.sh
<metrics folder>` works unchanged; this agent's extra columns are appended.

| File | Row | TripleMind columns, as used here | Extra columns |
|---|---|---|---|
| `game_results.csv` | per game | `won`, `rounds_played`, `game_time`, ... | `winners` |
| `efficiency.csv` | per combat move | `time_per_turn` search time; `num_iterations` states visited (forward-model evaluations); `root_value` best value of the last level; `num_children_at_root` actions at level 1; `num_action_sequences` sum of actions over levels; `num_of_terr_attacked` sequence length; `max_tree_depth` levels searched; `max_rollout` 0 | `reached_end_of_turn`, `num_actions_per_level`, `best_value_per_level` (`;`-separated) |
| `rollout_stats.csv` | per evaluation | `iteration` evaluation index, `reward`, `is_terminal`; `rollout_depth_reached` 0 | `tl` (sequence level) |
| `action_seq_stats.csv` | per level | `num_children` actions at the level; `visit_count` samples of the chosen action; `avg_value` best value; `second_*` the sibling with the **second-best value**; `mean_avg_value`, `avg_value_std_dev`, `avg_value_z` over siblings | `value_gap` (best − second), `end_phase_value`, `states_visited`, `level_seconds` |
| `sibling_values.csv` | per candidate per level | — | `action`, `chosen`, `samples`, `mean_value`, `std_value`, `diff_to_best` |

## Running it

```bash
# Engine tournament against TripleMind (both seats):
python3 tools/tournament.py --spec tournaments/specs/greedy_vs_tripleMind.json --no-gui
```
