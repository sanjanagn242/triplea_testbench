# OEP paper: Online Evolutionary Planning (purchase, combat move, non-combat move)

Agent ID `oep`. The Online Evolutionary Planning of Justesen, Mahlmann, Risi and Togelius,
*Playing Multiaction Adversarial Games: Online Evolutionary Planning Versus Tree Search* (IEEE
Transactions on Games, 2018, Section IV), on the same actions, heuristic and phases as the other
agents in [oep_paper](..) ([Vanilla MCTS](../vanilla_mcts/README.md), [NE-MCTS](../ne_mcts/README.md),
[BB-MCTS](../bb_mcts/README.md)): one evolution per phase request, **purchase, combat move and
non-combat move**. Placement is the engine's and battles are `fightAll`, as TripleMind. Action
generators, map graph and metric helpers are copies of TripleMind's (pure Python);
`noncombat_generators.py` is all-phases-mcts's. Not to be confused with [roxo-oep](../../roxo-oep/README.md),
Roxo's thesis version (territory genomes, Weak AI playouts).

## Algorithm

```
root = state at the phase's start (non-combat: after the forced air landing)
population = 100 random genomes: random legal actions until 3 are used up or END is drawn
until the time budget is used:
    evaluate every genome not evaluated yet (survivors keep their value):
        its actions on a clone of root, END, the rest of my turn without decisions
        (battles: default casualties, no retreat, the search's fixed dice; place: everything at
        the home factory; other phases ended), evaluate_state in [-1, 1]; cached per state
    survivors = the best 50 (survival rate 0.5)
    offspring until the population is 100 again:
        crossover of two random survivors; mutated with probability 0.1
play the best genome found
```

- **Genome**: a sequence of up to 3 actions of the phase (the paper's 5 action points, reduced to
  3). A gene is one action of the phase, END included: an END gene ends the genome early, and a
  genome that uses up its 3 actions ends with END. Genes are action keys, replayed through the
  forward model from the root, so a gene is *legal* when the state the genes before it reach
  still offers that action.
- **Crossover** (uniform): gene by gene, the gene of a random parent if it is legal in the
  child's simulated state; otherwise the other parent's; then the next genes of the parents'
  sequences (gene i+1 of each parent, then i+2, …); then a random legal action. A parent that
  ended early contributes END past its end.
- **Mutation**: one random gene replaced by another legal action. The genes after it are kept
  while they are legal; from the first illegal one, the rest of the sequence is regenerated
  randomly but legally.
- **Fitness** is static evaluation: the paper found rollout-based fitness no better. The battles
  of the turn are still fought (otherwise an attack changes nothing the heuristic sees), with the
  same dice for every genome of a search, so evaluation is deterministic and cached by the
  resulting phase state (the transposition key of the MCTS agents): survivors and repeated
  genomes are never re-evaluated.
- A phase whose only action is END is answered without evolving.

| Phase | One action (gene) | Gene key |
|---|---|---|
| purchase | buy 1 unit of a candidate rule (land combat units for PUs, as `purchase_actions`) | the rule |
| combat move | attack one enemy territory with every unmoved unit that can reach it | the target |
| non-combat move | for the most important territory short of cover: a reinforcement set or its skip; END plays the positioning moves | the territory and the units sent |

With 3 actions per genome, a purchase buys at most 3 units per turn; raise
`TRIPLEA_OEP_GENOME_LENGTH` to lift that.

## Settings (`run-agent.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `TRIPLEA_OEP_SECONDS` | 1.0 | evolution time per phase request |
| `TRIPLEA_OEP_POPULATION` | 100 | population size |
| `TRIPLEA_OEP_SURVIVAL` | 0.5 | share of the population kept each generation |
| `TRIPLEA_OEP_MUTATION` | 0.1 | probability that an offspring is mutated |
| `TRIPLEA_OEP_GENOME_LENGTH` | 3 | actions per genome |
| `TRIPLEA_OEP_SEED` | 0 | search seed (also draws each search's fixed dice) |

## Metrics

Same column names as the other OEP agents (TripleMind's columns first), plus a `phase` column:

| File | Row | Notes |
|---|---|---|
| `game_results.csv` | per game | TripleMind's + `winners` |
| `efficiency.csv` | per **combat-move** evolution | kept combat-only so it compares row for row with the combat-only OEP agents. `num_iterations` generations; `root_value` best genome's value; `num_children_at_root` legal first actions incl. END; `max_rollout` 0; `num_action_sequences` distinct genomes evaluated; `num_of_terr_attacked` actions chosen; `max_tree_depth` genome length; then `evaluations`, `cache_hits`, `best_value_per_generation` (`;`-separated), `best_genome`, `phase` |
| `efficiency_purchase.csv`, `efficiency_noncombat.csv` | per purchase / non-combat evolution | same columns |
| `rollout_stats.csv` | per computed (not cached) evaluation | `iteration` the generation, `rollout_depth_reached` 0, `tl` actions in the genome, `phase` |

## Running it

```bash
python3 tools/arena.py --xml games/capture-the-flag-diag/map/capture_the_flag/map/games/capture_the_flag.xml \
    --agent Russians=agents/oep_paper/oep/run-agent.sh --games 2
```

or add `"oep TRIPLEA_OEP_SECONDS=1.0"` to a `tournaments/specs/*.json` agent list.
