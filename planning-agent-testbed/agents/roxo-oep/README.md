# Roxo OEP

Online Evolutionary Planning as adapted to TripleA in M. Roxo's thesis *MCTS-based Planning for
Grand Strategy Games* (Instituto Superior Técnico, Section 4.4), after Justesen et al.'s OEP
(IEEE Transactions on Games, 2018). Built on [roxo-ne-mcts](../roxo-ne-mcts/README.md): the
territory actions, Weak AI playout heuristic, deterministic battles and reward are copies of its
modules (`world_model.py`, `weak_ai.py`, `fast_battle.py`, `land_graph.py`). The thesis's map and
setting are [games/minimap-roxo](../../games/minimap-roxo).

UML (open in draw.io): [docs/roxo-oep-uml.drawio](docs/roxo-oep-uml.drawio): 1 class diagram,
2 activity of one search.

## Files

| File | Purpose |
|---|---|
| `agent.json` | Manifest (ID `roxo-oep`, display name **Roxo OEP**) |
| `run-agent.sh` | Launcher and settings (below); puts `forward-model/python` on `PYTHONPATH` |
| `agent.py` | Protocol loop (as roxo-ne-mcts), one evolution per move phase, final routes, metrics |
| `oep.py` | `RoxoOEP`: genomes, creation, evaluation, procreation, mutation, generations |
| `world_model.py`, `weak_ai.py`, `fast_battle.py`, `land_graph.py` | Roxo's forward model, playout heuristic, deterministic battles and land routes (from roxo-ne-mcts) |

## How a turn is played

| Phase | Response |
|---|---|
| `purchase`, `place` | `{}`: the thesis setting has no production |
| `combatMove`, `nonCombatMove` | One evolution from the current state; the best genome's units, routed from where they stand |
| `battle` | `{"fightAll": true}`: TripleA resolves the real battles with dice |

## The algorithm

**Genome.** One action per candidate territory of the move phase, with the units moved there, or
a skip. Candidate territories: in a combat move the enemy territories next to the player's (as
`WorldModel` lists them, each once); in a noncombat move the player's own territories. A
territory's actions are those the forward model generates for it:

- *combat move*: if the available units can take it (Weak AI strength > 1.1x), the unit sets of
  `getUnitsUpToStrengthSet` and skip; otherwise skip only;
- *noncombat move*: every non-empty combination of the unit groups that can reach it (one group
  per territory the units stand in), and skip.

A unit belongs to at most one action of a genome: units already used by another gene are not
available.

**Creation (4.4.1).** The territories in random order; for each, a random one of its actions from
the units still available. The order is random because a territory served first has more units to
choose from.

**Evaluation (4.4.2).** The genome's actions are applied to the current state, the move phase is
ended (its battles resolved with the deterministic Fast AI outcome), then a playout guided by the
Weak AI heuristic runs for at most 10 rounds or until the game ends, and Roxo's reward rates the
final state (a win `0.6 + 0.4 x (P - phases) / P`, a loss the negative, otherwise
`(own - other units) / (2.5 x all units)`, with `P` = 10 rounds of move phases). The Weak AI's
shuffle is reseeded identically for every genome of a search, so the playout is deterministic and a
genome (identified by which units go where) is evaluated once.

**Procreation (4.4.3).** Territories in random order; for each, the gene of a random one of the two
parents. If its units are already used, a random action for the same territory from the units
still available replaces it.

**Mutation (4.4.4).** A new genome, with probability 0.1, has one gene replaced by a random action
for the same territory (from the units the other genes leave free).

**Generations** (OEP, Algorithm 1). Population 50; evaluate the new genomes, keep the best half
(survival rate 0.5), pair each survivor with a random other survivor to refill the population,
repeat until the time budget is used; play the best genome found. When no territory has a choice
(e.g. nothing can be attacked), the single possible genome is played without evolving.

## Where this differs from the thesis text

- *Mutation* is read as one gene of a new genome replaced with probability 0.1, as in Justesen et
  al.'s OEP (the thesis's sentence does not say whether it is per genome or per gene).
- *Pairing* follows Justesen et al.: each survivor procreates with another survivor chosen at
  random.
- The thesis does not say how the playout's randomness is fixed; here the Weak AI shuffle uses the
  same seed for every genome of one search (a new seed per search).
- The roxo-ne-mcts differences from his code (kernel validation, unit order, routes, shuffle,
  survivor order) apply here as well.

## Settings (`run-agent.sh`)

| Variable | Default | Meaning |
|---|---|---|
| `ROXO_SECONDS` | `1.0` | Search time per move phase |
| `ROXO_OEP_POPULATION` | `50` | Population size |
| `ROXO_OEP_SURVIVAL` | `0.5` | Share of the population kept each generation |
| `ROXO_OEP_MUTATION` | `0.1` | Probability that a new genome is mutated |
| `ROXO_OEP_PLAYOUT_ROUNDS` | `10` | Playout length in rounds |
| `ROXO_SEED` | `0` | Seed of the evolution and the Weak AI shuffle |
| `TRIPLEA_AGENT_LOG` | `0` | `1` logs each search to `planning-agent-testbed/logs/agent-<faction>-game-<n>.txt` |

## Metrics

As roxo-ne-mcts (with `agents/tripleMind/helper.py`), in `metrics/<game>/vs-<opponent>/`, or
`$TRIPLEA_TESTBENCH_OUTPUT_DIR/metrics/...` in a tournament:

| File | Row | Columns |
|---|---|---|
| `game_results.csv` | per game | TripleMind's |
| `efficiency.csv` | per search (combat and noncombat moves) | `num_iterations` generations; `root_value` best genome's value; `num_children_at_root` genes (candidate territories); `max_rollout` rounds; `num_action_sequences` distinct genomes evaluated; `num_of_terr_attacked` actions of the best genome; `max_tree_depth` blank; then `phase`, `evaluations`, `duplicates` (genomes already evaluated), `best_value_per_generation` (`;`-separated) |
| `rollout_stats.csv` | per evaluation | `rollout_depth_reached` rounds, `reward`, `is_terminal` |

## Running it

Against Roxo NE-MCTS, offline (from `planning-agent-testbed/`):

```bash
python3 tools/arena.py --xml games/minimap-roxo/map/minimap/map/games/minimap.xml \
    --agent Russians=agents/roxo-oep/run-agent.sh --agent Italians=agents/roxo-ne-mcts/run-agent.sh \
    --games 2 --max-rounds 35
```
