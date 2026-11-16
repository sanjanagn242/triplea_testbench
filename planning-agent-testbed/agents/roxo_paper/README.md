# roxo_paper agents

NE-MCTS, BB-MCTS and OEP from [oep_paper](../oep_paper), changed to follow M. Roxo's thesis
*MCTS-based Planning for Grand Strategy Games* (IST, 2021) more closely. They search the
**purchase** as well as the combat move and the non-combat move. The thesis has no purchase (its map
has no production), so purchase uses the same kind of actions and playout heuristic as the other
phases.

| Folder | Agent ID | Search |
|---|---|---|
| `ne_mcts/` | `roxo-paper-ne-mcts` | Non-exploring MCTS (thesis 3.10.1, 5.7) |
| `bb_mcts/` | `roxo-paper-bb-mcts` | Bridge Burning MCTS (thesis 3.10.2, 5.7) |
| `oep/` | `roxo-paper-oep` | Online Evolutionary Planning (thesis 5.6) |

These differ from the older [roxo-ne-mcts](../roxo-ne-mcts) and [roxo-oep](../roxo-oep) agents.
Those are ports of Roxo's own Java code with their own world model, and their search continues
into later phases and the opponent's turn. The agents here keep the oep_paper framework (one search
per phase request, the testbench forward model, the metrics files) and swap in Roxo's actions,
playout, reward, battles and parameters.

## Files

Each agent folder holds identical copies of the shared files. `check-shared-files.sh` lists any
copies that have drifted apart.

| File | Purpose |
|---|---|
| `phases.py` | Roxo's territory-based action generation, plus purchase (shared) |
| `playout.py` | Search base: forward model, battles, placement, the 10-round Weak AI playout, Roxo's reward, BB's random playout phases (shared) |
| `weak_ai.py`, `land_graph.py` | Roxo's simplified Weak AI and land routes, copied from roxo-ne-mcts (shared) |
| `ne_mcts.py` / `bb_mcts.py` / `oep.py` | The search |
| `agent.py`, `run-agent.sh`, `agent.json` | Protocol loop, settings, manifest |
| `helper.py`, `map_graph.py`, `fast_battle.py`, `generators.py`, `action_generators*`, `noncombat_generators.py` | From oep_paper: metrics, map, battles, and the forced air landing at the start of the non-combat move |

## Actions (`phases.py`)

Each phase is a list of decision items. Each item is decided once, by one of its options or its
skip. A tree level, or a gene, is one item. Items whose only option is the skip are passed over.
Units moved during the phase are never offered again.

| Phase | Items | Options per item |
|---|---|---|
| purchase | the player's land combat units bought with PUs, highest attack + defence per PU first | buy the most it can afford *and* place, buy half of that, skip |
| combat move | enemy land next to the player's territories, in map order | when the reachable units' Weak AI strength is more than 1.1x the defence: the unit sets of `getUnitsUpToStrengthSet`, largest first, then skip. Otherwise skip only, so unwinnable attacks are pruned |
| non-combat move | the player's territories, capitals first, then nearest the enemy | from each territory with units that can reach it, none / first half / all of them, in every combination, largest first. The all-none combination is the skip (Algorithm 5) |

"Place" means the factories' capacity from `rules.placement_snapshot`. A player with no factory
buys nothing.

## Playout and reward (`playout.py`)

From the decision: the rest of the phase, then every player's phases, by Roxo's simplified Weak AI,
for at most 10 rounds or until the game ends (Sec. 5.4–5.5).

- **Combat move:** first a single unit into each undefended enemy territory, then just enough
  units into the defended ones it beats by 1.1x.
- **Non-combat move:** each stack takes one step toward the nearest enemy capital. A threatened
  capital keeps its units.
- **Purchase:** the first option of each item, i.e. the most it can afford and place of the
  best-value unit, then the next unit with what is left.
- **Placement:** the capital's factory first, then the others.
- **Battles:** default casualties, no retreats.

**Reward** (Algorithm 6):
- a win scores 0.6 + 0.4 × (10 − rounds)/10, and a loss the negative;
- otherwise (my units − enemy units) / (2 × all units).

**Battles in the search:** the Fast AI's most likely outcome (`TRIPLEA_BATTLE_RESOLUTION=fast`,
Sec. 5.2.2). The live game still rolls dice.

## Searches

**NE-MCTS**
- Selection takes the best mean; every child is visited once before any is expanded further.
- Playouts are deterministic: the Weak AI's shuffle and the dice are fixed per search, so each
  node's playout is computed once.
- A transposition table merges states reached by different action orders.
- The most visited path is played. If it stops at a node that was never expanded, the Weak AI
  plays the rest of the phase.

**BB-MCTS**
- Selection is UCT, X + C·√(ln n / nⱼ) with C = 1/√2 (eq. 2.1).
- Each phase of a playout is played with random actions from `phases.py` with probability
  E = 0.5, and otherwise by the Weak AI (Sec. 5.7).
- The time budget is split into burn phases, one per decision. Each burn phase gets the time left
  divided by the items still undecided, and ends by keeping only the most visited root child.

**OEP**
- One gene per item, built in a random item order.
- Population 50, survival rate 0.5.
- Procreation: each survivor is paired with a random other survivor. Each gene comes from a random
  parent; if that gene's units are already used, a random option of the same item replaces it.
- Mutation: with probability 0.1, one gene is replaced by a random option of its item.
- Fitness is the playout's reward, cached per resulting state.

## Settings (`run-agent.sh`)

| Agent | Variables (default) |
|---|---|
| all | `TRIPLEA_BATTLE_RESOLUTION` (`fast`), `TRIPLEA_AGENT_LOG` (0) |
| NE | `TRIPLEA_ROXO_NE_SECONDS` (1.0), `_PLAYOUT_ROUNDS` (10), `_DICE_SEEDS` (1, dice mode only), `_SEED` (0) |
| BB | `TRIPLEA_ROXO_BB_SECONDS` (1.0), `_C` (0.7071), `_EXPLORE` (0.5), `_PLAYOUT_ROUNDS` (10), `_SEED` (0) |
| OEP | `TRIPLEA_ROXO_OEP_SECONDS` (1.0), `_POPULATION` (50), `_SURVIVAL` (0.5), `_MUTATION` (0.1), `_PLAYOUT_ROUNDS` (10), `_DICE_SEEDS` (1, dice mode only), `_SEED` (0) |

## Where these still differ from the thesis

- **Purchase is searched.** The thesis has no purchase.
- **Searches stay within one phase.** Each search covers one phase of the agent's turn (one
  request), as in oep_paper. Roxo's code (roxo-ne-mcts) keeps building its tree through later
  phases and the opponent's turn.
- **Combat unit sets follow Roxo's code.** They come from `getUnitsUpToStrengthSet` (a rising
  threshold, capped at 3.5x + 4), not from every winning prefix as Algorithm 4 lists them. So an
  undefended territory gets one option: every unit that can reach it.
- **Non-combat combinations are capped.** At most 243 combinations per territory. Source
  territories beyond the cap offer only all or none. The thesis map never reaches the cap.
- **The air landing is forced.** At the start of the non-combat move, air units the engine would
  destroy land first, as in oep_paper. The thesis setting has no air units.
- **The reward divides by 2.** It uses (mine − enemy) / (2 × total) as written in Algorithm 6.
  Roxo's code divides by 2.5 × total.
- **NE-MCTS's playout is fixed per search.** The thesis does not say how the playout's randomness
  is fixed. Here NE-MCTS and OEP reuse one Weak AI shuffle seed per search, and BB-MCTS draws a
  fresh one for every playout.
