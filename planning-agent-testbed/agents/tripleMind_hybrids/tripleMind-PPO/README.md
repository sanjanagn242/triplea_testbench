# TripleMind PPO

TripleMind ([agents/tripleMind](../../tripleMind)) with one change: the attack formation for each
target is chosen by a **PPO policy learned from scratch** from complete games and their outcomes.
It uses no MCTS demonstrations, no behaviour-cloning weights and no expert labels. Everything else
is TripleMind's: the map graph, the action generators, the target heuristic and the other phases.

| Phase | TripleMind (MCTS) | TripleMind BC | TripleMind PPO |
|---|---|---|---|
| purchase | first purchase from `purchase_actions` | same | same |
| combatMove: which target | `combat_actions` heuristic (most important first) | same | same |
| combatMove: which formation | MCTS over the generator's formations (attack sets + skip) | BC policy | **PPO policy** over the same formations |
| nonCombatMove | `noncombat_move` | same | same |
| place | `{}` (engine places all purchased units) | same | same |
| battle | fight all | same | same |

The agent class `TripleMindPPO` (`ppo/agent.py`) subclasses tripleMind-bc's `TripleMindBC` and
only replaces its formation chooser. As a result, the combat-move loop is the exact code BC uses:

1. Generate the formations for the next heuristic target.
2. Let the chooser pick one (possibly the skip).
3. Apply its moves with the forward model and mark its units as committed.
4. Repeat until no target is left.

The chosen moves go through `SearchAgent`'s checking and encoding, like every other search agent.

## Layout

```
agent.py              testbench entry point (inference only: loads a checkpoint, argmax)
run-agent.sh          launcher: settings, PYTHONPATH, TripleMind's Cython build, interpreter with torch
agent.json            manifest (id tripleMind-ppo; also listed in agents/agents.json)
run_ppo.py            CLI: train, evaluate, sanity
ppo/paths.py          where the reused code lives (nothing is copied)
ppo/features.py       DecisionEncoder: BC's preprocessing functions -> one decision's features
ppo/model.py          ActorCritic (actor = BC's CandidateScorer, separate critic MLP), checkpoints
ppo/policy.py         FormationPolicy: BCPolicy's choose() interface; sample / greedy / uniform
ppo/agent.py          TripleMindPPO(TripleMindBC)
ppo/rollout.py        games in the testbench arena (tools/arena.py), opponents, parallel collector
ppo/algorithm.py      GAE, clipped PPO loss, value loss, entropy bonus, update-time checks
ppo/train.py          training loop, metrics, checkpoints, periodic evaluation
ppo/evaluate.py       evaluation games: performance and computational cost, reported separately
ppo/checks.py         sanity checks
```

`docs/tripleMind-PPO-uml.drawio` (open in draw.io / diagrams.net) has seven pages: class diagrams of
the agent and policy and of training / rollouts / evaluation, the training activity, and sequence
diagrams of one training iteration, one combat move with the PPO policy, deployment start-up and
phases, and evaluation. [../docs/tripleMind-hybrids-uml.drawio](../docs/tripleMind-hybrids-uml.drawio)
puts all TripleMind variants (MCTS, BC, PPO) side by side.

It reuses, without copying:

- TripleMind's `map_graph`, `generators` and the Cython action generators.
- `behaviour-cloning/data-processing/attack_formation_dataset.py`: candidate canonicalisation and the actor-relative node, global and candidate features.
- `behaviour-cloning/bc_baseline/model.py`: `CandidateScorer`, `mlp` and `flatten_state`.
- tripleMind-bc's `TripleMindBC` and `SearchAgent`.
- `tools/arena.py`'s `play_game`: the game loop. Its one change is that it now also accepts in-process agents next to commands.

## Model

**Actor**: behaviour cloning's `CandidateScorer`, unchanged in architecture. The 29 × 45
territory features are flattened, joined with the 3 global features and encoded by an MLP (256,
128). Each candidate's 94 features go through a one-layer encoder, and a scorer MLP gives one
score per candidate. A masked softmax runs over the decision's legal candidates, so the number of
candidates can vary. Dropout is 0, so stored and recomputed log-probabilities match exactly. The
last layer starts at 1% scale, which makes the initial policy almost uniform.

**Critic**: a separate MLP (256, 128 → 1) over the same flattened territory and global features.
It sees no candidate features. It estimates the deciding faction's expected return from this
decision on.

**Normalisation**: BC fits its standardiser on a training split, and PPO has none. Instead:

- A short warm-up of random-formation self-play games gives the initial feature statistics.
- Running statistics are then updated after each of the first `--norm-update-iters` iterations and frozen after that.
- The statistics live in the model's buffers and in the checkpoint.

All features are relative to the deciding faction, so **one shared policy plays Russians and
Germans**.

## Training

One iteration works like this:

1. Snapshot the actor-critic (a frozen deep copy, no gradients).
2. Play `--games-per-iter` complete games in the testbench arena with that snapshot, in `--workers` processes. The policy samples from its softmax during these games.
3. Record every formation decision of a learning faction as one step: features, legal candidates, the chosen index, log π_old, V_old, the faction and the round.
4. Assign rewards when the game ends. Each faction's trajectory is its own decisions in order, several per turn, with the opponent's turns in between as part of the environment. Its last decision gets the faction's outcome: **+1 win, −1 loss, 0 no winner at the round limit**. That last decision is terminal; all other rewards are 0.
5. Compute GAE (γ = 0.995 per decision, λ = 0.95) per trajectory. Values and rewards of the other faction never enter a trajectory.
6. Run PPO epochs over minibatches:
   - clipped surrogate (ε = 0.2), value MSE (coefficient 0.5) and entropy bonus (0.01);
   - Adam (lr 3e-4), gradient clipping at 0.5, normalised advantages;
   - stop early when the approximate KL exceeds 1.5 × 0.03.
7. Log metrics and save checkpoints. Every `--eval-every` iterations, evaluate the greedy policy against `--eval-opponents`.

**Opponents** (`--opponents`, rotated over an iteration's games):

| Opponent | Meaning |
|---|---|
| `self` | Self-play: both factions are the learner (same snapshot); both trajectories are trained on |
| `past` | A random earlier `iter_NNNN.pt` of this run, frozen and greedy (self-play until one exists) |
| `heuristic` | The arena's built-in `HeuristicPolicy` (triplea_fm baseline), no agent process |
| `first` | TripleMind with the generator's first (strongest) formation |
| `random` | TripleMind with a uniformly random legal formation |
| `mcts` | TripleMind MCTS (`agents/tripleMind/agent.py`) as a subprocess. Recording is off, and its `metrics/` CSV goes to the run directory. It is not started through its `run-agent.sh`, which forces `TRIPLEMIND_RECORD=1` and would overwrite the recorded games BC trains on |
| `bc` | tripleMind-bc via its `run-agent.sh` |
| `ppo:PATH` | A frozen PPO checkpoint, greedy |
| `cmd:PATH` | Any agent launcher, run with bash |

Against a fixed opponent the learner alternates sides game by game. Only the learner's decisions
are trained on.

## Setup

Use the behaviour-cloning virtualenv (PyTorch CPU, NumPy 2, networkx, matplotlib); see the
[behaviour-cloning README](../../behaviour-cloning/README.md). `run-agent.sh` finds it
automatically. In the commands below, `py` stands for
`../../behaviour-cloning/.venv/bin/python`, run from this folder.

## Usage

**Train from scratch** (the defaults write `runs/ppo-selfplay`, the agent's default checkpoint):

```bash
py run_ppo.py train --out runs/ppo-selfplay --iterations 150 --games-per-iter 64 --workers 16 \
   --opponents self past heuristic first \
   --eval-every 15 --eval-games 20 --eval-opponents heuristic first random mcts
```

**Continue training** (model, optimizer, normalisation statistics and counters are restored):

```bash
py run_ppo.py train --out runs/ppo-selfplay --resume runs/ppo-selfplay/checkpoints/last.pt --iterations 300
```

All PPO hyperparameters are flags: `--lr --gamma --gae-lambda --clip --value-coef --entropy-coef
--epochs --minibatch-size --max-grad-norm --target-kl --value-clip --actor-hidden --critic-hidden`.
Other options are `--norm-warmup-games --norm-update-iters --save-every --max-rounds
--mcts-seconds --seed --xml`.

A run directory holds:

| File | Content |
|---|---|
| `config.json` | Training, PPO and model configuration |
| `metrics.jsonl`, `metrics.csv` | One row per iteration: policy / value loss, entropy, approximate KL, clip fraction, explained variance, gradient norm, parameter change, log-probability check, episode return, win and loss rate per opponent, rollout winner by side, decisions, candidates, skip rate, and collect / update seconds |
| `eval.jsonl` | Each evaluation report (performance and cost per opponent) |
| `checkpoints/last.pt` | The latest iteration |
| `checkpoints/best.pt` | Best mean evaluation score (win = 1, draw = ½) |
| `checkpoints/iter_NNNN.pt` | Every `--save-every` iterations; also the `past` opponents |
| `training_curves.png` | Training curves |
| `train.log` | If you redirect the output there |

**Evaluate** a checkpoint with complete greedy games, no updates and sides alternating:

```bash
py run_ppo.py evaluate --checkpoint runs/ppo-selfplay/checkpoints/best.pt \
   --opponents heuristic first random mcts bc ppo:runs/ppo-selfplay/checkpoints/iter_0030.pt \
   --games 40 --workers 16 --out runs/eval/best.json
# The same games for TripleMind's own first-formation rule, or a random formation, as baselines
py run_ppo.py evaluate --chooser first  --opponents heuristic random mcts --games 40 --workers 16
py run_ppo.py evaluate --chooser random --opponents heuristic first mcts --games 40 --workers 16
```

The report has two separate tables:

- **Game performance**: wins, losses and draws, win %, score, mean rounds, and wins by side.
- **Computational cost**: seconds per game, policy decisions per game, policy milliseconds per decision (feature encoding plus forward pass), moves applied, and moves rejected.

Training metrics stay in the run directory.

**Sanity checks**:

```bash
py run_ppo.py sanity          # writes runs/sanity/sanity.json; exit status 1 if a check fails
```

| # | Check | Where |
|---|---|---|
| 1 | GAE on interleaved two-faction games uses each faction's own rewards and values with the right sign; a trajectory mixing factions is rejected | `checks.check_gae_perspective`, `algorithm.build_batch` every update |
| 2 | Probabilities sum to 1 per decision, are 0 on padding, and padding does not change real scores | `checks.check_softmax_masking`; `policy.py` checks every decision |
| 3 | Every chosen action is one of the decision's generated legal candidates, for its target, and not an end-phase | `policy.py`, every decision (AssertionError) |
| 4 | Stored and recomputed log-probabilities agree (≤ 1e-4) before the update | `algorithm.check_log_probs`, every update |
| 5 | Rewards come from each faction's own perspective: winner +1, loser −1, zero-sum in self-play, last decision only | `rollout.check_episode`, every game |
| 6 | Losses and gradients are finite; the update changes actor and critic parameters; the snapshot stays unchanged | `algorithm.ppo_update` every update, `checks.check_rollouts_and_update` |
| 7 | Evaluation leaves no gradients and does not change parameters | `evaluate.evaluate`, `checks` |
| 8 | Checkpoint save/load gives identical outputs | `checks` |
| 9 | Fed the same observations through a full game, TripleMind PPO and the original TripleMind agent answer every purchase, nonCombatMove and place request identically | `checks.check_noncombat_parity` |

Training never falls back silently. If the policy fails in a rollout game, training stops with
the error. The deployed agent instead logs the error and plays the generator's first formation,
as tripleMind-bc does.

## First result (Capture the Flag, `runs/ppo-selfplay`)

The training command above was run with seed 0: 150 iterations and 9,600 games, about 12 minutes
on 16 workers including the evaluations. The entropy on multi-candidate decisions fell from 0.83
to about 0.08, and the critic's explained variance rose to about 0.8. `best.pt` is from
iteration 105.

Because `best.pt` was selected on the training-time evaluations, it was re-evaluated on fresh
seeds (`--seed 1`, 40 games per opponent, greedy, sides alternating, MCTS at 0.5 s per decision).
The results are in `runs/ppo-selfplay/heldout-eval/`:

| Opponent | PPO `best.pt` (W/L/D) | TripleMind first-formation baseline (W/L/D) |
|---|---|---|
| heuristic (`HeuristicPolicy`) | 0 / 40 / 0 | 0 / 40 / 0 |
| first | 40 / 0 / 0 | n/a |
| random | 16 / 3 / 21 | 28 / 8 / 4 |
| mcts (TripleMind MCTS) | 40 / 0 / 0 | 29 / 11 / 0 |
| bc (tripleMind-bc) | 40 / 0 / 0 | 20 / 0 / 20 |
| ppo: `iter_0030.pt` | 0 / 0 / 40 | 0 / 40 / 0 |

Caveats:

- Greedy play against deterministic opponents gives near-identical games, so 40 wins against `first`, MCTS or BC can be few distinct games repeated.
- The training-time MCTS win rate swung between 40% and 100% across checkpoints.
- No TripleMind variant wins against the arena's `HeuristicPolicy`, which captures the flag within 10 rounds. Better formations do not make up for TripleMind's purchase and non-combat phases, which this agent leaves unchanged.
- Against `random`, PPO draws more often at the round limit than the baseline does.
- Self-play has a side asymmetry: Russians won about 60–65% of the rollout games.

## Deployment in the testbench

```bash
# From planning-agent-testbed/: PPO (Russians) against TripleMind MCTS (Germans)
python3 tools/arena.py --xml games/capture-the-flag-diag/map/capture_the_flag/map/games/capture_the_flag.xml \
  --agent Russians=agents/tripleMind_hybrids/tripleMind-PPO/run-agent.sh \
  --agent Germans=agents/tripleMind/run-agent.sh --games 2
```

In the Java testbench, assign `tripleMind-ppo` to a faction in a game profile and run it from
**Planning Agent Testbench → Run Simulations**. The testbench finds `agent.json` in this nested
folder.

| Variable | Default | Meaning |
|---|---|---|
| `TRIPLEMIND_PPO_CHECKPOINT` | `runs/ppo-selfplay/checkpoints/best.pt` | Policy checkpoint |
| `TRIPLEMIND_PPO_MODE` | `greedy` | `greedy` (argmax) or `sample` |
| `TRIPLEMIND_PPO_THREADS` | 1 | PyTorch threads |
| `TRIPLEMIND_PPO_PYTHON` | (unset) | Interpreter; else `behaviour-cloning/.venv/bin/python`, else `TRIPLEA_TESTBENCH_PYTHON` / `python3` |
| `TRIPLEMIND_VIEW` | 0 | 1 shows TripleMind's live map window |
| `TRIPLEMIND_PURE_PYTHON` | 0 | 1 uses TripleMind's Python generators |
| `TRIPLEA_AGENT_SEED` | 0 | Seed of the agent's randomness |
| `TRIPLEA_AGENT_LOG` | 0 | 1 logs each decision (target, chosen candidate, its probability) to stderr and `logs/` |

The policy is tied to the map it was trained on. The checkpoint stores its feature names, and
the agent compares them with the game's at the first combat move. On another map, or with more
than one opponent, the agent logs that the policy is disabled and plays the generator's first
formation instead.
