# Behaviour cloning of TripleMind attack formations

TripleMind's combat-move MCTS picks a target territory heuristically, then searches over *attack
formations* (which units attack it, or skip). This folder learns to imitate that formation choice:
each decision has a variable-sized set of legal candidates, and the policy scores every candidate
and is trained with cross-entropy against the candidate MCTS chose.

```
data-processing/attack_formation_dataset.py   preprocessing: records -> validated samples
prepare_dataset.py                             preprocessing CLI + process_records()
bc_baseline/data.py                            both data modes -> identical training tensors
bc_baseline/model.py                           MLP candidate scorer, masked softmax
bc_baseline/train.py                           training, early stopping, checkpoints, overfit test
bc_baseline/evaluate.py                        metrics, baselines, plots, checkpoint evaluation
run_bc.py                                      CLI for all of the above
docs/behaviour-cloning-uml.drawio              UML (open in draw.io / diagrams.net)
```

`docs/behaviour-cloning-uml.drawio` has seven pages: an end-to-end overview (records → data
processing → training → evaluation → the tripleMind-bc agent), the data-processing class and
activity diagrams, the training/evaluation class diagram, and sequence diagrams of
`run_bc.py train` and `run_bc.py evaluate` / `sanity`, and the `CandidateScorer` neural-network architecture (layers, tensor
shapes, parameter counts). The running agent is drawn in
[tripleMind-bc/docs/tripleMind-bc-uml.drawio](../tripleMind_hybrids/tripleMind-bc/docs/tripleMind-bc-uml.drawio).

## Setup

The scripts need PyTorch (CPU is enough), NumPy 2, networkx and matplotlib. A local virtualenv:

```bash
cd planning-agent-testbed/agents/behaviour-cloning
python3 -m venv .venv
.venv/bin/pip install torch --index-url https://download.pytorch.org/whl/cpu
.venv/bin/pip install "numpy>=2" networkx matplotlib
```

`dataset.pkl` contains NumPy arrays, so it must be read with the NumPy major version that wrote
it (NumPy 2 also reads NumPy 1 pickles). Use `.venv/bin/python` below, or any environment with the
packages above.

## Usage

**Preprocess only** (records -> `data/`, no training):

```bash
.venv/bin/python run_bc.py preprocess --records ../tripleMind/records --data-dir data
```

Writes `dataset.pkl`, `feature_names.json`, `splits.json` (game-grouped train/val/test indices),
`report.json` and `report.txt` (validation report of the preprocessing assertions).

**Load and train** (uses an existing `data/`, skips preprocessing):

```bash
.venv/bin/python run_bc.py train --data-dir data --out runs/mlp-baseline
```

**Process and train** (preprocess `--records`, save to `--data-dir`, reload it to confirm the
saved files give identical inputs, then train):

```bash
.venv/bin/python run_bc.py train --records ../tripleMind/records --data-dir data --out runs/mlp-baseline
```

Training parameters are set in
[`bc_baseline/train_config.toml`](bc_baseline/train_config.toml) (learning rate, batch size,
epochs, seed, hidden sizes, dropout, weight decay, early-stopping patience and min delta, each
commented). Use another file with `--config path.toml`; a flag overrides the file for one run,
e.g. `--lr 3e-4 --hidden-dims 512 256 --epochs 50` (also `--batch-size --seed --dropout
--weight-decay --patience --min-delta`). The values used are saved in the run's `config.json`. Preprocessing options (preprocess and process
mode): `--val 0.15 --test 0.15 --split-seed 0 --game-xml PATH --strict --no-legality`.

A run directory holds `best.pt` (best validation-loss checkpoint with model config, train config,
feature names and the input fingerprint), `config.json`, `history.json` (per-epoch metrics),
`train_summary.json`, `eval.json` (metrics of `best.pt` on every split) and
`training_curves.png` (loss and accuracy per epoch).

**Evaluate a checkpoint** without retraining:

```bash
.venv/bin/python run_bc.py evaluate --checkpoint runs/mlp-baseline/best.pt --data-dir data
```

It warns if the data's input fingerprint differs from the one the checkpoint was trained on.

**Sanity checks**:

```bash
.venv/bin/python run_bc.py sanity --data-dir data
.venv/bin/python run_bc.py sanity --data-dir data --equivalence-records data-processing/records
```

| # | check | where |
|---|-------|-------|
| 1 | feature dimensions match `feature_names.json`, values finite, expert index inside its candidate set | `data.check_inputs`, every run |
| 2 | every decision has >= 1 candidate (and exactly one skip) | `data.check_inputs`, every run |
| 3 | softmax sums to 1 within each decision and is 0 on padding | `model.check_softmax`, every evaluation |
| 4 | no game (both factions' records) in two splits | `data.check_inputs`, every run |
| 5 | a dropout-free model memorises 64 multi-candidate training decisions | `sanity` |
| 6 | only the training split is ever passed to the optimiser; the standardiser is fitted on train only; evaluation runs without autograd and leaves no gradients | `train.assert_no_gradients` every epoch, `sanity` |
| 7 | process mode and load mode give the same tensors (SHA-256 fingerprint) | process-mode `train`, `sanity --equivalence-records` |

## Model

`CandidateScorer`: the 29 x 45 territory features are flattened and joined with the 3 global
features, standardised with training-split statistics, and encoded by an MLP (`--hidden-dims`).
Each candidate's 94 features go through a one-layer encoder; the state and candidate encodings are
concatenated and an MLP gives one score per candidate. Padding candidates are masked out before
the softmax. All features are relative to the acting faction, so one policy serves Russians and
Germans.

## Metrics

Per split: cross-entropy and top-1 accuracy; accuracy without single-candidate decisions (where
the only candidate is the skip); skip precision / recall / F1; accuracy by candidate-set size and
by faction. Baselines: a uniformly random legal candidate (accuracy 1/C, loss log C), always skip,
and always the generator's first candidate (it lists attacks strongest first).

## Baseline result (200 games, 13,497 decisions)

`run_bc.py train --data-dir data` with the defaults stops at epoch 14, best epoch 4:

| split | loss | acc | acc (>1 cand.) | skip F1 | random acc | always-skip acc | first-candidate acc |
|-------|------|-----|----------------|---------|------------|-----------------|---------------------|
| train | 0.525 | 80.9% | 75.7% | 90.6% | 48.5% | 25.5% | 80.8% |
| val   | 0.589 | 80.0% | 74.5% | 89.7% | 49.0% | 26.5% | 80.7% |
| test  | 0.558 | 81.4% | 76.3% | 90.1% | 49.1% | 26.0% | 81.7% |

The MLP beats the random and always-skip baselines but **not** the first-candidate rule: it
chooses candidate 0 in 97% of decisions, and MCTS chose it in 81% of them. Validation loss rises
after epoch 4 while training loss keeps falling, so the flat-feature MLP overfits rather than
learning when MCTS deviates from the strongest attack. That is the bar a GNN (or richer
candidate features) has to clear.
