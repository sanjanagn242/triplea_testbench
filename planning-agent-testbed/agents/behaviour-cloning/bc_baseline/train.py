"""Behaviour-cloning training: cross-entropy over each decision's candidates, Adam, early stopping
on validation loss, best checkpoint + config + history."""

from __future__ import annotations

import json
import random
import time
import tomllib
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
import torch.nn.functional as F

from .data import BCInputs, SplitTensors
from .evaluate import metrics, plot_history, predict
from .model import CandidateScorer, check_softmax


@dataclass
class TrainConfig:
    lr: float = 1e-3
    batch_size: int = 256
    epochs: int = 100
    seed: int = 0
    hidden_dims: List[int] = field(default_factory=lambda: [256, 128])
    dropout: float = 0.1
    weight_decay: float = 0.0
    patience: int = 10          # epochs without a validation-loss improvement before stopping
    min_delta: float = 1e-4


DEFAULT_CONFIG = Path(__file__).with_name("train_config.toml")


def load_train_config(path=DEFAULT_CONFIG, **overrides) -> TrainConfig:
    """TrainConfig from the [train] table of a TOML file; `overrides` that are not None win
    (command-line flags). Unknown keys and wrong types are errors."""
    with open(path, "rb") as f:
        values = tomllib.load(f).get("train", {})
    known = {f.name: f for f in fields(TrainConfig)}
    unknown = sorted(set(values) - set(known))
    if unknown:
        raise ValueError(f"{path}: unknown training parameters {unknown}; known: {sorted(known)}")
    values.update({k: v for k, v in overrides.items() if v is not None})
    cfg = TrainConfig(**values)
    for name in ("lr", "dropout", "weight_decay", "min_delta"):
        setattr(cfg, name, float(getattr(cfg, name)))
    for name in ("batch_size", "epochs", "seed", "patience"):
        if not isinstance(getattr(cfg, name), int) or isinstance(getattr(cfg, name), bool):
            raise ValueError(f"{path}: {name} must be an integer, got {getattr(cfg, name)!r}")
    if not cfg.hidden_dims or not all(isinstance(h, int) and h > 0 for h in cfg.hidden_dims):
        raise ValueError(f"{path}: hidden_dims must be a non-empty list of positive integers")
    if cfg.lr <= 0 or cfg.batch_size < 1 or cfg.epochs < 1 or cfg.patience < 1 \
            or not 0 <= cfg.dropout < 1:
        raise ValueError(f"{path}: out-of-range value in {asdict(cfg)}")
    return cfg


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def new_model(inputs: BCInputs, cfg: TrainConfig) -> CandidateScorer:
    model = CandidateScorer(inputs.node_shape, inputs.glob_dim, inputs.cand_dim, cfg.hidden_dims,
                            cfg.dropout)
    tr = inputs.splits["train"]
    model.norm.fit(tr.node, tr.glob, tr.cand, tr.mask)  # statistics of the training split only
    return model


def train_epoch(model, opt, split: SplitTensors, cfg: TrainConfig, gen: torch.Generator) -> float:
    """One pass over `split` (only ever the training split). Returns the mean training loss."""
    model.train()
    order = torch.randperm(len(split), generator=gen)
    total = 0.0
    for start in range(0, len(split), cfg.batch_size):
        rows = order[start:start + cfg.batch_size]
        node, glob, cand, mask, expert = split.batch(rows)
        scores = model(node, glob, cand, mask)
        loss = F.cross_entropy(scores, expert)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        total += loss.item() * len(rows)
    return total / len(split)


def assert_no_gradients(model):
    """Sanity check 6: after evaluation no parameter holds a gradient."""
    leaked = [n for n, p in model.named_parameters() if p.grad is not None]
    if leaked:
        raise AssertionError(f"evaluation produced gradients in {leaked[:3]}")


def train(inputs: BCInputs, cfg: TrainConfig, out_dir: str, verbose: bool = True) -> dict:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    set_seed(cfg.seed)
    gen = torch.Generator().manual_seed(cfg.seed)
    model = new_model(inputs, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)
    train_split, val_split = inputs.splits["train"], inputs.splits["val"]
    monitor = "val" if len(val_split) else "train"
    if monitor == "train":
        print("warning: empty validation split; early stopping monitors the training loss")

    # Check 3 once up front on an untrained batch, too.
    node, glob, cand, mask, _ = train_split.batch(torch.arange(min(64, len(train_split))))
    with torch.no_grad():
        check_softmax(model(node, glob, cand, mask), mask)

    (out / "config.json").write_text(json.dumps({
        "train": asdict(cfg), "model": model.config, "data_source": inputs.source,
        "data_fingerprint": inputs.fingerprint,
        "split_sizes": {k: len(v) for k, v in inputs.splits.items()}}, indent=1))

    history, best, best_epoch, stale = [], float("inf"), None, 0
    for epoch in range(1, cfg.epochs + 1):
        t0 = time.time()
        running = train_epoch(model, opt, train_split, cfg, gen)
        opt.zero_grad(set_to_none=True)
        row = {"epoch": epoch, "train_running_loss": running}
        for name, split in (("train", train_split), ("val", val_split)):
            if len(split):
                m = metrics(split, predict(model, split))
                row.update({f"{name}_loss": m["loss"], f"{name}_acc": m["acc"],
                            f"{name}_acc_multi": m["acc_multi"]})
        assert_no_gradients(model)
        row["seconds"] = time.time() - t0
        history.append(row)
        current = row[f"{monitor}_loss"]
        improved = current < best - cfg.min_delta
        if improved:
            best, best_epoch, stale = current, epoch, 0
            torch.save({"model_state": model.state_dict(), "model_config": model.config,
                        "train_config": asdict(cfg), "epoch": epoch, f"{monitor}_loss": current,
                        "data_fingerprint": inputs.fingerprint,
                        "feature_names": inputs.feature_names}, out / "best.pt")
        else:
            stale += 1
        if verbose:
            val = (f"val loss {row['val_loss']:.4f} acc {row['val_acc']:.3f} "
                   f"(>1: {row['val_acc_multi']:.3f})" if "val_loss" in row else "")
            print(f"epoch {epoch:3d}  train loss {row['train_loss']:.4f} acc {row['train_acc']:.3f}  "
                  f"{val}  {'*' if improved else ''}", flush=True)
        if stale >= cfg.patience:
            if verbose:
                print(f"early stop: no {monitor}-loss improvement for {cfg.patience} epochs")
            break

    (out / "history.json").write_text(json.dumps(history, indent=1))
    plot_history(history, out_dir)
    summary = {"best_epoch": best_epoch, f"best_{monitor}_loss": best, "epochs_run": len(history),
               "checkpoint": str(out / "best.pt")}
    (out / "train_summary.json").write_text(json.dumps(summary, indent=1))
    return summary


def overfit_test(inputs: BCInputs, n: int = 64, steps: int = 500, lr: float = 3e-3,
                 seed: int = 0, threshold: float = 0.95) -> dict:
    """Sanity check 5: a dropout-free model must memorise a small subset of multi-candidate
    training decisions."""
    set_seed(seed)
    tr = inputs.splits["train"]
    rows = torch.nonzero(tr.n_cand > 1).squeeze(1)[:n]
    cfg = TrainConfig(lr=lr, dropout=0.0, seed=seed)
    model = new_model(inputs, cfg)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    node, glob, cand, mask, expert = tr.batch(rows)
    for _ in range(steps):
        loss = F.cross_entropy(model(node, glob, cand, mask), expert)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
    model.eval()
    with torch.no_grad():
        scores = model(node, glob, cand, mask)
        acc = (scores.argmax(1) == expert).float().mean().item()
        final = F.cross_entropy(scores, expert).item()
    # Identical inputs with different expert labels cannot all be fitted; report them.
    keys = {}
    for i in range(len(rows)):
        k = (node[i].numpy().tobytes(), glob[i].numpy().tobytes(),
             cand[i][mask[i]].numpy().tobytes())
        keys.setdefault(k, set()).add(int(expert[i]))
    conflicts = sum(1 for v in keys.values() if len(v) > 1)
    return {"samples": len(rows), "steps": steps, "accuracy": acc, "loss": final,
            "conflicting_duplicates": conflicts, "passed": acc >= threshold}
