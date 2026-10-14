"""Metrics, baselines, checkpoint evaluation and training plots."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from .data import SPLITS, BCInputs, SplitTensors
from .model import MASKED, CandidateScorer, check_softmax

SIZE_BUCKET_CAP = 10  # candidate-set sizes above this are grouped as "11+"


@torch.no_grad()
def predict(model: CandidateScorer, split: SplitTensors, batch_size: int = 1024) -> torch.Tensor:
    """Scores (N, C) for a split, padded with MASKED; checks softmax normalisation (check 3).
    Runs without autograd, so evaluation never produces gradients (check 6)."""
    was_training = model.training
    model.eval()
    out = torch.full(split.mask.shape, MASKED)
    for start in range(0, len(split), batch_size):
        rows = torch.arange(start, min(start + batch_size, len(split)))
        node, glob, cand, mask, _ = split.batch(rows)
        scores = model(node, glob, cand, mask)
        check_softmax(scores, mask)
        out[rows, :scores.shape[1]] = scores
    model.train(was_training)
    return out


def _prf(pred: torch.Tensor, true: torch.Tensor) -> dict:
    tp = (pred & true).sum().item()
    fp = (pred & ~true).sum().item()
    fn = (~pred & true).sum().item()
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return {"precision": p, "recall": r, "f1": 2 * p * r / (p + r) if p + r else 0.0,
            "support": int(true.sum().item())}


def metrics(split: SplitTensors, scores: torch.Tensor) -> dict:
    """Imitation metrics of `scores` (N, C) against the expert, plus the baselines."""
    n = len(split)
    if n == 0:
        return {"n": 0}
    rows = torch.arange(n)
    nc = split.n_cand
    multi = nc > 1
    pred = scores.argmax(1)
    correct = pred == split.expert
    pred_skip = split.is_skip[rows, pred]
    true_skip = split.is_skip[rows, split.expert]

    def acc(sel):
        return correct[sel].float().mean().item() if sel.any() else None

    by_size = defaultdict(lambda: torch.zeros(n, dtype=torch.bool))
    for k in nc.unique().tolist():
        by_size[str(k) if k <= SIZE_BUCKET_CAP else f"{SIZE_BUCKET_CAP + 1}+"] |= nc == k
    by_actor = defaultdict(lambda: torch.zeros(n, dtype=torch.bool))
    for i, a in enumerate(split.actors):
        by_actor[a][i] = True

    always_skip = true_skip  # the skip candidate is correct exactly when the expert skipped
    return {
        "n": n, "n_multi": int(multi.sum()),
        "loss": F.cross_entropy(scores, split.expert).item(),
        "acc": acc(torch.ones(n, dtype=torch.bool)),
        "acc_multi": acc(multi),
        "skip": _prf(pred_skip, true_skip),
        "skip_multi": _prf(pred_skip[multi], true_skip[multi]),
        "acc_by_size": {k: {"n": int(v.sum()), "acc": acc(v)}
                        for k, v in sorted(by_size.items(), key=lambda kv: _size_key(kv[0]))},
        "acc_by_actor": {a: {"n": int(v.sum()), "acc": acc(v)} for a, v in sorted(by_actor.items())},
        "baselines": {
            # A uniformly random legal candidate: expected accuracy 1/C, cross-entropy log C.
            "random": {"acc": (1.0 / nc.float()).mean().item(),
                       "acc_multi": (1.0 / nc[multi].float()).mean().item() if multi.any() else None,
                       "loss": nc.float().log().mean().item()},
            # Applicable because every decision has exactly one skip candidate (checked on load).
            "always_skip": {"acc": always_skip.float().mean().item(),
                            "acc_multi": always_skip[multi].float().mean().item() if multi.any() else None,
                            "skip": _prf(torch.ones(n, dtype=torch.bool), true_skip)},
            # The generator lists attacks strongest first, so candidate 0 is its top-ranked one.
            "first_candidate": {"acc": (split.expert == 0).float().mean().item(),
                                "acc_multi": (split.expert[multi] == 0).float().mean().item()
                                if multi.any() else None},
        },
    }


def _size_key(k: str):
    return int(k.rstrip("+")) + (0.5 if k.endswith("+") else 0)


def evaluate_model(model: CandidateScorer, inputs: BCInputs, splits=SPLITS) -> Dict[str, dict]:
    return {name: metrics(inputs.splits[name], predict(model, inputs.splits[name]))
            for name in splits if len(inputs.splits[name])}


def load_checkpoint(path: str) -> tuple:
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = CandidateScorer.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def evaluate_checkpoint(path: str, inputs: BCInputs, out_path: str = None) -> Dict[str, dict]:
    """Evaluate a saved checkpoint without retraining."""
    model, ckpt = load_checkpoint(path)
    if ckpt.get("data_fingerprint") != inputs.fingerprint:
        print(f"warning: {path} was trained on inputs {str(ckpt.get('data_fingerprint'))[:12]}, "
              f"evaluating on {inputs.fingerprint[:12]}")
    results = evaluate_model(model, inputs)
    results["checkpoint"] = {"path": str(path), "epoch": ckpt.get("epoch"),
                             "data_fingerprint": inputs.fingerprint}
    if out_path:
        Path(out_path).write_text(json.dumps(results, indent=1))
    return results


def format_results(results: Dict[str, dict]) -> str:
    def pct(x):
        return "   -  " if x is None else f"{100 * x:5.1f}%"

    lines = [f"{'split':6s} {'n':>6s} {'loss':>7s} {'acc':>7s} {'acc>1':>7s} {'skipP':>7s} "
             f"{'skipR':>7s} {'skipF1':>7s} | {'rand acc':>8s} {'rand loss':>9s} {'skip acc':>8s} "
             f"{'1st acc':>7s} {'1st>1':>7s}"]
    for name in SPLITS:
        m = results.get(name)
        if not m or not m.get("n"):
            continue
        b = m["baselines"]
        lines.append(f"{name:6s} {m['n']:6d} {m['loss']:7.4f} {pct(m['acc'])} {pct(m['acc_multi'])} "
                     f"{pct(m['skip']['precision'])} {pct(m['skip']['recall'])} {pct(m['skip']['f1'])} | "
                     f"{pct(b['random']['acc']):>8s} {b['random']['loss']:9.4f} "
                     f"{pct(b['always_skip']['acc']):>8s} {pct(b['first_candidate']['acc'])} "
                     f"{pct(b['first_candidate']['acc_multi'])}")
    for name in SPLITS:
        m = results.get(name)
        if m and m.get("n"):
            sizes = "  ".join(f"{k}:{pct(v['acc']).strip()}({v['n']})" for k, v in m["acc_by_size"].items())
            lines.append(f"{name} accuracy by candidate-set size  {sizes}")
    lines.append("acc>1 = excluding single-candidate decisions; baselines: rand = random legal "
                 "candidate, skip = always skip, 1st = always the generator's first (strongest) "
                 "candidate")
    return "\n".join(lines)


def plot_history(history: list, out_dir: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    epochs = [h["epoch"] for h in history]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for split, color in (("train", "C0"), ("val", "C1")):
        if f"{split}_loss" in history[0]:
            axes[0].plot(epochs, [h[f"{split}_loss"] for h in history], color=color, label=split)
            axes[1].plot(epochs, [h[f"{split}_acc"] for h in history], color=color, label=split)
            if history[0].get(f"{split}_acc_multi") is not None:
                axes[1].plot(epochs, [h[f"{split}_acc_multi"] for h in history], "--",
                             color=color, label=f"{split} (>1 candidate)")
    axes[0].set(title="Cross-entropy loss", xlabel="epoch")
    axes[1].set(title="Top-1 imitation accuracy", xlabel="epoch", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
        ax.legend()
    fig.tight_layout()
    path = Path(out_dir) / "training_curves.png"
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return path
