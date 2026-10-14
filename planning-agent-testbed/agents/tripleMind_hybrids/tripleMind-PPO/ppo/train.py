"""PPO training loop: collect complete games with a frozen snapshot, then update.

  0. (new runs) normalisation warm-up: a few self-play games with uniformly random formations
     give the initial input statistics; no update.
  per iteration:
  1. snapshot the current actor-critic and play `games_per_iter` games with it (sampling);
     opponents rotate through `opponents`;
  2. per (game, faction) trajectory: terminal reward = the faction's outcome, GAE advantages;
  3. PPO epochs over the batch (log-probability check first, see algorithm.py);
  4. for the first `norm_update_iters` iterations, fold the batch into the running input
     statistics (they then stay fixed, so the policy only changes through gradient steps);
  5. log the training metrics, save checkpoints, and every `eval_every` iterations evaluate the
     greedy policy against `eval_opponents` (best.pt = best mean evaluation score).

Run directory: config.json, metrics.jsonl / metrics.csv (one row per iteration), eval.jsonl
(one evaluation report per line), checkpoints/{last,best,iter_NNNN}.pt, training_curves.png.
"""

from __future__ import annotations

import csv
import json
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from . import paths
from .algorithm import PPOConfig, build_batch, ppo_update
from .evaluate import evaluate, format_report
from .features import game_feature_names
from .model import RunningMeanStd, load_checkpoint, new_model, save_checkpoint
from .rollout import (Collector, EpisodeResult, GameSpec, RolloutContext, check_opponent,
                      factions)


@dataclass
class TrainConfig:
    iterations: int = 100
    games_per_iter: int = 32
    workers: int = 4
    opponents: List[str] = field(default_factory=lambda: ["self"])
    seed: int = 0
    xml: str = str(paths.DEFAULT_XML)
    max_rounds: int = 100
    mcts_seconds: float = 0.5
    actor_hidden: List[int] = field(default_factory=lambda: [256, 128])
    critic_hidden: List[int] = field(default_factory=lambda: [256, 128])
    norm_warmup_games: int = 16
    norm_update_iters: int = 20
    eval_every: int = 10
    eval_games: int = 20
    eval_opponents: List[str] = field(default_factory=lambda: ["heuristic", "first"])
    save_every: int = 10


class Normalizer:
    """Running statistics of the flattened state and of the candidate features."""

    def __init__(self, names: dict):
        state_dim = len(names["territories"]) * len(names["node"]) + len(names["global"])
        self.state = RunningMeanStd(state_dim)
        self.cand = RunningMeanStd(len(names["candidate"]))

    def update(self, results: Sequence[EpisodeResult]):
        steps = [t for r in results for traj in r.trajectories.values() for t in traj]
        if not steps:
            return
        self.state.update(torch.from_numpy(np.stack([np.concatenate([t.node.ravel(), t.glob])
                                                      for t in steps])))
        self.cand.update(torch.from_numpy(np.concatenate([t.cand for t in steps])))

    def state_dict(self):
        return {"state": self.state.state_dict(), "cand": self.cand.state_dict()}

    def load_state_dict(self, s):
        self.state.load_state_dict(s["state"])
        self.cand.load_state_dict(s["cand"])


def rollout_stats(results: Sequence[EpisodeResult], sides: Sequence[str]) -> Dict[str, float]:
    """Game-level training metrics of one iteration's rollouts."""
    row: Dict[str, float] = {}
    outcomes = [o for r in results for o in r.outcomes.values()]
    row["episode_return_mean"] = float(np.mean(outcomes))
    row["rounds_mean"] = float(np.mean([r.rounds for r in results]))
    row["no_winner_rate"] = float(np.mean([r.winner is None for r in results]))
    for side in sides:
        row[f"winner_{side}_rate"] = float(np.mean([r.winner == side for r in results]))
    vs_fixed = [r for r in results if len(r.spec.learners) == 1]  # self-play: winner_<side>_rate
    for label in dict.fromkeys(r.spec.label for r in vs_fixed):
        rs = [r for r in vs_fixed if r.spec.label == label]
        o = np.array([next(iter(r.outcomes.values())) for r in rs])
        row[f"win_rate/{label}"] = float((o > 0).mean())
        row[f"loss_rate/{label}"] = float((o < 0).mean())
    trajs = [t for r in results for t in r.trajectories.values()]
    steps = [s for t in trajs for s in t]
    row["decisions"] = len(steps)
    row["decisions_per_trajectory"] = len(steps) / max(len(trajs), 1)
    row["candidates_mean"] = float(np.mean([s.n_cand for s in steps])) if steps else 0.0
    row["single_candidate_frac"] = float(np.mean([s.n_cand == 1 for s in steps])) if steps else 0.0
    row["skip_rate"] = float(np.mean([s.action == s.n_cand - 1 for s in steps])) if steps else 0.0
    decisions = sum(r.policy_decisions for r in results)
    row["policy_ms_per_decision"] = 1000 * sum(r.policy_seconds for r in results) / max(decisions, 1)
    row["game_seconds_mean"] = float(np.mean([r.seconds for r in results]))
    return row


class MetricsWriter:
    def __init__(self, out: Path):
        self.jsonl, self.csv = out / "metrics.jsonl", out / "metrics.csv"
        self.fields: Optional[List[str]] = None
        if self.csv.exists():
            with open(self.csv, newline="") as f:
                self.fields = next(csv.reader(f), None)

    def write(self, row: dict):
        with open(self.jsonl, "a") as f:
            f.write(json.dumps(row) + "\n")
        new = self.fields is None
        if new:
            self.fields = list(row)
        with open(self.csv, "a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=self.fields, extrasaction="ignore")
            if new:
                w.writeheader()
            w.writerow(row)


def _specs(cfg: TrainConfig, first_game: int, sides, ckpt_dir: Path, rnd: random.Random):
    specs = []
    past = sorted(ckpt_dir.glob("iter_*.pt"))
    for g in range(cfg.games_per_iter):
        n = first_game + g
        label = cfg.opponents[g % len(cfg.opponents)]
        opp = label
        if label == "past":  # a frozen earlier snapshot of this run (self-play before there is one)
            opp = f"ppo:{rnd.choice(past)}" if past else "self"
        learners = tuple(sides) if opp == "self" else (sides[n % 2],)
        specs.append(GameSpec(n, cfg.seed * 1_000_003 + n, learners, opp, "sample", True, label))
    return specs


def train(cfg: TrainConfig, ppo: PPOConfig, out_dir: str, resume: Optional[str] = None,
          verbose: bool = True) -> dict:
    out = Path(out_dir)
    ckpt_dir = out / "checkpoints"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    for opp in cfg.opponents:
        if opp != "past":
            check_opponent(opp)
    for opp in cfg.eval_opponents:
        check_opponent(opp)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    torch.manual_seed(cfg.seed)
    names = game_feature_names(cfg.xml)
    sides = factions(cfg.xml)
    norm = Normalizer(names)
    start_it, games_played, best_score = 0, 0, -float("inf")
    if resume:
        model, ckpt = load_checkpoint(resume)
        if ckpt["feature_names"] != names:
            raise ValueError(f"{resume} was trained on another game's features")
        if ckpt.get("norm"):
            norm.load_state_dict(ckpt["norm"])
        start_it, games_played = ckpt.get("iteration", 0), ckpt.get("games", 0)
        best_score = ckpt.get("best_score", best_score)
    else:
        model = new_model(names, cfg.actor_hidden, cfg.critic_hidden)
    opt = torch.optim.Adam(model.parameters(), lr=ppo.lr, eps=1e-5)
    if resume and ckpt.get("optimizer_state"):
        opt.load_state_dict(ckpt["optimizer_state"])
    gen = torch.Generator().manual_seed(cfg.seed + start_it)
    rnd = random.Random(cfg.seed + start_it)
    (out / "config.json").write_text(json.dumps({
        "train": asdict(cfg), "ppo": ppo.to_dict(), "model": model.config, "resumed_from": resume,
        "feature_shapes": {k: len(v) for k, v in names.items()}}, indent=1))
    ctx = RolloutContext(cfg.xml, cfg.max_rounds, cfg.mcts_seconds, work_dir=str(out))
    collector = Collector(ctx, cfg.workers)
    metrics = MetricsWriter(out)

    def checkpoint(path, it, **extra):
        save_checkpoint(path, model, names, norm=norm.state_dict(), optimizer=opt, iteration=it,
                        games=games_played, train_config=asdict(cfg), ppo_config=ppo.to_dict(),
                        best_score=best_score, **extra)

    try:
        if not resume and cfg.norm_warmup_games > 0:
            t0 = time.time()
            specs = [GameSpec(n + 1, cfg.seed * 1_000_003 + 900_000 + n, tuple(sides), "self",
                              "uniform", True, "warmup") for n in range(cfg.norm_warmup_games)]
            warm = collector.run(None, names, specs)
            norm.update(warm)
            model.set_normalization(norm.state, norm.cand)
            if verbose:
                print(f"normalisation warm-up: {len(warm)} random self-play games, "
                      f"{norm.state.count} decisions, {time.time() - t0:.1f}s", flush=True)
        for it in range(start_it + 1, cfg.iterations + 1):
            t0 = time.time()
            specs = _specs(cfg, games_played + 1, sides, ckpt_dir, rnd)
            results = collector.run(model, names, specs)  # frozen snapshot inside
            games_played += len(results)
            t1 = time.time()
            batch = build_batch([t for r in results for t in r.trajectories.values()], ppo)
            stats = ppo_update(model, opt, batch, ppo, gen)
            t2 = time.time()
            if it <= cfg.norm_update_iters:
                norm.update(results)
                model.set_normalization(norm.state, norm.cand)
            row = {"iteration": it, "games_total": games_played, **rollout_stats(results, sides),
                   **stats, "collect_seconds": t1 - t0, "update_seconds": t2 - t1}
            metrics.write(row)
            if verbose:
                wr = "  ".join(f"{k.split('/')[1]} {100 * v:.0f}%" for k, v in row.items()
                               if k.startswith("win_rate/"))
                print(f"it {it:4d}  games {games_played:6d}  dec {row['decisions']:5d}  "
                      f"ret {row['episode_return_mean']:+.2f}  {wr}  "
                      f"pi {stats['policy_loss']:+.4f}  v {stats['value_loss']:.4f}  "
                      f"H {stats['entropy_multi']:.3f}  kl {stats['approx_kl']:.4f}  "
                      f"clip {stats['clip_frac']:.2f}  ev {stats['explained_variance']:+.2f}  "
                      f"[{t1 - t0:.0f}s + {t2 - t1:.1f}s]", flush=True)
            if cfg.save_every and it % cfg.save_every == 0:
                checkpoint(ckpt_dir / f"iter_{it:04d}.pt", it)
            if cfg.eval_every and (it % cfg.eval_every == 0 or it == cfg.iterations):
                report = evaluate(model, names, cfg.eval_opponents, cfg.eval_games, collector, cfg.seed)
                score = float(np.mean([v["performance"]["score"] for k, v in report.items()
                                       if not k.startswith("_")]))
                with open(out / "eval.jsonl", "a") as f:
                    f.write(json.dumps({"iteration": it, "games_total": games_played,
                                        "mean_score": score, "report": report}) + "\n")
                if verbose:
                    print(format_report(report), flush=True)
                if score > best_score:
                    best_score = score
                    checkpoint(ckpt_dir / "best.pt", it, eval_report=report)
                    if verbose:
                        print(f"  new best mean score {score:.3f} -> checkpoints/best.pt", flush=True)
            checkpoint(ckpt_dir / "last.pt", it)
    finally:
        collector.close()
    if not (ckpt_dir / "best.pt").exists() and (ckpt_dir / "last.pt").exists():
        (ckpt_dir / "best.pt").write_bytes((ckpt_dir / "last.pt").read_bytes())
    try:
        plot_metrics(out)
    except Exception as e:  # plotting is optional (matplotlib)
        print(f"no training curves: {e!r}")
    return {"iterations": cfg.iterations, "games": games_played, "best_score": best_score,
            "checkpoint": str(ckpt_dir / "best.pt")}


def plot_metrics(out: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rows = [json.loads(l) for l in (out / "metrics.jsonl").read_text().splitlines() if l.strip()]
    evals = [json.loads(l) for l in (out / "eval.jsonl").read_text().splitlines() if l.strip()] \
        if (out / "eval.jsonl").exists() else []
    it = [r["iteration"] for r in rows]
    panels = [("Policy loss", ["policy_loss"]), ("Value loss", ["value_loss"]),
              ("Entropy (multi-candidate decisions)", ["entropy_multi"]),
              ("Approx. KL / clip fraction", ["approx_kl", "clip_frac"]),
              ("Episode return (rollouts)", ["episode_return_mean"]),
              ("Rollout win rate by opponent", sorted({k for r in rows for k in r if k.startswith("win_rate/")}))]
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5))
    axes = axes.ravel()
    for ax, (title, keys) in zip(axes, panels):
        for k in keys:
            ax.plot(it, [r.get(k, np.nan) for r in rows], label=k)
        ax.set(title=title, xlabel="iteration")
        if len(keys) > 1:
            ax.legend(fontsize=7)
    ax = axes[6]
    for side in sorted({k for r in rows for k in r if k.startswith("winner_")}):
        ax.plot(it, [r.get(side, np.nan) for r in rows], label=side)
    ax.set(title="Winner of rollout games", xlabel="iteration", ylim=(0, 1))
    ax.legend(fontsize=7)
    ax = axes[7]
    if evals:
        opps = [k for k in evals[0]["report"] if not k.startswith("_")]
        for o in opps:
            ax.plot([e["iteration"] for e in evals],
                    [e["report"].get(o, {}).get("performance", {}).get("win_rate", np.nan) for e in evals],
                    marker="o", label=o[-30:])
        ax.legend(fontsize=7)
    ax.set(title="Evaluation win rate (greedy)", xlabel="iteration", ylim=(0, 1))
    for ax in axes:
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "training_curves.png", dpi=110)
    plt.close(fig)
