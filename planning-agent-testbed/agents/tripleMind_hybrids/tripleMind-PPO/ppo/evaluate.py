"""Evaluation: complete games of a fixed policy against fixed opponents, no gradient updates.

The policy plays greedily (argmax) and alternates sides game by game. The report keeps game
performance (wins, losses, draws, rounds) separate from computational cost (wall time per game,
policy time per decision); training metrics are reported by train.py.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np

from .model import ActorCritic, load_checkpoint
from .rollout import Collector, EpisodeResult, GameSpec, RolloutContext, check_opponent, factions

EVAL_SEED_OFFSET = 1_000_000  # evaluation games never reuse training seeds


def eval_specs(opponents: Sequence[str], games: int, xml: str, seed: int, mode: str = "greedy",
               first_game: int = 1) -> List[GameSpec]:
    sides = factions(xml)
    specs, n = [], first_game
    for opp in opponents:
        check_opponent(opp)
        if opp == "self":
            raise ValueError("'self' is a training opponent; evaluate against a frozen ppo:PATH")
        for g in range(games):
            specs.append(GameSpec(n, EVAL_SEED_OFFSET + seed * 10_007 + n, (sides[g % 2],), opp,
                                  mode=mode, record=False, label=opp))
            n += 1
    return specs


def summarise(results: Sequence[EpisodeResult], wall_seconds: float) -> Dict[str, dict]:
    out = {}
    for label in dict.fromkeys(r.spec.label for r in results):
        rs = [r for r in results if r.spec.label == label]
        outcome = np.array([next(iter(r.outcomes.values())) for r in rs])
        by_side = {}
        for r in rs:
            side = r.spec.learners[0]
            s = by_side.setdefault(side, {"games": 0, "wins": 0, "losses": 0, "draws": 0})
            o = r.outcomes[side]
            s["games"] += 1
            s["wins" if o > 0 else "losses" if o < 0 else "draws"] += 1
        decisions = sum(r.policy_decisions for r in rs)
        n = len(rs)
        out[label] = {
            "performance": {
                "games": n, "wins": int((outcome > 0).sum()), "losses": int((outcome < 0).sum()),
                "draws": int((outcome == 0).sum()), "win_rate": float((outcome > 0).mean()),
                "draw_rate": float((outcome == 0).mean()),
                "score": float(((outcome > 0) + 0.5 * (outcome == 0)).mean()),  # draw = half a win
                "rounds_mean": float(np.mean([r.rounds for r in rs])), "by_side": by_side},
            "cost": {
                "game_seconds_mean": float(np.mean([r.seconds for r in rs])),
                "game_seconds_total": float(sum(r.seconds for r in rs)),
                "policy_decisions_per_game": decisions / n,
                "policy_ms_per_decision": 1000 * sum(r.policy_seconds for r in rs) / decisions
                if decisions else 0.0,
                "moves_applied_per_game": sum(r.applied for r in rs) / n,
                "moves_rejected_per_game": sum(r.rejected for r in rs) / n},
        }
    out["_wall_seconds"] = wall_seconds
    return out


def evaluate(model: Optional[ActorCritic], names: Optional[dict], opponents: Sequence[str],
             games: int, collector: Collector, seed: int = 0, mode: str = "greedy") -> Dict[str, dict]:
    """Play `games` games against each opponent; the model's parameters are never touched."""
    if model is not None and any(p.grad is not None for p in model.parameters()):
        model.zero_grad(set_to_none=True)
    specs = eval_specs(opponents, games, collector.ctx.xml, seed, mode)
    started = time.time()
    results = collector.run(model, names, specs)
    report = summarise(results, time.time() - started)
    if model is not None and any(p.grad is not None for p in model.parameters()):
        raise AssertionError("evaluation produced gradients")
    return report


def evaluate_checkpoint(checkpoint: Optional[str], opponents: Sequence[str], games: int,
                        ctx: RolloutContext, workers: int = 1, seed: int = 0,
                        chooser: str = "ppo", out: Optional[str] = None) -> Dict[str, dict]:
    """chooser "ppo" evaluates `checkpoint`; "first" and "random" evaluate TripleMind's own
    first-formation rule and a random formation under the same conditions (baselines)."""
    model = names = None
    if chooser == "ppo":
        model, ckpt = load_checkpoint(checkpoint)
        names = ckpt["feature_names"]
    mode = {"ppo": "greedy", "first": "first", "random": "uniform"}[chooser]
    collector = Collector(ctx, workers)
    try:
        report = evaluate(model, names, opponents, games, collector, seed, mode)
    finally:
        collector.close()
    report["_meta"] = {"checkpoint": checkpoint if chooser == "ppo" else None, "chooser": chooser,
                       "games_per_opponent": games, "xml": ctx.xml, "max_rounds": ctx.max_rounds,
                       "mcts_seconds": ctx.mcts_seconds, "workers": workers, "seed": seed}
    if out:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_text(json.dumps(report, indent=1))
    return report


def _short(label: str) -> str:
    kind, _, arg = label.partition(":")
    if kind in ("ppo", "cmd") and arg:  # ppo:runs/x/checkpoints/iter_0030.pt -> ppo:x/iter_0030.pt
        p = Path(arg)
        label = f"{kind}:{p.parent.parent.name}/{p.name}" if p.parent.name == "checkpoints" else f"{kind}:{p.name}"
    return label[-32:]


def format_report(report: Dict[str, dict]) -> str:
    rows = [(_short(k), v) for k, v in report.items() if not k.startswith("_")]
    lines = ["Game performance (policy greedy, sides alternate)",
             f"  {'opponent':32s} {'games':>5s} {'win':>4s} {'loss':>4s} {'draw':>4s} "
             f"{'win%':>6s} {'score':>6s} {'rounds':>6s}  by side (wins/games)"]
    for k, v in rows:
        p = v["performance"]
        sides = "  ".join(f"{s}:{x['wins']}/{x['games']}" for s, x in sorted(p["by_side"].items()))
        lines.append(f"  {k:32s} {p['games']:5d} {p['wins']:4d} {p['losses']:4d} {p['draws']:4d} "
                     f"{100 * p['win_rate']:5.1f}% {p['score']:6.3f} {p['rounds_mean']:6.1f}  {sides}")
    lines += ["", "Computational cost",
              f"  {'opponent':32s} {'s/game':>7s} {'total s':>8s} {'decisions/game':>14s} "
              f"{'policy ms/decision':>18s} {'moves/game':>10s} {'rejected/game':>13s}"]
    for k, v in rows:
        c = v["cost"]
        lines.append(f"  {k:32s} {c['game_seconds_mean']:7.2f} {c['game_seconds_total']:8.1f} "
                     f"{c['policy_decisions_per_game']:14.1f} {c['policy_ms_per_decision']:18.2f} "
                     f"{c['moves_applied_per_game']:10.1f} {c['moves_rejected_per_game']:13.2f}")
    if "_wall_seconds" in report:
        lines.append(f"  wall time {report['_wall_seconds']:.1f}s")
    return "\n".join(lines)
