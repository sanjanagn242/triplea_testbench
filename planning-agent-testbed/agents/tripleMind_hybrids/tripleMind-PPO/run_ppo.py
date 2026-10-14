#!/usr/bin/env python3
"""TripleMind PPO: train, evaluate and check the attack-formation policy.

  run_ppo.py train     --out runs/ppo-selfplay [--iterations 100 --games-per-iter 32 --workers 4
                       --opponents self ...] [--resume runs/ppo-selfplay/checkpoints/last.pt]
  run_ppo.py evaluate  --checkpoint runs/ppo-selfplay/checkpoints/best.pt
                       --opponents heuristic first mcts ppo:OTHER.pt [--games 20 --workers 4]
  run_ppo.py evaluate  --chooser first --opponents heuristic mcts   (baseline, same conditions)
  run_ppo.py sanity    [--out runs/sanity]

See README.md.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from ppo import paths  # noqa: E402


def _common(p):
    p.add_argument("--xml", default=str(paths.DEFAULT_XML), help="game XML (two players)")
    p.add_argument("--max-rounds", type=int, default=100, help="round limit (no winner after it)")
    p.add_argument("--mcts-seconds", type=float, default=0.5, help="MCTS opponent time per decision")
    p.add_argument("--workers", type=int, default=4, help="parallel game processes (1 = in process)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--threads", type=int, default=0, help="torch threads in the main process (0 = default)")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    t = sub.add_parser("train", help="PPO from scratch (or resume)")
    _common(t)
    t.add_argument("--out", default=str(paths.DEFAULT_RUNS / "ppo-selfplay"))
    t.add_argument("--resume", help="checkpoint to continue from (model, optimizer, statistics)")
    t.add_argument("--iterations", type=int, default=100)
    t.add_argument("--games-per-iter", type=int, default=32)
    t.add_argument("--opponents", nargs="+", default=["self"],
                   help="rotated over the games of an iteration: self, past, heuristic, first, "
                        "random, mcts, bc, ppo:PATH, cmd:PATH")
    t.add_argument("--eval-every", type=int, default=10)
    t.add_argument("--eval-games", type=int, default=20)
    t.add_argument("--eval-opponents", nargs="+", default=["heuristic", "first"])
    t.add_argument("--save-every", type=int, default=10)
    t.add_argument("--norm-warmup-games", type=int, default=16)
    t.add_argument("--norm-update-iters", type=int, default=20)
    t.add_argument("--actor-hidden", type=int, nargs="+", default=[256, 128])
    t.add_argument("--critic-hidden", type=int, nargs="+", default=[256, 128])
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--gamma", type=float, default=0.995)
    t.add_argument("--gae-lambda", type=float, default=0.95)
    t.add_argument("--clip", type=float, default=0.2)
    t.add_argument("--value-coef", type=float, default=0.5)
    t.add_argument("--entropy-coef", type=float, default=0.01)
    t.add_argument("--epochs", type=int, default=4)
    t.add_argument("--minibatch-size", type=int, default=256)
    t.add_argument("--max-grad-norm", type=float, default=0.5)
    t.add_argument("--target-kl", type=float, default=0.03, help="0 disables the early stop")
    t.add_argument("--value-clip", type=float, default=0.0)

    e = sub.add_parser("evaluate", help="complete games against fixed opponents, no updates")
    _common(e)
    e.add_argument("--checkpoint", default=str(paths.DEFAULT_CHECKPOINT))
    e.add_argument("--chooser", choices=["ppo", "first", "random"], default="ppo",
                   help="ppo = the checkpoint (greedy); first / random = baselines")
    e.add_argument("--opponents", nargs="+", default=["heuristic", "first"],
                   help="heuristic, first, random, mcts, bc, ppo:PATH, cmd:PATH")
    e.add_argument("--games", type=int, default=20, help="games per opponent (sides alternate)")
    e.add_argument("--out", help="write the report as JSON")

    s = sub.add_parser("sanity", help="validation checks (see ppo/checks.py)")
    s.add_argument("--xml", default=str(paths.DEFAULT_XML))
    s.add_argument("--out", default=str(paths.DEFAULT_RUNS / "sanity"))
    s.add_argument("--games", type=int, default=2)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--max-rounds", type=int, default=100)
    s.add_argument("--threads", type=int, default=0)

    args = ap.parse_args(argv)
    import torch
    if args.threads:
        torch.set_num_threads(args.threads)

    if args.cmd == "train":
        from ppo.algorithm import PPOConfig
        from ppo.train import TrainConfig, train
        cfg = TrainConfig(args.iterations, args.games_per_iter, args.workers, args.opponents,
                          args.seed, args.xml, args.max_rounds, args.mcts_seconds,
                          args.actor_hidden, args.critic_hidden, args.norm_warmup_games,
                          args.norm_update_iters, args.eval_every, args.eval_games,
                          args.eval_opponents, args.save_every)
        ppo = PPOConfig(args.gamma, args.gae_lambda, args.clip, args.value_coef, args.entropy_coef,
                        args.lr, args.epochs, args.minibatch_size, args.max_grad_norm,
                        args.target_kl, args.value_clip)
        print(json.dumps(train(cfg, ppo, args.out, args.resume), indent=1))
    elif args.cmd == "evaluate":
        from ppo.evaluate import evaluate_checkpoint, format_report
        from ppo.rollout import RolloutContext
        out_dir = Path(args.out).parent if args.out else paths.DEFAULT_RUNS / "eval"
        ctx = RolloutContext(args.xml, args.max_rounds, args.mcts_seconds, work_dir=str(out_dir))
        report = evaluate_checkpoint(args.checkpoint, args.opponents, args.games, ctx,
                                     args.workers, args.seed, args.chooser, args.out)
        who = args.checkpoint if args.chooser == "ppo" else f"baseline: {args.chooser}"
        print(f"{who}\n{format_report(report)}")
    else:
        from ppo.checks import run_all
        results = run_all(args.out, args.xml, args.games, args.seed, args.max_rounds)
        Path(args.out, "sanity.json").write_text(json.dumps(results, indent=1, default=str))
        for name, r in results.items():
            extra = {k: v for k, v in r.items() if k != "passed" and not isinstance(v, (list, dict))}
            print(f"{'PASS' if r['passed'] else 'FAIL'}  {name:22s} {extra}")
        failed = [n for n, r in results.items() if not r["passed"]]
        print(f"details: {Path(args.out, 'sanity.json')}")
        sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
