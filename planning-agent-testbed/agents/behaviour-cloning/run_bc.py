#!/usr/bin/env python3
"""Baseline behaviour cloning of TripleMind MCTS attack-formation choices (see README.md).

  preprocess  records -> data dir (dataset.pkl, feature_names.json, splits.json, reports)
  train       train from a data dir (load mode) or from --records (process mode, saved first)
  evaluate    evaluate a saved checkpoint without retraining
  sanity      input checks, softmax, gradient isolation, overfit test, optional mode equivalence
"""

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from bc_baseline.data import (DEFAULT_DATA_DIR, check_inputs, describe,  # noqa: E402
                              load_processed, process_and_save)
from bc_baseline.evaluate import (evaluate_checkpoint, format_results, predict,  # noqa: E402
                                  evaluate_model)
from bc_baseline.train import (DEFAULT_CONFIG, TrainConfig, assert_no_gradients,  # noqa: E402
                               load_train_config, new_model,
                               overfit_test, train)

DEFAULT_RUNS = HERE / "runs"


def add_preprocess_args(p, records_required=False):
    p.add_argument("--records", nargs="+", required=records_required,
                   help="record files or folders of *.jsonl to preprocess")
    p.add_argument("--game-xml", help="rules XML (default: found by definition digest)")
    p.add_argument("--strict", action="store_true", help="stop at the first failed assertion")
    p.add_argument("--no-legality", action="store_true", help="skip the A6 legality check (faster)")
    p.add_argument("--val", type=float, default=0.15, help="fraction of games for validation")
    p.add_argument("--test", type=float, default=0.15, help="fraction of games for test")
    p.add_argument("--split-seed", type=int, default=0)


def get_inputs(args):
    """Process mode when --records is given (saved to --data-dir, then checked against a reload:
    sanity check 7), load mode otherwise."""
    if getattr(args, "records", None):
        inputs = process_and_save(args.records, args.data_dir, args.game_xml, args.strict,
                                  not args.no_legality, args.val, args.test, args.split_seed)
        reloaded = load_processed(args.data_dir)
        if reloaded.fingerprint != inputs.fingerprint:
            sys.exit("processed and saved data give different training inputs")
        print(f"saved to {args.data_dir}; reload gives identical inputs")
    else:
        inputs = load_processed(args.data_dir)
    print(describe(inputs))
    problems = check_inputs(inputs)
    if problems:
        sys.exit("input checks failed:\n  " + "\n  ".join(problems))
    return inputs


def cmd_preprocess(args):
    inputs = process_and_save(args.records, args.data_dir, args.game_xml, args.strict,
                              not args.no_legality, args.val, args.test, args.split_seed)
    print(describe(inputs))
    print(f"saved to {args.data_dir}")


def cmd_train(args):
    inputs = get_inputs(args)
    cfg = load_train_config(args.config, lr=args.lr, batch_size=args.batch_size,
                            epochs=args.epochs, seed=args.seed, hidden_dims=args.hidden_dims,
                            dropout=args.dropout, weight_decay=args.weight_decay,
                            patience=args.patience, min_delta=args.min_delta)
    print(f"training parameters ({args.config} + flags): {cfg}")
    out = Path(args.out)
    summary = train(inputs, cfg, out)
    print(f"best epoch {summary['best_epoch']}; checkpoint {summary['checkpoint']}")
    results = evaluate_checkpoint(summary["checkpoint"], inputs, out / "eval.json")
    print(format_results(results))
    print(f"outputs in {out}: best.pt config.json history.json train_summary.json eval.json "
          f"training_curves.png")


def cmd_evaluate(args):
    inputs = get_inputs(args)
    out = args.out or str(Path(args.checkpoint).with_name("eval.json"))
    results = evaluate_checkpoint(args.checkpoint, inputs, out)
    print(format_results(results))
    print(f"written to {out}")


def cmd_sanity(args):
    inputs = get_inputs(args)  # checks 1, 2, 4 (and 7 when --records is given)
    print("checks 1, 2, 4: feature dimensions, finite values, expert indices, >= 1 candidate, "
          "one skip per decision, no game in two splits - ok")

    # 3 and 6: softmax normalisation on every split; evaluation leaves no gradients and does
    # not change the parameters.
    model = new_model(inputs, TrainConfig(seed=args.seed))
    before = {k: v.clone() for k, v in model.state_dict().items()}
    evaluate_model(model, inputs)
    assert_no_gradients(model)
    if any(not torch.equal(before[k], v) for k, v in model.state_dict().items()):
        sys.exit("evaluation changed model parameters")
    print("checks 3, 6: softmax sums to 1 per decision with 0 on padding; evaluating all splits "
          "creates no gradients and changes no parameters - ok")

    r = overfit_test(inputs, n=args.overfit_samples, steps=args.overfit_steps, seed=args.seed)
    print(f"check 5: overfit {r['samples']} multi-candidate training decisions in {r['steps']} "
          f"steps -> accuracy {r['accuracy']:.3f}, loss {r['loss']:.4f} "
          f"({r['conflicting_duplicates']} identical inputs with different labels) - "
          f"{'ok' if r['passed'] else 'FAILED'}")

    if args.equivalence_records:
        with tempfile.TemporaryDirectory() as tmp:
            a = process_and_save(args.equivalence_records, tmp, val=args.val, test=args.test,
                                 split_seed=args.split_seed)
            b = load_processed(tmp)
        same = a.fingerprint == b.fingerprint
        print(f"check 7: process mode {a.fingerprint[:12]} vs saved/load mode {b.fingerprint[:12]}"
              f" - {'ok' if same else 'FAILED'}")
        if not same:
            sys.exit(1)
    if not r["passed"]:
        sys.exit(1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="command", required=True)

    p = sub.add_parser("preprocess", help="preprocess records and save, no training")
    add_preprocess_args(p, records_required=True)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.set_defaults(func=cmd_preprocess)

    p = sub.add_parser("train", help="train (load mode, or process mode with --records)")
    add_preprocess_args(p)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR),
                   help="processed data to load, or where process mode saves it")
    p.add_argument("--out", default=str(DEFAULT_RUNS / "mlp-baseline"), help="run directory")
    p.add_argument("--config", default=str(DEFAULT_CONFIG),
                   help="training parameters TOML (default: bc_baseline/train_config.toml); "
                        "the flags below override it")
    p.add_argument("--lr", type=float)
    p.add_argument("--batch-size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--seed", type=int)
    p.add_argument("--hidden-dims", type=int, nargs="+")
    p.add_argument("--dropout", type=float)
    p.add_argument("--weight-decay", type=float)
    p.add_argument("--patience", type=int)
    p.add_argument("--min-delta", type=float)
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("evaluate", help="evaluate a saved checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--out", help="results JSON (default: eval.json next to the checkpoint)")
    p.set_defaults(func=cmd_evaluate, records=None)

    p = sub.add_parser("sanity", help="run the sanity checks")
    p.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR))
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--overfit-samples", type=int, default=64)
    p.add_argument("--overfit-steps", type=int, default=500)
    p.add_argument("--equivalence-records", nargs="+",
                   help="also preprocess these records into a temp dir and compare process mode "
                        "with load mode (check 7)")
    p.add_argument("--val", type=float, default=0.15)
    p.add_argument("--test", type=float, default=0.15)
    p.add_argument("--split-seed", type=int, default=0)
    p.set_defaults(func=cmd_sanity, records=None)

    args = ap.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
