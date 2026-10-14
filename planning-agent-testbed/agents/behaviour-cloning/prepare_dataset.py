#!/usr/bin/env python3
"""Build, validate and (optionally) save the attack-formation behaviour-cloning dataset.

Examples (from planning-agent-testbed/agents/behaviour-cloning):
  python3 prepare_dataset.py                          # build from data-processing/records, print report
  python3 prepare_dataset.py --save                   # ...and save to ./data
  python3 prepare_dataset.py --save --data-dir data/ctf-v1 --records ../tripleMind/records
  python3 prepare_dataset.py --load                   # reuse a saved dataset (no rebuild)
  python3 prepare_dataset.py --strict                 # stop at the first failed assertion

Saved files (in --data-dir): dataset.pkl, report.json, report.txt, feature_names.json, splits.json.
Training is not part of this script; it shows what a training loop would consume.
"""

import argparse
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE / "data-processing"))

from attack_formation_dataset import (DEFAULT_DATA_DIR, DEFAULT_RECORDS,  # noqa: E402
                                      AttackFormationDataset, collate)


def process_records(records=(str(DEFAULT_RECORDS),), game_xml=None, data_dir=None, strict=False,
                    check_legality=True, val=0.15, test=0.15, seed=0):
    """Preprocess records into (dataset, game-grouped splits); with `data_dir`, also save
    dataset.pkl, report.json/.txt, feature_names.json and splits.json there. Used by this script
    and by the behaviour-cloning pipeline (run_bc.py)."""
    ds = AttackFormationDataset.from_records(records, game_xml=game_xml, strict=strict,
                                             check_legality=check_legality)
    splits = ds.split_by_game(val, test, seed)
    if data_dir is not None:
        ds.save(data_dir, splits=splits)
    return ds, splits


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--records", nargs="+", default=[str(DEFAULT_RECORDS)],
                    help="record files or folders of *.jsonl (default: data-processing/records)")
    ap.add_argument("--game-xml", help="rules XML (default: found under games/ by definition digest)")
    ap.add_argument("--save", action="store_true", help="save the processed dataset to --data-dir")
    ap.add_argument("--load", action="store_true", help="load a saved dataset from --data-dir")
    ap.add_argument("--data-dir", default=str(DEFAULT_DATA_DIR), help="default: behaviour-cloning/data")
    ap.add_argument("--strict", action="store_true", help="raise on the first failed assertion")
    ap.add_argument("--no-legality", action="store_true", help="skip A6 (faster)")
    ap.add_argument("--val", type=float, default=0.15, help="validation fraction of games")
    ap.add_argument("--test", type=float, default=0.15, help="test fraction of games")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    # 1. Build (or load) the dataset. Invalid samples are listed in the report, never dropped silently.
    # 2. Split by game, so both factions' samples of one game stay in one split (A10).
    if args.load:
        ds = AttackFormationDataset.load(args.data_dir)
        splits = ds.split_by_game(args.val, args.test, args.seed)
    else:
        ds, splits = process_records(args.records, args.game_xml, None, args.strict,
                                     not args.no_legality, args.val, args.test, args.seed)
    print(ds.report.summary())

    # 3. What a training loop gets: one sample...
    if len(ds) == 0:
        sys.exit("no valid samples")
    item = ds[0]
    m = item["metadata"]
    print("\nOne sample:")
    print(f"  {m['record']} step {m['step_index']} level {m['level']}: {m['actor']} "
          f"vs {m['opponent']}, target {m['target']}, z={item['z']}")
    print(f"  node_features {item['node_features'].shape}  edge_index {item['edge_index'].shape}  "
          f"global_features {item['global_features'].shape}  "
          f"candidate_features {item['candidate_features'].shape}  expert_index {item['expert_index']}")
    for i, c in enumerate(m["candidates"]):
        units = sum(len(mv["unitIds"]) for mv in c["moves"])
        mark = "  <- MCTS choice" if i == item["expert_index"] else ""
        print(f"    [{i}] {c['kind']:6s} {c['target']} units={units} strength={c['strength']} "
              f"visits={m['mcts_visits'][i]}{mark}")

    # ...and a padded batch of the training split (variable candidate counts -> mask).
    train = ds.subset(splits["train"])
    batch = collate([train[i] for i in range(min(8, len(train)))])
    print("\nTrain batch of", len(batch["expert_index"]), "samples:",
          {k: v.shape for k, v in batch.items() if hasattr(v, "shape")})

    # 4. Save for training.
    if args.save:
        out = ds.save(args.data_dir, splits=splits)
        print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
