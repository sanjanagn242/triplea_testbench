"""Training inputs for behaviour cloning, from either preprocessing mode.

  process mode  prepare_dataset.process_records(): records -> dataset (+ saved files)
  load mode     AttackFormationDataset.load() + splits.json + feature_names.json

Both modes end in make_inputs(dataset, splits, feature_names), so they produce the same tensors;
inputs_fingerprint() hashes those tensors and is stored in checkpoints.
"""

from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

BC_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BC_DIR / "data-processing"))
sys.path.insert(0, str(BC_DIR))

from attack_formation_dataset import (DEFAULT_DATA_DIR, AttackFormationDataset,  # noqa: E402
                                      collate)
from prepare_dataset import process_records  # noqa: E402

SPLITS = ("train", "val", "test")


@dataclass
class SplitTensors:
    """One split, padded to its largest candidate set. Row i is dataset sample indices[i]."""
    name: str
    node: torch.Tensor        # (N, T, F_node)
    glob: torch.Tensor        # (N, F_global)
    cand: torch.Tensor        # (N, C, F_cand), zero padding
    mask: torch.Tensor        # (N, C) bool, True for real candidates
    expert: torch.Tensor      # (N,) long
    is_skip: torch.Tensor     # (N, C) bool
    indices: List[int]
    game_ids: List[str]
    actors: List[str]

    def __len__(self):
        return self.expert.shape[0]

    @property
    def n_cand(self) -> torch.Tensor:
        return self.mask.sum(1)

    def batch(self, rows: torch.Tensor):
        """(node, glob, cand, mask, expert) for `rows`, padding trimmed to the batch's largest set."""
        c = int(self.mask[rows].sum(1).max()) if len(rows) else 1
        return (self.node[rows], self.glob[rows], self.cand[rows, :c], self.mask[rows, :c],
                self.expert[rows])


@dataclass
class BCInputs:
    splits: Dict[str, SplitTensors]
    feature_names: dict
    fingerprint: str
    source: str

    @property
    def node_shape(self):
        return tuple(self.splits["train"].node.shape[1:])

    @property
    def glob_dim(self):
        return self.splits["train"].glob.shape[1]

    @property
    def cand_dim(self):
        return self.splits["train"].cand.shape[2]


# ---- the two modes ----------------------------------------------------------------------------

def load_processed(data_dir: str = str(DEFAULT_DATA_DIR)) -> BCInputs:
    """Load mode: dataset.pkl, splits.json and feature_names.json from `data_dir`."""
    d = Path(data_dir)
    for f in ("dataset.pkl", "splits.json", "feature_names.json"):
        if not (d / f).exists():
            raise FileNotFoundError(f"{d / f} missing; run `run_bc.py preprocess` first")
    ds = AttackFormationDataset.load(str(d))
    splits = json.loads((d / "splits.json").read_text())
    names = json.loads((d / "feature_names.json").read_text())
    return make_inputs(ds, splits, names, source=f"load:{d}")


def process_and_save(records: Sequence[str], data_dir: Optional[str], game_xml=None,
                     strict=False, check_legality=True, val=0.15, test=0.15,
                     split_seed=0) -> BCInputs:
    """Process mode: run the preprocessing (prepare_dataset.process_records), save it to
    `data_dir` (unless None) and build the inputs from the in-memory dataset."""
    ds, splits = process_records(records, game_xml, data_dir, strict, check_legality, val, test,
                                 split_seed)
    print(ds.report.summary())
    return make_inputs(ds, splits, feature_names_of(ds), source=f"process:{list(records)}")


def feature_names_of(ds: AttackFormationDataset) -> dict:
    """The content of feature_names.json, as AttackFormationDataset.save writes it."""
    return {"territories": list(ds.graph.territories), "node": ds.spec.node_names,
            "global": ds.spec.global_names, "candidate": ds.spec.candidate_names}


# ---- shared conversion ------------------------------------------------------------------------

def make_inputs(ds: AttackFormationDataset, splits: dict, feature_names: dict,
                source: str = "") -> BCInputs:
    if feature_names != feature_names_of(ds):
        raise ValueError("feature_names.json does not match the dataset's feature spec")
    skip_col = feature_names["candidate"].index("is_skip")
    out = {}
    for name in SPLITS:
        idx = [int(i) for i in splits.get(name, [])]
        if not idx:
            out[name] = _empty(name, ds, feature_names)
            continue
        b = collate([ds[i] for i in idx])  # the preprocessing's own padding
        cand = torch.from_numpy(b["candidate_features"])
        out[name] = SplitTensors(
            name, torch.from_numpy(b["node_features"]), torch.from_numpy(b["global_features"]),
            cand, torch.from_numpy(b["candidate_mask"]), torch.from_numpy(b["expert_index"]),
            (cand[:, :, skip_col] > 0.5) & torch.from_numpy(b["candidate_mask"]), idx,
            [m["game_id"] for m in b["metadata"]], [m["actor"] for m in b["metadata"]])
    inputs = BCInputs(out, feature_names, "", source)
    inputs.fingerprint = inputs_fingerprint(inputs)
    return inputs


def _empty(name, ds, names) -> SplitTensors:
    t, fn = len(names["territories"]), len(names["node"])
    return SplitTensors(name, torch.zeros(0, t, fn), torch.zeros(0, len(names["global"])),
                        torch.zeros(0, 1, len(names["candidate"])), torch.zeros(0, 1, dtype=torch.bool),
                        torch.zeros(0, dtype=torch.long), torch.zeros(0, 1, dtype=torch.bool),
                        [], [], [])


def inputs_fingerprint(inputs: BCInputs) -> str:
    h = hashlib.sha256(json.dumps(inputs.feature_names, sort_keys=True).encode())
    for name in SPLITS:
        s = inputs.splits[name]
        h.update(name.encode())
        h.update(json.dumps([s.indices, s.game_ids]).encode())
        for t in (s.node, s.glob, s.cand, s.mask, s.expert):
            h.update(t.contiguous().numpy().tobytes())
    return h.hexdigest()


# ---- sanity checks on the inputs (1, 2, 4) -----------------------------------------------------

def check_inputs(inputs: BCInputs) -> List[str]:
    """Feature dimensions, finite values, expert indices, >= 1 candidate, padding layout, one
    skip per decision, and no game shared by two splits. Returns the list of problems."""
    names = inputs.feature_names
    want_node = (len(names["territories"]), len(names["node"]))
    problems = []
    for name, s in inputs.splits.items():
        n = len(s)
        if n == 0:
            continue
        if tuple(s.node.shape[1:]) != want_node:
            problems.append(f"{name}: node features {tuple(s.node.shape[1:])} != {want_node}")
        if s.glob.shape[1] != len(names["global"]):
            problems.append(f"{name}: global features {s.glob.shape[1]} != {len(names['global'])}")
        if s.cand.shape[2] != len(names["candidate"]):
            problems.append(f"{name}: candidate features {s.cand.shape[2]} != {len(names['candidate'])}")
        for label, t in (("node", s.node), ("global", s.glob), ("candidate", s.cand)):
            if not torch.isfinite(t).all():
                problems.append(f"{name}: non-finite {label} features")
        nc = s.n_cand
        if (nc < 1).any():
            problems.append(f"{name}: {(nc < 1).sum().item()} samples without a candidate")
        if ((s.expert < 0) | (s.expert >= nc)).any():
            problems.append(f"{name}: expert index outside the candidate set")
        prefix = torch.arange(s.mask.shape[1])[None, :] < nc[:, None]
        if not torch.equal(prefix, s.mask):
            problems.append(f"{name}: candidate mask is not a prefix")
        if (s.cand[~s.mask] != 0).any():
            problems.append(f"{name}: non-zero padding candidates")
        if not torch.equal(s.is_skip.sum(1), torch.ones(n, dtype=torch.long)):
            problems.append(f"{name}: decisions without exactly one skip candidate")
        if len(set(s.indices)) != n:
            problems.append(f"{name}: repeated sample indices")
    games = {name: set(s.game_ids) for name, s in inputs.splits.items()}
    rows = {name: set(s.indices) for name, s in inputs.splits.items()}
    for i, a in enumerate(SPLITS):
        for b in SPLITS[i + 1:]:
            if games[a] & games[b]:
                problems.append(f"games in both {a} and {b}: {sorted(games[a] & games[b])}")
            if rows[a] & rows[b]:
                problems.append(f"samples in both {a} and {b}")
    if len(inputs.splits["train"]) == 0:
        problems.append("empty training split")
    return problems


def describe(inputs: BCInputs) -> str:
    lines = [f"inputs from {inputs.source} (fingerprint {inputs.fingerprint[:12]})",
             f"  node {inputs.node_shape}, global {inputs.glob_dim}, candidate {inputs.cand_dim}"]
    for name, s in inputs.splits.items():
        if len(s):
            nc = s.n_cand.float()
            lines.append(f"  {name:5s} {len(s):6d} decisions from {len(set(s.game_ids)):3d} games, "
                         f"candidates mean {nc.mean():.2f} max {int(nc.max())}, "
                         f"single-candidate {(nc == 1).float().mean():.1%}")
        else:
            lines.append(f"  {name:5s} empty")
    return "\n".join(lines)
