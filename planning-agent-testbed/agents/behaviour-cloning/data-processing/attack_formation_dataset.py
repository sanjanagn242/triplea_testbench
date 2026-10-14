"""Behaviour-cloning dataset of TripleMind MCTS attack-formation decisions (preprocessing only).

The policy to be learned replaces the MCTS choice of *which attack formation* to send against a
territory that TripleMind's heuristic has already chosen. One sample is one sequential MCTS
sub-action: one node on the recorded `path` of a combat-move search tree (TripleMind records,
agents/tripleMind/agent.py) together with the child the search chose there.

For every sample the module
  1. rebuilds a forward-model state from the node's snapshot (triplea_fm LiveBridge). A snapshot
     does not say which territories this turn's earlier sub-actions conquered by blitzing (moving
     into them unopposed), and that decides which routes are open; that turn history is taken
     from replaying the recorded expert actions from the root (A8 checks that replay),
  2. regenerates the legal formations for the heuristic-fixed target with TripleMind's own
     generator (generators.ag.combat_actions, the code MCTS expands with), including the skip,
  3. canonicalises the formations (unit ids are interchangeable, routes and movement states are
     not) and finds the one MCTS chose (the expert index),
  4. checks the assertions below and builds actor-relative graph, global and candidate features.

Invalid samples are never dropped silently: every failure is listed in the ValidationReport, and
`strict=True` raises on the first one.

Assertions (names used in the report):
  A1_observer_is_active        observationPlayer == activePlayer (== the recording faction)
  A2_game_completed            the record has a result line with a valid winner; z = 1 win, 0 loss
  A3_static_graph              territories, neighbours and static attributes identical everywhere
  A4_pre_state_and_target      the snapshot is the chosen child's state of the previous node and
                               unchanged from the root apart from committed units; the expert's
                               target is the generator's target
  A5_expert_in_candidates      the expert formation is one of the generated candidates
  A6_candidates_legal          every generated candidate is accepted by the forward model
  A7_units_consistent          unit availability (owner, location, movement) and commitments
  A8_expert_reproduces_next    applying the expert action gives the next recorded path state
  A9_skip_end_phase            skip (a target with no units) and end-phase (no target) are distinct
  A10_split_grouped            dataset splits share no game

Usage: see ../prepare_dataset.py.
"""

from __future__ import annotations

import glob
import hashlib
import json
import os
import pickle
import random
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parents[2]  # data-processing -> behaviour-cloning -> agents -> planning-agent-testbed
TRIPLEMIND = TESTBED / "agents" / "tripleMind"
for _p in (TESTBED / "forward-model" / "python", TRIPLEMIND):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import LiveBridge, observation_from_state  # noqa: E402

import map_graph  # noqa: E402  (TripleMind)
from generators import ag  # noqa: E402  (TripleMind's action generators, Cython when built)

PREPROCESSING_VERSION = 1
PHASE = "combatMove"
DEFAULT_RECORDS = HERE / "records"
DEFAULT_DATA_DIR = HERE.parent / "data"
DEFAULT_XML_SEARCH = TESTBED / "games"

# Unit types of the per-source allocation block of candidate features (29 x 3).
ALLOCATION_TYPES = ("infantry", "armour", "artillery")
OWNER_GROUPS = ("own", "opp", "neutral")
STATIC_FIELDS = ("water", "neighbors", "production", "capitalFor", "victoryCityValue", "impassable")
ASSERTIONS = ("A1_observer_is_active", "A2_game_completed", "A3_static_graph",
              "A4_pre_state_and_target", "A5_expert_in_candidates", "A6_candidates_legal",
              "A7_units_consistent", "A8_expert_reproduces_next", "A9_skip_end_phase",
              "A10_split_grouped")


class DatasetAssertionError(AssertionError):
    pass


# =================================================================================================
# Records
# =================================================================================================

@dataclass
class Record:
    path: str
    header: dict
    steps: List[dict]
    result: Optional[dict]

    @property
    def meta(self) -> dict:
        return self.header["meta"]

    @property
    def actor(self) -> str:
        return self.meta["player"]

    @property
    def game_id(self) -> str:
        """Both factions' records of one game share it (used to group splits)."""
        return f"{self.meta.get('map_name')}/game-{self.meta.get('game_number')}"

    @property
    def name(self) -> str:
        return os.path.basename(self.path)


def read_record(path: str) -> Record:
    header, steps, result = None, [], None
    with open(path, encoding="utf-8") as f:
        for n, line in enumerate(f, 1):
            if not line.strip():
                continue
            r = json.loads(line)
            kind = r.get("type")
            if kind == "header":
                header = r
            elif kind == "step":
                steps.append(r)
            elif kind == "result":
                result = r
            else:
                raise ValueError(f"{path}:{n}: unknown line type {kind!r}")
    if header is None:
        raise ValueError(f"{path}: no header line")
    return Record(str(path), header, steps, result)


def game_outcome(record: Record) -> Tuple[Optional[int], List[str]]:
    """(z, problems): z = 1 if the recording faction won, 0 if it lost (A2)."""
    res, problems = record.result, []
    if res is None:
        return None, ["no result line: the game was stopped before it ended"]
    winners = res.get("winners") or []
    if not winners:
        problems.append(f"no winners (status {res.get('status')!r})")
    players = {p["name"] for p in record.header.get("initial_state", {}).get("players", [])}
    if players and not set(winners) <= players:
        problems.append(f"winners {winners} are not players {sorted(players)}")
    won = record.actor in winners
    if bool(res.get("won")) != won:
        problems.append(f"result.won={res.get('won')} but winners={winners}")
    return (None if problems else int(won)), problems


def find_game_xml(digest: str, search_root: Path = DEFAULT_XML_SEARCH) -> str:
    """The rules XML under `search_root` whose forward-model definition digest is `digest`."""
    for path in sorted(glob.glob(str(Path(search_root) / "**" / "*.xml"), recursive=True)):
        try:
            text = Path(path).read_text(encoding="utf-8")
            if "<game" not in text[:2000]:
                continue
            d, _ = fm.load_definition(text, strict=False)
        except Exception:
            continue
        if d.definition_digest == digest:
            return path
    raise FileNotFoundError(f"no rules XML under {search_root} has definition digest {digest}")


# =================================================================================================
# Static graph
# =================================================================================================

@dataclass
class StaticGraph:
    territories: Tuple[str, ...]
    index: Dict[str, int]
    attrs: Dict[str, dict]           # STATIC_FIELDS per territory, as in the observation
    edge_index: np.ndarray           # (2, E), both directions

    @classmethod
    def from_observation(cls, obs: dict) -> "StaticGraph":
        terrs = sorted(obs["territories"], key=lambda t: t["name"])
        names = tuple(t["name"] for t in terrs)
        index = {n: i for i, n in enumerate(names)}
        attrs = {t["name"]: {k: (sorted(t[k]) if k == "neighbors" else t[k]) for k in STATIC_FIELDS}
                 for t in terrs}
        edges = sorted((index[a], index[b]) for a in names for b in attrs[a]["neighbors"])
        return cls(names, index, attrs, np.array(edges, dtype=np.int64).T.reshape(2, -1))

    def differences(self, obs: dict) -> List[str]:
        """Why `obs` does not describe this graph (empty when identical)."""
        out = []
        names = {t["name"] for t in obs["territories"]}
        if names != set(self.territories):
            out.append(f"territory set differs: missing {sorted(set(self.territories) - names)}, "
                       f"extra {sorted(names - set(self.territories))}")
        for t in obs["territories"]:
            ref = self.attrs.get(t["name"])
            if ref is None:
                continue
            for k in STATIC_FIELDS:
                v = sorted(t.get(k, [])) if k == "neighbors" else t.get(k)
                if v != ref[k]:
                    out.append(f"{t['name']}.{k}: {v!r} != {ref[k]!r}")
        return out

    def definition_differences(self, d) -> List[str]:
        out = []
        if set(d.territory_names) != set(self.territories):
            out.append("definition territories differ from the observation's")
            return out
        for ti, name in enumerate(d.territory_names):
            nb = sorted(d.territory_names[n] for n in d.neighbors[ti])
            if nb != self.attrs[name]["neighbors"]:
                out.append(f"{name}: definition neighbours {nb} != observation {self.attrs[name]['neighbors']}")
        for a in self.territories:
            for b in self.attrs[a]["neighbors"]:
                if a not in self.attrs[b]["neighbors"]:
                    out.append(f"edge {a}->{b} is not symmetric")
        return out


# =================================================================================================
# Observations, units and actions
# =================================================================================================

def normalize_observation(obs: dict) -> dict:
    """Copy of an observation with unit lists sorted by id (unit order carries no meaning)."""
    obs = json.loads(json.dumps(obs))
    for t in obs.get("territories", []):
        t["units"] = sorted(t.get("units", []), key=lambda u: u["id"])
    for p in obs.get("players", []):
        p["unitsInReserve"] = sorted(p.get("unitsInReserve", []), key=lambda u: u["id"])
    obs["territories"] = sorted(obs.get("territories", []), key=lambda t: t["name"])
    obs["players"] = sorted(obs.get("players", []), key=lambda p: p["name"])
    return obs


def observation_differences(a: dict, b: dict, limit: int = 8) -> List[str]:
    """Readable differences between two (normalised) observations."""
    out = []
    for k in sorted(set(a) | set(b)):
        if k in ("territories", "players"):
            continue
        if a.get(k) != b.get(k):
            out.append(f"{k}: {a.get(k)!r} != {b.get(k)!r}")
    ta = {t["name"]: t for t in a.get("territories", [])}
    tb = {t["name"]: t for t in b.get("territories", [])}
    for name in sorted(set(ta) | set(tb)):
        x, y = ta.get(name), tb.get(name)
        if x != y:
            if x is None or y is None:
                out.append(f"territory {name} only on one side")
                continue
            for k in sorted(set(x) | set(y)):
                if x.get(k) != y.get(k):
                    if k == "units":
                        ux = {u["id"]: u for u in x["units"]}
                        uy = {u["id"]: u for u in y["units"]}
                        diff = [i[:8] for i in set(ux) | set(uy) if ux.get(i) != uy.get(i)]
                        out.append(f"{name}.units differ for {sorted(diff)}")
                    else:
                        out.append(f"{name}.{k}: {x.get(k)!r} != {y.get(k)!r}")
    if a.get("players") != b.get("players"):
        out.append("players differ")
    return out[:limit]


def unit_table(obs: dict) -> Dict[str, dict]:
    """{unit id: type, owner, territory (None = reserve), movementUsed, movementLeft, hits}."""
    table = {}
    for t in obs["territories"]:
        for u in t["units"]:
            table[u["id"]] = {"type": u["type"], "owner": u["owner"], "territory": t["name"],
                              "movementUsed": _num(u.get("movementUsed")),
                              "movementLeft": _num(u.get("movementLeft")), "hits": u.get("hits", 0)}
    for p in obs["players"]:
        for u in p.get("unitsInReserve", []):
            table[u["id"]] = {"type": u["type"], "owner": u["owner"], "territory": None,
                              "movementUsed": _num(u.get("movementUsed")),
                              "movementLeft": _num(u.get("movementLeft")), "hits": u.get("hits", 0)}
    return table


def _num(text) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return 0.0


@dataclass
class Candidate:
    """One formation for the fixed target, in engine terms (unit ids, territory names)."""
    kind: str                        # "attack" | "skip" | "end_phase"
    target: Optional[str]
    strength: float
    moves: List[dict]                # [{"from", "to", "route", "unitIds", "unitTypes", "intent"}]
    key: tuple = ()
    source: int = -1                 # index of the generator action it was built from

    def to_json(self) -> dict:
        return {"kind": self.kind, "target": self.target, "strength": self.strength,
                "moves": self.moves}


def canonical_key(kind: str, target: Optional[str], moves: Sequence[dict],
                  units: Dict[str, dict]) -> tuple:
    """Equivalence class of an action: which units of which type, movement state and damage take
    which route; the ids of interchangeable units are dropped."""
    if kind == "end_phase":
        return ("end_phase",)
    if kind == "skip":
        return ("skip", target)
    per_unit = []
    for mv in moves:
        for uid in mv["unitIds"]:
            u = units.get(uid, {})
            per_unit.append((tuple(mv["route"]), mv.get("intent", "NORMAL"), u.get("type"),
                             u.get("movementUsed"), u.get("hits")))
    return ("attack", target, tuple(sorted(per_unit, key=repr)))


def generated_candidates(actions, bridge: LiveBridge, units: Dict[str, dict]
                         ) -> Tuple[List[Candidate], int]:
    """Generator actions (generators.ag.CombatAction, in generator order) as Candidates, without
    duplicates of an equivalent earlier one. Returns (candidates, number of duplicates merged)."""
    out, seen, dups = [], set(), 0
    for i, a in enumerate(actions):
        kind = "end_phase" if a.end_phase else "attack" if a.moves else "skip"
        moves = []
        for mv in a.moves:
            j = bridge.move_to_json(mv)
            j["unitTypes"] = [units.get(u, {}).get("type") for u in j["unitIds"]]
            moves.append(j)
        c = Candidate(kind, a.to_terr, a.strength, moves, source=i)
        c.key = canonical_key(kind, c.target, moves, units)
        if c.key in seen:
            dups += 1
            continue
        seen.add(c.key)
        out.append(c)
    return out, dups


def engine_observation(d, sim, observer: str, uuid_of: Dict[int, str]) -> dict:
    """observation_from_state with the engine's unit ids (a unit the engine never sent keeps
    its model id): the shape of the recorded snapshots."""
    obs = observation_from_state(d, sim, observer)
    for t in obs["territories"]:
        for u in t["units"]:
            u["id"] = uuid_of.get(int(u["id"]), u["id"])
    for p in obs["players"]:
        for u in p["unitsInReserve"]:
            u["id"] = uuid_of.get(int(u["id"]), u["id"])
    return obs


def recorded_candidate(action: dict, units: Dict[str, dict]) -> Candidate:
    """A path/child action as recorded by TripleMind's agent.combat_action_json."""
    if action.get("end_phase"):
        kind = "end_phase"
    elif not action.get("moves"):
        kind = "skip"
    else:
        kind = "attack"
    c = Candidate(kind, action.get("to"), action.get("strength", 0), action.get("moves", []))
    c.key = canonical_key(kind, c.target, c.moves, units)
    return c


# =================================================================================================
# Features (actor-relative; ids never enter a feature)
# =================================================================================================

@dataclass(frozen=True)
class FeatureSpec:
    territories: Tuple[str, ...]
    unit_types: Tuple[str, ...]

    @property
    def node_names(self) -> List[str]:
        names = ["production", "water", "impassable", "capital_own", "capital_opp",
                 "victory_city_value", "owner_own", "owner_opp", "owner_neutral"]
        names += [f"units_{g}_{t}" for g in OWNER_GROUPS for t in self.unit_types]
        names += [f"available_{t}" for t in self.unit_types]
        names += [f"committed_{t}" for t in self.unit_types]
        names += ["is_target"]
        return names

    @property
    def global_names(self) -> List[str]:
        return ["round", "income_own", "income_opp"]

    @property
    def candidate_names(self) -> List[str]:
        names = [f"alloc_{t}_{u}" for t in self.territories for u in ALLOCATION_TYPES]
        names += [f"total_{u}" for u in ALLOCATION_TYPES]
        names += ["attack_strength", "commitment_cost", "num_sources", "is_skip"]
        return names


def game_unit_types(d, mapg) -> Tuple[str, ...]:
    """Unit types the game can contain (initial units and every production frontier): the unit
    columns of the node features."""
    types = {d.unit_types[t].name for t, *_ in d.initial_units}
    for rules in mapg.frontiers.values():
        for rule in rules:
            types |= set(mapg.production_rules[rule].results)
    unit_types = tuple(sorted(t for t in types if t in mapg.unit_types))
    missing = [t for t in ALLOCATION_TYPES if t not in unit_types]
    if missing:
        raise ValueError(f"allocation unit types {missing} are not in this game")
    return unit_types


def _group(owner: Optional[str], actor: str, opponent: str) -> str:
    return "own" if owner == actor else "opp" if owner == opponent else "neutral"


def node_features(spec: FeatureSpec, graph: StaticGraph, obs: dict, actor: str, opponent: str,
                  committed: Dict[str, str], target: str) -> np.ndarray:
    ti = {t: i for i, t in enumerate(spec.unit_types)}
    nt = len(spec.unit_types)
    x = np.zeros((len(graph.territories), len(spec.node_names)), dtype=np.float32)
    terr = {t["name"]: t for t in obs["territories"]}
    for i, name in enumerate(graph.territories):
        t, a = terr[name], graph.attrs[name]
        g = _group(t.get("owner"), actor, opponent)
        row = [a["production"], float(a["water"]), float(a["impassable"]),
               float(a["capitalFor"] == actor), float(a["capitalFor"] == opponent),
               a["victoryCityValue"], float(g == "own"), float(g == "opp"), float(g == "neutral")]
        counts = np.zeros((len(OWNER_GROUPS), nt), dtype=np.float32)
        available = np.zeros(nt, dtype=np.float32)
        comm = np.zeros(nt, dtype=np.float32)
        for u in t["units"]:
            k = ti[u["type"]]
            counts[OWNER_GROUPS.index(_group(u["owner"], actor, opponent)), k] += 1
            if u["owner"] == actor:
                if u["id"] in committed:
                    comm[k] += 1
                elif _num(u.get("movementLeft")) > 0:
                    available[k] += 1
        x[i] = np.concatenate([np.array(row, dtype=np.float32), counts.ravel(), available, comm,
                               [float(name == target)]])
    return x


def global_features(graph: StaticGraph, obs: dict, actor: str, opponent: str) -> np.ndarray:
    income = Counter()
    for t in obs["territories"]:
        income[t.get("owner")] += graph.attrs[t["name"]]["production"]
    return np.array([obs["round"], income[actor], income[opponent]], dtype=np.float32)


def candidate_features(spec: FeatureSpec, graph: StaticGraph, cand: Candidate,
                       units: Dict[str, dict], unit_cost) -> np.ndarray:
    alloc = np.zeros((len(graph.territories), len(ALLOCATION_TYPES)), dtype=np.float32)
    cost, sources = 0.0, set()
    for mv in cand.moves:
        sources.add(mv["route"][0])
        for uid in mv["unitIds"]:
            utype = units[uid]["type"]
            alloc[graph.index[mv["route"][0]], ALLOCATION_TYPES.index(utype)] += 1
            cost += unit_cost(utype)
    return np.concatenate([alloc.ravel(), alloc.sum(axis=0),
                           [cand.strength, cost, len(sources), float(cand.kind == "skip")]]
                          ).astype(np.float32)


# =================================================================================================
# Validation report
# =================================================================================================

@dataclass
class ValidationReport:
    counts: Counter = field(default_factory=Counter)
    assertion_failures: Counter = field(default_factory=Counter)
    failures: List[dict] = field(default_factory=list)
    rejected_records: List[dict] = field(default_factory=list)
    failed_samples: List[dict] = field(default_factory=list)
    leaves: Counter = field(default_factory=Counter)
    candidate_counts: List[int] = field(default_factory=list)
    samples_by_actor: Counter = field(default_factory=Counter)
    outcomes: Counter = field(default_factory=Counter)
    notes: List[str] = field(default_factory=list)
    splits: dict = field(default_factory=dict)
    strict: bool = False

    def fail(self, assertion: str, where: dict, message: str):
        assert assertion in ASSERTIONS, assertion
        self.assertion_failures[assertion] += 1
        entry = {"assertion": assertion, **where, "message": message}
        self.failures.append(entry)
        if self.strict:
            raise DatasetAssertionError(json.dumps(entry))

    def to_json(self) -> dict:
        cc = self.candidate_counts
        return {
            "preprocessing_version": PREPROCESSING_VERSION,
            "counts": dict(self.counts),
            "samples_by_actor": dict(self.samples_by_actor),
            "outcomes": dict(self.outcomes),
            "leaves": dict(self.leaves),
            "candidates": {
                "total": int(sum(cc)), "min": int(min(cc)) if cc else 0,
                "mean": float(np.mean(cc)) if cc else 0.0, "max": int(max(cc)) if cc else 0,
                "histogram": dict(sorted(Counter(cc).items())),
            },
            "assertion_failures": {a: self.assertion_failures.get(a, 0) for a in ASSERTIONS},
            "rejected_records": self.rejected_records,
            "failed_samples": self.failed_samples,
            "failures": self.failures,
            "splits": self.splits,
            "notes": self.notes,
        }

    def summary(self) -> str:
        j = self.to_json()
        c, cand = j["counts"], j["candidates"]
        lines = [
            "Attack-formation dataset validation",
            f"  records        {c.get('records', 0)} read, {c.get('records_accepted', 0)} accepted, "
            f"{len(self.rejected_records)} rejected",
            f"  steps          {c.get('steps', 0)} combat moves, {c.get('steps_failed', 0)} failed",
            f"  path nodes     {c.get('path_nodes', 0)} ({c.get('decision_nodes', 0)} decisions, "
            f"{c.get('leaf_nodes', 0)} leaves {dict(self.leaves)})",
            f"  samples        {c.get('samples_ok', 0)} valid, {c.get('samples_failed', 0)} failed "
            f"(expert skip {c.get('expert_skip', 0)}); by actor {dict(self.samples_by_actor)}; "
            f"outcomes {dict(self.outcomes)}",
            f"  candidates     total {cand['total']}, per sample min {cand['min']} / mean "
            f"{cand['mean']:.2f} / max {cand['max']}; equivalent duplicates merged "
            f"{c.get('duplicate_candidates_merged', 0)}; samples with only one candidate "
            f"{c.get('single_candidate_samples', 0)}",
            f"  MCTS children  {c.get('mcts_children', 0)} recorded, "
            f"{c.get('mcts_children_not_generated', 0)} not among the generated candidates",
            "  assertion failures:",
        ]
        lines += [f"    {a:28s} {n}" for a, n in j["assertion_failures"].items()]
        for r in self.rejected_records:
            lines.append(f"  rejected {r['record']}: {'; '.join(r['reasons'])}")
        if self.splits:
            for name, s in self.splits.items():
                if isinstance(s, dict):
                    lines.append(f"  split {name:5s} {s['samples']} samples from games {s['games']}")
        lines += [f"  note: {n}" for n in self.notes]
        return "\n".join(lines)


# =================================================================================================
# Dataset
# =================================================================================================

@dataclass
class Sample:
    node_features: np.ndarray        # (T, F_node)
    global_features: np.ndarray      # (F_global,)
    candidate_features: np.ndarray   # (C, F_cand)
    expert_index: int
    z: float                         # actor-perspective outcome: 1 win, 0 loss
    metadata: dict                   # ids, routes, unit states, commitments (not features)


class AttackFormationDataset:
    """Samples of MCTS attack-formation choices. `dataset[i]` is a dict of numpy arrays plus
    metadata; it works as a torch Dataset (len / getitem) without importing torch."""

    def __init__(self, samples: List[Sample], graph: StaticGraph, spec: FeatureSpec,
                 report: ValidationReport, config: dict):
        self.samples = samples
        self.graph = graph
        self.spec = spec
        self.report = report
        self.config = config

    # ---- construction ---------------------------------------------------------------------

    @classmethod
    def from_records(cls, records: Sequence[str] = (str(DEFAULT_RECORDS),),
                     game_xml: Optional[str] = None, strict: bool = False,
                     check_legality: bool = True, cache_dir: Optional[str] = None,
                     ) -> "AttackFormationDataset":
        """Build from record files and/or folders of *.jsonl. With `cache_dir`, a dataset saved
        there from the same records, XML and settings is loaded instead of rebuilt (candidate
        generation and the legality checks are the slow part)."""
        paths = _expand_paths(records)
        if not paths:
            raise FileNotFoundError(f"no .jsonl records in {list(records)}")
        if cache_dir is not None:
            fp = _fingerprint(paths, game_xml, check_legality)
            cached = Path(cache_dir) / "dataset.pkl"
            if cached.exists():
                ds = cls.load(cache_dir)
                if ds.config.get("fingerprint") == fp:
                    return ds
        builder = _Builder(paths, game_xml, strict, check_legality)
        ds = builder.build()
        if cache_dir is not None:
            ds.config["fingerprint"] = _fingerprint(paths, game_xml, check_legality)
            ds.save(cache_dir)
        return ds

    # ---- access ---------------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, i: int) -> dict:
        s = self.samples[i]
        return {"node_features": s.node_features, "edge_index": self.graph.edge_index,
                "global_features": s.global_features, "candidate_features": s.candidate_features,
                "expert_index": s.expert_index, "z": s.z, "metadata": s.metadata}

    @property
    def game_ids(self) -> List[str]:
        return [s.metadata["game_id"] for s in self.samples]

    def subset(self, indices: Sequence[int]) -> "AttackFormationDataset":
        return AttackFormationDataset([self.samples[i] for i in indices], self.graph, self.spec,
                                      self.report, self.config)

    def split_by_game(self, val_fraction: float = 0.15, test_fraction: float = 0.15,
                      seed: int = 0) -> Dict[str, List[int]]:
        """Sample indices per split; every game (both factions' records) lands in one split.
        Checks A10 and stores the split in the report."""
        games = sorted(set(self.game_ids))
        random.Random(seed).shuffle(games)
        n = len(games)
        n_test = round(n * test_fraction)
        n_val = round(n * val_fraction)
        if n >= 2 and val_fraction > 0 and n_val == 0:
            n_val = 1
        if n - n_val - n_test < 1:
            n_test = max(0, n - n_val - 1)
        assign = {}
        for k, g in enumerate(games):
            assign[g] = "val" if k < n_val else "test" if k < n_val + n_test else "train"
        splits = {"train": [], "val": [], "test": []}
        for i, g in enumerate(self.game_ids):
            splits[assign[g]].append(i)
        seen = {name: {self.game_ids[i] for i in idx} for name, idx in splits.items()}
        names = list(seen)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                shared = seen[names[a]] & seen[names[b]]
                if shared:
                    self.report.fail("A10_split_grouped", {"splits": [names[a], names[b]]},
                                     f"games in both: {sorted(shared)}")
        self.report.splits = {name: {"samples": len(idx), "games": sorted(seen[name])}
                              for name, idx in splits.items()}
        self.report.splits["seed"] = seed
        if n < 3:
            self.report.notes.append(f"only {n} game(s): split is {dict(Counter(assign.values()))}")
        return splits

    # ---- persistence ----------------------------------------------------------------------

    def save(self, data_dir: str = str(DEFAULT_DATA_DIR), splits: Optional[dict] = None):
        """dataset.pkl (samples, graph, feature spec, config), report.json / report.txt,
        feature_names.json and, if given, splits.json."""
        out = Path(data_dir)
        out.mkdir(parents=True, exist_ok=True)
        with open(out / "dataset.pkl", "wb") as f:
            pickle.dump({"version": PREPROCESSING_VERSION, "samples": self.samples,
                         "graph": self.graph, "spec": self.spec, "report": self.report,
                         "config": self.config}, f, protocol=pickle.HIGHEST_PROTOCOL)
        (out / "report.json").write_text(json.dumps(self.report.to_json(), indent=1, default=str))
        (out / "report.txt").write_text(self.report.summary() + "\n")
        (out / "feature_names.json").write_text(json.dumps({
            "territories": list(self.graph.territories), "node": self.spec.node_names,
            "global": self.spec.global_names, "candidate": self.spec.candidate_names}, indent=1))
        if splits is not None:
            (out / "splits.json").write_text(json.dumps(splits))
        return out

    @classmethod
    def load(cls, data_dir: str = str(DEFAULT_DATA_DIR)) -> "AttackFormationDataset":
        with open(Path(data_dir) / "dataset.pkl", "rb") as f:
            d = pickle.load(f)
        if d.get("version") != PREPROCESSING_VERSION:
            raise ValueError(f"{data_dir}: preprocessing version {d.get('version')} != "
                             f"{PREPROCESSING_VERSION}; rebuild the dataset")
        return cls(d["samples"], d["graph"], d["spec"], d["report"], d["config"])


def collate(items: Sequence[dict]) -> dict:
    """Pad a list of dataset items into a batch: candidate_features (B, Cmax, F) with
    candidate_mask (B, Cmax); node_features (B, T, F) share one edge_index (static graph)."""
    cmax = max(it["candidate_features"].shape[0] for it in items)
    fc = items[0]["candidate_features"].shape[1]
    cand = np.zeros((len(items), cmax, fc), dtype=np.float32)
    mask = np.zeros((len(items), cmax), dtype=bool)
    for b, it in enumerate(items):
        c = it["candidate_features"].shape[0]
        cand[b, :c] = it["candidate_features"]
        mask[b, :c] = True
    return {"node_features": np.stack([it["node_features"] for it in items]),
            "edge_index": items[0]["edge_index"],
            "global_features": np.stack([it["global_features"] for it in items]),
            "candidate_features": cand, "candidate_mask": mask,
            "expert_index": np.array([it["expert_index"] for it in items], dtype=np.int64),
            "z": np.array([it["z"] for it in items], dtype=np.float32),
            "metadata": [it["metadata"] for it in items]}


def _expand_paths(records: Sequence[str]) -> List[str]:
    paths = []
    for r in ([records] if isinstance(records, (str, Path)) else records):
        if os.path.isdir(r):
            paths.extend(sorted(glob.glob(os.path.join(r, "*.jsonl"))))
        else:
            paths.append(str(r))
    return sorted(set(paths))


def _fingerprint(paths, game_xml, check_legality) -> str:
    h = hashlib.sha256()
    h.update(f"{PREPROCESSING_VERSION}|{check_legality}|{ALLOCATION_TYPES}".encode())
    for p in list(paths) + ([game_xml] if game_xml else []):
        h.update(os.path.basename(p).encode())
        h.update(Path(p).read_bytes())
    return h.hexdigest()


# =================================================================================================
# Building samples
# =================================================================================================

class _Builder:
    def __init__(self, paths, game_xml, strict, check_legality):
        self.paths = paths
        self.game_xml = game_xml
        self.check_legality = check_legality
        self.report = ValidationReport(strict=strict)
        self.rng = fm.SeededRng(0)  # moves roll no dice; apply_in_place still wants an rng

    # ---- setup ----------------------------------------------------------------------------

    def build(self) -> AttackFormationDataset:
        R = self.report
        records = []
        for p in self.paths:
            R.counts["records"] += 1
            try:
                records.append(read_record(p))
            except Exception as e:
                R.rejected_records.append({"record": os.path.basename(p), "reasons": [repr(e)]})
        if not records:
            raise ValueError("no readable records")

        digests = {r.meta.get("definition_digest") for r in records}
        if len(digests) != 1 or None in digests:
            raise ValueError(f"records come from different or unknown definitions: {digests}")
        digest = digests.pop()
        xml_path = self.game_xml or find_game_xml(digest)
        xml = Path(xml_path).read_text(encoding="utf-8")
        self.d, support = fm.load_definition(xml, strict=False)
        if self.d.definition_digest != digest:
            raise ValueError(f"{xml_path} has digest {self.d.definition_digest}, records {digest}")
        if not support.supported:
            raise ValueError(f"{xml_path}: forward model does not support this game")
        self.map = map_graph.MapGraph.from_xml(xml)

        # A3: one static graph, checked against the definition and every record.
        self.graph = StaticGraph.from_observation(records[0].header["initial_state"])
        for msg in self.graph.definition_differences(self.d):
            R.fail("A3_static_graph", {"record": "definition"}, msg)
        self.spec = FeatureSpec(self.graph.territories, game_unit_types(self.d, self.map))

        self._check_game_pairs(records)
        samples = []
        for rec in records:
            samples.extend(self._record(rec))
        config = {"records": [os.path.basename(p) for p in self.paths], "game_xml": xml_path,
                  "definition_digest": digest, "check_legality": self.check_legality,
                  "allocation_types": ALLOCATION_TYPES}
        R.counts["samples_ok"] = len(samples)
        return AttackFormationDataset(samples, self.graph, self.spec, R, config)

    def _check_game_pairs(self, records):
        """Both factions' records of one game must report the same result (A2)."""
        by_game = defaultdict(list)
        for r in records:
            if r.result is not None:
                by_game[r.game_id].append(r)
        for gid, rs in by_game.items():
            results = {(tuple(sorted(r.result.get("winners") or [])), r.result.get("rounds_played"))
                       for r in rs}
            if len(results) > 1:
                for r in rs:
                    self.report.fail("A2_game_completed", {"record": r.name},
                                     f"records of {gid} disagree on the result: {results}")

    # ---- per record -----------------------------------------------------------------------

    def _record(self, rec: Record) -> List[Sample]:
        R = self.report
        z, problems = game_outcome(rec)
        reasons = list(problems)
        for msg in problems:
            R.fail("A2_game_completed", {"record": rec.name}, msg)
        for msg in self.graph.differences(rec.header["initial_state"]):
            R.fail("A3_static_graph", {"record": rec.name, "where": "initial_state"}, msg)
            reasons.append("static graph differs")
        if reasons:
            R.rejected_records.append({"record": rec.name, "reasons": sorted(set(reasons)),
                                       "steps": len(rec.steps)})
            R.counts["steps_in_rejected_records"] += len(rec.steps)
            return []
        R.counts["records_accepted"] += 1
        R.outcomes["win" if z else "loss"] += 1
        bridge = LiveBridge(self.d, rec.actor, rec.meta.get("game_number", 1))
        samples = []
        for step in rec.steps:
            R.counts["steps"] += 1
            if step.get("phase") != PHASE:
                R.counts["steps_other_phase"] += 1
                continue
            try:
                samples.extend(self._step(rec, step, bridge, z))
            except DatasetAssertionError:
                raise
            except Exception as e:  # report, never silently drop
                R.counts["steps_failed"] += 1
                R.failed_samples.append({"record": rec.name, "step": step.get("index"),
                                         "level": None, "reasons": [f"exception: {e!r}"]})
        return samples

    # ---- per step (one search tree) -------------------------------------------------------

    def _step(self, rec: Record, step: dict, bridge: LiveBridge, z: int) -> List[Sample]:
        R = self.report
        actor = rec.actor
        where = {"record": rec.name, "step": step.get("index"), "round": step.get("round")}
        path = step.get("path") or []
        if not path:
            R.counts["steps_failed"] += 1
            R.failed_samples.append({**where, "level": None, "reasons": ["step has no path"]})
            return []

        # A1 and A3 on the observation and every path snapshot.
        step_ok = True
        for k, obs in [("state", step["state"])] + [(f"path[{i}]", e["state"]) for i, e in enumerate(path)]:
            if not (obs.get("observationPlayer") == obs.get("activePlayer") == actor):
                R.fail("A1_observer_is_active", {**where, "where": k},
                       f"observationPlayer {obs.get('observationPlayer')}, activePlayer "
                       f"{obs.get('activePlayer')}, recording faction {actor}")
                step_ok = False
            for msg in self.graph.differences(obs):
                R.fail("A3_static_graph", {**where, "where": k}, msg)
                step_ok = False
        opponents = [p["name"] for p in step["state"]["players"] if p["name"] != actor]
        if len(opponents) != 1:
            R.notes.append(f"{rec.name} step {step.get('index')}: expected one opponent, "
                           f"got {opponents}")
            step_ok = False
        if not step_ok:
            R.counts["steps_failed"] += 1
            n = sum(1 for e in path if e.get("chosen") is not None)
            R.counts["samples_failed"] += n
            R.failed_samples.append({**where, "level": None, "samples": n,
                                     "reasons": ["A1/A3 failed for this step"]})
            return []
        opponent = opponents[0]

        # A4 (root): the tree's root snapshot is the observation the agent answered.
        root_obs = normalize_observation(path[0]["state"])
        root_units = unit_table(root_obs)
        outer_units = unit_table(normalize_observation(step["state"]))
        if outer_units != root_units:
            diff = [i[:8] for i in set(outer_units) | set(root_units)
                    if outer_units.get(i) != root_units.get(i)]
            R.fail("A4_pre_state_and_target", {**where, "level": 0},
                   f"root snapshot units differ from the observation for {sorted(diff)[:8]}")

        samples = []
        excluded = set()
        committed: Dict[str, str] = {}  # unit id -> territory it was committed to
        history = None  # (conquered, blitzed) after the expert actions so far
        for k, node in enumerate(path):
            R.counts["path_nodes"] += 1
            w = {**where, "level": k}
            pre_obs = normalize_observation(node["state"])
            sim = bridge.build_state(pre_obs, PHASE)
            if history is not None:
                sim.conquered, sim.blitzed = set(history[0]), set(history[1])
            view = map_graph.GameState.from_sim(self.map, self.d, sim, actor)
            actions, next_excluded = ag.combat_actions(view, self.d, set(excluded), actor)
            if not actions:  # as MCTSGameState.heuristic_combat_legal_moves
                actions = [ag.CombatAction(end_phase=True, strength=0)]
            units = unit_table(pre_obs)

            if node.get("chosen") is None:
                self._leaf(node, actions, w)
                continue
            R.counts["decision_nodes"] += 1
            failed: List[str] = []

            def fail(assertion, msg):
                failed.append(f"{assertion}: {msg}")
                R.fail(assertion, w, msg)

            # A4: snapshot = chosen child of the previous node.
            if k > 0:
                prev = path[k - 1]
                expected = normalize_observation(prev["children"][prev["chosen"]]["state"])
                if expected != pre_obs:
                    fail("A4_pre_state_and_target", "snapshot is not the previous chosen child's "
                         f"state: {observation_differences(expected, pre_obs)}")

            expert = recorded_candidate(node["children"][node["chosen"]]["action"], units)
            cands, dups = self._candidates(actions, bridge, units)
            R.counts["duplicate_candidates_merged"] += dups
            targets = {c.target for c in cands}

            # A4: one heuristic-fixed target, and it is the expert's.
            if len(targets) != 1 or expert.target not in targets:
                fail("A4_pre_state_and_target",
                     f"generator targets {sorted(map(str, targets))}, expert target {expert.target}")
            # A9: skip and end-phase are distinct.
            kinds = Counter(c.kind for c in cands)
            if kinds["end_phase"] or kinds["skip"] != 1 or cands[-1].kind != "skip":
                fail("A9_skip_end_phase", f"candidate kinds {dict(kinds)} (expected attacks then "
                     "exactly one skip, no end-phase at a decision node)")
            if expert.kind == "end_phase":
                fail("A9_skip_end_phase", "the expert action is an end-phase at a decision node")
            # A7: units, availability and commitments.
            for msg in self._unit_checks(units, root_units, committed, expert, cands, actor):
                fail("A7_units_consistent", msg)
            # A5: expert among the candidates.
            keys = [c.key for c in cands]
            expert_index = keys.index(expert.key) if expert.key in keys else None
            if expert_index is None:
                fail("A5_expert_in_candidates", f"expert {expert.kind} on {expert.target} "
                     f"({sum(len(m['unitIds']) for m in expert.moves)} units) not among "
                     f"{len(cands)} generated candidates")
            elif cands[expert_index].strength != expert.strength:
                fail("A5_expert_in_candidates", f"matched candidate strength "
                     f"{cands[expert_index].strength} != recorded {expert.strength}")
            visits = [0] * len(cands)
            for child in node["children"]:
                R.counts["mcts_children"] += 1
                ck = recorded_candidate(child["action"], units).key
                if ck in keys:
                    visits[keys.index(ck)] += child["visits"]
                else:
                    R.counts["mcts_children_not_generated"] += 1
            # A6: every candidate is legal.
            if self.check_legality:
                for i, c in enumerate(cands):
                    msg = self._illegal(sim, c, bridge)
                    if msg:
                        fail("A6_candidates_legal", f"candidate {i} ({c.kind}): {msg}")
            # A8: expert action reproduces the next path snapshot.
            nxt = path[k + 1]["state"] if k + 1 < len(path) else None
            if nxt is None:
                fail("A8_expert_reproduces_next", "no next path node after a chosen child")
            else:
                msg, after = self._reproduces(sim, expert, bridge, nxt, actor)
                if msg:
                    fail("A8_expert_reproduces_next", msg)
                if after is not None:
                    history = (set(after.conquered), set(after.blitzed))
            # Candidate unit types must fit the allocation block.
            bad = {units[u]["type"] for c in cands for m in c.moves for u in m["unitIds"]
                   if units.get(u, {}).get("type") not in ALLOCATION_TYPES}
            if bad:
                failed.append(f"candidate unit types {sorted(bad)} outside {ALLOCATION_TYPES}")

            if failed:
                R.counts["samples_failed"] += 1
                R.failed_samples.append({**w, "reasons": failed})
            else:
                samples.append(self._sample(rec, step, k, pre_obs, units, committed, excluded,
                                            cands, expert, expert_index, visits, opponent, z))
                R.candidate_counts.append(len(cands))
                R.samples_by_actor[actor] += 1
                R.counts["expert_skip"] += expert.kind == "skip"
                R.counts["single_candidate_samples"] += len(cands) == 1
            excluded = next_excluded
            for mv in expert.moves:
                for uid in mv["unitIds"]:
                    committed[uid] = mv["route"][-1]
        return samples

    def _leaf(self, node: dict, actions, w: dict):
        """The end of the chosen path: no sample. Check its reason against the generator."""
        R = self.report
        R.counts["leaf_nodes"] += 1
        reason = node.get("terminal_reason")
        R.leaves[str(reason)] += 1
        only_end = all(a.end_phase for a in actions)
        if reason == "no_target" and not only_end:
            R.fail("A9_skip_end_phase", w, "leaf marked no_target but the generator finds target "
                   f"{actions[0].to_terr}")
        if reason == "timeout" and only_end:
            R.fail("A9_skip_end_phase", w, "leaf marked timeout but the generator finds no target")
        if node.get("children"):
            R.fail("A9_skip_end_phase", w, "path ends at a node that has children but none chosen")

    def _candidates(self, actions, bridge: LiveBridge, units) -> Tuple[List[Candidate], int]:
        return generated_candidates(actions, bridge, units)

    def _move_actions(self, cand: Candidate, bridge: LiveBridge):
        ti = self.d.territory_index
        return [fm.MoveAction(tuple(bridge.uid_of[u] for u in mv["unitIds"]),
                              tuple(ti[t] for t in mv["route"]), mv.get("intent", "NORMAL"))
                for mv in cand.moves]

    def _illegal(self, sim, cand: Candidate, bridge) -> Optional[str]:
        work = sim.clone()
        try:
            for mv in self._move_actions(cand, bridge):
                if fm.apply_in_place(self.d, work, mv, self.rng) == fm.REJECTED:
                    return f"move {mv} rejected by the forward model"
        except KeyError as e:
            return f"unknown unit id {e}"
        return None

    def _reproduces(self, sim, expert: Candidate, bridge, next_obs, actor):
        """(problem or None, state after the expert action or None)."""
        work = sim.clone()
        try:
            for mv in self._move_actions(expert, bridge):
                if fm.apply_in_place(self.d, work, mv, self.rng) == fm.REJECTED:
                    return f"expert move {mv} rejected by the forward model", None
        except KeyError as e:
            return f"unknown unit id {e}", None
        got = engine_observation(self.d, work, actor, bridge.uuid_of)
        got, want = normalize_observation(got), normalize_observation(next_obs)
        if got != want:
            return f"differs: {observation_differences(want, got)}", work
        return None, work

    def _unit_checks(self, units, root_units, committed, expert, cands, actor) -> List[str]:
        out = []
        if set(units) != set(root_units):
            out.append(f"unit set changed since the root: +{len(set(units) - set(root_units))} "
                       f"-{len(set(root_units) - set(units))}")
        for uid, dest in committed.items():
            u = units.get(uid)
            r = root_units.get(uid)
            if u is None or r is None or u["territory"] != dest or u["movementUsed"] <= r["movementUsed"]:
                out.append(f"committed unit {uid[:8]} is not at {dest} with movement used ({u})")
        for uid, u in units.items():
            r = root_units.get(uid)
            if uid not in committed and r is not None and (u["territory"], u["movementUsed"], u["hits"]) \
                    != (r["territory"], r["movementUsed"], r["hits"]):
                out.append(f"uncommitted unit {uid[:8]} changed since the root: {r} -> {u}")
        for label, c in [("expert", expert)] + [(f"candidate {i}", c) for i, c in enumerate(cands)]:
            for mv in c.moves:
                for uid in mv["unitIds"]:
                    u = units.get(uid)
                    if u is None:
                        out.append(f"{label}: unit {uid[:8]} not in the snapshot")
                    elif u["owner"] != actor or u["territory"] != mv["route"][0] or uid in committed \
                            or u["movementLeft"] < len(mv["route"]) - 1:
                        out.append(f"{label}: unit {uid[:8]} not available for {mv['route']} ({u}, "
                                   f"committed={uid in committed})")
        return out

    def _sample(self, rec, step, k, pre_obs, units, committed, excluded, cands, expert,
                expert_index, visits, opponent, z) -> Sample:
        actor = rec.actor
        target = expert.target
        unit_cost = lambda t: self.map.unit_cost(t) or 0.0  # noqa: E731
        x = node_features(self.spec, self.graph, pre_obs, actor, opponent, committed, target)
        g = global_features(self.graph, pre_obs, actor, opponent)
        c = np.stack([candidate_features(self.spec, self.graph, cd, units, unit_cost)
                      for cd in cands])
        meta = {
            "game_id": rec.game_id, "record": rec.name, "actor": actor, "opponent": opponent,
            "round": step.get("round"), "step_index": step.get("index"), "level": k,
            "target": target, "excluded_before": sorted(excluded),
            "candidates": [cd.to_json() for cd in cands],
            "candidate_keys": [cd.key for cd in cands],
            "expert_action": expert.to_json(),
            "mcts_visits": visits,
            "committed": dict(committed),
            "units": units,  # id -> type, owner, territory, movement, hits (never a feature)
            "search": step.get("search"),
        }
        return Sample(x, g, c, int(expert_index), float(z), meta)

