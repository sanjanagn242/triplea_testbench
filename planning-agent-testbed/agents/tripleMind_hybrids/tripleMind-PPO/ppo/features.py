"""Actor-relative features of one attack-formation decision.

The same preprocessing functions as the behaviour-cloning dataset and the tripleMind-bc agent
(behaviour-cloning/data-processing/attack_formation_dataset.py): the generator's formations are
canonicalised with generated_candidates (equivalent formations merged, skip kept), and the
territory, global and candidate features are built from an engine-style observation of the
forward-model state, relative to the deciding faction.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import paths  # noqa: F401  (sys.path)

import triplea_fm as fm  # noqa: E402
from triplea_fm.testbench import observation_from_state  # noqa: E402

import map_graph  # noqa: E402  (TripleMind)
from attack_formation_dataset import (FeatureSpec, StaticGraph, candidate_features,  # noqa: E402
                                      engine_observation, game_unit_types, generated_candidates,
                                      global_features, node_features, normalize_observation,
                                      unit_table)


@dataclass
class Decision:
    """One formation decision: features plus the mapping back to the generator's actions."""
    node: np.ndarray          # (T, F_node) territory features
    glob: np.ndarray          # (F_global,)
    cand: np.ndarray          # (C, F_cand), one row per legal (canonical) candidate
    sources: List[int]        # generator action index of each candidate
    kinds: List[str]          # "attack" | "skip"
    target: str


def feature_names(graph: StaticGraph, spec: FeatureSpec) -> dict:
    """The checkpoint's record of what the inputs mean (as feature_names.json in BC)."""
    return {"territories": list(graph.territories), "node": spec.node_names,
            "global": spec.global_names, "candidate": spec.candidate_names}


def game_feature_names(xml_path: str, player: Optional[str] = None) -> dict:
    """Feature names of a game, from its initial state (no live game needed)."""
    xml = Path(xml_path).read_text()
    d, _ = fm.load_definition(xml)
    mapg = map_graph.MapGraph.from_xml(xml)
    player = player or d.player_names[0]
    obs = normalize_observation(observation_from_state(d, fm.create_initial_state(d), player))
    graph = StaticGraph.from_observation(obs)
    return feature_names(graph, FeatureSpec(graph.territories, game_unit_types(d, mapg)))


def shapes(names: dict) -> dict:
    return {"node_shape": [len(names["territories"]), len(names["node"])],
            "glob_dim": len(names["global"]), "cand_dim": len(names["candidate"])}


class DecisionEncoder:
    """Builds Decisions for one faction in one game (BCPolicy's feature path, without a model)."""

    def __init__(self, definition, mapg, player: str, bridge, expected_names: Optional[dict] = None):
        self.d, self.map, self.player, self.bridge = definition, mapg, player, bridge
        self.unit_types = game_unit_types(definition, mapg)
        self.expected_names = expected_names
        self.graph: Optional[StaticGraph] = None
        self.spec: Optional[FeatureSpec] = None
        self.opponent: Optional[str] = None

    def _bind(self, obs: dict):
        self.graph = StaticGraph.from_observation(obs)
        self.spec = FeatureSpec(self.graph.territories, self.unit_types)
        live = feature_names(self.graph, self.spec)
        if self.expected_names is not None and live != self.expected_names:
            diff = [k for k in live if live[k] != self.expected_names.get(k)]
            raise ValueError(f"this game's features differ from the policy's ({diff}); "
                             f"the policy was trained on another map")
        opponents = [p["name"] for p in obs["players"] if p["name"] != self.player]
        if len(opponents) != 1:
            raise ValueError(f"the policy needs exactly one opponent, got {opponents}")
        self.opponent = opponents[0]

    def encode(self, sim, actions, committed: Dict[str, str]) -> Decision:
        """`actions`: the generator's formations for one target (attack sets, then the skip);
        `committed`: engine unit id -> territory, for units moved earlier this phase."""
        obs = normalize_observation(engine_observation(self.d, sim, self.player, self.bridge.uuid_of))
        if self.graph is None:
            self._bind(obs)
        units = unit_table(obs)
        cands, _ = generated_candidates(actions, self.bridge, units)
        if not cands:
            raise ValueError("the generator returned no candidate")
        if any(c.kind == "end_phase" for c in cands):
            raise ValueError("end-phase candidate in a formation decision")
        target = cands[0].target
        if any(c.target != target for c in cands):
            raise ValueError(f"candidates for several targets: {sorted({c.target for c in cands})}")
        x = node_features(self.spec, self.graph, obs, self.player, self.opponent, committed, target)
        g = global_features(self.graph, obs, self.player, self.opponent)
        unit_cost = lambda t: self.map.unit_cost(t) or 0.0  # noqa: E731  (as in preprocessing)
        c = np.stack([candidate_features(self.spec, self.graph, cd, units, unit_cost) for cd in cands])
        return Decision(x, g, c, [cd.source for cd in cands], [cd.kind for cd in cands], target)
