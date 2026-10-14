"""The behaviour-cloning attack-formation policy, as the agent uses it.

Given the forward-model state, the generator's formations for the heuristic-fixed target and the
units already committed this phase, BCPolicy builds exactly the features the policy was trained
on (the preprocessing functions of behaviour-cloning/data-processing/attack_formation_dataset.py)
and returns the index of the highest-scoring formation.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

AGENTS = Path(__file__).resolve().parents[2]  # planning-agent-testbed/agents
BC_DIR = AGENTS / "behaviour-cloning"
for _p in (BC_DIR / "data-processing", BC_DIR):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import numpy as np  # noqa: E402
import torch  # noqa: E402

from attack_formation_dataset import (FeatureSpec, StaticGraph, candidate_features,  # noqa: E402
                                      engine_observation, game_unit_types, generated_candidates,
                                      global_features, node_features, normalize_observation,
                                      unit_table)
from bc_baseline.evaluate import load_checkpoint  # noqa: E402

DEFAULT_CHECKPOINT = BC_DIR / "runs" / "mlp-baseline" / "best.pt"


class BCPolicy:
    def __init__(self, checkpoint: str, definition, mapg, player: str, bridge):
        self.checkpoint = str(checkpoint)
        self.model, ckpt = load_checkpoint(self.checkpoint)
        self.trained_features = ckpt["feature_names"]
        self.d, self.map, self.player, self.bridge = definition, mapg, player, bridge
        self.unit_types = game_unit_types(definition, mapg)
        self.graph: Optional[StaticGraph] = None
        self.spec: Optional[FeatureSpec] = None

    def _bind(self, obs: dict):
        """Static graph and feature spec of this game, checked against the checkpoint's."""
        self.graph = StaticGraph.from_observation(obs)
        self.spec = FeatureSpec(self.graph.territories, self.unit_types)
        live = {"territories": list(self.graph.territories), "node": self.spec.node_names,
                "global": self.spec.global_names, "candidate": self.spec.candidate_names}
        if live != self.trained_features:
            diff = [k for k in live if live[k] != self.trained_features.get(k)]
            raise ValueError(f"this game's features differ from the checkpoint's ({diff}); "
                             f"the policy was trained on another map")

    @torch.no_grad()
    def choose(self, sim, actions, committed: Dict[str, str]) -> Tuple[int, List[float], dict]:
        """(index into `actions`, scores of the distinct candidates, info) for one decision.
        `committed` maps engine unit ids already moved this phase to their destination."""
        obs = normalize_observation(engine_observation(self.d, sim, self.player, self.bridge.uuid_of))
        if self.graph is None:
            self._bind(obs)
        units = unit_table(obs)
        cands, _ = generated_candidates(actions, self.bridge, units)
        opponents = [p["name"] for p in obs["players"] if p["name"] != self.player]
        if len(opponents) != 1:
            raise ValueError(f"the policy needs exactly one opponent, got {opponents}")
        target = cands[0].target
        x = node_features(self.spec, self.graph, obs, self.player, opponents[0], committed, target)
        g = global_features(self.graph, obs, self.player, opponents[0])
        unit_cost = lambda t: self.map.unit_cost(t) or 0.0  # noqa: E731  (as in preprocessing)
        c = np.stack([candidate_features(self.spec, self.graph, cd, units, unit_cost) for cd in cands])
        scores = self.model(torch.from_numpy(x)[None], torch.from_numpy(g)[None],
                            torch.from_numpy(c)[None], torch.ones(1, len(cands), dtype=torch.bool))[0]
        best = int(scores.argmax())
        info = {"target": target, "candidates": len(cands), "chosen": best,
                "kind": cands[best].kind, "probs": torch.softmax(scores, 0).tolist()}
        return cands[best].source, scores.tolist(), info
