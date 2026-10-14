"""TripleMindPPO: TripleMind with a PPO policy choosing the attack formation.

A subclass of tripleMind-bc's TripleMindBC, which is TripleMind with a pluggable formation chooser
(its `bc` attribute). Everything but the chooser is inherited unchanged:
  purchase       the first purchase TripleMind's generator proposes
  combatMove     targets from TripleMind's heuristic (generators.ag.combat_actions, most important
                 first); for each target the chooser picks one of the generator's formations (an
                 attack set or the skip); its moves are applied with the forward model and the next
                 target is generated from the new state, until no target is left
  nonCombatMove  the generator's noncombat moves
  place          {} : the engine places all purchased units
  battle         fight all
The responses go through SearchAgent's checking and encoding, the action pathway of every other
search agent; game rules are the engine's (or, in rollouts, the forward model's).

Choosers (`chooser`): "ppo" (the policy, `mode` greedy or sample), "uniform" (uniformly random
legal formation) and "first" (no policy: TripleMindBC's fallback, the generator's first
formation).
"""

from __future__ import annotations

import os
import random
from pathlib import Path
from typing import Dict, Optional, Tuple

from . import paths
from .features import DecisionEncoder
from .model import ActorCritic, load_checkpoint
from .policy import FormationPolicy

bc_agent = paths.bc_agent_module()
map_graph = bc_agent.map_graph
generators = bc_agent.generators

CHOOSERS = ("ppo", "uniform", "first")
_CHECKPOINTS: Dict[Tuple[str, float], Tuple[ActorCritic, dict]] = {}


def cached_checkpoint(path) -> Tuple[ActorCritic, dict]:
    """A checkpoint is loaded once per process (an agent process plays many games)."""
    path = Path(path)
    key = (str(path.resolve()), path.stat().st_mtime)
    if key not in _CHECKPOINTS:
        _CHECKPOINTS[key] = load_checkpoint(path)
    return _CHECKPOINTS[key]


class TripleMindPPO(bc_agent.TripleMindBC):
    def __init__(self, model: Optional[ActorCritic] = None, feature_names: Optional[dict] = None,
                 chooser: str = "ppo", mode: Optional[str] = None, record: bool = False,
                 seed: Optional[int] = None, policy_seed: Optional[int] = None,
                 checkpoint: Optional[str] = None):
        super().__init__()
        if chooser not in CHOOSERS:
            raise ValueError(f"chooser {chooser!r} not in {CHOOSERS}")
        if seed is not None:
            self.seed, self.rnd = seed, random.Random(seed)
        self.checkpoint = str(checkpoint or os.environ.get("TRIPLEMIND_PPO_CHECKPOINT")
                              or paths.DEFAULT_CHECKPOINT)
        self.chooser = chooser
        self.mode = mode or os.environ.get("TRIPLEMIND_PPO_MODE", "greedy")
        self.model, self.feature_names = model, feature_names
        self.record = record  # training rollouts: keep transitions, never fall back silently
        self.policy_seed = self.seed if policy_seed is None else policy_seed
        self.policy: Optional[FormationPolicy] = None
        self.result: Optional[dict] = None  # game_end result

    # ---- setup ----------------------------------------------------------------------------

    def setup(self):
        # As TripleMindBC.setup, with the PPO chooser instead of the BC checkpoint.
        random.seed(self.seed)
        self.map = map_graph.MapGraph.from_xml(self.game_xml)
        self.view = (map_graph.MapView(self.map, self.player_name)
                     if os.environ.get("TRIPLEMIND_VIEW", "1") == "1" else None)
        self.log(f"map graph: {self.map}")
        self.log(f"action generators: {generators.IMPLEMENTATION}")
        self.policy = self.bc = None
        self.result = None
        if self.chooser == "first":
            self.log("formation chooser: the generator's first (strongest) formation")
            return
        try:
            model, names = self.model, self.feature_names
            if self.chooser == "ppo" and model is None:
                model, ckpt = cached_checkpoint(self.checkpoint)
                names = ckpt["feature_names"]
                self.log(f"PPO policy: {self.checkpoint} (iteration {ckpt.get('iteration')})")
            encoder = DecisionEncoder(self.d, self.map, self.player_name, self.bridge, names)
            if self.chooser == "uniform":
                self.policy = FormationPolicy(encoder, None, "uniform", self.record, self.policy_seed)
            else:
                self.policy = FormationPolicy(encoder, model, self.mode, self.record, self.policy_seed)
            self.bc = self.policy  # TripleMindBC.combat_move asks self.bc.choose(...)
        except Exception as e:
            if self.record:
                raise
            self.log(f"PPO policy unavailable ({e!r}); combat moves use the generator's first "
                     f"(strongest) formation")

    # ---- protocol -------------------------------------------------------------------------

    def handle(self, request: dict) -> dict:
        if request.get("type") == "game_end":
            self.result = request.get("result") or {}
            self.log(f"game over: {self.result.get('status')} winners {self.result.get('winners')}")
            return {"ok": True}
        return super().handle(request)

    @property
    def policy_disabled(self) -> bool:
        """True when TripleMindBC switched the policy off after an error during this game."""
        return self.policy is not None and self.bc is None

    def outcome(self) -> float:
        """+1 won, -1 lost, 0 no winner (round limit), from this faction's perspective."""
        if self.result is None:
            raise RuntimeError("the game has not ended")
        if self.result.get("won"):
            return 1.0
        return -1.0 if self.result.get("winners") else 0.0
