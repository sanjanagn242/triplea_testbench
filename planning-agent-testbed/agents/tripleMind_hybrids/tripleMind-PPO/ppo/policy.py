"""The PPO formation chooser, with BCPolicy's interface (tripleMind-bc/bc_policy.py):

    choose(sim, actions, committed) -> (index into `actions`, candidate scores, info)

so TripleMindBC.combat_move (target heuristic, applying the moves, committed units) is used as it
is. Modes:
  sample   draw from the softmax over the legal candidates (training rollouts)
  greedy   argmax (evaluation and deployment)
  uniform  uniformly random legal candidate (normalisation warm-up, "random" baseline)

With record=True every decision is kept as a Transition (features, legal-candidate count, chosen
index, log-probability and value of the frozen snapshot); rewards are filled in by the rollout
collector when the game ends.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import torch

from .features import DecisionEncoder
from .model import ActorCritic

MODES = ("sample", "greedy", "uniform")
PROB_ATOL = 1e-5


@dataclass
class Transition:
    node: np.ndarray            # (T, F_node)
    glob: np.ndarray            # (F_global,)
    cand: np.ndarray            # (C, F_cand)
    action: int                 # index into the legal candidates (not the generator's list)
    logp: float                 # log pi_old(action), from the snapshot that acted
    value: float                # V_old(state)
    actor: str                  # faction that decided (rewards are from its perspective)
    round: int
    target: str
    reward: float = 0.0
    done: bool = False

    @property
    def n_cand(self) -> int:
        return len(self.cand)


class FormationPolicy:
    def __init__(self, encoder: DecisionEncoder, model: Optional[ActorCritic], mode: str = "greedy",
                 record: bool = False, seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode {mode!r} not in {MODES}")
        if model is None and mode != "uniform":
            raise ValueError("a model is needed unless mode is 'uniform'")
        self.encoder, self.model, self.mode, self.record = encoder, model, mode, record
        self.gen = torch.Generator().manual_seed(seed)
        self.trajectory: List[Transition] = []
        self.error: Optional[BaseException] = None  # TripleMindBC catches errors; rollouts re-raise
        self.decisions = 0
        self.seconds = 0.0   # encoding + forward pass, for the computational-cost report

    @torch.no_grad()
    def choose(self, sim, actions, committed: Dict[str, str]):
        try:
            return self._choose(sim, actions, committed)
        except BaseException as e:
            self.error = e
            raise

    def _choose(self, sim, actions, committed):
        started = time.perf_counter()
        dec = self.encoder.encode(sim, actions, committed)
        n = len(dec.sources)
        node, glob = torch.from_numpy(dec.node)[None], torch.from_numpy(dec.glob)[None]
        cand = torch.from_numpy(dec.cand)[None]
        mask = torch.ones(1, n, dtype=torch.bool)
        if self.model is not None:
            scores = self.model.scores(node, glob, cand, mask)[0]
        else:
            scores = torch.zeros(n)
        probs = torch.softmax(scores, 0)
        if not torch.isfinite(probs).all() or abs(float(probs.sum()) - 1.0) > PROB_ATOL:
            raise AssertionError(f"action probabilities do not normalise: sum {float(probs.sum())}")
        if self.mode == "greedy":
            k = int(scores.argmax())
        else:
            k = int(torch.multinomial(probs, 1, generator=self.gen))
        # Legality: the choice is one of this decision's generated candidates, for its target.
        if not 0 <= k < n:
            raise AssertionError(f"chosen candidate {k} outside the {n} legal candidates")
        source = dec.sources[k]
        if not 0 <= source < len(actions) or actions[source].to_terr != dec.target \
                or actions[source].end_phase:
            raise AssertionError(f"chosen candidate {k} maps to an invalid generator action {source}")
        if self.record:
            logp = float(torch.log_softmax(scores, 0)[k])
            value = float(self.model.value(node, glob)[0]) if self.model is not None else 0.0
            self.trajectory.append(Transition(dec.node, dec.glob, dec.cand, k, logp, value,
                                              self.encoder.player, int(sim.round), dec.target))
        self.decisions += 1
        self.seconds += time.perf_counter() - started
        info = {"target": dec.target, "candidates": n, "chosen": k, "kind": dec.kinds[k],
                "probs": probs.tolist()}
        return source, scores.tolist(), info
