"""Actor-critic for attack-formation selection.

Actor: behaviour cloning's CandidateScorer (behaviour-cloning/bc_baseline/model.py), unchanged:
flattened territory features + global features -> state encoding, each candidate's features ->
candidate encoding, one score per candidate, masked softmax over the decision's legal candidates.
It is built from scratch (no BC weights); dropout is 0 so stored and recomputed log-probabilities
agree exactly.

Critic: a separate MLP from the same state input (territory + global features, no candidate
features) to one value, the expected return of the deciding faction from this decision on.

Both standardise their inputs with running statistics collected from rollouts (BC fits them on
its training split; PPO has none), stored as buffers so a checkpoint carries them.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence, Tuple

import torch
import torch.nn.functional as F
from torch import nn

from . import paths  # noqa: F401  (sys.path)

from bc_baseline.model import MASKED, CandidateScorer, flatten_state, mlp  # noqa: E402

CHECKPOINT_FORMAT = "tripleMind-ppo/1"


class RunningMeanStd:
    """Mean and variance over everything seen so far (Chan et al. parallel update), float64."""

    def __init__(self, dim: int):
        self.count = 0
        self.mean = torch.zeros(dim, dtype=torch.float64)
        self.m2 = torch.zeros(dim, dtype=torch.float64)

    def update(self, x: torch.Tensor):
        if len(x) == 0:
            return
        x = x.to(torch.float64)
        n, mean = len(x), x.mean(0)
        m2 = ((x - mean) ** 2).sum(0)
        total = self.count + n
        delta = mean - self.mean
        self.mean = self.mean + delta * n / total
        self.m2 = self.m2 + m2 + delta ** 2 * self.count * n / total
        self.count = total

    @property
    def std(self) -> torch.Tensor:
        std = (self.m2 / max(self.count, 1)).sqrt()
        return torch.where(std > 1e-6, std, torch.ones_like(std))  # constant features: as BC

    def state_dict(self) -> dict:
        return {"count": self.count, "mean": self.mean.clone(), "m2": self.m2.clone()}

    def load_state_dict(self, s: dict):
        self.count, self.mean, self.m2 = s["count"], s["mean"].clone(), s["m2"].clone()


class ValueNet(nn.Module):
    """State-value critic: standardised (territory features, global features) -> V."""

    def __init__(self, node_shape: Sequence[int], glob_dim: int, hidden_dims=(256, 128)):
        super().__init__()
        state_dim = node_shape[0] * node_shape[1] + glob_dim
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.body = mlp([state_dim, *hidden_dims, 1], last_activation=False)

    def forward(self, node: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
        state = (flatten_state(node, glob) - self.state_mean) / self.state_std
        return self.body(state).squeeze(-1)


class ActorCritic(nn.Module):
    def __init__(self, node_shape: Sequence[int], glob_dim: int, cand_dim: int,
                 actor_hidden=(256, 128), critic_hidden=(256, 128)):
        super().__init__()
        self.config = {"node_shape": list(node_shape), "glob_dim": glob_dim, "cand_dim": cand_dim,
                       "actor_hidden": list(actor_hidden), "critic_hidden": list(critic_hidden)}
        self.actor = CandidateScorer(node_shape, glob_dim, cand_dim, actor_hidden, dropout=0.0)
        self.critic = ValueNet(node_shape, glob_dim, critic_hidden)
        # Small final layers: the initial policy is close to uniform over the legal candidates
        # and the initial values close to 0 (standard PPO initialisation).
        for last in (self.actor.scorer[-1], self.critic.body[-1]):
            with torch.no_grad():
                last.weight.mul_(0.01)
                last.bias.zero_()

    @classmethod
    def from_config(cls, c: dict) -> "ActorCritic":
        return cls(c["node_shape"], c["glob_dim"], c["cand_dim"], c["actor_hidden"], c["critic_hidden"])

    @torch.no_grad()
    def set_normalization(self, state: RunningMeanStd, cand: RunningMeanStd):
        """Copy running statistics into the actor's Standardizer and the critic."""
        n = self.actor.norm
        n.state_mean.copy_(state.mean.float())
        n.state_std.copy_(state.std.float())
        n.cand_mean.copy_(cand.mean.float())
        n.cand_std.copy_(cand.std.float())
        self.critic.state_mean.copy_(state.mean.float())
        self.critic.state_std.copy_(state.std.float())

    def scores(self, node, glob, cand, mask) -> torch.Tensor:
        """Candidate scores (B, C); padding gets MASKED."""
        return self.actor(node, glob, cand, mask)

    def value(self, node, glob) -> torch.Tensor:
        return self.critic(node, glob)

    def evaluate(self, node, glob, cand, mask, actions) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """(log pi(action), entropy, V) for a batch of decisions."""
        logp_all = F.log_softmax(self.scores(node, glob, cand, mask), dim=1)
        logp = logp_all.gather(1, actions[:, None]).squeeze(1)
        p = logp_all.exp()
        entropy = -torch.where(mask, p * logp_all, torch.zeros_like(p)).sum(1)
        return logp, entropy, self.value(node, glob)


def new_model(names: dict, actor_hidden=(256, 128), critic_hidden=(256, 128)) -> ActorCritic:
    t, f = len(names["territories"]), len(names["node"])
    return ActorCritic((t, f), len(names["global"]), len(names["candidate"]), actor_hidden, critic_hidden)


def save_checkpoint(path, model: ActorCritic, feature_names: dict, *, norm: Optional[dict] = None,
                    optimizer: Optional[torch.optim.Optimizer] = None, **extra):
    """Everything to resume training (optimizer, running statistics, counters) and to deploy."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    data = {"format": CHECKPOINT_FORMAT, "model_config": model.config,
            "model_state": model.state_dict(), "feature_names": feature_names, "norm": norm,
            "optimizer_state": optimizer.state_dict() if optimizer is not None else None, **extra}
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(data, tmp)
    tmp.replace(path)  # never leave a half-written checkpoint behind


def load_checkpoint(path) -> Tuple[ActorCritic, dict]:
    ckpt = torch.load(str(path), map_location="cpu", weights_only=False)
    if ckpt.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(f"{path} is not a TripleMind PPO checkpoint (format {ckpt.get('format')!r})")
    model = ActorCritic.from_config(ckpt["model_config"])
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


__all__ = ["MASKED", "ActorCritic", "RunningMeanStd", "ValueNet", "load_checkpoint", "new_model",
           "save_checkpoint"]
