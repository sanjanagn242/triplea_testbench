"""Baseline candidate scorer: an MLP over (flattened territory features + global features) and
each candidate's features, one scalar score per candidate, softmax within each decision."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import nn

MASKED = -1e9  # score of padding candidates: exp() underflows to 0 without producing NaNs


def mlp(sizes: Sequence[int], dropout: float = 0.0, last_activation: bool = True) -> nn.Sequential:
    layers = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(sizes[i], sizes[i + 1]))
        if i < len(sizes) - 2 or last_activation:
            layers.append(nn.ReLU())
            if dropout > 0:
                layers.append(nn.Dropout(dropout))
    return nn.Sequential(*layers)


class Standardizer(nn.Module):
    """Per-feature mean/std fitted on the training split only (stored with the model)."""

    def __init__(self, state_dim: int, cand_dim: int):
        super().__init__()
        self.register_buffer("state_mean", torch.zeros(state_dim))
        self.register_buffer("state_std", torch.ones(state_dim))
        self.register_buffer("cand_mean", torch.zeros(cand_dim))
        self.register_buffer("cand_std", torch.ones(cand_dim))

    @torch.no_grad()
    def fit(self, node, glob, cand, mask):
        state = flatten_state(node, glob)
        real = cand[mask]
        for name, x in (("state", state), ("cand", real)):
            mean, std = x.mean(0), x.std(0, unbiased=False)
            getattr(self, f"{name}_mean").copy_(mean)
            getattr(self, f"{name}_std").copy_(torch.where(std > 1e-6, std, torch.ones_like(std)))

    def forward(self, state, cand):
        return (state - self.state_mean) / self.state_std, (cand - self.cand_mean) / self.cand_std


def flatten_state(node: torch.Tensor, glob: torch.Tensor) -> torch.Tensor:
    return torch.cat([node.flatten(1), glob], dim=1)


class CandidateScorer(nn.Module):
    def __init__(self, node_shape, glob_dim: int, cand_dim: int, hidden_dims=(256, 128),
                 dropout: float = 0.1):
        super().__init__()
        self.config = {"node_shape": list(node_shape), "glob_dim": glob_dim, "cand_dim": cand_dim,
                       "hidden_dims": list(hidden_dims), "dropout": dropout}
        state_dim = node_shape[0] * node_shape[1] + glob_dim
        h = hidden_dims[-1]
        self.norm = Standardizer(state_dim, cand_dim)
        self.state_encoder = mlp([state_dim, *hidden_dims], dropout)
        self.cand_encoder = mlp([cand_dim, h], dropout)
        self.scorer = mlp([2 * h, h, 1], dropout, last_activation=False)

    def forward(self, node, glob, cand, mask) -> torch.Tensor:
        """Scores (B, C); padding candidates get MASKED."""
        state, cand = self.norm(flatten_state(node, glob), cand)
        s = self.state_encoder(state)                       # (B, H)
        c = self.cand_encoder(cand)                         # (B, C, H)
        s = s[:, None, :].expand(-1, c.shape[1], -1)
        scores = self.scorer(torch.cat([s, c], dim=-1)).squeeze(-1)
        return scores.masked_fill(~mask, MASKED)

    @classmethod
    def from_config(cls, config: dict) -> "CandidateScorer":
        return cls(config["node_shape"], config["glob_dim"], config["cand_dim"],
                   config["hidden_dims"], config["dropout"])


def check_softmax(scores: torch.Tensor, mask: torch.Tensor, atol: float = 1e-5):
    """Sanity check 3: probabilities sum to 1 within each decision and are 0 on padding."""
    p = torch.softmax(scores, dim=1)
    sums = p.sum(1)
    if not torch.allclose(sums, torch.ones_like(sums), atol=atol):
        raise AssertionError(f"softmax does not sum to 1 (worst {sums.sub(1).abs().max():.2e})")
    if (p[~mask] > atol).any():
        raise AssertionError("softmax gives probability to padding candidates")
    return p
