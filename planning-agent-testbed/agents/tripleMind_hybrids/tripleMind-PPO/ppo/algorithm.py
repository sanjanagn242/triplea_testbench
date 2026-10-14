"""PPO: generalised advantage estimation, the clipped surrogate objective, the value loss and the
entropy bonus, over variable-sized candidate sets.

Advantages are computed per trajectory, i.e. per (game, faction): a faction's consecutive
formation decisions (several per turn, the opponent's turns in between) form one sequence, and
values and rewards of the other faction never enter it.

Checks done on every update (AssertionError on failure):
  * the stored log-probabilities equal the ones the current parameters give before the first
    gradient step (the snapshot that acted is the network being updated);
  * losses and gradients are finite;
  * the update changes the parameters.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch

from .model import ActorCritic
from .policy import Transition

LOGP_ATOL = 1e-4


@dataclass
class PPOConfig:
    gamma: float = 0.995             # per decision of the same faction
    gae_lambda: float = 0.95
    clip: float = 0.2
    value_coef: float = 0.5
    entropy_coef: float = 0.01
    lr: float = 3e-4
    epochs: int = 4
    minibatch_size: int = 256
    max_grad_norm: float = 0.5
    target_kl: float = 0.03          # stop the epochs early when the approximate KL exceeds 1.5x
    value_clip: float = 0.0          # 0 = unclipped value loss
    normalize_advantages: bool = True

    def to_dict(self):
        return asdict(self)


def gae(rewards: Sequence[float], values: Sequence[float], dones: Sequence[bool], gamma: float,
        lam: float, last_value: float = 0.0):
    """Advantages and returns of one trajectory (a done step does not bootstrap)."""
    n = len(rewards)
    adv = np.zeros(n, dtype=np.float32)
    running = 0.0
    for t in reversed(range(n)):
        nonterminal = 0.0 if dones[t] else 1.0
        next_value = values[t + 1] if t + 1 < n else last_value
        delta = rewards[t] + gamma * next_value * nonterminal - values[t]
        running = delta + gamma * lam * nonterminal * running
        adv[t] = running
    return adv, adv + np.asarray(values, dtype=np.float32)


@dataclass
class Batch:
    node: torch.Tensor        # (N, T, F)
    glob: torch.Tensor        # (N, G)
    cand: torch.Tensor        # (N, C, K), zero padding
    mask: torch.Tensor        # (N, C)
    action: torch.Tensor      # (N,)
    old_logp: torch.Tensor
    old_value: torch.Tensor
    advantage: torch.Tensor
    ret: torch.Tensor
    actors: List[str]

    def __len__(self):
        return len(self.action)

    def rows(self, idx: torch.Tensor):
        c = int(self.mask[idx].sum(1).max())
        return (self.node[idx], self.glob[idx], self.cand[idx, :c], self.mask[idx, :c],
                self.action[idx], self.old_logp[idx], self.old_value[idx], self.advantage[idx],
                self.ret[idx])


def build_batch(trajectories: Sequence[List[Transition]], cfg: PPOConfig) -> Batch:
    steps, advs, rets = [], [], []
    for traj in trajectories:
        if not traj:
            continue
        if len({t.actor for t in traj}) != 1:
            raise AssertionError("a trajectory mixes the decisions of several factions")
        if not traj[-1].done:
            raise AssertionError("trajectory without a terminal decision")
        a, r = gae([t.reward for t in traj], [t.value for t in traj], [t.done for t in traj],
                   cfg.gamma, cfg.gae_lambda)
        steps += traj
        advs.append(a)
        rets.append(r)
    if not steps:
        raise ValueError("no decisions collected")
    n, cmax = len(steps), max(t.n_cand for t in steps)
    cand = torch.zeros(n, cmax, steps[0].cand.shape[1])
    mask = torch.zeros(n, cmax, dtype=torch.bool)
    for i, t in enumerate(steps):
        cand[i, :t.n_cand] = torch.from_numpy(t.cand)
        mask[i, :t.n_cand] = True
    return Batch(torch.from_numpy(np.stack([t.node for t in steps])),
                 torch.from_numpy(np.stack([t.glob for t in steps])), cand, mask,
                 torch.tensor([t.action for t in steps]),
                 torch.tensor([t.logp for t in steps], dtype=torch.float32),
                 torch.tensor([t.value for t in steps], dtype=torch.float32),
                 torch.from_numpy(np.concatenate(advs)), torch.from_numpy(np.concatenate(rets)),
                 [t.actor for t in steps])


@torch.no_grad()
def check_log_probs(model: ActorCritic, batch: Batch, chunk: int = 2048) -> float:
    """Largest |stored - recomputed| log-probability; raises above LOGP_ATOL."""
    worst = 0.0
    for start in range(0, len(batch), chunk):
        idx = torch.arange(start, min(start + chunk, len(batch)))
        node, glob, cand, mask, action, old_logp, *_ = batch.rows(idx)
        logp, _, _ = model.evaluate(node, glob, cand, mask, action)
        worst = max(worst, float((logp - old_logp).abs().max()))
    if not worst <= LOGP_ATOL:
        raise AssertionError(f"stored and recomputed log-probabilities differ by {worst:.2e}")
    return worst


def _flat_params(model) -> torch.Tensor:
    return torch.cat([p.detach().flatten() for p in model.parameters()]).clone()


def ppo_update(model: ActorCritic, opt: torch.optim.Optimizer, batch: Batch, cfg: PPOConfig,
               gen: torch.Generator) -> Dict[str, float]:
    model.train()
    logp_gap = check_log_probs(model, batch)
    adv = batch.advantage
    if cfg.normalize_advantages and len(adv) > 1:
        adv = (adv - adv.mean()) / (adv.std() + 1e-8)
    before = _flat_params(model)
    sums: Dict[str, float] = {}
    count, epochs_run, stopped = 0, 0, False
    for _ in range(cfg.epochs):
        order = torch.randperm(len(batch), generator=gen)
        epoch_kl, epoch_n = 0.0, 0
        for start in range(0, len(batch), cfg.minibatch_size):
            idx = order[start:start + cfg.minibatch_size]
            node, glob, cand, mask, action, old_logp, old_value, _, ret = batch.rows(idx)
            a = adv[idx]
            logp, entropy, value = model.evaluate(node, glob, cand, mask, action)
            log_ratio = logp - old_logp
            ratio = log_ratio.exp()
            policy_loss = -torch.min(ratio * a, ratio.clamp(1 - cfg.clip, 1 + cfg.clip) * a).mean()
            if cfg.value_clip > 0:
                clipped = old_value + (value - old_value).clamp(-cfg.value_clip, cfg.value_clip)
                value_loss = 0.5 * torch.max((value - ret) ** 2, (clipped - ret) ** 2).mean()
            else:
                value_loss = 0.5 * ((value - ret) ** 2).mean()
            ent = entropy.mean()
            loss = policy_loss + cfg.value_coef * value_loss - cfg.entropy_coef * ent
            if not torch.isfinite(loss):
                raise AssertionError(f"non-finite PPO loss {loss.item()}")
            opt.zero_grad(set_to_none=True)
            loss.backward()
            bad = [n for n, p in model.named_parameters()
                   if p.grad is not None and not torch.isfinite(p.grad).all()]
            if bad:
                raise AssertionError(f"non-finite gradients in {bad[:3]}")
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
            opt.step()
            with torch.no_grad():
                approx_kl = ((ratio - 1) - log_ratio).mean()
                clip_frac = ((ratio - 1).abs() > cfg.clip).float().mean()
                multi = mask.sum(1) > 1
            m = len(idx)
            for k, v in (("policy_loss", policy_loss), ("value_loss", value_loss), ("entropy", ent),
                         ("approx_kl", approx_kl), ("clip_frac", clip_frac), ("grad_norm", grad_norm),
                         ("entropy_multi", entropy[multi].mean() if multi.any() else torch.tensor(0.0))):
                sums[k] = sums.get(k, 0.0) + float(v.detach()) * m
            count += m
            epoch_kl += float(approx_kl) * m
            epoch_n += m
        epochs_run += 1
        if cfg.target_kl > 0 and epoch_kl / epoch_n > 1.5 * cfg.target_kl:
            stopped = True
            break
    opt.zero_grad(set_to_none=True)
    model.eval()
    change = float((_flat_params(model) - before).norm())
    if not np.isfinite(change):
        raise AssertionError("non-finite parameters after the update")
    if change == 0.0:
        raise AssertionError("the PPO update did not change the parameters")
    with torch.no_grad():
        var = batch.ret.var()
        explained = float(1 - (batch.ret - batch.old_value).var() / var) if var > 0 else float("nan")
    out = {k: v / count for k, v in sums.items()}
    out.update({"epochs_run": epochs_run, "kl_early_stop": stopped, "param_change": change,
                "logp_check_max_abs": logp_gap, "explained_variance": explained,
                "advantage_mean": float(batch.advantage.mean()),
                "advantage_std": float(batch.advantage.std()) if len(batch) > 1 else 0.0,
                "return_mean": float(batch.ret.mean()), "value_mean": float(batch.old_value.mean())})
    return out
