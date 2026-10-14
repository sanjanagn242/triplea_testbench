"""Sanity checks (run_ppo.py sanity). Each returns a dict with "passed" and details.

  1 gae_perspective       GAE on interleaved two-faction games: each faction's advantages use
                          only its own values and rewards, with its own outcome's sign
  2 softmax_masking       probabilities sum to 1 per decision, are 0 on padding, and padding does
                          not change the real candidates' scores (variable-sized sets)
  3 rollout_games         real games with a sampling snapshot: every choice is a legal generated
                          candidate (asserted per decision in policy.py), rewards from each
                          faction's perspective (rollout.check_episode), probabilities normalised
  4 ppo_update            stored == recomputed log-probabilities before the update, finite
                          losses and gradients, parameters change, the snapshot does not
  5 evaluation_no_grad    evaluation games leave no gradients and do not change parameters
  6 checkpoint_roundtrip  save -> load gives identical outputs
  7 noncombat_parity      in a full game, TripleMind PPO and the original TripleMind agent
                          (agents/tripleMind/agent.py), fed the same observations, answer every
                          purchase, nonCombatMove and place request identically
"""

from __future__ import annotations

import contextlib
import copy
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

from . import paths
from .algorithm import PPOConfig, build_batch, check_log_probs, gae, ppo_update
from .evaluate import evaluate
from .features import game_feature_names
from .model import load_checkpoint, new_model, save_checkpoint
from .policy import Transition
from .rollout import Collector, GameSpec, RolloutContext, factions, game
from .train import Normalizer

import arena  # noqa: E402


def _t(actor, reward=0.0, done=False, value=0.0):
    z = np.zeros(1, dtype=np.float32)
    return Transition(z[None], z, z[None], 0, 0.0, value, actor, 1, "x", reward, done)


def check_gae_perspective() -> dict:
    cfg = PPOConfig(gamma=0.9, gae_lambda=1.0)
    # Germans win: G's last decision +1, R's last decision -1. Interleaved in time, separate lists.
    g = [_t("Germans", value=0.5), _t("Germans", value=0.2), _t("Germans", 1.0, True, value=0.1)]
    r = [_t("Russians", value=-0.3), _t("Russians", -1.0, True, value=0.4)]
    adv_g, ret_g = gae([t.reward for t in g], [t.value for t in g], [t.done for t in g], 0.9, 1.0)
    adv_r, ret_r = gae([t.reward for t in r], [t.value for t in r], [t.done for t in r], 0.9, 1.0)
    want_ret_g = np.array([0.81, 0.9, 1.0], dtype=np.float32)
    want_ret_r = np.array([-0.9, -1.0], dtype=np.float32)
    batch = build_batch([g, r], cfg)
    ok = (np.allclose(ret_g, want_ret_g) and np.allclose(ret_r, want_ret_r)
          and (ret_g > 0).all() and (ret_r < 0).all()
          and batch.actors == ["Germans"] * 3 + ["Russians"] * 2
          and torch.allclose(batch.ret, torch.from_numpy(np.concatenate([ret_g, ret_r]))))
    mixed_rejected = False
    try:
        build_batch([[_t("Germans"), _t("Russians", 1.0, True)]], cfg)
    except AssertionError:
        mixed_rejected = True
    return {"passed": bool(ok and mixed_rejected), "returns_germans": ret_g.tolist(),
            "returns_russians": ret_r.tolist(), "mixed_trajectory_rejected": mixed_rejected}


def check_softmax_masking(names: dict) -> dict:
    torch.manual_seed(0)
    model = new_model(names).eval()
    t, f = len(names["territories"]), len(names["node"])
    k = len(names["candidate"])
    sizes = [1, 2, 5, 9]
    node, glob = torch.randn(len(sizes), t, f), torch.randn(len(sizes), len(names["global"]))
    cand = torch.zeros(len(sizes), max(sizes), k)
    mask = torch.zeros(len(sizes), max(sizes), dtype=torch.bool)
    for i, n in enumerate(sizes):
        cand[i, :n] = torch.randn(n, k)
        mask[i, :n] = True
    with torch.no_grad():
        scores = model.scores(node, glob, cand, mask)
        p = torch.softmax(scores, 1)
        alone = [model.scores(node[i:i + 1], glob[i:i + 1], cand[i:i + 1, :n],
                              mask[i:i + 1, :n])[0] for i, n in enumerate(sizes)]
        logp, ent, v = model.evaluate(node, glob, cand, mask, torch.zeros(len(sizes), dtype=torch.long))
    sums = p.sum(1)
    pad_invariant = all(torch.allclose(scores[i, :n], alone[i], atol=1e-5) for i, n in enumerate(sizes))
    ok = (torch.allclose(sums, torch.ones_like(sums), atol=1e-5) and float(p[~mask].max()) < 1e-6
          and pad_invariant and torch.isfinite(ent).all() and abs(float(ent[0])) < 1e-6
          and bool((ent <= torch.tensor(sizes).float().log() + 1e-5).all()))
    return {"passed": bool(ok), "prob_sums": sums.tolist(), "max_padding_prob": float(p[~mask].max()),
            "padding_invariant": pad_invariant, "entropy": ent.tolist(),
            "initial_entropy_vs_uniform": [float(e) / max(np.log(n), 1e-9) for e, n in zip(ent, sizes)]}


def check_rollouts_and_update(names: dict, ctx: RolloutContext, games: int, seed: int) -> Dict[str, dict]:
    sides = factions(ctx.xml)
    collector = Collector(ctx, 1)
    norm = Normalizer(names)
    warm = collector.run(None, names, [GameSpec(1, seed, tuple(sides), "self", "uniform", True, "warmup")])
    norm.update(warm)
    torch.manual_seed(seed)
    model = new_model(names)
    model.set_normalization(norm.state, norm.cand)
    specs = [GameSpec(i + 1, seed + 100 + i, tuple(sides), "self", "sample", True, "self")
             for i in range(games)]
    results = collector.run(model, names, specs)
    steps = [t for r in results for traj in r.trajectories.values() for t in traj]
    legal = all(0 <= t.action < t.n_cand for t in steps)
    # Probabilities of the stored decisions, recomputed one by one.
    with torch.no_grad():
        sums = [float(torch.softmax(model.scores(torch.from_numpy(t.node)[None], torch.from_numpy(t.glob)[None],
                                                 torch.from_numpy(t.cand)[None],
                                                 torch.ones(1, t.n_cand, dtype=torch.bool))[0], 0).sum())
                for t in steps]
    rollout = {
        "passed": bool(legal and steps and max(abs(s - 1) for s in sums) < 1e-5),
        "games": len(results), "decisions": len(steps),
        "outcomes": [r.outcomes for r in results], "winners": [r.winner for r in results],
        "rounds": [r.rounds for r in results], "max_prob_sum_error": max(abs(s - 1) for s in sums),
        "multi_candidate_decisions": sum(t.n_cand > 1 for t in steps),
        "rewards_checked": "rollout.check_episode passed for every game"}

    cfg = PPOConfig(epochs=2, minibatch_size=64, target_kl=0.0)
    batch = build_batch([t for r in results for t in r.trajectories.values()], cfg)
    snapshot = copy.deepcopy(model)
    before = {n: p.detach().clone() for n, p in model.named_parameters()}
    gap = check_log_probs(model, batch)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr, eps=1e-5)
    stats = ppo_update(model, opt, batch, cfg, torch.Generator().manual_seed(seed))
    changed = [n for n, p in model.named_parameters() if not torch.equal(p, before[n])]
    finite = all(torch.isfinite(p).all() for p in model.parameters())
    snapshot_intact = all(torch.equal(p, before[n]) for n, p in snapshot.named_parameters())
    actor_changed = any(n.startswith("actor.") for n in changed)
    critic_changed = any(n.startswith("critic.") for n in changed)
    update = {"passed": bool(gap <= 1e-4 and finite and actor_changed and critic_changed and snapshot_intact),
              "logp_max_abs_gap": gap, "changed_tensors": len(changed),
              "actor_changed": actor_changed, "critic_changed": critic_changed,
              "snapshot_unchanged": snapshot_intact, "param_change_norm": stats["param_change"],
              "policy_loss": stats["policy_loss"], "value_loss": stats["value_loss"],
              "entropy": stats["entropy"], "approx_kl": stats["approx_kl"]}

    params = {n: p.detach().clone() for n, p in model.named_parameters()}
    report = evaluate(model, names, ["heuristic"], 2, collector, seed)
    collector.close()
    no_grad = all(p.grad is None for p in model.parameters())
    unchanged = all(torch.equal(p, params[n]) for n, p in model.named_parameters())
    evaluation = {"passed": bool(no_grad and unchanged), "no_gradients": no_grad,
                  "parameters_unchanged": unchanged,
                  "games": report["heuristic"]["performance"]["games"]}

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "c.pt"
        save_checkpoint(path, model, names, norm=norm.state_dict(), optimizer=opt, iteration=1)
        loaded, ckpt = load_checkpoint(path)
        t = steps[0]
        args = (torch.from_numpy(t.node)[None], torch.from_numpy(t.glob)[None],
                torch.from_numpy(t.cand)[None], torch.ones(1, t.n_cand, dtype=torch.bool))
        with torch.no_grad():
            same = (torch.equal(model.scores(*args), loaded.scores(*args))
                    and torch.equal(model.value(*args[:2]), loaded.value(*args[:2])))
        roundtrip = {"passed": bool(same and ckpt["feature_names"] == names), "identical_outputs": same}
    return {"rollout_games": rollout, "ppo_update": update, "evaluation_no_grad": evaluation,
            "checkpoint_roundtrip": roundtrip}


class _Shadowed:
    """Arena entry: the PPO agent plays; the original TripleMind agent gets the same messages and
    its answers to non-combat requests are compared with the PPO agent's."""

    def __init__(self, ppo_agent, original):
        self.ppo, self.original = ppo_agent, original
        self.compared: Dict[str, int] = {}
        self.mismatches: List[dict] = []

    def ask(self, message: dict) -> dict:
        reply = self.ppo.handle(message)
        shadow = self.original.handle_request(copy.deepcopy(message))
        phase = message.get("phase")
        if message.get("type") == "turn_request" and phase in ("purchase", "nonCombatMove", "place", "battle"):
            self.compared[phase] = self.compared.get(phase, 0) + 1
            if _canonical(reply) != _canonical(shadow):
                self.mismatches.append({"round": message["state"].get("round"), "phase": phase,
                                        "ppo": reply, "tripleMind": shadow})
        return reply

    def close(self):
        pass


def _canonical(reply: dict) -> dict:
    """An empty move list and no move list both mean "no moves"."""
    reply = dict(reply or {})
    if not reply.get("moves"):
        reply.pop("moves", None)
    return reply


@contextlib.contextmanager
def _cwd(path: Path):
    old = os.getcwd()
    path.mkdir(parents=True, exist_ok=True)
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(old)


def check_noncombat_parity(names: dict, ctx: RolloutContext, work: Path, seed: int,
                           opponent: str = "heuristic") -> dict:
    os.environ["TRIPLEMIND_RECORD"] = "0"  # the original agent must not write game records
    os.environ["TRIPLEMIND_VIEW"] = "0"
    original_mod = paths.triplemind_agent_module()
    from .agent import TripleMindPPO

    d, xml_text = game(ctx.xml)
    sides = factions(ctx.xml)
    torch.manual_seed(seed)
    model = new_model(names).eval()
    entries, shadows = {}, []
    for i, side in enumerate(sides):
        # MCTS is only run for the original's combat moves, which are not compared; a short budget.
        original = original_mod.TripleMindAgent(0.02, 100000, 2, 1.414, seed)
        ppo_agent = TripleMindPPO(model, names, mode="sample", seed=seed, policy_seed=seed + i)
        entries[side] = _Shadowed(ppo_agent, original)
        shadows.append(entries[side])
    with _cwd(work):  # the original MCTS writes metrics/efficiency.csv relative to the cwd
        winner, rounds, _ = arena.play_game(d, xml_text, entries, 1, 0.02, seed, ctx.max_rounds, False)
    compared: Dict[str, int] = {}
    for s in shadows:
        for k, v in s.compared.items():
            compared[k] = compared.get(k, 0) + v
    mismatches = [m for s in shadows for m in s.mismatches]
    return {"passed": not mismatches and compared.get("nonCombatMove", 0) > 0,
            "compared": compared, "mismatches": len(mismatches), "examples": mismatches[:3],
            "rounds": rounds, "winner": winner,
            "note": "the original's combat moves (MCTS) differ by design and are not compared"}


def run_all(out_dir: str, xml: str = str(paths.DEFAULT_XML), games: int = 2, seed: int = 0,
            max_rounds: int = 100) -> Dict[str, dict]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    names = game_feature_names(xml)
    ctx = RolloutContext(xml, max_rounds, 0.2, work_dir=str(out))
    results = {"gae_perspective": check_gae_perspective(), "softmax_masking": check_softmax_masking(names)}
    results.update(check_rollouts_and_update(names, ctx, games, seed))
    results["noncombat_parity"] = check_noncombat_parity(names, ctx, out / "parity", seed)
    return results
