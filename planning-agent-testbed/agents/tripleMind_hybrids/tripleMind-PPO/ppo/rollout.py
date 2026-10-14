"""Rollout collection: complete games in the testbench arena (tools/arena.py play_game).

The arena is the testbench's offline stand-in for the Java engine: it sends the agent protocol
messages (game_start, turn_request per phase, game_end) and applies the replies with the forward
model. Learner factions are TripleMindPPO agents running in this process (arena.play_game accepts
objects with ask()/close() as well as commands); opponents are in-process agents, agent
subprocesses, or the arena's built-in HeuristicPolicy.

Every formation decision of a learner faction is one policy step, in the order that faction made
them; the opponent's turns between two of its decisions are part of the environment. When the
game ends each learner trajectory gets its faction's outcome (+1 win, -1 loss, 0 round limit) as
the reward of its last decision, which is marked terminal.

Opponents:
  self         the other faction is a learner too (same frozen snapshot; self-play)
  heuristic    the arena's HeuristicPolicy (triplea_fm.baselines), no agent process
  first        TripleMind with the generator's first (strongest) formation, in process
  random       TripleMind with a uniformly random legal formation, in process
  mcts         TripleMind MCTS (agents/tripleMind/agent.py) as a subprocess, records off
  bc           tripleMind-bc (its run-agent.sh) as a subprocess
  ppo:PATH     a frozen PPO checkpoint, greedy, in process
  cmd:PATH     any agent launcher script, run with bash
"""

from __future__ import annotations

import copy
import os
import shlex
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from . import paths

os.environ.setdefault("TRIPLEMIND_VIEW", "0")  # no live map window in rollouts
from .agent import TripleMindPPO
from .model import ActorCritic
from .policy import Transition

import arena  # noqa: E402  (tools/arena.py)
import triplea_fm as fm  # noqa: E402

OPPONENTS = ("self", "heuristic", "first", "random", "mcts", "bc")
LEARNER_CHOOSERS = {"uniform": "uniform", "first": "first"}  # other modes: the PPO policy
SUBPROCESS_OPPONENTS = ("mcts", "bc")


@dataclass
class RolloutContext:
    xml: str = str(paths.DEFAULT_XML)
    max_rounds: int = 100
    mcts_seconds: float = 0.5        # TRIPLEA_MCTS_SECONDS of MCTS opponents (arena --seconds)
    work_dir: str = "."              # working directory of subprocess opponents (their outputs)
    python: str = sys.executable     # interpreter for the MCTS opponent


@dataclass
class GameSpec:
    game_number: int
    seed: int
    learners: Tuple[str, ...]        # factions played by the policy being trained / evaluated
    opponent: str                    # what plays the other faction (see module docstring)
    mode: str = "sample"             # learner chooser: sample | greedy | uniform | first
    record: bool = True
    label: str = ""                  # opponent as configured (e.g. "past" before it is resolved)


@dataclass
class EpisodeResult:
    spec: GameSpec
    winner: Optional[str]
    rounds: int
    outcomes: Dict[str, float]                       # learner faction -> +1 / -1 / 0
    trajectories: Dict[str, List[Transition]] = field(default_factory=dict)
    seconds: float = 0.0                             # wall time of the whole game
    policy_decisions: int = 0
    policy_seconds: float = 0.0                      # encoding + forward passes of the learners
    applied: int = 0
    rejected: int = 0


class InProcessAgent:
    """arena.AgentProcess for an agent object in this process."""

    def __init__(self, agent):
        self.agent = agent

    def ask(self, message: dict) -> dict:
        return self.agent.handle(message)

    def close(self):
        pass


_GAMES: Dict[str, Tuple[object, str]] = {}


def game(xml_path: str):
    """(definition, xml text), loaded once per process."""
    if xml_path not in _GAMES:
        text = Path(xml_path).read_text()
        d, _ = fm.load_definition(text)
        if len(d.player_names) != 2:
            raise ValueError(f"PPO formation training needs a two-player game, got {d.player_names}")
        _GAMES[xml_path] = (d, text)
    return _GAMES[xml_path]


def factions(xml_path: str) -> Tuple[str, ...]:
    return tuple(game(xml_path)[0].player_names)


def check_opponent(spec: str):
    kind, _, arg = spec.partition(":")
    if kind in ("ppo", "cmd"):
        if not arg or not Path(arg).exists():
            raise FileNotFoundError(f"opponent {spec!r}: {arg!r} does not exist")
    elif spec not in OPPONENTS:
        raise ValueError(f"unknown opponent {spec!r} (one of {OPPONENTS}, ppo:PATH, cmd:PATH)")


def opponent_agent(spec: str, seed: int, ctx: RolloutContext):
    """The arena entry for an opponent: an in-process agent, a command, or None (heuristic)."""
    kind, _, arg = spec.partition(":")
    if spec == "heuristic":
        return None
    if spec == "first":
        return InProcessAgent(TripleMindPPO(chooser="first", seed=seed))
    if spec == "random":
        return InProcessAgent(TripleMindPPO(chooser="uniform", seed=seed, policy_seed=seed + 7919))
    if kind == "ppo":
        return InProcessAgent(TripleMindPPO(chooser="ppo", mode="greedy", seed=seed, checkpoint=arg))
    if spec == "mcts":
        # Not run-agent.sh: that forces TRIPLEMIND_RECORD=1, which would overwrite the recorded
        # games in tripleMind/records (the BC dataset). Its metrics CSV goes to work_dir.
        work = Path(ctx.work_dir) / "mcts-opponent"
        work.mkdir(parents=True, exist_ok=True)
        script = (f"cd {shlex.quote(str(work))} && TRIPLEMIND_RECORD=0 TRIPLEMIND_VIEW=0 "
                  f"PYTHONPATH={shlex.quote(str(paths.FORWARD_MODEL))} "
                  f"exec {shlex.quote(ctx.python)} -u {shlex.quote(str(paths.TRIPLEMIND / 'agent.py'))}")
        return ["bash", "-c", script]
    if spec == "bc":
        return ["bash", str(paths.BC_AGENT_DIR / "run-agent.sh")]
    if kind == "cmd":
        return ["bash", arg]
    raise ValueError(f"unknown opponent {spec!r}")


def assign_terminal_reward(trajectory: List[Transition], outcome: float):
    for t in trajectory:
        t.reward, t.done = 0.0, False
    if trajectory:
        trajectory[-1].reward, trajectory[-1].done = outcome, True


def play_episode(model: Optional[ActorCritic], names: Optional[dict], spec: GameSpec,
                 ctx: RolloutContext) -> EpisodeResult:
    """One complete game. `model` is the frozen snapshot every learner faction acts with."""
    d, xml_text = game(ctx.xml)
    learners: Dict[str, TripleMindPPO] = {}
    entries = {}
    for i, faction in enumerate(d.player_names):
        if faction in spec.learners:
            chooser = LEARNER_CHOOSERS.get(spec.mode, "ppo")
            agent = TripleMindPPO(model, names, chooser=chooser,
                                  mode=spec.mode if chooser == "ppo" else None,
                                  record=spec.record and chooser != "first", seed=spec.seed,
                                  policy_seed=spec.seed * 2 + i)
            learners[faction] = agent
            entries[faction] = InProcessAgent(agent)
        else:
            entry = opponent_agent(spec.opponent, spec.seed, ctx)
            if entry is not None:
                entries[faction] = entry
    started = time.time()
    alliance, rounds, stats = arena.play_game(d, xml_text, entries, spec.game_number,
                                              ctx.mcts_seconds, spec.seed, ctx.max_rounds, False)
    winner = None  # the arena reports the winning alliance; one faction per alliance here
    if alliance is not None:
        winner = next(p for pid, p in enumerate(d.player_names)
                      if d.alliance_names[d.alliance_of[pid]] == alliance)
    result = EpisodeResult(spec, winner, rounds, {}, seconds=time.time() - started,
                           applied=stats["applied"], rejected=stats["rejected"])
    for faction, agent in learners.items():
        result.outcomes[faction] = agent.outcome()
        if agent.chooser == "first":
            continue  # no policy: TripleMind's own first-formation rule (a baseline)
        policy = agent.policy
        if policy is None or agent.policy_disabled or policy.error is not None:
            raise RuntimeError(f"game {spec.game_number}: the {faction} policy failed "
                               f"({policy.error if policy else 'not created'!r})")
        result.policy_decisions += policy.decisions
        result.policy_seconds += policy.seconds
        if spec.record:
            assign_terminal_reward(policy.trajectory, result.outcomes[faction])
            result.trajectories[faction] = policy.trajectory
    check_episode(result)
    return result


def check_episode(r: EpisodeResult):
    """Rewards are from each acting faction's own perspective."""
    for faction, outcome in r.outcomes.items():
        expected = 0.0 if r.winner is None else (1.0 if r.winner == faction else -1.0)
        if outcome != expected:
            raise AssertionError(f"game {r.spec.game_number}: {faction} outcome {outcome} but "
                                 f"winner {r.winner}")
        traj = r.trajectories.get(faction, [])
        if any(t.actor != faction for t in traj):
            raise AssertionError(f"game {r.spec.game_number}: foreign decision in {faction}'s trajectory")
        if traj and (traj[-1].reward != outcome or not traj[-1].done
                     or any(t.reward != 0 or t.done for t in traj[:-1])):
            raise AssertionError(f"game {r.spec.game_number}: {faction} rewards misassigned")
    if len(r.outcomes) == 2 and sum(r.outcomes.values()) != 0:
        raise AssertionError(f"game {r.spec.game_number}: self-play outcomes not zero-sum {r.outcomes}")


# ---- parallel collection --------------------------------------------------------------------

def _init_worker():
    os.environ["TRIPLEMIND_VIEW"] = "0"
    torch.set_num_threads(1)


def _worker_play(config, state, names, specs, ctx) -> List[EpisodeResult]:
    model = None
    if config is not None:
        model = ActorCritic.from_config(config)
        model.load_state_dict(state)
        model.eval().requires_grad_(False)
    return [play_episode(model, names, s, ctx) for s in specs]


class Collector:
    """Plays games with a frozen snapshot, in this process (workers <= 1) or in a pool of
    worker processes (each game is independent; the snapshot is sent with every chunk)."""

    def __init__(self, ctx: RolloutContext, workers: int = 1):
        self.ctx = ctx
        self.workers = workers
        self.pool = None
        if workers > 1:
            import multiprocessing as mp
            self.pool = ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn"),
                                            initializer=_init_worker)
        else:
            os.environ["TRIPLEMIND_VIEW"] = "0"

    def run(self, model: Optional[ActorCritic], names: Optional[dict],
            specs: Sequence[GameSpec]) -> List[EpisodeResult]:
        snapshot = None
        if model is not None:
            snapshot = copy.deepcopy(model).eval().requires_grad_(False)
        if self.pool is None:
            return [play_episode(snapshot, names, s, self.ctx) for s in specs]
        config = snapshot.config if snapshot is not None else None
        state = snapshot.state_dict() if snapshot is not None else None
        # Subprocess opponents are slow and uneven: one game per task; otherwise a few per task.
        size = 1 if any(s.opponent in SUBPROCESS_OPPONENTS for s in specs) else \
            max(1, len(specs) // (self.workers * 4))
        chunks = [list(specs[i:i + size]) for i in range(0, len(specs), size)]
        futures = [self.pool.submit(_worker_play, config, state, names, c, self.ctx) for c in chunks]
        results = [r for f in futures for r in f.result()]
        return sorted(results, key=lambda r: r.spec.game_number)

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(cancel_futures=True)
            self.pool = None
