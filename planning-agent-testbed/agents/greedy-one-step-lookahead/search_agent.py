"""SearchAgent: a base class for forward-model search agents (newline-delimited JSON over stdin/stdout).

Subclass SearchAgent and implement search(). Everything else is done here:

  * the protocol loop (one JSON request line in, one JSON response line out);
  * logging, only when TRIPLEA_AGENT_LOG is 1 (stderr and planning-agent-testbed/logs/...);
  * game_start: loading the map into the forward model (or answering with empty actions when the
    map is outside the forward model's profile);
  * each phase request: building the forward-model state from the observation, checking it,
    computing the time budget, calling search(), checking the returned actions with the kernel in
    order (rejected ones are dropped and logged) and encoding them as the testbench response.

search(root, phase, deadline) receives a SimulationState positioned at this player's decision for
`phase` ("purchase", "combatMove", "nonCombatMove" or "place") and returns the actions for the
rest of that phase, in order. Battles are answered by battle() (default: let the engine fight
them; in protocol v1 the engine also picks this player's casualties and retreats).

Helpers for search code: legal_actions, simulate / apply, same_phase, new_rng, rollout and
evaluate. They are thin wrappers over the triplea_fm library, which can also be used directly.
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import traceback
from pathlib import Path
from typing import List, Optional

import triplea_fm as fm
from triplea_fm import toolkit
from triplea_fm.actionspace import legal_actions as _all_legal_actions
from triplea_fm.baselines import HeuristicPolicy, Precomp, evaluate as _evaluate
from triplea_fm.testbench import LiveBridge


PROTOCOL_VERSION = 1
SEARCH_PHASES = ("purchase", "combatMove", "nonCombatMove", "place")
LOGGING = os.environ.get("TRIPLEA_AGENT_LOG", "0") == "1"  # set in run-agent.sh
LOG_DIR = Path(__file__).resolve().parent.parent.parent / "logs"  # planning-agent-testbed/logs


class SearchAgent:
    """Base class: override search(); optionally setup(), battle() and time_budget()."""

    # Share of the base time per phase (time_budget = seconds × share).
    PHASE_TIME_SHARE = {"purchase": 0.6, "combatMove": 1.5, "nonCombatMove": 0.8, "place": 0.3}

    def __init__(self, seconds: Optional[float] = None, seed: Optional[int] = None,
                 log_dir: Optional[Path] = None):
        self.seconds = float(os.environ.get("TRIPLEA_AGENT_SECONDS", "2.0")) if seconds is None else seconds
        self.seed = int(os.environ.get("TRIPLEA_AGENT_SEED", "0")) if seed is None else seed
        self.rnd = random.Random(self.seed)  # the agent's own randomness; game dice come from new_rng()
        self.log_dir = log_dir or LOG_DIR
        self.player_name: Optional[str] = None  # faction played, e.g. "Russians"
        self.me: Optional[int] = None  # its player index in the definition
        self.d: Optional[fm.GameDefinition] = None  # None when the map is outside the profile
        self.bridge: Optional[LiveBridge] = None
        self.precomp: Optional[Precomp] = None  # distances, unit costs, territory values
        self.policy: Optional[HeuristicPolicy] = None  # library baseline, used by rollout()
        self.log_path: Optional[Path] = None

    # =============================================================================================
    # Override these
    # =============================================================================================

    def search(self, root: fm.SimulationState, phase: str, deadline: float) -> List[object]:
        """Return the actions for the rest of the phase, in order (time.time() < deadline).

        `root` is this player's decision for `phase`; never modify it, work on root.clone().
        Actions are triplea_fm actions (PurchaseAction, MoveAction, PlaceAction, EndPhaseAction).
        Returning [] ends the phase without acting.
        """
        raise NotImplementedError

    def setup(self):
        """Per-game hook, called after the definition is loaded (self.d, self.me are set)."""

    def battle(self, observation: dict) -> dict:
        """Answer to the battle phase. Protocol v1 only supports letting the engine fight."""
        return {"fightAll": True}

    def time_budget(self, phase: str) -> float:
        return self.seconds * self.PHASE_TIME_SHARE.get(phase, 1.0)

    # =============================================================================================
    # Helpers for search code
    # =============================================================================================

    def legal_actions(self, s: fm.SimulationState, exhaustive: bool = False) -> List[object]:
        """Actions for s.decision: toolkit candidates (a compact legal subset), or with
        exhaustive=True every legal action (can be thousands of moves)."""
        if s.decision is None:
            return []
        return list(_all_legal_actions(self.d, s)) if exhaustive else toolkit.candidates(self.d, s)

    def new_rng(self) -> fm.SeededRng:
        """Fresh game dice for one simulation (seeded from the agent's own generator)."""
        return fm.SeededRng(self.rnd.getrandbits(62))

    def simulate(self, s: fm.SimulationState, action, rng=None) -> Optional[fm.SimulationState]:
        """The state after `action` (s is not changed), or None if the kernel rejects it."""
        r = fm.transition(self.d, s, action, rng or self.new_rng())
        return r.next_state if r.status in (fm.DECISION, fm.TERMINAL) else None

    def apply(self, s: fm.SimulationState, action, rng) -> bool:
        """Apply `action` to s in place (fast; use on clones). False if rejected (s unchanged)."""
        return fm.apply_in_place(self.d, s, action, rng) != fm.REJECTED

    @staticmethod
    def same_phase(root: fm.SimulationState, s: fm.SimulationState) -> bool:
        """True while s is still the decision phase that `root` started in."""
        dec = s.decision
        return (dec is not None and s.terminal is None and dec.actor == root.decision.actor
                and dec.step_id == root.decision.step_id and dec.kind == root.decision.kind
                and s.round == root.round)

    def rollout(self, s: fm.SimulationState, rng, rounds: int = 2, max_steps: int = 5000) -> fm.SimulationState:
        """Play every player with the library's HeuristicPolicy for `rounds` rounds, in place."""
        horizon = s.round + rounds
        steps = 0
        while s.terminal is None and s.decision is not None and s.round <= horizon and steps < max_steps:
            if not self.apply(s, self.policy.act(s, self.rnd), rng) and \
                    not self.apply(s, fm.EndPhaseAction(), rng):
                break
            steps += 1
        return s

    def evaluate(self, s: fm.SimulationState) -> float:
        """This player's value of s in [0, 1] (1/0 when the game is won/lost)."""
        return _evaluate(self.d, self.precomp, s, self.me)

    def log(self, message: str):
        if not LOGGING:
            return
        line = f"{time.strftime('%H:%M:%S')} {message}"
        print(line, file=sys.stderr, flush=True)
        if self.log_path:
            with open(self.log_path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    # =============================================================================================
    # Framework (normally not overridden)
    # =============================================================================================

    def run(self):
        """The protocol loop: never exits on an error, answers it with {"error": ...}."""
        for line in sys.stdin:
            line = line.strip()
            if not line:
                continue
            try:
                response = self.handle(json.loads(line))
            except Exception as e:
                traceback.print_exc(file=sys.stderr)
                self.log(f"error: {e!r}")
                response = {"error": repr(e)}
            sys.stdout.write(json.dumps(response) + "\n")
            sys.stdout.flush()

    def handle(self, request: dict) -> dict:
        kind = request.get("type")
        if kind == "game_start":
            return self.game_start(request)
        if kind == "turn_request":
            return self.turn(request)
        return {"error": f"unknown request type {kind!r}"}

    def game_start(self, request: dict) -> dict:
        self.player_name = request["playerName"]
        game_number = request.get("gameNumber", 1)
        if LOGGING:
            self.log_dir.mkdir(parents=True, exist_ok=True)
            self.log_path = self.log_dir / f"agent-{self.player_name}-game-{game_number}.txt"
            self.log_path.write_text(f"{game_number}\n", encoding="utf-8")
        try:
            d, report = fm.load_definition(request["gameXml"], strict=False)
        except Exception as e:  # not a TripleA game XML the forward model can read
            self.log(f"could not load gameXml ({e!r}); answering with empty actions")
            self.d = None
            return {"ready": True, "protocolVersion": PROTOCOL_VERSION}
        if not report.supported:
            for f in report.features:
                self.log(f"profile refusal: {f.code} {f.feature_id}={f.effective_value}: {f.explanation}")
            self.log("forward model unavailable for this game; answering with empty actions")
            self.d = None
            return {"ready": True, "protocolVersion": PROTOCOL_VERSION}
        self.d = d
        self.me = d.player_index[self.player_name]
        self.bridge = LiveBridge(d, self.player_name, game_number)
        self.precomp = Precomp(d)
        self.policy = HeuristicPolicy(d, self.precomp)
        self.setup()
        self.log(f"ready as {self.player_name}")
        return {"ready": True, "protocolVersion": PROTOCOL_VERSION}

    def turn(self, request: dict) -> dict:
        phase = request.get("phase")
        if phase == "battle":
            return self.battle(request.get("state", {}))
        if phase not in SEARCH_PHASES or self.d is None:
            return {}
        root = self.bridge.build_state(request["state"], phase)
        try:
            fm.create_state(self.d, root)
        except fm.StateError as e:
            self.log(f"round {root.round} {phase}: state rejected by the forward model: {e}")
            return {}
        started = time.time()
        actions = self.search(root, phase, started + self.time_budget(phase)) or []
        accepted = self._check(root, actions)
        response = self._encode(phase, accepted)
        self.log(f"round {root.round} {phase}: {time.time() - started:.2f}s, {len(accepted)}/{len(actions)} "
                 f"actions accepted -> {json.dumps(response)}")
        return response

    def _check(self, root, actions) -> List[object]:
        """Replay the actions on a clone in order; keep those the kernel accepts in this phase."""
        s, rng = root.clone(), self.new_rng()
        accepted = []
        for a in actions:
            if not self.same_phase(root, s) or a.kind == "end_phase":
                break
            if self.apply(s, a, rng):
                accepted.append(a)
            else:
                self.log(f"  rejected {fm.action_to_json(self.d, a)}: {fm.validate(self.d, s, a).message}")
        return accepted

    def _encode(self, phase: str, actions) -> dict:
        b = self.bridge
        if phase == "purchase":
            merged = {}
            for a in actions:
                for rule, count in a.quantities:
                    merged[rule] = merged.get(rule, 0) + count
            return b.purchase_to_json(fm.PurchaseAction(tuple(sorted(merged.items())))) if merged else {}
        if phase in ("combatMove", "nonCombatMove"):
            return {"moves": [b.move_to_json(a) for a in actions if a.kind == "move"]}
        return {"placements": [b.place_to_json(a) for a in actions if a.kind == "place"]}
