"""Validation and transition kernel (spec §8-§10).

transition(definition, state, action, rng) applies one supplied Action, then advances through
automatic and chance effects until the next explicit decision or a terminal state. The input
state and RNG are never mutated; rejection/failure returns them unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

from . import rules
from .definition import GameDefinition
from .rng import RngError
from .state import SimulationState, StateError, check_invariants

DECISION, TERMINAL, REJECTED, FAILED = "DECISION", "TERMINAL", "REJECTED", "FAILED"


@dataclass(frozen=True)
class ValidationResult:
    ok: bool
    code: str = ""
    message: str = ""

    def __bool__(self):
        return self.ok


VALID = ValidationResult(True)


@dataclass
class TransitionResult:
    status: str
    next_state: SimulationState
    validation: ValidationResult
    next_rng: object
    events: List[tuple] = field(default_factory=list)
    chance_outcomes: List[tuple] = field(default_factory=list)
    error: Optional[str] = None

    @property
    def next_decision(self):
        return self.next_state.decision

    @property
    def terminal_outcome(self):
        return self.next_state.terminal


class _ChanceLog:
    """Wraps the caller's RNG copy so every draw is recorded in the transition result."""

    def __init__(self, rng):
        self.rng = rng
        self.log: List[tuple] = []

    def draw(self, request):
        out = self.rng.draw(request)
        self.log.append((request.purpose, request.upper_bound, tuple(out)))
        return out


def validate(definition: GameDefinition, state: SimulationState, action) -> ValidationResult:
    """Pure legality check of a supplied Action against the supplied state (consumes no RNG)."""
    if state.definition_digest != definition.definition_digest:
        return ValidationResult(False, "DEFINITION_MISMATCH", "state belongs to another definition")
    if state.terminal is not None:
        return ValidationResult(False, "WRONG_PHASE", "game is over")
    err = rules.validate_action(definition, state, action)
    if err is None:
        return VALID
    return ValidationResult(False, err[0], err[1])


def validate_batch(definition: GameDefinition, state: SimulationState,
                   actions: Sequence) -> List[ValidationResult]:
    """Same results as validate() for each action, in input order (spec §23.3)."""
    return [validate(definition, state, a) for a in actions]


def transition(definition: GameDefinition, state: SimulationState, action, rng) -> TransitionResult:
    check = validate(definition, state, action)
    if not check.ok:
        return TransitionResult(REJECTED, state, check, rng)
    working = state.clone()
    chance = _ChanceLog(rng.copy())
    events: List[tuple] = []
    try:
        rules.apply_action(definition, working, action, chance, events)
        rules.advance(definition, working, chance, events)
    except RngError as e:
        return TransitionResult(FAILED, state, check, rng, error=str(e))
    except StateError as e:
        return TransitionResult(FAILED, state, check, rng, error=str(e))
    status = TERMINAL if working.terminal is not None else DECISION
    return TransitionResult(status, working, check, chance.rng, events, chance.log)


def apply_in_place(definition: GameDefinition, state: SimulationState, action, rng) -> str:
    """Mutable fast path for disposable rollout states (spec §12 optional apply, without undo).

    Validates first; on success mutates `state` and `rng` in place and returns DECISION/TERMINAL.
    Returns REJECTED (state untouched) for invalid actions. If an engine error is raised the state
    must be discarded. Semantics are identical to transition().
    """
    check = validate(definition, state, action)
    if not check.ok:
        return REJECTED
    rules.apply_action(definition, state, action, rng, [])
    rules.advance(definition, state, rng, [])
    return TERMINAL if state.terminal is not None else DECISION


def create_initial_state(definition: GameDefinition, rng=None) -> SimulationState:
    """Apply the setup recipe once and advance to the first decision.

    Initialization consumes no randomness in V1; rng is accepted for API symmetry.
    """
    s = rules.initial_state(definition)
    rules.advance(definition, s, rng, [])
    return s


def create_state(definition: GameDefinition, snapshot: SimulationState) -> SimulationState:
    """Install a complete hypothetical world after invariant checks (no merging with setup)."""
    if snapshot.definition_digest != definition.definition_digest:
        raise StateError("DEFINITION_MISMATCH", "snapshot belongs to another definition")
    check_invariants(definition, snapshot)
    s = snapshot.clone()
    rules.check_profile(definition, s)
    return s


def is_terminal(state: SimulationState) -> bool:
    return state.terminal is not None


def outcome(state: SimulationState):
    return state.terminal
