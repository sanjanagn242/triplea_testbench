"""triplea_fm: standalone forward model for TripleA (profile forward-model-v1, land/air subset).

Public API (spec §12):
    load_definition(xml_text, overrides=None, strict=True) -> (GameDefinition, SupportReport)
    create_initial_state(definition) -> SimulationState
    create_state(definition, snapshot) -> SimulationState
    validate(definition, state, action) -> ValidationResult
    validate_batch(definition, state, actions) -> [ValidationResult]
    transition(definition, state, action, rng) -> TransitionResult
    clone(state), is_terminal(state), outcome(state)
    serialize_state(state) / deserialize_state(definition, data)

Agent-side helpers live in triplea_fm.toolkit (candidate generation) and triplea_fm.testbench
(building states from live testbench observations). The kernel does not depend on them.
"""

from .actions import (CONTINUE, NORMAL, RAID, RETREAT, EndPhaseAction, MoveAction, PlaceAction,
                      PurchaseAction, RetreatAction, SelectBattleAction, SelectCasualtiesAction,
                      action_from_json, action_to_json)
from .definition import GameDefinition, SupportReport, UnsupportedDefinition, load_definition, NULL_PLAYER
from .kernel import (DECISION, FAILED, REJECTED, TERMINAL, TransitionResult, ValidationResult, apply_in_place,
                     create_initial_state, create_state, is_terminal, outcome, transition, validate,
                     validate_batch)
from .rng import RandomRequest, RecordingRng, RngError, SeededRng, TapeRng
from .state import SimulationState, StateError, state_from_json


def clone(state: SimulationState) -> SimulationState:
    return state.clone()


def serialize_state(state: SimulationState) -> dict:
    return state.to_json()


def deserialize_state(definition: GameDefinition, data: dict) -> SimulationState:
    return state_from_json(definition, data)


__all__ = [n for n in dir() if not n.startswith("_")]
