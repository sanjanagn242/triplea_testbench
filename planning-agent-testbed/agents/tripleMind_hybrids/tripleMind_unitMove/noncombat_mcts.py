"""TripleMind non-combat-move MCTS with opponent nodes: mcts.MCTS searching the non-combat move.

A node's children are reinforcement options for one territory short of cover (or a skip of it),
from noncombat_generators.noncombat_actions, the way the combat search decides one target at a
time. When no territory is left, the phase ends with the positioning moves and the tree passes on
to the next player's non-combat move (opponent nodes); the phases in between (battles, place,
purchase, combat move) are played by the rollout policies. Rollouts continue the node's phase with
random options, then play on as in the combat search.
"""

from triplea_fm.definition import NONCOMBAT_MOVE

import noncombat_generators as ncg
from mcts import MCTS, MCTSGameState


class NoncombatMCTSGameState(MCTSGameState):
    """MCTSGameState whose searched moves are non-combat options. `excluded` holds the territories
    already decided, `moved` the forward-model ids of the units moved this phase."""

    def __init__(self, search, sim, current_player, played_by=None):
        super().__init__(search, sim, current_player, played_by)
        self.moved = set()

    def clone(self):
        cloned = NoncombatMCTSGameState(self.search, self.sim.clone(), self.current_player,
                                        self.played_by)
        cloned.actions = list(self.actions)
        cloned.num_actions = len(self.actions)
        cloned.actionIndex = self.actionIndex
        cloned.excluded = set(self.excluded)
        cloned.moved = set(self.moved)
        return cloned

    def legal_moves(self):
        self.actions = []
        self.actionIndex = 0
        self.actions, self.excluded = ncg.noncombat_actions(
            self.view(), self.search.d, self.excluded, self.moved, self.current_player)
        self.num_actions = len(self.actions)

    def apply_moves(self, moves):
        for mv in moves:
            self.search.apply(self.sim, mv)
            self.moved.update(mv.units)

    def apply_move(self, action):
        if not action.end_phase:
            self.apply_moves(action.moves)


class NoncombatMCTS(MCTS):
    search_kind = NONCOMBAT_MOVE

    def start_phase(self, state):
        """A player's non-combat move starts: nothing decided or moved yet, and the air the engine
        would destroy lands first (forced). Returns those moves."""
        state.excluded = set()
        state.moved = set()
        moves = ncg.land_air_moves(state.view(), self.d, state.current_player)
        state.apply_moves(moves)
        return moves

    def finish_phase(self, state):
        """Play the end-of-phase action's positioning moves, then end the phase."""
        end = next((a for a in state.actions if a.end_phase), None)
        moves = (end.moves if end is not None else
                 ncg.positioning_moves(state.view(), self.d, state.current_player, state.moved))
        state.apply_moves(moves)
        return self.end_phase(state.sim)

    def _rollout_searched(self, current_state):
        """Rest of the node's non-combat move: a random option per territory, then the end."""
        while True:
            current_state.legal_moves()
            mv = self.pick_rollout_move_random(current_state.actions)
            if mv.end_phase:
                break
            current_state.apply_move(mv)
        return self.finish_phase(current_state)
