"""The Hard AI (AbstractProAi/ProAi) driven by the forward model.

Phase flow, as in AbstractProAi:
  * purchase: simulate the rest of the turn on a copy of the state with the forward model
    (combat move -> battles -> noncombat move), then decide purchases and their placement on the
    simulated end-of-turn board. The combat moves and the placement plan are stored.
  * combat move: replay the stored combat moves if the kernel still accepts all of them,
    otherwise plan again on the live state.
  * noncombat move: plan on the live state, keeping planned placements as defenders.
  * place: place as planned, then place anything left.
Every move and placement is checked by the kernel before it is returned.
"""

from __future__ import annotations

import random
import time
from typing import Callable, Dict, List, Optional

import triplea_fm as fm
from triplea_fm.definition import COMBAT_MOVE, NONCOMBAT_MOVE, GameDefinition
from triplea_fm.state import MOVE_D, SimulationState

from . import battle, moves
from .combat_move import CombatMoveAi
from .data import ProData, PurchaseTerritory
from .noncombat_move import NonCombatMoveAi
from .odds import OddsCalculator
from .purchase import PurchaseAi, find_purchase_territories


class HardAi:
    def __init__(self, d: GameDefinition, player: int, seed: int = 0, battle_samples: int = 5,
                 run_scale: float = 1.0, log: Optional[Callable[[str], None]] = None):
        self.d = d
        self.player = player
        self.seed = seed
        self.battle_samples = battle_samples
        self.rnd = random.Random(seed)  # the AI's own randomness (ProPurchaseUtils' Math.random)
        self.calc = OddsCalculator(seed, run_scale)
        self.log = log or (lambda msg: None)
        self.combat_ai = CombatMoveAi(self)
        self.noncombat_ai = NonCombatMoveAi(self)
        self.purchase_ai = PurchaseAi(self)
        self.stored_combat_moves: Optional[tuple] = None  # (round, [MoveAction])
        self.stored_purchase_territories: Optional[tuple] = None  # (round, {t: PurchaseTerritory})
        self.stored_strafing: List[int] = []

    def _rng(self, s: SimulationState, stream: int) -> fm.SeededRng:
        """Dice for the AI's own simulations (never the game's)."""
        return fm.SeededRng(self.seed * 1_000_003 + s.round, stream=stream)

    # ---- purchase ----
    def purchase(self, root: SimulationState) -> Optional[fm.PurchaseAction]:
        d, p = self.d, self.player
        started = time.time()
        self.calc.reset_cache()
        self.stored_combat_moves = None
        self.stored_purchase_territories = None
        pus = root.balances[p][d.treasury_resource]
        if pus <= 0:
            return None
        if not find_purchase_territories(ProData(d, root, p), p):
            self.log("purchase: no place territories")
            return None

        end_state = self._simulate_turn(root)
        pd = ProData(d, end_state, p)
        held = root.held_units(p)
        purchase_territories, counts = self.purchase_ai.purchase(pd, pus, held)
        self.stored_purchase_territories = (root.round, purchase_territories)
        action = fm.PurchaseAction(tuple(sorted((r, c) for r, c in counts.items() if c > 0)))
        check = fm.validate(d, root, action)
        if not check.ok:
            self.log(f"purchase rejected by the forward model ({check.message}); buying nothing")
            return None
        names = {d.production_rules[r].name: c for r, c in action.quantities}
        self.log(f"purchase: {names} in {time.time() - started:.2f}s "
                 f"({self.calc.calls} battle estimates, {self.calc.simulated} simulated battles)")
        return action

    def _simulate_turn(self, root: SimulationState) -> SimulationState:
        """Play this player's turn up to its noncombat move on a copy, with the forward model."""
        d, p = self.d, self.player
        w = root.clone()
        rng = self._rng(root, 1)
        fm.apply_in_place(d, w, fm.EndPhaseAction(), rng)  # leave the purchase without buying
        while w.decision is not None and w.decision.actor == p and w.round == root.round:
            kind = d.steps[w.step].kind
            if w.decision.kind == MOVE_D and kind == COMBAT_MOVE:
                attack_moves = self._plan_combat_moves(w)
                self.stored_combat_moves = (root.round, moves.execute(d, w, attack_moves, rng, self.log))
                w, rng = battle.simulate_battles(self, d, w, self.seed + root.round, self.battle_samples)
            elif w.decision.kind == MOVE_D and kind == NONCOMBAT_MOVE:
                pd = ProData(d, w, p)
                move_map = self.noncombat_ai.do_noncombat_move(pd, None)
                moves.execute(d, w, moves.calculate_move_routes(pd, move_map, False, self.log), rng, self.log)
                return w
            else:
                break
        if w.decision is None or w.decision.actor != p or d.steps[w.step].kind != NONCOMBAT_MOVE:
            # The kernel moved past this player's turn (nothing left to move): restore the turn's
            # conquests, which decide where units may be placed.
            w.conquered = {t for t, o in enumerate(w.owner) if o == p and root.owner[t] != p}
        return w

    def _plan_combat_moves(self, s: SimulationState) -> List[fm.MoveAction]:
        pd = ProData(self.d, s, self.player)
        attack_map = self.combat_ai.do_combat_move(pd)
        return (moves.calculate_move_routes(pd, attack_map, True, self.log)
                + moves.calculate_bombing_routes(pd, attack_map))

    # ---- moves ----
    def combat_move(self, root: SimulationState) -> List[fm.MoveAction]:
        d = self.d
        stored = self.stored_combat_moves
        self.stored_combat_moves = None
        if stored is not None and stored[0] == root.round:
            w = root.clone()
            accepted = moves.execute(d, w, stored[1], self._rng(root, 2))
            if len(accepted) == len(stored[1]):
                self.log(f"combat move: replaying {len(accepted)} stored moves")
                return accepted
            self.log("combat move: stored moves no longer valid; planning again")
        self.calc.reset_cache()
        planned = self._plan_combat_moves(root)
        return moves.execute(d, root.clone(), planned, self._rng(root, 3), self.log)

    def noncombat_move(self, root: SimulationState) -> List[fm.MoveAction]:
        d = self.d
        stored = self.stored_purchase_territories
        purchase_territories = stored[1] if stored is not None and stored[0] == root.round else None
        pd = ProData(d, root, self.player)
        move_map = self.noncombat_ai.do_noncombat_move(pd, purchase_territories)
        planned = moves.calculate_move_routes(pd, move_map, False, self.log)
        return moves.execute(d, root.clone(), planned, self._rng(root, 4), self.log)

    # ---- placement ----
    def place(self, root: SimulationState) -> List[fm.PlaceAction]:
        stored = self.stored_purchase_territories
        self.stored_purchase_territories = None
        purchase_territories: Optional[Dict[int, PurchaseTerritory]] = \
            stored[1] if stored is not None and stored[0] == root.round else None
        return self.purchase_ai.place(ProData(self.d, root, self.player), purchase_territories,
                                      self._rng(root, 5))
