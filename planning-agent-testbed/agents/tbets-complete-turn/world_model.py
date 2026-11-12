"""The parts of Roxo's forward model (NewAction / NewActionSkip and ActionUtils in his
MCTSGrandStrategy fork) the search uses, on top of the testbench kernel (triplea_fm): territory
actions (units sent to a territory, or a skip of it) and routing a territory action's units there.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import triplea_fm as fm
from triplea_fm import MoveAction
from triplea_fm import toolkit
from triplea_fm.definition import COMBAT_MOVE

import fast_battle

fast_battle.install()

TERRITORY, SKIP = "territory", "skip"


@dataclass(frozen=True)
class TerritoryAction:
    """NewAction (units sent to territory) or NewActionSkip."""
    kind: str
    territory: Optional[int] = None
    units: Tuple[int, ...] = ()


class Game:
    """Rules shared by every node of one game (his ActionUtils and ForwardModel helpers)."""

    def __init__(self, d):
        self.d = d
        self.rng = fm.SeededRng(0)  # draws dice only with TRIPLEA_ROLLOUT_BATTLE=dice

    @staticmethod
    def actor(s) -> int:
        return s.decision.actor

    def is_combat(self, s) -> bool:
        return self.d.steps[s.step].kind == COMBAT_MOVE

    def _route(self, s, u, t: int, max_steps: int, cache=None) -> Optional[Tuple[int, ...]]:
        """getRouteForUnit with territoryCanMoveLandUnitsThrough: a shortest route for u to t
        within max_steps. `cache` shares the search between units of one kind in one state."""
        ut = self.d.unit_types[u.type]
        key = (u.loc, max_steps, ut.can_blitz)
        routes = cache.get(key) if cache is not None else None
        if routes is None:
            routes = toolkit.unit_routes(self.d, s, self.actor(s), u.loc, max_steps, False, ut.can_blitz,
                                         self.is_combat(s))
            if cache is not None:
                cache[key] = routes
        return routes.get(t)

    # ---- applying actions ----
    def routed_moves(self, s, t: int, unit_ids: Sequence[int]) -> List[MoveAction]:
        """ForwardModel.calculateMoveRoutesNew + doMove: one route per unit (no movement limit;
        the kernel checks it), units sharing a route merged into one move."""
        by_route: Dict[Tuple[int, ...], List[int]] = {}
        cache = {}
        for uid in unit_ids:
            u = s.units.get(uid)
            if u is None or u.loc == t or u.loc < 0:
                continue
            route = self._route(s, u, t, len(self.d.territory_names), cache)
            if route is not None:
                by_route.setdefault(route, []).append(uid)
        return [MoveAction(tuple(ids), r) for r, ids in by_route.items()]

    def move_units(self, s, t: int, unit_ids: Sequence[int]):
        for m in self.routed_moves(s, t, unit_ids):
            fm.apply_in_place(self.d, s, m, self.rng)  # a rejected move is skipped, as performMove
