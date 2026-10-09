"""From move maps to forward-model moves (ProMoveUtils), checked by the forward model.

calculate_move_routes / calculate_bombing_routes find a route per unit like ProMoveUtils; units with
the same route are merged into one MoveAction. execute() then replays the moves on a copy of the
state with the kernel, keeping only those the kernel accepts, in order (later moves see the effect
of earlier ones, e.g. a territory conquered by a blitz).
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional

import triplea_fm as fm
from triplea_fm import toolkit

from .data import ProData, ProTerritory, TempUnit


def route_for_unit(pd: ProData, u, t: int, combat: bool):
    d, s, player = pd.d, pd.s, pd.player
    ut = pd.ut(u)
    if ut.is_air:
        aa = [x for x in range(pd.n) if pd.has_enemy_aa(player, x)]
        routes = toolkit.unit_routes(d, s, player, u.loc, pd.movement_left(u), True, False, combat, blocked=aa)
    else:
        routes = toolkit.unit_routes(d, s, player, u.loc, pd.movement_left(u), False, ut.can_blitz and combat,
                                     combat)
    return routes.get(t)


def calculate_move_routes(pd: ProData, move_map: Dict[int, ProTerritory], combat: bool,
                          log: Callable[[str], None]) -> List[fm.MoveAction]:
    moves = []
    for t, pt in move_map.items():
        for u in pt.units:
            if isinstance(u, TempUnit) or u.owner != pd.player or u.loc < 0 or u.loc == t:
                continue
            route = route_for_unit(pd, u, t, combat)
            if route is None:
                log(f"  no route for unit {u.id} {pd.ut(u).name} "
                    f"{pd.d.territory_names[u.loc]} -> {pd.d.territory_names[t]}")
                continue
            moves.append(fm.MoveAction((u.id,), route))
    return merge(moves)


def calculate_bombing_routes(pd: ProData, attack_map: Dict[int, ProTerritory]) -> List[fm.MoveAction]:
    moves = []
    for t, pt in attack_map.items():
        for u in pt.bombers:
            if u.loc == t:
                continue
            route = route_for_unit(pd, u, t, True)
            if route is not None:
                moves.append(fm.MoveAction((u.id,), route, fm.RAID))
    return merge(moves)


def merge(moves: List[fm.MoveAction]) -> List[fm.MoveAction]:
    """Merge moves sharing a route and intent, keeping the order of first appearance."""
    grouped: Dict[tuple, List[int]] = {}
    for m in moves:
        grouped.setdefault((m.route, m.intent), []).extend(m.units)
    return [fm.MoveAction(tuple(units), route, intent) for (route, intent), units in grouped.items()]


def execute(d, s, actions, rng, log: Optional[Callable[[str], None]] = None) -> List:
    """Apply actions to `s` in place with the kernel; return the accepted ones. A rejected merged
    move is retried unit by unit."""
    accepted = []
    for a in actions:
        if fm.apply_in_place(d, s, a, rng) != fm.REJECTED:
            accepted.append(a)
            continue
        if a.kind == "move" and len(a.units) > 1:
            for uid in a.units:
                if uid not in s.units:
                    continue
                single = fm.MoveAction((uid,), a.route, a.intent)
                if fm.apply_in_place(d, s, single, rng) != fm.REJECTED:
                    accepted.append(single)
                elif log:
                    log(f"  rejected {fm.action_to_json(d, single)}: {fm.validate(d, s, single).message}")
        elif log:
            log(f"  rejected {fm.action_to_json(d, a)}: {fm.validate(d, s, a).message}")
    return accepted
