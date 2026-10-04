"""V1 land/air rule programs (spec §9), traced from the pinned TripleA engine.

Engine references are given as `Class.method` names; see forward-model-rule-model.md §7 for the
full source index. Functions here mutate the private working state owned by kernel.transition.
"""

from __future__ import annotations

from collections import deque
from typing import Dict, List, Optional, Set, Tuple

from .actions import RAID, CONTINUE, RETREAT
from .definition import (BATTLE, BID, BID_PLACE, COMBAT_MOVE, END_ROUND, END_TURN, INIT,
                         NONCOMBAT_MOVE, NULL_PLAYER, PLACE, PURCHASE, GameDefinition)
from .rng import RandomRequest
from .state import (BATTLE_SELECTION_D, CASUALTIES_D, HELD, MOVE_D, NORMAL_BATTLE, PLACE_D,
                    PURCHASE_D, RAID_BATTLE, RETREAT_D, Battle, Decision, HitBatch,
                    SimulationState, StateError, TerminalOutcome, Unit)

# phase_pc values for decision steps
PC_ENTRY, PC_AWAIT, PC_DONE = 0, 1, 2
# battle step program counters
B_INIT, B_RAIDS, B_DEFENSELESS, B_ONLY_ONE, B_SELECT = 0, 1, 2, 3, 4

Err = Optional[Tuple[str, str]]


# --------------------------------------------------------------------------------------------
# Small predicates
# --------------------------------------------------------------------------------------------

def _ut(d: GameDefinition, u: Unit):
    return d.unit_types[u.type]


def movement_left(d: GameDefinition, u: Unit) -> int:
    return max(0, d.unit_types[u.type].movement - u.moved)


def _exhaust(d: GameDefinition, u: Unit):
    # ChangeFactory.markNoMovementChange: alreadyMoved = max movement (+1 in the engine)
    u.moved = max(u.moved, d.unit_types[u.type].movement)


def is_neutral_land(d, s, t) -> bool:
    return not d.is_water[t] and s.owner[t] == NULL_PLAYER


def is_enemy_territory(d, s, player, t) -> bool:
    return s.owner[t] != player and d.is_at_war(player, s.owner[t])


def enemy_units(d, s, player, t) -> List[Unit]:
    return [u for u in s.units_in(t) if d.is_at_war(player, u.owner)]


def enemy_combat_units(d, s, player, t) -> List[Unit]:
    return [u for u in s.units_in(t) if d.is_at_war(player, u.owner) and not _ut(d, u).is_infrastructure]


def can_land_here(d, s, player, t) -> bool:
    """Matches.airCanLandOnThisAlliedNonConqueredLandTerritory (land, allied owner, not conquered)."""
    return (not d.is_water[t] and t not in s.conquered and s.owner[t] != NULL_PLAYER
            and d.is_allied(player, s.owner[t]))


def is_blitzable(d, s, player, t) -> bool:
    """Matches.territoryIsBlitzable under WW2V2 (no enemy units at all, incl. infrastructure)."""
    if d.is_water[t] or s.owner[t] == NULL_PLAYER:
        return False
    if t in s.conquered and t not in s.blitzed:
        return False
    return not enemy_units(d, s, player, t)


def has_enough_capitals(d, s, player) -> bool:
    """TerritoryAttachment.doWeHaveEnoughCapitalsToProduce with no PlayerAttachment."""
    capitals = [t for t, c in enumerate(d.capital_of) if c == player]
    return not capitals or any(s.owner[t] == player for t in capitals)


def _air_distances(d, s, start: int, limit: int) -> Dict[int, int]:
    """Shortest flight distances from start through territories air may fly over (no neutral land,
    no impassable). The start itself is always allowed."""
    dist = {start: 0}
    q = deque([start])
    while q:
        t = q.popleft()
        if dist[t] >= limit:
            continue
        for n in d.neighbors[t]:
            if n in dist or d.impassable[n] or is_neutral_land(d, s, n):
                continue
            dist[n] = dist[t] + 1
            q.append(n)
    return dist


def air_can_find_land(d, s, player, end: int, moves_left: int) -> bool:
    """AirMovementValidator.canFindLand for land-only maps."""
    if moves_left < 0:
        return False
    if moves_left == 0:
        return False
    dist = _air_distances(d, s, end, moves_left)
    return any(t != end and can_land_here(d, s, player, t) for t in dist)


# --------------------------------------------------------------------------------------------
# Initial state and profile checks
# --------------------------------------------------------------------------------------------

def initial_state(d: GameDefinition) -> SimulationState:
    s = SimulationState()
    s.definition_digest = d.definition_digest
    s.round = 1
    s.step = 0
    s.step_runs = [0] * len(d.steps)
    s.owner = list(d.initial_owner)
    s.orig_owner = list(d.initial_owner)  # InitializationDelegate.initOriginalOwner
    s.balances = [list(b) for b in d.initial_balances]
    for type_id, terr, owner, qty in d.initial_units:
        for _ in range(qty):
            uid = s.next_unit_id
            s.next_unit_id += 1
            s.units[uid] = Unit(uid, type_id, owner, owner, terr)
    return s


def check_profile(d: GameDefinition, s: SimulationState) -> None:
    for u in s.units.values():
        if _ut(d, u).domain == 2:
            raise StateError("UNSUPPORTED_BY_PROFILE_V1", "naval units require Forward Model V2")


# --------------------------------------------------------------------------------------------
# Phase controller
# --------------------------------------------------------------------------------------------

def _effective_max_runs(d: GameDefinition, step) -> Optional[int]:
    if step.kind in (BID, BID_PLACE):
        # InitializationDelegate.initSkipUnusedBids: zero bid -> maxRunCount 0
        bid = d.properties.get(f"{d.player_names[step.actor]} bid", 0)
        if not bid:
            return 0
    return step.max_runs


def _new_decision(s: SimulationState, actor, kind, battle_id=None, batch=None) -> Decision:
    dec = Decision(s.next_decision_id, actor, kind, s.step, battle_id, batch)
    s.next_decision_id += 1
    s.decision = dec
    return dec


def advance(d: GameDefinition, s: SimulationState, rng, events: list) -> None:
    """Run automatic effects until an explicit decision or terminal state (spec §9.1, §10)."""
    guard = 0
    while True:
        guard += 1
        if guard > 100000:
            raise StateError("AUTOMATIC_PROGRESS_FAILURE", "automatic loop limit exceeded")
        if s.terminal is not None:
            s.decision = None
            return
        if s.decision is not None:
            return
        step = d.steps[s.step]
        if not s.entered:
            mr = _effective_max_runs(d, step)
            if mr is not None and s.step_runs[s.step] >= mr:
                _next_step(d, s)
                continue
            s.step_runs[s.step] += 1
            s.entered = True
            s.phase_pc = PC_ENTRY
            _enter_step(d, s, step, rng, events)
            if s.terminal is not None:
                continue
        kind = step.kind
        if kind == BATTLE:
            if _battle_step(d, s, rng, events):
                return
            _exit_step(d, s, step, events)
            continue
        if kind in (PURCHASE, COMBAT_MOVE, NONCOMBAT_MOVE, PLACE):
            if s.phase_pc == PC_ENTRY:
                s.phase_pc = PC_AWAIT if _requires_input(d, s, step) else PC_DONE
            if s.phase_pc == PC_AWAIT:
                _new_decision(s, step.actor, {PURCHASE: PURCHASE_D, PLACE: PLACE_D}.get(kind, MOVE_D))
                return
        _exit_step(d, s, step, events)


def _next_step(d, s):
    s.entered = False
    s.phase_pc = PC_ENTRY
    s.step += 1
    if s.step >= len(d.steps):
        s.step = 0
        s.round += 1


def _requires_input(d, s, step) -> bool:
    p = step.actor
    if step.kind == PURCHASE:
        # PurchaseDelegate.delegateCurrentlyRequiresUserInput
        bal = s.balances[p]
        affordable = any(all(bal[r] >= a for r, a in d.production_rules[rid].costs) for rid in d.frontier[p])
        return affordable and has_enough_capitals(d, s, p)
    if step.kind in (COMBAT_MOVE, NONCOMBAT_MOVE):
        # MoveDelegate.delegateCurrentlyRequiresUserInput
        for u in s.units.values():
            if u.owner == p and u.loc >= 0 and movement_left(d, u) > 0:
                if step.kind == COMBAT_MOVE and not _ut(d, u).can_move_in_combat:
                    continue
                return True
        return False
    if step.kind == PLACE:
        return bool(s.held_units(p))
    return False


def _enter_step(d, s, step, rng, events):
    p = step.actor
    if step.kind == PLACE:
        s.placement_cap = placement_snapshot(d, s, p)
        s.placement_used = {}
    elif step.kind == END_TURN:
        # AbstractEndTurnDelegate.start: capital gate, then production x multiplier
        if p != NULL_PLAYER and has_enough_capitals(d, s, p):
            income = sum(d.production[t] for t in range(len(d.territory_names)) if s.owner[t] == p)
            income *= d.pu_multiplier
            s.balances[p][d.treasury_resource] += income
            events.append(("income", p, income))
    elif step.kind == END_ROUND:
        _check_victory(d, s, events)
    elif step.kind == BATTLE:
        s.phase_pc = B_INIT


def placement_snapshot(d, s, p) -> Dict[int, int]:
    """Producer snapshot at place-step start (AbstractPlaceDelegate.canProduce and
    getMaxUnitsToBePlacedFrom): owned, not conquered this turn, with an owned producer unit.
    Unlimited (-1) for an original factory whose producer unit's original owner is the player,
    otherwise the territory production (WW2V2: at least 0)."""
    caps = {}
    for t in range(len(d.territory_names)):
        if d.is_water[t] or s.owner[t] != p or t in s.conquered:
            continue
        producers = [u for u in s.units_in(t) if u.owner == p and _ut(d, u).can_produce]
        if not producers:
            continue
        if d.original_factory[t] and any(u.orig_owner == p for u in producers):
            caps[t] = -1
        else:
            caps[t] = max(0, d.production[t])
    return caps


def _exit_step(d, s, step, events):
    p = step.actor
    if step.kind == COMBAT_MOVE:
        # MoveDelegate.end (no reset at end): all moved non-air units lose remaining movement
        for u in s.units.values():
            if u.moved > 0 and not _ut(d, u).is_air:
                _exhaust(d, u)
    elif step.kind == NONCOMBAT_MOVE:
        _remove_air_that_cant_land(d, s, events)
        # MoveDelegate.getResetUnitStateChange for all units of all players
        for u in s.units.values():
            u.moved = 0
            u.in_combat = False
    elif step.kind == PLACE:
        for u in s.held_units(p):  # Unplaced units live = false
            del s.units[u.id]
            events.append(("unplaced_destroyed", u.id))
        s.invalidate()
        _remove_air_that_cant_land(d, s, events)
        s.placement_cap = None
        s.placement_used = {}
    elif step.kind == END_TURN:
        # AbstractEndTurnDelegate.end -> BattleTracker.clear()
        s.conquered.clear()
        s.blitzed.clear()
        s.fought.clear()
        s.raid_loss.clear()
        s.battles.clear()
        s.active_battle = None
    _next_step(d, s)


def _remove_air_that_cant_land(d, s, events):
    """AirThatCantLandUtil: air on land that was conquered this turn or is not allied is removed."""
    removed = []
    for u in list(s.units.values()):
        if u.loc < 0 or not _ut(d, u).is_air:
            continue
        t = u.loc
        ok = (not d.is_water[t] and t not in s.conquered and s.owner[t] != NULL_PLAYER
              and d.is_allied(u.owner, s.owner[t]))
        if not ok:
            removed.append(u.id)
    for uid in removed:
        del s.units[uid]
        events.append(("air_lost", uid))
    if removed:
        s.invalidate()
        _prune_battles(s)


def _check_victory(d, s, events):
    """EndRoundDelegate.checkVictoryCities for Total Victory."""
    if not d.total_victory:
        return
    winners = []
    for aid, members in enumerate(d.alliance_members):
        vcs = sum(d.victory_value[t] for t in range(len(d.territory_names)) if s.owner[t] in members)
        if vcs >= d.victory_threshold[aid]:
            winners.append(aid)
    if len(winners) > 1:
        raise StateError("UNCERTIFIED_CONFIGURATION", "simultaneous victory is an unresolved rule")
    if winners:
        s.terminal = TerminalOutcome("TOTAL_VICTORY_VC", s.round, winners)
        events.append(("victory", winners[0]))


# --------------------------------------------------------------------------------------------
# Validation dispatch
# --------------------------------------------------------------------------------------------

def validate_action(d: GameDefinition, s: SimulationState, a) -> Err:
    dec = s.decision
    if dec is None:
        return ("WRONG_PHASE", "no pending decision")
    k = getattr(a, "kind", None)
    if k == "end_phase":
        if dec.kind not in (PURCHASE_D, MOVE_D, PLACE_D):
            return ("PHASE_OBLIGATION", f"cannot end phase during a {dec.kind} decision")
        return None
    if k == "purchase":
        return ("WRONG_PHASE", "not a purchase decision") if dec.kind != PURCHASE_D else _validate_purchase(d, s, a)
    if k == "move":
        return ("WRONG_PHASE", "not a move decision") if dec.kind != MOVE_D else _validate_move(d, s, a)
    if k == "place":
        return ("WRONG_PHASE", "not a place decision") if dec.kind != PLACE_D else _validate_place(d, s, a)
    if k == "select_battle":
        if dec.kind != BATTLE_SELECTION_D:
            return ("WRONG_PHASE", "not a battle selection decision")
        b = s.battles.get(a.battle_id)
        if b is None or b.status != "PENDING":
            return ("UNKNOWN_ID", "no such pending battle")
        if b.kind == NORMAL_BATTLE and any(o.kind == RAID_BATTLE and o.site == b.site for o in s.battles.values()):
            return ("PHASE_OBLIGATION", "the bombing raid at this territory must be fought first")
        return None
    if k == "casualties":
        if dec.kind != CASUALTIES_D:
            return ("WRONG_PHASE", "not a casualty decision")
        hb = dec.hit_batch
        if a.hit_batch_id != hb.id:
            return ("STALE_DECISION", "hit batch mismatch")
        killed = list(a.killed)
        if len(set(killed)) != len(killed):
            return ("DUPLICATE_UNIT", "duplicate casualty")
        if any(u not in hb.eligible for u in killed):
            return ("BAD_CASUALTIES", "casualty is not an eligible target")
        need = min(hb.hits, len(hb.eligible))
        if len(killed) != need:
            return ("BAD_CASUALTIES", f"exactly {need} casualties required")
        return None
    if k == "retreat":
        if dec.kind != RETREAT_D:
            return ("WRONG_PHASE", "not a retreat decision")
        if a.choice == CONTINUE:
            return None if a.destination is None else ("ILLEGAL_RETREAT", "CONTINUE takes no destination")
        if a.choice != RETREAT:
            return ("ILLEGAL_RETREAT", "unknown retreat choice")
        b = s.battles[dec.battle_id]
        if a.destination not in retreat_destinations(d, s, b):
            return ("ILLEGAL_RETREAT", "destination is not a legal retreat territory")
        return None
    return ("INVALID_ACTION", f"unknown action {a!r}")


def _validate_purchase(d, s, a) -> Err:
    p = s.decision.actor
    if not has_enough_capitals(d, s, p):
        return ("PHASE_OBLIGATION", "cannot purchase without a capital")
    total = [0] * len(d.resource_names)
    for rid, count in a.quantities:
        if not isinstance(count, int) or count < 0:
            return ("INVALID_ACTION", "purchase quantities must be nonnegative integers")
        if not 0 <= rid < len(d.production_rules) or rid not in d.frontier[p]:
            return ("FRONTIER_MISMATCH", "production rule is not in the player's frontier")
        for r, amount in d.production_rules[rid].costs:
            total[r] += amount * count
    if any(total[r] > s.balances[p][r] for r in range(len(total))):
        return ("INSUFFICIENT_RESOURCES", "not enough resources")
    return None


def _validate_place(d, s, a) -> Err:
    p = s.decision.actor
    t = a.territory
    if not a.units:
        return ("INVALID_ACTION", "no units")
    if len(set(a.units)) != len(a.units):
        return ("DUPLICATE_UNIT", "duplicate unit")
    for uid in a.units:
        u = s.units.get(uid)
        if u is None or u.loc != HELD or u.owner != p:
            return ("UNKNOWN_ID", f"unit {uid} is not held by the player")
        ut = _ut(d, u)
        if ut.domain == 2:
            return ("UNSUPPORTED_BY_PROFILE_V1", "naval placement requires Forward Model V2")
        if ut.is_factory:
            return ("UNCERTIFIED_CONFIGURATION", "construction placement is not implemented")
    if not 0 <= t < len(d.territory_names) or d.is_water[t]:
        return ("CAPACITY_EXCEEDED", "territory cannot receive land/air placements")
    cap = (s.placement_cap or {}).get(t)
    if cap is None or s.owner[t] != p:
        return ("CAPACITY_EXCEEDED", "no eligible producer in this territory")
    if cap >= 0 and s.placement_used.get(t, 0) + len(a.units) > cap:
        return ("CAPACITY_EXCEEDED", f"producer capacity {cap} exceeded")
    return None


def _validate_move(d, s, a) -> Err:
    p = s.decision.actor
    combat = d.steps[s.step].kind == COMBAT_MOVE
    route = list(a.route)
    if not a.units:
        return ("INVALID_ACTION", "no units")
    if len(set(a.units)) != len(a.units):
        return ("DUPLICATE_UNIT", "not all units unique")
    if len(route) < 2:
        return ("BAD_ROUTE", "route must contain start and at least one step")
    nt = len(d.territory_names)
    if any(not 0 <= t < nt for t in route):
        return ("UNKNOWN_ID", "unknown territory in route")
    for x, y in zip(route, route[1:]):
        if y not in d.neighbors[x]:
            return ("BAD_ROUTE", "route territories are not adjacent")
    start, end, middle = route[0], route[-1], route[1:-1]
    units = []
    for uid in a.units:
        u = s.units.get(uid)
        if u is None:
            return ("UNKNOWN_ID", f"unknown unit {uid}")
        if u.loc != start:
            return ("UNKNOWN_ID", f"unit {uid} is not in the start territory")
        if u.owner != p:
            return ("ACTOR_MISMATCH", f"unit {uid} is not owned by the moving player")
        if _ut(d, u).domain == 2:
            return ("UNSUPPORTED_BY_PROFILE_V1", "naval movement requires Forward Model V2")
        units.append(u)
    types = [_ut(d, u) for u in units]
    all_air = all(t.is_air for t in types)
    any_air = any(t.is_air for t in types)
    any_land = any(t.is_land for t in types)
    if any(d.impassable[t] for t in route):
        return ("BLOCKED_ROUTE", "cannot move through impassable territories")
    if any_land and any(d.is_water[t] for t in route):
        return ("UNSUPPORTED_BY_PROFILE_V1", "land units entering water require transports (V2)")
    if d.neutral_charge != 0 and any(is_neutral_land(d, s, t) for t in route[1:]):
        # admitted only as inert; a hypothetical state must not make it active
        return ("UNSUPPORTED_BY_PROFILE_V1", "neutral entry charges are OPTIONAL_LATER")

    if combat:
        err = _validate_combat_route(d, s, p, units, types, route, all_air, any_air, any_land)
    else:
        err = _validate_noncombat_route(d, s, p, units, types, route, all_air)
    if err:
        return err
    # MoveValidator.validateNonEnemyUnitsOnPath (AA and factories don't block)
    if not all_air and any(enemy_combat_units(d, s, p, t) for t in middle):
        return ("BLOCKED_ROUTE", "enemy units on the route")
    # validateBasic: movement budget (cost 1 per entered territory)
    cost = len(route) - 1
    for u in units:
        if movement_left(d, u) < cost:
            return ("INSUFFICIENT_MOVEMENT", f"unit {u.id} lacks movement")
    # AirMovementValidator.validateAirCanLand
    if any_air and not can_land_here(d, s, p, end):
        for u in units:
            if _ut(d, u).is_air and not air_can_find_land(d, s, p, end, movement_left(d, u) - cost):
                return ("ILLEGAL_LANDING", "not all air units can land")
        for u in s.units_in(end):
            if u.owner == p and _ut(d, u).is_air and not air_can_find_land(d, s, p, end, movement_left(d, u)):
                return ("ILLEGAL_LANDING", "air already at the destination cannot land")
    if a.intent == RAID:
        if not combat:
            return ("INVALID_ACTION", "raids are combat moves")
        if not all(t.is_strategic_bomber for t in types):
            return ("INVALID_ACTION", "only strategic bombers can raid (no escorts by default)")
        if not any(_ut(d, u).can_be_damaged for u in enemy_units(d, s, p, end)):
            return ("INVALID_ACTION", "no enemy damageable target at the destination")
    return None


def _validate_combat_route(d, s, p, units, types, route, all_air, any_air, any_land) -> Err:
    start, end, middle = route[0], route[-1], route[1:-1]
    if any(not t.can_move_in_combat for t in types):
        return ("INVALID_ACTION", "cannot move AA guns in combat movement phase")
    enemy = lambda t: is_enemy_territory(d, s, p, t)
    # Contested start owned by the enemy (MoveValidator.validateCombat)
    middle_enemy = any(enemy(t) for t in middle)
    if not d.is_water[start] and enemy(start) and middle_enemy:
        if not is_blitzable(d, s, p, start) and not all_air:
            return ("BLOCKED_ROUTE", "cannot blitz out of battle further into enemy territory")
        if any(t.is_land and not t.can_blitz for t in types):
            return ("BLOCKED_ROUTE", "not all units can blitz out of empty enemy territory")
    if not d.is_water[start] and not enemy(start) and middle_enemy and not is_blitzable(d, s, p, start) \
            and not all_air:
        return ("BLOCKED_ROUTE", "cannot blitz out of battle into enemy territory")
    # Neutral middle steps stop land units (neutrals not blitzable)
    if not all_air and any(is_neutral_land(d, s, t) for t in middle):
        return ("BLOCKED_ROUTE", "must stop land units when passing through neutral territories")
    if any_land:
        enemy_count, all_blitzable = 0, True
        for t in middle:
            if d.is_water[t]:
                continue
            if d.is_at_war(p, s.owner[t]) or t in s.conquered:
                enemy_count += 1
                all_blitzable &= is_blitzable(d, s, p, t)
        if enemy_count > 0 and not all_blitzable:
            return ("BLOCKED_ROUTE", "cannot blitz on that route")
        start_fought = start in s.conquered or start in s.blitzed
        non_end = route[:-1]
        friendly = lambda t: s.owner[t] != NULL_PLAYER and d.is_allied(p, s.owner[t])
        fought_over = lambda t: t in s.conquered or t in s.blitzed or t in s.fought
        non_blitzers = [u for u, t in zip(units, types) if t.is_land and not t.can_blitz]
        if not start_fought and all(friendly(t) for t in non_end) and not any(fought_over(t) for t in non_end):
            non_blitzers = [u for u in non_blitzers if u.in_combat]
        if non_blitzers:
            return ("BLOCKED_ROUTE", "not all units can blitz")
    if any_air:
        moved_air = any(u.moved > 0 for u, t in zip(units, types) if t.is_air)
        if any(is_neutral_land(d, s, t) for t in middle) or (is_neutral_land(d, s, start) and moved_air):
            return ("BLOCKED_ROUTE", "air units cannot fly over neutral territories")
    if not all_air and any(t in s.conquered and t not in s.blitzed and not d.is_water[t] for t in middle):
        return ("BLOCKED_ROUTE", "cannot move through newly captured territories")
    return None


def _validate_noncombat_route(d, s, p, units, types, route, all_air) -> Err:
    end = route[-1]
    neutral_or_enemy = lambda t: is_neutral_land(d, s, t) or (
        is_enemy_territory(d, s, p, t) and not (d.is_water[t] and s.owner[t] == NULL_PLAYER))
    if neutral_or_enemy(end):
        return ("WRONG_PHASE", "cannot advance units to battle in non combat")
    if enemy_units(d, s, p, end):
        return ("WRONG_PHASE", "cannot advance to battle in non combat")
    if all_air:
        if any(is_neutral_land(d, s, t) for t in route):
            return ("BLOCKED_ROUTE", "air units cannot fly over neutral territories in non combat")
    elif any(neutral_or_enemy(t) for t in route):
        return ("BLOCKED_ROUTE", "cannot move units through neutral or enemy territories in non combat")
    return None


def retreat_destinations(d, s, b: Battle) -> List[int]:
    """MustFightBattle.getAttackerRetreatTerritories for land/air battles."""
    alive = [s.units[u] for u in b.attackers if u in s.units]
    if alive and all(_ut(d, u).is_air for u in alive):
        return [b.site]
    p = b.attacker
    by_origin: Dict[int, List[Unit]] = {}
    for u in alive:
        o = b.origins.get(u.id)
        if o is not None:
            by_origin.setdefault(o, []).append(u)
    out = []
    for t, us in sorted(by_origin.items()):
        if enemy_combat_units(d, s, p, t):
            continue
        if all(_ut(d, u).is_air for u in us):  # WW2V2: air-only origins excluded
            continue
        if is_enemy_territory(d, s, p, t) or s.owner[t] == NULL_PLAYER or d.impassable[t] or d.is_water[t]:
            continue
        out.append(t)
    return out


# --------------------------------------------------------------------------------------------
# Action application
# --------------------------------------------------------------------------------------------

def apply_action(d: GameDefinition, s: SimulationState, a, rng, events) -> None:
    dec = s.decision
    k = a.kind
    s.decision = None
    if k == "end_phase":
        s.phase_pc = PC_DONE
        return
    if k == "purchase":
        p = dec.actor
        for rid, count in a.quantities:
            rule = d.production_rules[rid]
            for r, amount in rule.costs:
                s.balances[p][r] -= amount * count
            for type_id, qty in rule.results:
                for _ in range(qty * count):
                    uid = s.next_unit_id
                    s.next_unit_id += 1
                    s.units[uid] = Unit(uid, type_id, p, p, HELD)
        events.append(("purchase", p, a.quantities))
        return
    if k == "place":
        for uid in a.units:
            s.units[uid].loc = a.territory
        s.placement_used[a.territory] = s.placement_used.get(a.territory, 0) + len(a.units)
        s.invalidate()
        events.append(("place", a.territory, a.units))
        return
    if k == "move":
        _perform_move(d, s, dec.actor, a, rng, events)
        return
    if k == "select_battle":
        s.active_battle = a.battle_id
        s.battles[a.battle_id].status = "ACTIVE"
        return
    b = s.battles[dec.battle_id]
    if k == "casualties":
        _commit_casualties(d, s, b, list(a.killed), events)
        return
    if k == "retreat":
        if a.choice == CONTINUE:
            b.stage = "STALEMATE_CHECK"
        else:
            _do_retreat(d, s, b, a.destination, events)
        return


def _perform_move(d, s, p, a, rng, events):
    combat = d.steps[s.step].kind == COMBAT_MOVE
    route = list(a.route)
    start, end, middle = route[0], route[-1], route[1:-1]
    moving = [s.units[u] for u in a.units]
    cost = len(route) - 1
    # Fly-over AA (AaInMoveUtil.getTerritoriesWhereAaWillFire): middle steps, plus start if no battle
    # was fought there. Combat move only, unless "Always on AA" also enables it in noncombat move.
    if combat or d.always_on_aa:
        aa_terrs = [t for t in middle if any(_ut(d, u).is_aa for u in enemy_units(d, s, p, t))]
        if start not in s.fought and any(_ut(d, u).is_aa for u in enemy_units(d, s, p, start)):
            aa_terrs.insert(0, start)
        for t in aa_terrs:
            air = sorted((u for u in moving if _ut(d, u).is_air), key=lambda u: (movement_left(d, u), u.id))
            if not air:
                break
            dice = rng.draw(RandomRequest("aa:flyover", d.dice_sides, len(air)))
            for u, die in zip(air, dice):
                if die < 1:  # attackAa = 1
                    moving.remove(u)
                    del s.units[u.id]
                    events.append(("aa_kill", t, u.id))
    moving_ids = {u.id for u in moving}
    for b in s.battles.values():
        if b.site == start and b.status == "PENDING":
            b.attackers = [u for u in b.attackers if u not in moving_ids]
    for bid in [bid for bid, b in s.battles.items()
                if b.site == start and b.status == "PENDING" and not b.attackers]:
        del s.battles[bid]
    for u in moving:
        u.loc = end
        u.moved += cost
    s.invalidate()
    _prune_battles(s)
    if not moving:
        return
    if not combat:
        events.append(("move", a.units, route))
        return
    land = [u for u in moving if _ut(d, u).is_land]
    if a.intent == RAID:
        b = _battle_at(s, end, RAID_BATTLE, p)
        if b is None:
            b = _new_battle(s, RAID_BATTLE, end, p, s.owner[end])
        for u in moving:
            b.attackers.append(u.id)
            b.origins[u.id] = route[-2]
            u.in_combat = True
        events.append(("raid_declared", end, a.units))
        _mark_movement(d, s, route, land)
        return
    # BattleTracker.addEmptyBattle: empty enemy territories on the route (steps; end handled below)
    if land:
        conquer = []
        for t in middle:
            if is_enemy_territory(d, s, p, t) and not enemy_combat_units(d, s, p, t):
                conquer.append((t, is_blitzable(d, s, p, t)))
        for t, blitz in conquer:
            _take_over(d, s, t, p, blitz, events)
    if enemy_combat_units(d, s, p, end):
        b = _battle_at(s, end, NORMAL_BATTLE, p)
        if b is None:
            b = _new_battle(s, NORMAL_BATTLE, end, p, s.owner[end])
        for u in moving:
            b.attackers.append(u.id)
            b.origins[u.id] = route[-2]
            if not _ut(d, u).is_air:
                _exhaust(d, u)  # MustFightBattle.addAttackChange
        events.append(("attack", end, a.units))
    elif land and is_enemy_territory(d, s, p, end):
        blitz = is_blitzable(d, s, p, end)
        existing = _battle_at(s, end, NORMAL_BATTLE, p)
        if existing is None:
            _take_over(d, s, end, p, blitz, events)
            for u in moving:
                u.in_combat = True
    else:
        existing = _battle_at(s, end, NORMAL_BATTLE, p)
        if existing is not None:
            for u in moving:
                existing.attackers.append(u.id)
                existing.origins[u.id] = route[-2]
    _mark_movement(d, s, route, land)
    events.append(("move", a.units, route))


def _mark_movement(d, s, route, land):
    """MovePerformer.markMovementChange: crossing an empty neutral or a conquered-not-blitzed
    territory stops all land units in the move."""
    steps = route[1:]
    stop = any(t in s.conquered and t not in s.blitzed for t in steps)
    if stop:
        for u in land:
            _exhaust(d, u)


def _prune_battles(s):
    """Drop references to units that no longer exist (casualties already removed)."""
    for b in s.battles.values():
        b.attackers = [u for u in b.attackers if u in s.units]
        b.defenders = [u for u in b.defenders if u in s.units]
        for u in [u for u in b.origins if u not in s.units]:
            del b.origins[u]


def _battle_at(s, site, kind, attacker) -> Optional[Battle]:
    for b in s.battles.values():
        if b.site == site and b.kind == kind and b.attacker == attacker:
            return b
    return None


def _new_battle(s, kind, site, attacker, defender) -> Battle:
    b = Battle(s.next_battle_id, kind, site, attacker, defender)
    s.next_battle_id += 1
    s.battles[b.id] = b
    return b


def _take_over(d, s, t, p, blitz, events):
    """BattleTracker.takeOver for land territories."""
    prev = s.owner[t]
    cp = d.capital_of[t]
    if cp != NULL_PLAYER and prev == cp and d.is_at_war(p, prev):
        owned_capitals = sum(1 for x, c in enumerate(d.capital_of) if c == cp and s.owner[x] == cp)
        if not 1 < owned_capitals:  # retainCapitalNumber default 1; destroysPUs default false
            r = d.treasury_resource
            amount = s.balances[cp][r]
            s.balances[cp][r] = 0
            s.balances[p][r] += amount
            events.append(("capital_captured", t, cp, p, amount))
    new_owner = p
    orig = s.orig_owner[t]
    if orig != NULL_PLAYER and orig != prev and orig != p and d.is_allied(p, orig):
        orig_has_capital = any(s.owner[x] == orig for x, c in enumerate(d.capital_of) if c == orig)
        if orig_has_capital or d.capital_of[t] == orig:
            new_owner = orig  # liberation
    s.owner[t] = new_owner
    for u in s.units_in(t):
        if _ut(d, u).is_infrastructure and d.is_at_war(new_owner, u.owner):
            u.owner = new_owner
            _exhaust(d, u)
    s.conquered.add(t)
    if blitz:
        s.blitzed.add(t)
    events.append(("takeover", t, prev, new_owner, bool(blitz)))


# --------------------------------------------------------------------------------------------
# Battle phase
# --------------------------------------------------------------------------------------------

def _battle_step(d, s, rng, events) -> bool:
    """Returns True if paused at a decision."""
    p = d.steps[s.step].actor
    while True:
        if s.active_battle is not None:
            b = s.battles[s.active_battle]
            if _run_battle(d, s, b, rng, events):
                return True
            continue
        pending = sorted((b for b in s.battles.values() if b.attacker == p and b.status == "PENDING"),
                         key=lambda b: b.id)
        if s.phase_pc == B_INIT:
            _setup_same_territory_battles(d, s, p, events)
            s.phase_pc = B_RAIDS
            continue
        if s.phase_pc == B_RAIDS:
            raids = [b for b in pending if b.kind == RAID_BATTLE]
            if raids:
                _start(s, raids[0])
                continue
            s.phase_pc = B_DEFENSELESS
            continue
        if s.phase_pc == B_DEFENSELESS:
            weak = [b for b in pending if _defender_power(d, s, b) == 0]
            if weak:
                _start(s, weak[0])
                continue
            s.phase_pc = B_ONLY_ONE
            continue
        if s.phase_pc == B_ONLY_ONE:
            s.phase_pc = B_SELECT
            if len(pending) == 1:
                _start(s, pending[0])
            continue
        if pending:
            _new_decision(s, p, BATTLE_SELECTION_D)
            return True
        return False


def _start(s, b):
    b.status = "ACTIVE"
    s.active_battle = b.id


def _setup_same_territory_battles(d, s, p, events):
    """BattleDelegate.setupUnitsInSameTerritoryBattles for land/air: the player's units sharing a
    territory with enemy units, or standing in enemy territory, get a scripted battle there.
    Units already attacking in a pending bombing raid at that territory are excluded."""
    for t in range(len(d.territory_names)):
        mine = [u for u in s.units_in(t) if u.owner == p]
        if not mine or not (enemy_units(d, s, p, t) or is_enemy_territory(d, s, p, t)):
            continue
        raid = _battle_at(s, t, RAID_BATTLE, p)
        if raid is not None:
            mine = [u for u in mine if u.id not in raid.attackers]
        if all(_ut(d, u).is_infrastructure for u in mine):
            continue
        battle = _battle_at(s, t, NORMAL_BATTLE, p)
        if battle is None:
            if enemy_combat_units(d, s, p, t):
                battle = _new_battle(s, NORMAL_BATTLE, t, p, s.owner[t])
            elif any(_ut(d, u).is_land for u in mine) and is_enemy_territory(d, s, p, t):
                _take_over(d, s, t, p, False, events)  # BattleTracker.addEmptyBattle (scripted route)
                continue
            else:
                continue
        if not battle.attackers:
            for u in mine:
                if not _ut(d, u).is_infrastructure:
                    battle.attackers.append(u.id)
                    battle.origins[u.id] = t  # RouteScripted(territory)


def _defender_power(d, s, b) -> int:
    if b.kind != NORMAL_BATTLE:
        return 1
    return sum(_ut(d, u).defense for u in enemy_combat_units(d, s, b.attacker, b.site))


def _alive(s, ids):
    return [s.units[u] for u in ids if u in s.units]


def _attack_strengths(d, s, units: List[Unit]) -> List[Tuple[Unit, int]]:
    """MainOffenseCombatValue with legacy artillery support (+1, one target per artillery)."""
    support = sum(1 for u in units if _ut(d, u).is_artillery)
    out = []
    for u in sorted(units, key=lambda x: x.id):
        ut = _ut(d, u)
        strength = ut.attack
        if ut.is_artillery_supportable and support > 0:
            strength += 1
            support -= 1
        out.append((u, min(strength, d.dice_sides)))
    return out


def _defense_strengths(d, units: List[Unit]) -> List[Tuple[Unit, int]]:
    return [(u, min(_ut(d, u).defense, d.dice_sides)) for u in sorted(units, key=lambda x: x.id)]


def _roll(d, rng, purpose, strengths) -> Tuple[int, Tuple[int, ...]]:
    firing = sorted(((u, st) for u, st in strengths if st > 0), key=lambda x: (-x[1], x[0].id))
    if not firing:
        return 0, ()
    dice = rng.draw(RandomRequest(purpose, d.dice_sides, len(firing)))
    hits = sum(1 for (u, st), die in zip(firing, dice) if die < st)
    return hits, tuple(dice)


def _auto_casualties(d, s, eligible: List[Unit], hits: int) -> Optional[List[int]]:
    """CasualtySelector auto-selection: all die, single target, or one type with 1 HP."""
    if hits >= len(eligible):
        return [u.id for u in eligible]
    types = {u.type for u in eligible}
    if len(eligible) == 1 or (len(types) == 1 and _ut(d, eligible[0]).hit_points == 1):
        order = sorted(eligible, key=lambda u: (movement_left(d, u) if _ut(d, u).is_air else 0, u.id))
        return [u.id for u in order[:hits]]
    return None


def _run_battle(d, s, b: Battle, rng, events) -> bool:
    if b.kind == RAID_BATTLE:
        _fight_raid(d, s, b, rng, events)
        return False
    p = b.attacker
    while True:
        st = b.stage
        if st == "START":
            b.defenders = [u.id for u in enemy_units(d, s, p, b.site)]
            b.round = 1
            att = [u for u in _alive(s, b.attackers) if not _ut(d, u).is_infrastructure]
            if not att:
                _end_battle(d, s, b, "DEFENDER", events)
                return False
            if not enemy_combat_units(d, s, p, b.site):
                _end_battle(d, s, b, "ATTACKER", events)
                return False
            b.stage = "AA"
        elif st == "AA":
            if b.round == 1:
                aa = [u for u in _alive(s, b.defenders) if _ut(d, u).is_aa]
                air = sorted((u for u in _alive(s, b.attackers) if _ut(d, u).is_air),
                             key=lambda u: (movement_left(d, u), u.id))
                if aa and air:
                    dice = rng.draw(RandomRequest("aa:battle", d.dice_sides, len(air)))
                    for u, die in zip(air, dice):
                        if die < 1:
                            del s.units[u.id]
                            events.append(("aa_kill", b.site, u.id))
                    s.invalidate()
                    _prune_battles(s)
                for u in _alive(s, b.attackers):  # MarkNoMovementLeft
                    if not _ut(d, u).is_air:
                        _exhaust(d, u)
            b.stage = "ATT_FIRE"
        elif st == "ATT_FIRE":
            att = [u for u in _alive(s, b.attackers) if not _ut(d, u).is_infrastructure]
            hits, dice = _roll(d, rng, "combat:attacker", _attack_strengths(d, s, att))
            b.attacker_dice = dice
            b.waiting_dead = []
            b.stage = "DEF_FIRE"
            if _queue_hits(d, s, b, hits, "DEFENDER", events):
                return True
        elif st == "DEF_FIRE":
            defn = [u for u in _alive(s, b.defenders) if not _ut(d, u).is_infrastructure]
            hits, dice = _roll(d, rng, "combat:defender", _defense_strengths(d, defn))
            b.defender_dice = dice
            b.stage = "REMOVE"
            if _queue_hits(d, s, b, hits, "ATTACKER", events):
                return True
        elif st == "REMOVE":
            for uid in b.waiting_dead:
                if uid in s.units:
                    del s.units[uid]
            if b.waiting_dead:
                events.append(("casualties_removed", b.site, tuple(b.waiting_dead)))
            b.waiting_dead = []
            s.invalidate()
            _prune_battles(s)
            b.stage = "END_CHECK"
        elif st == "END_CHECK":
            att = [u for u in _alive(s, b.attackers) if not _ut(d, u).is_infrastructure]
            dfd = [u for u in _alive(s, b.defenders) if not _ut(d, u).is_infrastructure]
            if not att:
                _end_battle(d, s, b, "DEFENDER", events)
                return False
            if not dfd:
                _end_battle(d, s, b, "ATTACKER", events)
                return False
            if _stalemate(d, s, att, dfd):
                _end_battle(d, s, b, "DRAW", events)
                return False
            b.stage = "RETREAT_QUERY"
        elif st == "RETREAT_QUERY":
            if retreat_destinations(d, s, b):
                b.stage = "RETREAT_WAIT"
                _new_decision(s, p, RETREAT_D, b.id)
                return True
            b.stage = "STALEMATE_CHECK"
        elif st == "RETREAT_WAIT":
            return True
        elif st == "STALEMATE_CHECK":
            att = [u for u in _alive(s, b.attackers) if not _ut(d, u).is_infrastructure]
            dfd = [u for u in _alive(s, b.defenders) if not _ut(d, u).is_infrastructure]
            if _stalemate(d, s, att, dfd):
                _end_battle(d, s, b, "DRAW", events)
                return False
            b.round += 1
            if b.round > 10000:
                raise StateError("AUTOMATIC_PROGRESS_FAILURE", "battle exceeded 10000 rounds")
            b.stage = "AA"
        elif ":" in st:  # paused at a casualty decision
            return True
        else:
            raise StateError("ENGINE_FAILURE", f"unknown battle stage {st}")


def _stalemate(d, s, att, dfd) -> bool:
    att_power = sum(st for _, st in _attack_strengths(d, s, att))
    def_power = sum(st for _, st in _defense_strengths(d, dfd))
    return att_power == 0 and def_power == 0


def _queue_hits(d, s, b, hits, side, events) -> bool:
    """Create the hit batch for `side` (whose units are hit). Returns True if a decision is needed."""
    if hits <= 0:
        return False
    if side == "DEFENDER":
        eligible = [u for u in _alive(s, b.defenders) if not _ut(d, u).is_infrastructure]
        actor = b.defender
    else:
        eligible = [u for u in _alive(s, b.attackers) if not _ut(d, u).is_infrastructure]
        actor = b.attacker
    if not eligible:
        return False
    auto = _auto_casualties(d, s, eligible, hits)
    if auto is not None:
        b.waiting_dead.extend(auto)
        events.append(("casualties", b.site, side, tuple(auto)))
        return False
    hb = HitBatch(s.next_batch_id, actor, hits, [u.id for u in eligible], side, "combat")
    s.next_batch_id += 1
    b.pending = hb
    b.stage = ("DEF_SELECT" if side == "DEFENDER" else "ATT_SELECT") + ":" + b.stage
    _new_decision(s, actor, CASUALTIES_D, b.id, hb)
    return True


def _commit_casualties(d, s, b, killed, events):
    b.waiting_dead.extend(killed)
    side = b.pending.side
    b.pending = None
    b.stage = b.stage.split(":", 1)[1]  # resume the stored stage, no reroll
    events.append(("casualties", b.site, side, tuple(killed)))


def _do_retreat(d, s, b, dest, events):
    for u in _alive(s, b.attackers):
        if not _ut(d, u).is_air:
            u.loc = dest
    s.invalidate()
    events.append(("retreat", b.site, dest))
    b.attackers = []
    _end_battle(d, s, b, "DEFENDER", events)


def _end_battle(d, s, b, winner, events):
    p = b.attacker
    if winner == "ATTACKER":
        survivors = [u for u in _alive(s, b.attackers) if _ut(d, u).is_land and u.loc == b.site]
        if survivors and is_enemy_territory(d, s, p, b.site):
            _take_over(d, s, b.site, p, False, events)
            for u in survivors:
                u.in_combat = True
    s.fought.add(b.site)
    events.append(("battle_end", b.id, b.site, winner))
    del s.battles[b.id]
    s.active_battle = None


def _fight_raid(d, s, b, rng, events):
    """StrategicBombingRaidBattle.fight with treasury damage (WW2V2 per-bomber production cap)."""
    p = b.attacker
    bombers = sorted(_alive(s, b.attackers), key=lambda u: (movement_left(d, u), u.id))
    aa = [u for u in enemy_units(d, s, p, b.site) if _ut(d, u).is_aa]
    if aa and bombers:
        dice = rng.draw(RandomRequest("aa:raid", d.dice_sides, len(bombers)))
        for u, die in zip(list(bombers), dice):
            if die < 1:
                bombers.remove(u)
                del s.units[u.id]
                events.append(("aa_kill", b.site, u.id))
        s.invalidate()
        _prune_battles(s)
    total = 0
    if bombers:
        rolls = rng.draw(RandomRequest("raid:damage", d.dice_sides, len(bombers)))
        cap = d.production[b.site]
        for r in rolls:
            total += min(r + 1, cap)
        if d.pu_cap:
            # "Territory Turn Limit" (PU_CAP): total loss this turn capped at production
            total = min(total, max(0, cap - s.raid_loss.get(b.site, 0)))
    defender = s.owner[b.site]
    s.raid_loss[b.site] = s.raid_loss.get(b.site, 0) + total
    if defender != NULL_PLAYER and d.is_at_war(p, defender):
        cost = total * d.pu_multiplier
        res = d.treasury_resource
        removed = min(cost, s.balances[defender][res])
        s.balances[defender][res] -= removed
        events.append(("raid_damage", b.site, defender, removed))
    events.append(("battle_end", b.id, b.site, "RAID"))
    del s.battles[b.id]
    s.active_battle = None
