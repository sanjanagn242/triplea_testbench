"""Deterministic battles for the agent's simulations (thesis §5.2.2).

Roxo's forward model resolves every battle with the most likely outcome from the Fast AI's battle
estimate (ActionUtils.simulateBattles with FastOddsEstimator), so one playout values a node. The
real game still rolls dice. This module ports the estimator (and the ProBattleUtils strength
estimate it calls) and simulateBattles, and swaps them in for the kernel's dice-rolling battle loop,
rules._run_battle, inside this agent's process only.
"""

from __future__ import annotations

import math

from triplea_fm import rules
from triplea_fm.state import NORMAL_BATTLE

_dice_battle = rules._run_battle


def estimate_power(d, s, units, attacking: bool) -> float:
    """ProBattleUtils.estimatePower: total combat power (artillery support included), in d6 terms."""
    if attacking:
        strengths = rules._attack_strengths(d, s, units)
    else:
        strengths = rules._defense_strengths(d, units)
    return sum(st for _, st in strengths) * 6.0 / d.dice_sides


def estimate_strength(d, s, units, attacking: bool) -> float:
    """ProBattleUtils.estimateStrength: 2 x hit points left + power."""
    hit_points = sum(d.unit_types[u.type].hit_points - u.hits for u in units)
    return 2.0 * hit_points + estimate_power(d, s, units, attacking)


def estimate_strength_difference(d, s, attackers, defenders) -> float:
    """ProBattleUtils.estimateStrengthDifference: 0 = sure defender win, 50 = even, 100+ = sure
    attacker win. Both lists hold combat units only (no infrastructure)."""
    if not attackers or estimate_power(d, s, attackers, True) <= 0:
        return 0.0
    if not defenders or estimate_power(d, s, defenders, False) <= 0:
        return 99_999.0
    attacker = estimate_strength(d, s, attackers, True)
    defender = estimate_strength(d, s, defenders, False)
    return (attacker - defender) / math.pow(defender, 0.85) * 50 + 50


def most_likely_outcome(d, s, attackers, defenders):
    """FastOddsEstimator.calculate: (attackers left, defenders left). The winning side keeps its most
    expensive units, ceil(n x margin / 50) of them, n counting every unit in the battle (Roxo's
    simulateBattles passes the defenders' infrastructure too); the losing side keeps none."""
    combat = lambda us: [u for u in us if not d.unit_types[u.type].is_infrastructure]  # noqa: E731
    win = estimate_strength_difference(d, s, combat(attackers), combat(defenders))
    by_cost = lambda u: (-d.unit_cost(u.type), u.id)  # noqa: E731 - ProPurchaseUtils cost, reversed
    if win > 50:
        keep = math.ceil(len(attackers) * (min(100.0, win) - 50) / 50)
        return sorted(attackers, key=by_cost)[:keep], []
    keep = math.ceil(len(defenders) * (50 - max(0.0, win)) / 50)
    return [], sorted(defenders, key=by_cost)[:keep]


def _most_likely_battle(d, s, b, rng, events) -> bool:
    """Drop-in for rules._run_battle, as Roxo's ActionUtils.simulateBattles: removes the casualties
    of the most likely outcome, then the attacker takes the territory unless it has no units left
    and the defender has some (so it also takes it when both sides are wiped out). Raids keep the
    kernel's own resolution. Never pauses for a decision."""
    if b.kind != NORMAL_BATTLE:
        return _dice_battle(d, s, b, rng, events)
    p = b.attacker
    b.defenders = [u.id for u in rules.enemy_units(d, s, p, b.site)]
    attackers = [u for u in rules._alive(s, b.attackers) if u.loc == b.site]
    defenders = rules.enemy_units(d, s, p, b.site)
    for u in attackers:  # MarkNoMovementLeft, as after the first round of a real battle
        if not d.unit_types[u.type].is_air:
            rules._exhaust(d, u)
    lost = False
    if defenders:
        if attackers:
            attackers_left, defenders_left = most_likely_outcome(d, s, attackers, defenders)
        else:  # ProOddsCalculator answers an empty result: every defender is removed
            attackers_left, defenders_left = [], []
        survivors = {u.id for u in attackers_left} | {u.id for u in defenders_left}
        for u in attackers + defenders:
            if u.id not in survivors and not (u in defenders and d.unit_types[u.type].is_infrastructure):
                del s.units[u.id]
        lost = not attackers_left and bool(defenders_left)
        s.invalidate()
        rules._prune_battles(s)
    if not lost and rules.is_enemy_territory(d, s, p, b.site):
        rules._take_over(d, s, b.site, p, False, events)
        for u in rules._alive(s, b.attackers):
            if d.unit_types[u.type].is_land and u.loc == b.site:
                u.in_combat = True
    rules._end_battle(d, s, b, "DEFENDER" if lost else "ATTACKER", events)
    return False


def install():
    """Make every battle the kernel fights in this process deterministic."""
    rules._run_battle = _most_likely_battle
