"""Battle-time decisions (AbstractProAi.selectCasualties, ProRetreatAi) and the battle phase of the
turn simulation (ProSimulateTurnUtils.simulateBattles), played with the forward-model kernel.

In protocol v1 the engine still decides this agent's casualties and retreats in live games, so
these policies act inside the agent's own simulations.
"""

from __future__ import annotations

from typing import List, Tuple

import triplea_fm as fm
from triplea_fm import rules, toolkit
from triplea_fm.state import BATTLE_SELECTION_D, CASUALTIES_D, RETREAT_D

from .data import ProData
from .odds import estimate_strength, estimate_strength_difference

BATTLE_DECISIONS = (BATTLE_SELECTION_D, CASUALTIES_D, RETREAT_D)


def select_casualties(ai, d, s) -> fm.SelectCasualtiesAction:
    """Default order of loss, then keep expensive units alive (bubble sort on cost) unless a
    defender is losing anyway."""
    hb = s.decision.hit_batch
    b = s.battles[s.decision.battle_id]
    pd = ProData(d, s, s.decision.actor)
    order = toolkit.default_casualty_order(d, s)
    need = min(hb.hits, len(order))
    attacking = hb.side == "ATTACKER"
    need_to_check = True
    if not attacking:
        attackers = [s.units[u] for u in b.attackers if u in s.units]
        default_killed = set(order[:need])
        defenders = [u for u in rules.enemy_units(d, s, b.attacker, b.site) if u not in default_killed]
        if estimate_strength_difference(pd, b.site, attackers, defenders) > 55:
            need_to_check = False
    cost = lambda u: pd.value_of(u)  # noqa: E731
    while need_to_check:
        need_to_check = False
        for i in range(len(order) - 1):
            if cost(order[i]) > 1.5 * cost(order[i + 1]):
                order[i], order[i + 1] = order[i + 1], order[i]
                need_to_check = True
    return fm.SelectCasualtiesAction(hb.id, tuple(u.id for u in order[:need]))


def retreat(ai, d, s) -> fm.RetreatAction:
    """AbstractProAi.retreatQuery + ProRetreatAi.retreatQuery for the attacker."""
    b = s.battles[s.decision.battle_id]
    player = s.decision.actor
    pd = ProData(d, s, player)
    site = b.site
    attackers = [s.units[u] for u in b.attackers if u in s.units]
    defenders = rules.enemy_units(d, s, player, site)
    destinations = rules.retreat_destinations(d, s, b)
    if not destinations:
        return fm.RetreatAction(fm.CONTINUE)
    diff = estimate_strength_difference(pd, site, attackers, defenders)
    strafing = site in ai.stored_strafing
    if (strafing or diff > 50) and any(pd.ut(u).is_land for u in attackers):
        return fm.RetreatAction(fm.CONTINUE)
    result = ai.calc.calculate(pd, site, attackers, defenders)
    is_factory = 1 if pd.has_infra_factory(site) else 0
    production, is_capital = d.production[site], 1 if d.capital_of[site] != -1 else 0
    territory_value = 0.0
    if result.has_land_unit_remaining or not any(pd.ut(u).is_air for u in attackers):
        territory_value = result.win_percentage / 100 * (2.0 * production * (1 + is_factory) * (1 + is_capital))
    if result.tuv_swing + territory_value >= 0:
        return fm.RetreatAction(fm.CONTINUE)
    capital = pd.my_capital
    best, best_strength = None, float("-inf")
    for t in destinations:
        if t == capital:
            best = t
            break
        strength = estimate_strength(pd, t, pd.allied_units(player, t), [], False)
        if strength > best_strength:
            best, best_strength = t, strength
    ai.log(f"  retreating from {d.territory_names[site]} to {d.territory_names[best]}")
    return fm.RetreatAction(fm.RETREAT, best)


def play_battle_decision(ai, d, s):
    """The action for a battle decision in a simulation: this player uses its own policies,
    everybody else the library defaults (as the engine's calculator does)."""
    dec = s.decision
    if dec.kind == BATTLE_SELECTION_D:
        return fm.SelectBattleAction(min(b.id for b in s.battles.values()
                                         if b.attacker == dec.actor and b.status == "PENDING"))
    if dec.kind == CASUALTIES_D:
        return select_casualties(ai, d, s) if dec.actor == ai.player else toolkit.default_casualties(d, s)
    if dec.actor == ai.player:
        return retreat(ai, d, s)
    return fm.RetreatAction(fm.CONTINUE)


def simulate_battles(ai, d, s, seed: int, samples: int) -> Tuple[object, object]:
    """Fight this turn's battles with the kernel `samples` times, from the end of the combat move
    (a MOVE decision in `s`), and return the (state, rng) of the run closest to the average outcome
    (AggregateResults.getBattleResultsClosestToAverage applied to the whole battle phase)."""
    p = ai.player
    runs = []
    for k in range(samples):
        w = s.clone()
        rng = fm.SeededRng(seed, stream=1000 + k)
        if fm.apply_in_place(d, w, fm.EndPhaseAction(), rng) == fm.REJECTED:
            return s, rng
        while w.decision is not None and w.decision.kind in BATTLE_DECISIONS:
            if fm.apply_in_place(d, w, play_battle_decision(ai, d, w), rng) == fm.REJECTED:
                break
        runs.append((w, rng, _features(d, w, p)))
    n = len(runs)
    mean = [sum(r[2][i] for r in runs) / n for i in range(3)]
    best = min(runs, key=lambda r: sum(abs(r[2][i] - mean[i]) for i in range(3)))
    return best[0], best[1]


def _features(d, s, p) -> List[float]:
    mine = sum(1 for u in s.units.values() if u.owner == p and u.loc >= 0
               and not d.unit_types[u.type].is_infrastructure)
    enemy = sum(1 for u in s.units.values() if u.loc >= 0 and d.is_at_war(p, u.owner)
                and not d.unit_types[u.type].is_infrastructure)
    owned = sum(1 for o in s.owner if o == p)
    return [mine, enemy, owned]
