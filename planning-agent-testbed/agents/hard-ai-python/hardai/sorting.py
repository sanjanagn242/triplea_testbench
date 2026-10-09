"""Unit ordering for move assignment (ProSortMoveOptionsUtils)."""

from __future__ import annotations

from typing import Dict

from .data import OrderedSet, ProData


def sort_unit_move_options(pd: ProData, options: Dict[object, OrderedSet]) -> Dict[object, OrderedSet]:
    """By number of move options, then unit value, then unit type name."""
    d = pd.d
    items = sorted(options.items(), key=lambda kv: (len(kv[1]), pd.value_of(kv[0]), d.unit_types[kv[0].type].name))
    return dict(items)


def _not_winning(pd: ProData, player, territories, attack_map, calc):
    out = []
    for t in territories:
        patd = attack_map[t]
        if patd.battle_result is None:
            patd.estimate_battle_result(calc, player)
        if not patd.currently_wins:
            out.append(t)
    return out


def sort_unit_needed_options(pd: ProData, player, options, attack_map, calc) -> Dict[object, OrderedSet]:
    """By number of territories that still need units, then unit value, then type name."""
    d = pd.d
    keys = {u: (len(_not_winning(pd, player, ts, attack_map, calc)), pd.value_of(u), d.unit_types[u.type].name)
            for u, ts in options.items()}
    return dict(sorted(options.items(), key=lambda kv: keys[kv[0]]))


def sort_unit_needed_options_then_attack(pd: ProData, player, options, attack_map, calc) -> Dict[object, OrderedSet]:
    """By number of territories that still need units, then attack efficiency (best first), then
    type name, then (air of one type) total distance to those territories."""
    d = pd.d
    keys = {}
    for u, ts in options.items():
        needed = _not_winning(pd, player, ts, attack_map, calc)
        if not needed:
            keys[u] = (0,)
            continue
        ut = d.unit_types[u.type]
        distance = 0
        if ut.is_air:
            fly = lambda x: pd.can_move_air_no_aa(player, x)  # noqa: E731
            distance = sum(pd.distance(u.loc, t, fly, ignore_end=True) for t in needed)
        keys[u] = (len(needed), -_attack_efficiency(pd, player, attack_map, needed, u), ut.name, distance)
    return dict(sorted(options.items(), key=lambda kv: keys[kv[0]]))


def _attack_efficiency(pd: ProData, player, attack_map, territories, unit) -> float:
    from .odds import estimate_power
    min_power = None
    for t in territories:
        defenders = pd.enemy_units(player, t)
        attackers = list(attack_map[t].units)
        without = estimate_power(pd, t, attackers, defenders, True)
        with_unit = estimate_power(pd, t, attackers + [unit], defenders, True)
        diff = (with_unit - without) * pd.d.dice_sides / 6.0  # raw power, as PowerStrengthAndRolls
        min_power = diff if min_power is None or diff < min_power else min_power
    if pd.ut(unit).is_air:
        min_power *= 10
    value = pd.value_of(unit)
    return 0.0 if value == 0 else min_power / value
