"""Abstract, structured form of a testbench observation, and the key that identifies a state.

The raw observation lists every unit with a random ID and repeats the static map and unit-type data
(neighbours, production, attack, defense, ...). The abstraction keeps only what changes during a
game, with the units counted per owner and type:

  {"round": 3, "step": "Combat Move", "active": "Russians",
   "players": {"Germans": {"PUs": 12}, "Russians": {"PUs": 9, "techs": ["..."], "reserve": {"infantry": 2}}},
   "territories": {"Flag": {"owner": null, "units": {"Neutral": {"infantry": 3}}},
                   "RussianStepTwo": {"owner": "Russians", "units": {"Russians": {"infantry": 2, "tank": 1}}}}}

A unit that has taken hits or damage is counted under "<type>@<hits>h<damage>d" (e.g. "battleship@1h0d"),
so damaged and undamaged units stay apart. Movement is left out: the states are recorded at the start
of a phase, before any unit has moved in it. Territories without an owner or units are left out;
"techs" and "reserve" only appear when not empty.

The abstraction is one pass over the territories (well under a millisecond for a small map) and
state_key hashes its canonical JSON.
"""

import hashlib
import json


def abstract(state):
    territories = {}
    for t in state.get("territories", ()):
        units = {}
        for u in t.get("units", ()):
            kind = u.get("type")
            if u.get("hits") or u.get("damage"):
                kind = f"{kind}@{u.get('hits') or 0}h{u.get('damage') or 0}d"
            by_type = units.setdefault(u.get("owner"), {})
            by_type[kind] = by_type.get(kind, 0) + 1
        owner = t.get("owner")
        if owner is not None or units:
            territories[t["name"]] = {"owner": owner, "units": units}
    players = {}
    for p in state.get("players", ()):
        entry = dict(p.get("resources") or {})
        if p.get("technologies"):
            entry["techs"] = sorted(map(str, p["technologies"]))
        reserve = {}
        for u in p.get("unitsInReserve", ()):
            kind = u.get("type") if isinstance(u, dict) else str(u)
            reserve[kind] = reserve.get(kind, 0) + 1
        if reserve:
            entry["reserve"] = reserve
        players[p["name"]] = entry
    return {
        "round": state.get("round"),
        "step": state.get("step"),
        "active": state.get("activePlayer"),
        "players": players,
        "territories": territories,
    }


def state_key(player, abstract_state, game_name=None):
    """Hash of (faction, game, abstract state): equal exactly when the abstract states are equal."""
    canonical = json.dumps([player, game_name, abstract_state], sort_keys=True, separators=(",", ":"))
    return hashlib.sha1(canonical.encode()).hexdigest()
