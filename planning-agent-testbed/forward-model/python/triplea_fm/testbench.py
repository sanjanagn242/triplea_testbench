"""Agent-side bridge between the live testbench protocol (v1) and the forward model.

This is belief construction, not kernel behavior (spec §2, §23.1): the live StateObservation lacks
some history, so this module reconstructs a complete SimulationState from the observation plus
memory kept across requests, and states every assumption it makes:

  * conquered-this-turn = territories the player owns now but did not own at its first request
    of the turn. All of them are treated as conquered-not-blitzed (blitzed set left empty), which
    is conservative for later combat moves and exact for noncombat moves and placement.
  * fought-this-turn is empty; raid losses are empty (they only matter inside the battle phase).
  * unit original owner = first owner the bridge ever saw for that unit; factories present at game
    start take the XML setup owner.
  * wasInCombat is false for every unit.

When the Java adapter gains the §23.1 exporter these assumptions disappear.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from . import rules
from .actions import NORMAL
from .definition import (COMBAT_MOVE, NONCOMBAT_MOVE, NULL_PLAYER, PLACE, PURCHASE, GameDefinition)
from .state import HELD, MOVE_D, PLACE_D, PURCHASE_D, Decision, SimulationState, Unit

PHASE_KINDS = {"purchase": PURCHASE, "combatMove": COMBAT_MOVE, "nonCombatMove": NONCOMBAT_MOVE,
               "place": PLACE}
_DECISION_KIND = {PURCHASE: PURCHASE_D, COMBAT_MOVE: MOVE_D, NONCOMBAT_MOVE: MOVE_D, PLACE: PLACE_D}


def _movement(text) -> int:
    try:
        return int(float(text))
    except (TypeError, ValueError):
        return 0


class LiveBridge:
    def __init__(self, definition: GameDefinition, player_name: str, game_number: int = 1):
        self.d = definition
        self.game_number = game_number
        self.player = definition.player_index[player_name]
        self.uid_of: Dict[str, int] = {}
        self.uuid_of: Dict[int, str] = {}
        self.first_owner: Dict[str, int] = {}
        self.turn_key: Optional[Tuple[int, str]] = None
        self.turn_start_owner: Optional[List[int]] = None
        self._setup_factory_owner = {}
        for type_id, terr, owner, _qty in definition.initial_units:
            if definition.unit_types[type_id].is_factory:
                self._setup_factory_owner[terr] = owner

    def _owner(self, name) -> int:
        if name is None:
            return NULL_PLAYER
        return self.d.player_index.get(name, NULL_PLAYER)

    def _uid(self, uuid: str) -> int:
        uid = self.uid_of.get(uuid)
        if uid is None:
            uid = len(self.uid_of) + 1
            self.uid_of[uuid] = uid
            self.uuid_of[uid] = uuid
        return uid

    def build_state(self, obs: dict, phase: str) -> SimulationState:
        """SimulationState at this player's `phase` decision, from a v1 observation."""
        d = self.d
        kind = PHASE_KINDS[phase]
        s = SimulationState()
        s.definition_digest = d.definition_digest
        s.game_number = self.game_number
        s.round = int(obs["round"])
        owners = [NULL_PLAYER] * len(d.territory_names)
        for t in obs["territories"]:
            owners[d.territory_index[t["name"]]] = self._owner(t.get("owner"))
        key = (s.round, obs.get("activePlayer"))
        if key != self.turn_key:
            self.turn_key = key
            self.turn_start_owner = list(owners)
        s.owner = owners
        s.orig_owner = list(d.initial_owner)
        s.balances = [[0] * len(d.resource_names) for _ in d.player_names]
        units: List[Tuple[dict, int]] = []
        for t in obs["territories"]:
            ti = d.territory_index[t["name"]]
            for u in t.get("units", []):
                units.append((u, ti))
        for p in obs["players"]:
            pi = d.player_index.get(p["name"])
            if pi is None:
                continue
            for r, amount in p.get("resources", {}).items():
                if r in d.resource_names:
                    s.balances[pi][d.resource_names.index(r)] = int(amount)
            for u in p.get("unitsInReserve", []):
                units.append((u, HELD))
        for u, loc in sorted(units, key=lambda x: x[0]["id"]):
            uid = self._uid(u["id"])
            type_id = d.unit_type_index[u["type"]]
            owner = self._owner(u.get("owner"))
            if loc >= 0 and d.unit_types[type_id].is_factory and loc in self._setup_factory_owner \
                    and u["id"] not in self.first_owner:
                self.first_owner[u["id"]] = self._setup_factory_owner[loc]
            orig = self.first_owner.setdefault(u["id"], owner)
            s.units[uid] = Unit(uid, type_id, owner, orig, loc, _movement(u.get("movementUsed")),
                                int(u.get("hits", 0)))
        s.next_unit_id = (max(s.units) + 1) if s.units else 1
        s.next_unit_id = max(s.next_unit_id, len(self.uid_of) + 1)
        if self.turn_start_owner is not None:
            s.conquered = {t for t, o in enumerate(owners)
                           if o == self.player and self.turn_start_owner[t] != self.player}
        step_id = next(st.id for st in d.steps if st.kind == kind and st.actor == self.player)
        s.step = step_id
        s.step_runs = [0] * len(d.steps)
        for st in d.steps:
            mr = rules._effective_max_runs(d, st)
            s.step_runs[st.id] = mr if mr is not None else s.round
        s.entered = True
        s.phase_pc = rules.PC_AWAIT
        if kind == PLACE:
            s.placement_cap = rules.placement_snapshot(d, s, self.player)
            s.placement_used = {}
        s.decision = Decision(1, self.player, _DECISION_KIND[kind], step_id)
        s.next_decision_id = 2
        return s

    def legal_actions(self, obs: dict, phase: str, distinct_units: bool = False):
        """(state, iterator over every legal action) for this player's `phase` decision as observed
        in a v1 observation; see actionspace.legal_actions. Actions refer to the returned state's
        unit ids, which the *_to_json methods translate back to the engine's."""
        from .actionspace import legal_actions
        s = self.build_state(obs, phase)
        return s, legal_actions(self.d, s, distinct_units)

    # ---- action -> v1 protocol JSON ----
    def move_to_json(self, action) -> dict:
        tn = self.d.territory_names
        out = {"from": tn[action.route[0]], "to": tn[action.route[-1]],
               "unitIds": [self.uuid_of[u] for u in action.units],
               "route": [tn[t] for t in action.route]}
        out["intent"] = action.intent if action.intent else NORMAL
        return out

    def purchase_to_json(self, action) -> dict:
        return {"purchases": [{"rule": self.d.production_rules[r].name, "count": c}
                              for r, c in action.quantities if c > 0]}

    def place_to_json(self, action) -> dict:
        return {"territory": self.d.territory_names[action.territory],
                "unitIds": [self.uuid_of[u] for u in action.units]}


def observation_from_state(d: GameDefinition, s: SimulationState, observer: str) -> dict:
    """A v1 StateObservation-shaped dict built from a model state (for offline arenas/tests).

    Mirrors FullStateObservationProvider's field names; unit ids are the model ids as strings.
    """
    def pname(o):
        return None if o == NULL_PLAYER else d.player_names[o]

    def unit_json(u):
        ut = d.unit_types[u.type]
        return {"id": str(u.id), "type": ut.name, "owner": pname(u.owner) or "Neutral",
                "hits": u.hits, "damage": 0, "movementUsed": str(u.moved),
                "movementLeft": str(rules.movement_left(d, u)), "movement": ut.movement,
                "attack": ut.attack, "defense": ut.defense, "air": ut.is_air, "sea": ut.domain == 2,
                "infrastructure": ut.is_infrastructure}

    step = d.steps[s.step]
    players = []
    for pi, name in sorted(enumerate(d.player_names), key=lambda x: x[1]):
        players.append({"name": name,
                        "resources": {r: s.balances[pi][ri] for ri, r in enumerate(d.resource_names)},
                        "technologies": [], "unitsInReserve": [unit_json(u) for u in s.held_units(pi)]})
    territories = []
    for ti, name in sorted(enumerate(d.territory_names), key=lambda x: x[1]):
        cap = d.capital_of[ti]
        territories.append({
            "name": name, "water": d.is_water[ti], "owner": pname(s.owner[ti]),
            "neighbors": sorted(d.territory_names[n] for n in d.neighbors[ti]),
            "production": d.production[ti], "unitProduction": d.production[ti],
            "capitalFor": None if cap == NULL_PLAYER else d.player_names[cap],
            "victoryCityValue": d.victory_value[ti], "impassable": d.impassable[ti],
            "units": [unit_json(u) for u in sorted(s.units_in(ti), key=lambda u: (d.unit_types[u.type].name, u.id))]})
    return {"observationMode": "FULL", "observationPlayer": observer, "gameName": d.game_name,
            "mapName": d.game_name, "round": s.round, "step": step.name,
            "activePlayer": pname(step.actor), "diceSides": d.dice_sides,
            "gameProperties": {k: str(v) for k, v in d.properties.items()},
            "players": players, "territories": territories}
