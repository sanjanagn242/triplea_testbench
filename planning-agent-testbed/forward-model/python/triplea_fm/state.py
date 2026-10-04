"""SimulationState (spec §6): the complete hypothetical world, plus cloning and invariants."""

from __future__ import annotations

from typing import Dict, List, Optional, Set, Tuple

from .definition import GameDefinition, NULL_PLAYER

HELD = -2  # unit location: in its owner's hand (purchased, not yet placed)

# Decision kinds (spec §6.3)
PURCHASE_D, MOVE_D, BATTLE_SELECTION_D = "PURCHASE", "MOVE", "BATTLE_SELECTION"
CASUALTIES_D, RETREAT_D, PLACE_D = "CASUALTIES", "RETREAT", "PLACE"

NORMAL_BATTLE, RAID_BATTLE = "NORMAL", "RAID"


class StateError(Exception):
    def __init__(self, code: str, message: str):
        super().__init__(f"{code}: {message}")
        self.code = code


class Unit:
    __slots__ = ("id", "type", "owner", "orig_owner", "loc", "moved", "hits", "in_combat")

    def __init__(self, uid, type_id, owner, orig_owner, loc, moved=0, hits=0, in_combat=False):
        self.id = uid
        self.type = type_id
        self.owner = owner
        self.orig_owner = orig_owner
        self.loc = loc
        self.moved = moved
        self.hits = hits
        self.in_combat = in_combat

    def copy(self) -> "Unit":
        return Unit(self.id, self.type, self.owner, self.orig_owner, self.loc, self.moved, self.hits,
                    self.in_combat)

    def to_json(self):
        return [self.id, self.type, self.owner, self.orig_owner, self.loc, self.moved, self.hits,
                int(self.in_combat)]

    def __repr__(self):
        return f"Unit({self.to_json()})"


class HitBatch:
    """Hits already rolled that the selecting actor must allocate (spec §6.3)."""

    __slots__ = ("id", "selecting_actor", "hits", "eligible", "side", "purpose")

    def __init__(self, bid, selecting_actor, hits, eligible, side, purpose):
        self.id = bid
        self.selecting_actor = selecting_actor
        self.hits = hits
        self.eligible = tuple(eligible)
        self.side = side  # "ATTACKER" (attacking units are hit) or "DEFENDER"
        self.purpose = purpose  # "combat" | "aa"

    def copy(self):
        return HitBatch(self.id, self.selecting_actor, self.hits, self.eligible, self.side, self.purpose)

    def to_json(self):
        return {"id": self.id, "actor": self.selecting_actor, "hits": self.hits,
                "eligible": list(self.eligible), "side": self.side, "purpose": self.purpose}


class Battle:
    __slots__ = ("id", "kind", "site", "attacker", "defender", "attackers", "origins", "defenders",
                 "status", "round", "stage", "pending", "waiting_dead", "attacker_dice",
                 "defender_dice")

    def __init__(self, bid, kind, site, attacker, defender):
        self.id = bid
        self.kind = kind
        self.site = site
        self.attacker = attacker
        self.defender = defender
        self.attackers: List[int] = []  # ordered attacking unit ids
        self.origins: Dict[int, int] = {}  # unit id -> territory it attacked from
        self.defenders: List[int] = []  # fixed when the battle starts
        self.status = "PENDING"  # PENDING | ACTIVE
        self.round = 0
        self.stage = "START"
        self.pending: Optional[HitBatch] = None
        self.waiting_dead: List[int] = []  # defenders hit this round who still return fire
        self.attacker_dice: Tuple[int, ...] = ()
        self.defender_dice: Tuple[int, ...] = ()

    def copy(self):
        b = Battle(self.id, self.kind, self.site, self.attacker, self.defender)
        b.attackers = list(self.attackers)
        b.origins = dict(self.origins)
        b.defenders = list(self.defenders)
        b.status = self.status
        b.round = self.round
        b.stage = self.stage
        b.pending = self.pending.copy() if self.pending else None
        b.waiting_dead = list(self.waiting_dead)
        b.attacker_dice = self.attacker_dice
        b.defender_dice = self.defender_dice
        return b

    def to_json(self):
        return {"id": self.id, "kind": self.kind, "site": self.site, "attacker": self.attacker,
                "defender": self.defender, "attackers": list(self.attackers),
                "origins": {str(k): v for k, v in self.origins.items()}, "defenders": list(self.defenders),
                "status": self.status, "round": self.round, "stage": self.stage,
                "pending": self.pending.to_json() if self.pending else None,
                "waitingDead": list(self.waiting_dead)}


class Decision:
    __slots__ = ("id", "actor", "kind", "step_id", "battle_id", "hit_batch")

    def __init__(self, did, actor, kind, step_id, battle_id=None, hit_batch=None):
        self.id = did
        self.actor = actor
        self.kind = kind
        self.step_id = step_id
        self.battle_id = battle_id
        self.hit_batch = hit_batch

    def to_json(self):
        return {"id": self.id, "actor": self.actor, "kind": self.kind, "stepId": self.step_id,
                "battleId": self.battle_id,
                "hitBatch": self.hit_batch.to_json() if self.hit_batch else None}

    def __repr__(self):
        return f"Decision({self.to_json()})"


class TerminalOutcome:
    __slots__ = ("rule_id", "round", "winners")

    def __init__(self, rule_id, round_, winners):
        self.rule_id = rule_id
        self.round = round_
        self.winners = tuple(winners)  # alliance ids

    def to_json(self):
        return {"ruleId": self.rule_id, "round": self.round, "winners": list(self.winners)}


class SimulationState:
    __slots__ = ("definition_digest", "game_number", "round", "step", "step_runs", "entered", "phase_pc", "decision", "owner",
                 "orig_owner", "units", "next_unit_id", "balances", "conquered", "blitzed", "fought",
                 "raid_loss", "placement_cap", "placement_used", "battles", "active_battle",
                 "terminal", "next_battle_id", "next_decision_id", "next_batch_id", "_terr_units")

    def __init__(self):
        self.definition_digest = ""
        self.game_number = 1  # testbench batch number (gameNumber), starting at 1
        self.round = 1
        self.step = 0
        self.step_runs: List[int] = []
        self.entered = False  # lifecycle: entry effects of the current step committed
        self.phase_pc = 0  # progress through the current step's automatic program
        self.decision: Optional[Decision] = None
        self.owner: List[int] = []
        self.orig_owner: List[int] = []
        self.units: Dict[int, Unit] = {}
        self.next_unit_id = 1
        self.balances: List[List[int]] = []
        self.conquered: Set[int] = set()
        self.blitzed: Set[int] = set()
        self.fought: Set[int] = set()
        self.raid_loss: Dict[int, int] = {}
        self.placement_cap: Optional[Dict[int, int]] = None  # producer territory -> cap (-1 unlimited)
        self.placement_used: Dict[int, int] = {}
        self.battles: Dict[int, Battle] = {}
        self.active_battle: Optional[int] = None
        self.terminal: Optional[TerminalOutcome] = None
        self.next_battle_id = 1
        self.next_decision_id = 1
        self.next_batch_id = 1
        self._terr_units: Optional[List[List[int]]] = None

    # ---- copies ----
    def clone(self) -> "SimulationState":
        s = SimulationState.__new__(SimulationState)
        s.definition_digest = self.definition_digest
        s.game_number = self.game_number
        s.round = self.round
        s.step = self.step
        s.step_runs = list(self.step_runs)
        s.entered = self.entered
        s.phase_pc = self.phase_pc
        d = self.decision
        s.decision = None if d is None else Decision(
            d.id, d.actor, d.kind, d.step_id, d.battle_id, d.hit_batch.copy() if d.hit_batch else None)
        s.owner = list(self.owner)
        s.orig_owner = list(self.orig_owner)
        s.units = {k: u.copy() for k, u in self.units.items()}
        s.next_unit_id = self.next_unit_id
        s.balances = [list(b) for b in self.balances]
        s.conquered = set(self.conquered)
        s.blitzed = set(self.blitzed)
        s.fought = set(self.fought)
        s.raid_loss = dict(self.raid_loss)
        s.placement_cap = None if self.placement_cap is None else dict(self.placement_cap)
        s.placement_used = dict(self.placement_used)
        s.battles = {k: b.copy() for k, b in self.battles.items()}
        s.active_battle = self.active_battle
        s.terminal = self.terminal
        s.next_battle_id = self.next_battle_id
        s.next_decision_id = self.next_decision_id
        s.next_batch_id = self.next_batch_id
        s._terr_units = None
        return s

    # ---- derived queries (caches, never authority) ----
    def units_in(self, terr: int) -> List[Unit]:
        if self._terr_units is None:
            idx: List[List[int]] = [[] for _ in self.owner]
            for uid in sorted(self.units):
                loc = self.units[uid].loc
                if loc >= 0:
                    idx[loc].append(uid)
            self._terr_units = idx
        return [self.units[u] for u in self._terr_units[terr]]

    def held_units(self, player: int) -> List[Unit]:
        return [u for uid, u in sorted(self.units.items()) if u.loc == HELD and u.owner == player]

    def invalidate(self):
        self._terr_units = None

    def to_json(self) -> Dict:
        """Portable canonical snapshot of all essential fields (spec §6.4)."""
        return {
            "schemaVersion": 1, "definitionDigest": self.definition_digest,
            "gameNumber": self.game_number, "round": self.round,
            "step": self.step, "stepRuns": list(self.step_runs), "entered": self.entered, "phasePc": self.phase_pc,
            "decision": self.decision.to_json() if self.decision else None,
            "owner": list(self.owner), "origOwner": list(self.orig_owner),
            "units": [self.units[k].to_json() for k in sorted(self.units)],
            "nextUnitId": self.next_unit_id, "balances": [list(b) for b in self.balances],
            "conquered": sorted(self.conquered), "blitzed": sorted(self.blitzed),
            "fought": sorted(self.fought), "raidLoss": {str(k): v for k, v in sorted(self.raid_loss.items())},
            "placementCap": None if self.placement_cap is None else
            {str(k): v for k, v in sorted(self.placement_cap.items())},
            "placementUsed": {str(k): v for k, v in sorted(self.placement_used.items())},
            "battles": [self.battles[k].to_json() for k in sorted(self.battles)],
            "activeBattle": self.active_battle,
            "terminal": self.terminal.to_json() if self.terminal else None,
            "nextBattleId": self.next_battle_id, "nextDecisionId": self.next_decision_id,
            "nextBatchId": self.next_batch_id,
        }


def state_from_json(definition: GameDefinition, data: Dict) -> SimulationState:
    if data.get("schemaVersion") != 1:
        raise StateError("SCHEMA_MISMATCH", f"schema {data.get('schemaVersion')}")
    if data.get("definitionDigest") != definition.definition_digest:
        raise StateError("DEFINITION_MISMATCH", "snapshot belongs to another definition")
    s = SimulationState()
    s.definition_digest = data["definitionDigest"]
    s.game_number = data.get("gameNumber", 1)  # absent in older snapshots
    s.round = data["round"]
    s.step = data["step"]
    s.step_runs = list(data["stepRuns"])
    s.entered = data["entered"]
    s.phase_pc = data["phasePc"]
    s.owner = list(data["owner"])
    s.orig_owner = list(data["origOwner"])
    for u in data["units"]:
        s.units[u[0]] = Unit(u[0], u[1], u[2], u[3], u[4], u[5], u[6], bool(u[7]))
    s.next_unit_id = data["nextUnitId"]
    s.balances = [list(b) for b in data["balances"]]
    s.conquered = set(data["conquered"])
    s.blitzed = set(data["blitzed"])
    s.fought = set(data["fought"])
    s.raid_loss = {int(k): v for k, v in data["raidLoss"].items()}
    pc = data["placementCap"]
    s.placement_cap = None if pc is None else {int(k): v for k, v in pc.items()}
    s.placement_used = {int(k): v for k, v in data["placementUsed"].items()}
    for bj in data["battles"]:
        b = Battle(bj["id"], bj["kind"], bj["site"], bj["attacker"], bj["defender"])
        b.attackers = list(bj["attackers"])
        b.origins = {int(k): v for k, v in bj["origins"].items()}
        b.defenders = list(bj["defenders"])
        b.status, b.round, b.stage = bj["status"], bj["round"], bj["stage"]
        p = bj["pending"]
        b.pending = None if p is None else HitBatch(p["id"], p["actor"], p["hits"], p["eligible"],
                                                    p["side"], p["purpose"])
        b.waiting_dead = list(bj["waitingDead"])
        s.battles[b.id] = b
    s.active_battle = data["activeBattle"]
    t = data["terminal"]
    s.terminal = None if t is None else TerminalOutcome(t["ruleId"], t["round"], t["winners"])
    s.next_battle_id = data["nextBattleId"]
    s.next_decision_id = data["nextDecisionId"]
    s.next_batch_id = data["nextBatchId"]
    d = data["decision"]
    if d is not None:
        hb = None
        if d["hitBatch"] is not None:
            # The decision shares the battle's pending batch.
            hb = s.battles[d["battleId"]].pending
        s.decision = Decision(d["id"], d["actor"], d["kind"], d["stepId"], d["battleId"], hb)
    check_invariants(definition, s)
    return s


def check_invariants(definition: GameDefinition, s: SimulationState) -> None:
    """Structural/semantic import checks (spec §6.4). Raises StateError."""
    nt, np_ = len(definition.territory_names), len(definition.player_names)
    if len(s.owner) != nt or len(s.orig_owner) != nt:
        raise StateError("INVALID_SIMULATION_STATE", "territory arrays have wrong length")
    if len(s.balances) != np_ or any(len(b) != len(definition.resource_names) for b in s.balances):
        raise StateError("INVALID_SIMULATION_STATE", "balances have wrong shape")
    if any(v < 0 for b in s.balances for v in b):
        raise StateError("INVALID_SIMULATION_STATE", "negative balance")
    if len(s.step_runs) != len(definition.steps) or not 0 <= s.step < len(definition.steps):
        raise StateError("INVALID_SIMULATION_STATE", "bad phase cursor")
    for o in list(s.owner) + list(s.orig_owner):
        if o != NULL_PLAYER and not 0 <= o < np_:
            raise StateError("INVALID_SIMULATION_STATE", f"bad owner {o}")
    for uid, u in s.units.items():
        if uid != u.id or uid >= s.next_unit_id:
            raise StateError("INVALID_SIMULATION_STATE", f"bad unit identity {uid}")
        if not 0 <= u.type < len(definition.unit_types):
            raise StateError("INVALID_SIMULATION_STATE", f"unit {uid} has unknown type")
        ut = definition.unit_types[u.type]
        if u.loc == HELD:
            if u.owner == NULL_PLAYER:
                raise StateError("INVALID_SIMULATION_STATE", f"held unit {uid} without owner")
        elif not 0 <= u.loc < nt:
            raise StateError("INVALID_SIMULATION_STATE", f"unit {uid} has bad location")
        elif definition.is_water[u.loc] and not ut.is_air:
            raise StateError("UNSUPPORTED_BY_PROFILE_V1", f"unit {uid} at sea requires Forward Model V2")
        if ut.domain == 2:
            raise StateError("UNSUPPORTED_BY_PROFILE_V1", f"sea unit {uid} requires Forward Model V2")
        if not 0 <= u.hits < ut.hit_points or u.moved < 0:
            raise StateError("INVALID_SIMULATION_STATE", f"unit {uid} has bad hits/movement")
    for b in s.battles.values():
        for uid in b.attackers + b.defenders:
            if uid not in s.units:
                raise StateError("INVALID_SIMULATION_STATE", f"battle {b.id} references missing unit")
        if b.kind == NORMAL_BATTLE and b.attackers and any(uid not in b.origins for uid in b.attackers
                                                           if not definition.unit_types[s.units[uid].type].is_air):
            raise StateError("INCOMPLETE_SIMULATION_STATE", f"battle {b.id} lacks attack origins")
    if s.terminal is None and s.decision is None:
        raise StateError("INCOMPLETE_SIMULATION_STATE", "nonterminal snapshot without a decision")
