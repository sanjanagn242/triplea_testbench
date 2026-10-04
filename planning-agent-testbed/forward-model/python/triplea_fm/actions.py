"""Typed agent-supplied Actions (spec §7). The model never generates these."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple

NORMAL, RAID = "NORMAL", "RAID"
CONTINUE, RETREAT = "CONTINUE", "RETREAT"


@dataclass(frozen=True)
class PurchaseAction:
    quantities: Tuple[Tuple[int, int], ...]  # (production rule id, count)
    kind = "purchase"


@dataclass(frozen=True)
class MoveAction:
    units: Tuple[int, ...]  # UnitIds
    route: Tuple[int, ...]  # territory ids, including the start territory
    intent: str = NORMAL  # NORMAL | RAID (strategic bombing; bombers only)
    kind = "move"


@dataclass(frozen=True)
class SelectBattleAction:
    battle_id: int
    kind = "select_battle"


@dataclass(frozen=True)
class SelectCasualtiesAction:
    hit_batch_id: int
    killed: Tuple[int, ...]
    kind = "casualties"


@dataclass(frozen=True)
class RetreatAction:
    choice: str  # CONTINUE | RETREAT
    destination: Optional[int] = None
    kind = "retreat"


@dataclass(frozen=True)
class PlaceAction:
    units: Tuple[int, ...]  # held UnitIds
    territory: int
    kind = "place"


@dataclass(frozen=True)
class EndPhaseAction:
    kind = "end_phase"


Action = object  # union of the dataclasses above


def action_to_json(definition, action) -> Dict:
    """Canonical JSON encoding using names (stable across processes)."""
    tn = definition.territory_names
    k = action.kind
    if k == "purchase":
        return {"kind": k, "quantities": {definition.production_rules[r].name: c for r, c in action.quantities}}
    if k == "move":
        return {"kind": k, "units": list(action.units), "route": [tn[t] for t in action.route],
                "intent": action.intent}
    if k == "select_battle":
        return {"kind": k, "battleId": action.battle_id}
    if k == "casualties":
        return {"kind": k, "hitBatchId": action.hit_batch_id, "killed": list(action.killed)}
    if k == "retreat":
        return {"kind": k, "choice": action.choice,
                "destination": None if action.destination is None else tn[action.destination]}
    if k == "place":
        return {"kind": k, "units": list(action.units), "territory": tn[action.territory]}
    return {"kind": "end_phase"}


def action_from_json(definition, data: Dict):
    ti = definition.territory_index
    k = data["kind"]
    if k == "purchase":
        return PurchaseAction(tuple(sorted((definition.production_rule_index[n], int(c))
                                           for n, c in data["quantities"].items())))
    if k == "move":
        return MoveAction(tuple(int(u) for u in data["units"]), tuple(ti[t] for t in data["route"]),
                          data.get("intent", NORMAL))
    if k == "select_battle":
        return SelectBattleAction(int(data["battleId"]))
    if k == "casualties":
        return SelectCasualtiesAction(int(data["hitBatchId"]), tuple(int(u) for u in data["killed"]))
    if k == "retreat":
        d = data.get("destination")
        return RetreatAction(data["choice"], None if d is None else ti[d])
    if k == "place":
        return PlaceAction(tuple(int(u) for u in data["units"]), ti[data["territory"]])
    if k == "end_phase":
        return EndPhaseAction()
    raise ValueError(f"unknown action kind {k!r}")
