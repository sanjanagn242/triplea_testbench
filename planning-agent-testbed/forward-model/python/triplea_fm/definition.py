"""Scenario normalization: TripleA XML -> immutable GameDefinition (spec §4-§5, §15).

This is a Python normalizer for the V1 land/air subset. The spec recommends reusing TripleA's
Java GameParser offline; this module instead reimplements the parser defaults that the admitted
V1 configurations need and refuses everything else explicitly (see SUPPORT_NOTES in README).
"""

from __future__ import annotations

import hashlib
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

SCHEMA_VERSION = 1
PROFILE_ID = "forward-model-v1"
PROFILE_VERSION = "1.0.0-python"
NORMALIZER_VERSION = "triplea_fm-0.2"

NULL_PLAYER = -1

LAND, AIR, SEA = 0, 1, 2

# Step kinds (registered phase programs, spec §9.1).
INIT, BID, BID_PLACE, TECH, TECH_ACTIVATION = "INIT", "BID", "BID_PLACE", "TECH", "TECH_ACTIVATION"
PURCHASE, COMBAT_MOVE, BATTLE, NONCOMBAT_MOVE = "PURCHASE", "COMBAT_MOVE", "BATTLE", "NONCOMBAT_MOVE"
PLACE, END_TURN, END_ROUND = "PLACE", "END_TURN", "END_ROUND"

_DELEGATE_KINDS = {
    "games.strategy.triplea.delegate.InitializationDelegate": INIT,
    "games.strategy.triplea.delegate.BidPurchaseDelegate": BID,
    "games.strategy.triplea.delegate.BidPlaceDelegate": BID_PLACE,
    "games.strategy.triplea.delegate.TechnologyDelegate": TECH,
    "games.strategy.triplea.delegate.TechActivationDelegate": TECH_ACTIVATION,
    "games.strategy.triplea.delegate.PurchaseDelegate": PURCHASE,
    "games.strategy.triplea.delegate.MoveDelegate": "MOVE",
    "games.strategy.triplea.delegate.BattleDelegate": BATTLE,
    "games.strategy.triplea.delegate.PlaceDelegate": PLACE,
    "games.strategy.triplea.delegate.EndTurnDelegate": END_TURN,
    "games.strategy.triplea.delegate.EndRoundDelegate": END_ROUND,
}

# Properties the V1 profile understands. Value: (kind, admitted_check or None).
# kind: "display" (no gameplay meaning), "inert" (verified inert legacy key), "gameplay".
# admitted_check returns None if the effective value is admitted, else a refusal reason.
def _require(expected):
    def check(value):
        return None if value == expected else f"only {expected!r} is admitted by V1"
    return check


def _bool_false(value):
    return None if value is False else "only false is admitted by V1"


def _bool(value):
    return None if isinstance(value, bool) else "expected a boolean"


def _int(value):
    return None if isinstance(value, int) and not isinstance(value, bool) else "expected an integer"


_KNOWN_PROPERTIES = {
    "mapName": ("display", None),
    "notes": ("display", None),
    "Total Victory": ("gameplay", None),
    "Low Luck": ("gameplay", _bool_false),
    "Low Luck for AntiAircraft": ("gameplay", _bool_false),
    "Low Luck for Technology": ("gameplay", None),  # technology refused separately
    "Low Luck for Bombing and Territory Damage": ("gameplay", _bool_false),
    "WW2V2": ("gameplay", _require(True)),
    "Use Destroyers and Artillery": ("gameplay", _bool_false),
    "Two hit battleship": ("gameplay", _bool_false),
    "Produce fighters on carriers": ("gameplay", _bool_false),
    "Submersible Subs": ("gameplay", _bool_false),
    "Always on AA": ("gameplay", _bool),  # fly-over AA also in noncombat move (AaInMoveUtil)
    "Battleships repair at end of round": ("gameplay", _bool_false),
    "neutralCharge": ("gameplay", _int),  # nonzero admitted only when inert, checked below
    "maxFactoriesPerTerritory": ("gameplay", None),
    "Allied Air Dependents": ("inert", None),  # unmapped legacy spelling, rule-model §5
    "Tech Development": ("gameplay", _bool_false),
    "Heavy Bomber Dice Rolls": ("inert", None),  # only read with heavyBomber tech, refused
    "Territory Turn Limit": ("gameplay", _bool),  # Constants.PU_CAP: per-turn raid loss cap
}
# Name patterns: "<player> bid", "<alliance> Total Victory VCs".


class UnsupportedDefinition(Exception):
    """Raised by load_definition when the support result is Unsupported."""

    def __init__(self, report: "SupportReport"):
        super().__init__("\n".join(f"{f.code}: {f.feature_id}: {f.explanation}" for f in report.features))
        self.report = report


@dataclass(frozen=True)
class UnsupportedFeature:
    code: str  # UNSUPPORTED_BY_PROFILE_V1 | UNSUPPORTED_OPTIONAL_LATER | UNCERTIFIED_CONFIGURATION
    feature_id: str
    effective_value: str
    explanation: str


@dataclass
class SupportReport:
    features: List[UnsupportedFeature] = field(default_factory=list)
    inert: List[str] = field(default_factory=list)

    @property
    def supported(self) -> bool:
        return not self.features

    def refuse(self, code, feature_id, value, explanation):
        self.features.append(UnsupportedFeature(code, feature_id, str(value), explanation))


@dataclass(frozen=True)
class UnitType:
    id: int
    name: str
    domain: int
    movement: int
    attack: int
    defense: int
    attack_rolls: int
    defense_rolls: int
    hit_points: int
    can_blitz: bool
    is_infrastructure: bool
    can_produce: bool
    is_factory: bool
    is_aa: bool
    aa_strength: int
    can_move_in_combat: bool
    is_strategic_bomber: bool
    is_artillery: bool
    is_artillery_supportable: bool
    can_be_damaged: bool
    deferred: Tuple[Tuple[str, str], ...]  # inert V2 metadata (transportCost etc.)

    @property
    def is_air(self):
        return self.domain == AIR

    @property
    def is_land(self):
        return self.domain == LAND

    @property
    def is_combatant(self):
        """Can take part in ordinary combat as a target/firer (not infrastructure)."""
        return not self.is_infrastructure


@dataclass(frozen=True)
class ProductionRule:
    id: int
    name: str
    costs: Tuple[Tuple[int, int], ...]  # (resource id, amount)
    results: Tuple[Tuple[int, int], ...]  # (unit type id, quantity)


@dataclass(frozen=True)
class Step:
    id: int
    name: str
    kind: str
    actor: int  # player id, or NULL_PLAYER for unowned steps
    max_runs: Optional[int]


@dataclass(frozen=True)
class GameDefinition:
    schema_version: int
    profile_id: str
    profile_version: str
    normalizer_version: str
    definition_digest: str
    source_digest: str
    game_name: str
    territory_names: Tuple[str, ...]
    territory_index: Dict[str, int]
    is_water: Tuple[bool, ...]
    neighbors: Tuple[Tuple[int, ...], ...]
    production: Tuple[int, ...]
    victory_value: Tuple[int, ...]
    capital_of: Tuple[int, ...]  # player id or NULL_PLAYER
    original_factory: Tuple[bool, ...]
    impassable: Tuple[bool, ...]
    player_names: Tuple[str, ...]
    player_index: Dict[str, int]
    alliance_names: Tuple[str, ...]
    alliance_of: Tuple[int, ...]  # per player
    alliance_members: Tuple[Tuple[int, ...], ...]
    resource_names: Tuple[str, ...]
    treasury_resource: int
    unit_types: Tuple[UnitType, ...]
    unit_type_index: Dict[str, int]
    production_rules: Tuple[ProductionRule, ...]
    production_rule_index: Dict[str, int]
    frontier: Tuple[Tuple[int, ...], ...]  # per player, production rule ids
    steps: Tuple[Step, ...]
    dice_sides: int
    properties: Dict[str, object]
    victory_threshold: Dict[int, int]  # alliance id -> VC threshold (Total Victory)
    total_victory: bool
    pu_multiplier: int
    always_on_aa: bool  # "Always on AA"
    pu_cap: bool  # "Territory Turn Limit" (Constants.PU_CAP)
    neutral_charge: int  # "neutralCharge"; nonzero only when no neutral land can be entered
    # Initial setup recipe.
    initial_owner: Tuple[int, ...]
    initial_units: Tuple[Tuple[int, int, int, int], ...]  # (type, territory, owner, quantity)
    initial_balances: Tuple[Tuple[int, ...], ...]  # per player, per resource
    support: SupportReport

    def is_allied(self, a: int, b: int) -> bool:
        if a == b:
            return True
        if a == NULL_PLAYER or b == NULL_PLAYER:
            return False
        return self.alliance_of[a] == self.alliance_of[b]

    def is_at_war(self, a: int, b: int) -> bool:
        # Null player has the WAR archetype with everyone (RelationshipTypeList); singleton or
        # multi-player alliances are at war with every other alliance (parser default).
        return not self.is_allied(a, b)

    def unit_cost(self, type_id: int) -> int:
        """Cheapest treasury cost to buy one unit of this type (for agent heuristics)."""
        best = None
        for rule in self.production_rules:
            for t, q in rule.results:
                if t == type_id and q > 0:
                    cost = sum(a for r, a in rule.costs if r == self.treasury_resource) / q
                    best = cost if best is None else min(best, cost)
        return int(best) if best is not None else 0


def _parse_value(text: str):
    t = text.strip()
    if t.lower() == "true":
        return True
    if t.lower() == "false":
        return False
    try:
        return int(t)
    except ValueError:
        return t


def _options(att: ET.Element) -> Dict[str, object]:
    out = {}
    for opt in att.findall("option"):
        out[opt.get("name")] = _parse_value(opt.get("value", ""))
    return out


def load_definition(xml_text: str, overrides: Optional[Dict[str, object]] = None, strict: bool = True):
    """Parse and normalize a TripleA XML game. Returns (definition, support_report).

    With strict=True (default) an unsupported configuration raises UnsupportedDefinition.
    overrides: explicit setup values for editable properties (e.g. {"Russians bid": 0}).
    """
    if "<!ENTITY" in xml_text:
        raise ValueError("XML entity declarations are not allowed")
    overrides = dict(overrides or {})
    root = ET.fromstring(xml_text.encode("utf-8") if isinstance(xml_text, str) else xml_text)
    report = SupportReport()

    info = root.find("info")
    game_name = info.get("name") if info is not None else "unknown"

    loader = root.find("loader")
    if loader is not None and loader.get("javaClass") != "games.strategy.triplea.TripleA":
        report.refuse("UNSUPPORTED", "loader", loader.get("javaClass"), "only the TripleA loader is supported")
    if root.find("variableList") is not None or next(root.iter("foreach"), None) is not None:
        report.refuse("UNCERTIFIED_CONFIGURATION", "variableList/foreach", "present",
                      "XML variables/foreach expansion is not implemented by the Python normalizer")

    # --- map ---
    map_el = root.find("map")
    t_names: List[str] = []
    t_water: List[bool] = []
    for t in map_el.findall("territory"):
        t_names.append(t.get("name"))
        t_water.append(str(t.get("water", "false")).lower() == "true")
    t_index = {n: i for i, n in enumerate(t_names)}
    adj: List[set] = [set() for _ in t_names]
    for c in map_el.findall("connection"):
        a, b = t_index[c.get("t1")], t_index[c.get("t2")]
        if a != b:
            adj[a].add(b)
            adj[b].add(a)

    # --- resources / players / alliances ---
    res_names = [r.get("name") for r in root.find("resourceList").findall("resource")]
    pl = root.find("playerList")
    p_names = [p.get("name") for p in pl.findall("player")]
    p_index = {n: i for i, n in enumerate(p_names)}
    alliance_names: List[str] = []
    alliance_of = [-1] * len(p_names)
    for a in pl.findall("alliance"):
        an = a.get("alliance")
        if an not in alliance_names:
            alliance_names.append(an)
        pid = p_index[a.get("player")]
        if alliance_of[pid] != -1:
            report.refuse("UNCERTIFIED_CONFIGURATION", "alliance", a.get("player"),
                          "players in more than one alliance are not supported")
        alliance_of[pid] = alliance_names.index(an)
    for pid, aid in enumerate(alliance_of):
        if aid == -1:  # parser: a player without alliance is alone
            alliance_names.append(p_names[pid])
            alliance_of[pid] = len(alliance_names) - 1
    alliance_members = tuple(tuple(p for p in range(len(p_names)) if alliance_of[p] == a)
                             for a in range(len(alliance_names)))
    if pl.find("relationshipInitialize") is not None or root.find("relationshipTypes") is not None:
        report.refuse("UNSUPPORTED_OPTIONAL_LATER", "relationships", "explicit",
                      "explicit relationship types/politics are not supported by V1")

    # --- properties ---
    props: Dict[str, object] = {}
    plist = root.find("propertyList")
    if plist is not None:
        for p in plist.findall("property"):
            name = p.get("name")
            val_child = p.find("value")
            raw = val_child.text if val_child is not None and val_child.text else p.get("value", "")
            props[name] = _parse_value(raw)
    for k, v in overrides.items():
        props[k] = v
    for name, value in props.items():
        known = _KNOWN_PROPERTIES.get(name)
        if name.endswith(" bid") and name[:-4] in p_index:
            if value not in (0, False):
                report.refuse("UNSUPPORTED_OPTIONAL_LATER", f"property:{name}", value,
                              "nonzero bids are OPTIONAL_LATER")
            continue
        if name.endswith(" Total Victory VCs") or name.endswith(" Honorable Victory VCs") or \
                name.endswith(" Projection of Power VCs"):
            continue
        if known is None:
            report.refuse("UNCERTIFIED_CONFIGURATION", f"property:{name}", value,
                          "unknown property; not in the V1 supported-property manifest")
            continue
        kind, check = known
        if kind == "inert":
            report.inert.append(name)
        if check is not None:
            reason = check(value)
            if reason:
                report.refuse("UNSUPPORTED_OPTIONAL_LATER", f"property:{name}", value, reason)
    for name in ("Honorable Victory", "Projection of Power", "Economic Victory", "Triggered Victory"):
        if props.get(name) is True:
            report.refuse("UNSUPPORTED_OPTIONAL_LATER", f"property:{name}", True,
                          "only Total Victory VC checks are supported")

    # --- unit types ---
    unit_names = [u.get("name") for u in root.find("unitList").findall("unit")]
    unit_opts: Dict[str, Dict[str, object]] = {n: {} for n in unit_names}
    terr_opts: Dict[str, Dict[str, object]] = {}
    att_list = root.find("attachmentList")
    if att_list is not None:
        for att in att_list.findall("attachment"):
            jc = att.get("javaClass", "")
            target = att.get("attachTo")
            name = att.get("name")
            short = jc.rsplit(".", 1)[-1]
            if short == "UnitAttachment" or name == "unitAttachment":
                unit_opts.setdefault(target, {}).update(_options(att))
            elif short == "TerritoryAttachment" or name == "territoryAttachment":
                terr_opts.setdefault(target, {}).update(_options(att))
            elif short == "TechAttachment":
                for k, v in _options(att).items():
                    if v is True:
                        report.refuse("UNSUPPORTED_OPTIONAL_LATER", f"tech:{target}:{k}", v,
                                      "granted technology is OPTIONAL_LATER")
            elif short in ("RulesAttachment", "TriggerAttachment", "PoliticalActionAttachment",
                           "UserActionAttachment", "CanalAttachment", "TerritoryEffectAttachment",
                           "UnitSupportAttachment", "PlayerAttachment", "RelationshipTypeAttachment"):
                report.refuse("UNSUPPORTED_OPTIONAL_LATER", f"attachment:{short}:{target}", name,
                              f"{short} is not supported by the V1 Python normalizer")
            else:
                report.refuse("UNCERTIFIED_CONFIGURATION", f"attachment:{short}:{target}", name,
                              "unknown attachment class")

    known_unit_opts = {
        "movement", "attack", "defense", "attackRolls", "defenseRolls", "hitPoints", "isAir", "isSea",
        "canBlitz", "isFactory", "isAA", "artillery", "artillerySupportable", "isStrategicBomber",
        "isInfantry", "transportCost", "carrierCost", "transportCapacity", "carrierCapacity",
        "isSub", "isDestroyer", "canBombard", "isInfrastructure", "canProduceUnits",
        "isMarine", "isLandTransportable",
    }
    unit_types: List[UnitType] = []
    for uid, uname in enumerate(unit_names):
        o = unit_opts.get(uname, {})
        for k in o:
            if k not in known_unit_opts:
                report.refuse("UNCERTIFIED_CONFIGURATION", f"unit:{uname}:{k}", o[k],
                              "unit option not in the V1 manifest")
        is_air = o.get("isAir") is True
        is_sea = o.get("isSea") is True
        domain = AIR if is_air else (SEA if is_sea else LAND)
        is_factory = o.get("isFactory") is True
        is_aa = o.get("isAA") is True
        infra = is_factory or is_aa or o.get("isInfrastructure") is True
        deferred = tuple(sorted((k, str(o[k])) for k in
                                ("transportCost", "carrierCost", "transportCapacity", "carrierCapacity",
                                 "isSub", "isDestroyer", "canBombard", "isMarine", "isLandTransportable")
                                if k in o))
        unit_types.append(UnitType(
            id=uid, name=uname, domain=domain,
            movement=int(o.get("movement", 0)), attack=int(o.get("attack", 0)),
            defense=int(o.get("defense", 0)),
            attack_rolls=int(o.get("attackRolls", 1)), defense_rolls=int(o.get("defenseRolls", 1)),
            hit_points=int(o.get("hitPoints", 1)), can_blitz=o.get("canBlitz") is True,
            is_infrastructure=infra, can_produce=is_factory or o.get("canProduceUnits") is True,
            is_factory=is_factory, is_aa=is_aa, aa_strength=1, can_move_in_combat=not is_aa,
            is_strategic_bomber=o.get("isStrategicBomber") is True,
            is_artillery=o.get("artillery") is True,
            is_artillery_supportable=o.get("artillerySupportable") is True,
            can_be_damaged=is_factory, deferred=deferred))
    u_index = {u.name: u.id for u in unit_types}

    # --- territories ---
    production = [0] * len(t_names)
    victory = [0] * len(t_names)
    capital_of = [NULL_PLAYER] * len(t_names)
    orig_factory = [False] * len(t_names)
    impassable = [False] * len(t_names)
    known_terr_opts = {"production", "unitProduction", "capital", "victoryCity", "originalFactory",
                       "isImpassable", "originalOwner"}
    for tname, o in terr_opts.items():
        i = t_index[tname]
        for k in o:
            if k not in known_terr_opts:
                report.refuse("UNCERTIFIED_CONFIGURATION", f"territory:{tname}:{k}", o[k],
                              "territory option not in the V1 manifest")
        production[i] = int(o.get("production", 0))
        victory[i] = int(o.get("victoryCity", 0))
        if "capital" in o:
            capital_of[i] = p_index[str(o["capital"])]
        orig_factory[i] = o.get("originalFactory") is True
        impassable[i] = o.get("isImpassable") is True

    # --- production ---
    prod_el = root.find("production")
    rules: List[ProductionRule] = []
    rule_index: Dict[str, int] = {}
    frontiers: Dict[str, List[int]] = {}
    player_frontier = [()] * len(p_names)
    if prod_el is not None:
        for r in prod_el.findall("productionRule"):
            costs = tuple((res_names.index(c.get("resource")), int(c.get("quantity")))
                          for c in r.findall("cost"))
            results = []
            for res in r.findall("result"):
                target = res.get("resourceOrUnit")
                if target in u_index:
                    results.append((u_index[target], int(res.get("quantity"))))
                else:
                    report.refuse("UNCERTIFIED_CONFIGURATION", f"rule:{r.get('name')}", target,
                                  "resource-output production rules need certification")
            rule_index[r.get("name")] = len(rules)
            rules.append(ProductionRule(len(rules), r.get("name"), costs, tuple(results)))
        for f in prod_el.findall("productionFrontier"):
            frontiers[f.get("name")] = [rule_index[x.get("name")] for x in f.findall("frontierRules")]
        pf = list(player_frontier)
        for pp in prod_el.findall("playerProduction"):
            pf[p_index[pp.get("player")]] = tuple(frontiers[pp.get("frontier")])
        player_frontier = pf
        if prod_el.find("repairRule") is not None:
            report.refuse("UNSUPPORTED_OPTIONAL_LATER", "repairRule", "present", "repair is OPTIONAL_LATER")

    # --- sequence ---
    steps: List[Step] = []
    delegates = {}
    gp = root.find("gamePlay")
    for d in gp.findall("delegate"):
        delegates[d.get("name")] = d.get("javaClass")
    for s in gp.find("sequence").findall("step"):
        jc = delegates[s.get("delegate")]
        kind = _DELEGATE_KINDS.get(jc)
        name = s.get("name")
        if kind is None:
            report.refuse("UNSUPPORTED", f"delegate:{jc}", name, "unregistered delegate class")
            kind = "UNKNOWN"
        if kind == "MOVE":
            # GameStepPropertiesHelper fallback: names ending in NonCombatMove are noncombat.
            kind = NONCOMBAT_MOVE if name.endswith("NonCombatMove") else COMBAT_MOVE
        for prop in s.findall("stepProperty"):
            report.refuse("UNCERTIFIED_CONFIGURATION", f"stepProperty:{name}", prop.get("name"),
                          "explicit step properties are not implemented")
        actor = p_index[s.get("player")] if s.get("player") else NULL_PLAYER
        max_runs = int(s.get("maxRunCount")) if s.get("maxRunCount") else None
        if kind in (TECH, TECH_ACTIVATION):
            report.refuse("UNSUPPORTED_OPTIONAL_LATER", f"step:{name}", kind, "technology steps")
        steps.append(Step(len(steps), name, kind, actor, max_runs))

    # --- initialize ---
    init_el = root.find("initialize")
    initial_owner = [NULL_PLAYER] * len(t_names)
    initial_units: List[Tuple[int, int, int, int]] = []
    balances = [[0] * len(res_names) for _ in p_names]
    if init_el is not None:
        oi = init_el.find("ownerInitialize")
        if oi is not None:
            for to in oi.findall("territoryOwner"):
                initial_owner[t_index[to.get("territory")]] = p_index[to.get("owner")]
        ui = init_el.find("unitInitialize")
        if ui is not None:
            for up in ui.findall("unitPlacement"):
                owner = p_index[up.get("owner")] if up.get("owner") else NULL_PLAYER
                if up.get("hitsTaken") not in (None, "0") or up.get("unitDamage") not in (None, "0"):
                    report.refuse("UNCERTIFIED_CONFIGURATION", "unitPlacement:damage", up.get("unitType"),
                                  "initial damage needs certification")
                initial_units.append((u_index[up.get("unitType")], t_index[up.get("territory")], owner,
                                      int(up.get("quantity", 1))))
            for hu in ui.findall("heldUnits"):
                report.refuse("UNCERTIFIED_CONFIGURATION", "heldUnits", hu.get("unitType"),
                              "initial held units need certification")
        ri = init_el.find("resourceInitialize")
        if ri is not None:
            for rg in ri.findall("resourceGiven"):
                balances[p_index[rg.get("player")]][res_names.index(rg.get("resource"))] += \
                    int(rg.get("quantity"))
        if init_el.find("relationshipInitialize") is not None:
            report.refuse("UNSUPPORTED_OPTIONAL_LATER", "relationshipInitialize", "present",
                          "explicit relationships are not supported")

    # --- V2 / optional activation checks (spec §15) ---
    purchasable_types = {t for r in rules for t, _ in r.results
                         if any(r.id in fr for fr in player_frontier)}
    initial_types = {u[0] for u in initial_units}
    for ut in unit_types:
        active = ut.id in purchasable_types or ut.id in initial_types
        if not active:
            continue
        if ut.domain == SEA:
            report.refuse("UNSUPPORTED_BY_PROFILE_V1", f"unit:{ut.name}", "sea unit active",
                          "naval movement/combat require Forward Model V2")
        for k, v in ut.deferred:
            if k == "transportCapacity":
                report.refuse("UNSUPPORTED_BY_PROFILE_V1", f"unit:{ut.name}:transportCapacity", v,
                              "transport mechanics require Forward Model V2")
            if k == "carrierCapacity":
                report.refuse("UNSUPPORTED_BY_PROFILE_V1", f"unit:{ut.name}:carrierCapacity", v,
                              "carriers require Forward Model V2")
        if ut.hit_points != 1:
            report.refuse("UNCERTIFIED_CONFIGURATION", f"unit:{ut.name}:hitPoints", ut.hit_points,
                          "multi-hit land/air units are an unresolved certification gate")
        if ut.attack_rolls != 1 or ut.defense_rolls != 1:
            report.refuse("UNCERTIFIED_CONFIGURATION", f"unit:{ut.name}:rolls", "multiple",
                          "multiple combat rolls need certification")
    for i, w in enumerate(t_water):
        if w and any(u[1] == i for u in initial_units):
            report.refuse("UNSUPPORTED_BY_PROFILE_V1", f"territory:{t_names[i]}", "units at sea",
                          "units in water territories require Forward Model V2")
    neutral_charge = props.get("neutralCharge", 0)
    if isinstance(neutral_charge, int) and neutral_charge != 0:
        # The charge applies only when entering or flying over null-owned land (BattleTracker,
        # MoveValidator, AirMovementValidator). It is inert when all such territories are impassable.
        enterable = [t_names[i] for i in range(len(t_names))
                     if not t_water[i] and initial_owner[i] == NULL_PLAYER and not impassable[i]]
        if enterable:
            report.refuse("UNSUPPORTED_OPTIONAL_LATER", "property:neutralCharge", neutral_charge,
                          "neutral entry charges are OPTIONAL_LATER (enterable neutral land: "
                          + ", ".join(enterable) + ")")
        else:
            report.inert.append("neutralCharge")
    if len(res_names) != 1:
        report.refuse("UNCERTIFIED_CONFIGURATION", "resources", len(res_names),
                      "non-PU basic resource paths are an unresolved certification gate")

    dice_sides = int(root.find("diceSides").get("value")) if root.find("diceSides") is not None else 6

    victory_threshold = {}
    for aid, an in enumerate(alliance_names):
        key = f"{an} Total Victory VCs"
        if key in props:
            victory_threshold[aid] = int(props[key])
        else:
            victory_threshold[aid] = 18  # EndRoundDelegate.getVcAmount default (rule-model §5)

    digest_src = repr((SCHEMA_VERSION, PROFILE_ID, PROFILE_VERSION, NORMALIZER_VERSION, t_names, t_water,
                       [sorted(a) for a in adj], production, victory, capital_of, orig_factory,
                       p_names, alliance_of, res_names, unit_types, rules, player_frontier,
                       [(s.name, s.kind, s.actor, s.max_runs) for s in steps], dice_sides,
                       sorted((k, repr(v)) for k, v in props.items()), initial_owner, initial_units,
                       balances, sorted(overrides.items(), key=lambda kv: kv[0])))
    definition = GameDefinition(
        schema_version=SCHEMA_VERSION, profile_id=PROFILE_ID, profile_version=PROFILE_VERSION,
        normalizer_version=NORMALIZER_VERSION,
        definition_digest=hashlib.sha256(digest_src.encode()).hexdigest(),
        source_digest=hashlib.sha256(xml_text.encode() if isinstance(xml_text, str) else xml_text).hexdigest(),
        game_name=game_name,
        territory_names=tuple(t_names), territory_index=t_index, is_water=tuple(t_water),
        neighbors=tuple(tuple(sorted(a)) for a in adj), production=tuple(production),
        victory_value=tuple(victory), capital_of=tuple(capital_of), original_factory=tuple(orig_factory),
        impassable=tuple(impassable), player_names=tuple(p_names), player_index=p_index,
        alliance_names=tuple(alliance_names), alliance_of=tuple(alliance_of),
        alliance_members=alliance_members, resource_names=tuple(res_names),
        treasury_resource=res_names.index("PUs") if "PUs" in res_names else 0,
        unit_types=tuple(unit_types), unit_type_index=u_index, production_rules=tuple(rules),
        production_rule_index=rule_index, frontier=tuple(player_frontier), steps=tuple(steps),
        dice_sides=dice_sides, properties=props, victory_threshold=victory_threshold,
        total_victory=props.get("Total Victory") is True,
        pu_multiplier=int(props.get("PU Multiplier", 1)),
        always_on_aa=props.get("Always on AA") is True,
        pu_cap=props.get("Territory Turn Limit") is True,
        neutral_charge=neutral_charge if isinstance(neutral_charge, int) else 0,
        initial_owner=tuple(initial_owner), initial_units=tuple(initial_units),
        initial_balances=tuple(tuple(b) for b in balances), support=report)
    if strict and not report.supported:
        raise UnsupportedDefinition(report)
    return definition, report
