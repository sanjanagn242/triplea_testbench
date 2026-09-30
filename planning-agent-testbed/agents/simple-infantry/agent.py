#!/usr/bin/env python3
"""Map-rule-aware, JSON-RPC-over-stdio policy for the planning-agent testbed."""

from __future__ import annotations

import json
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
import xml.etree.ElementTree as ET
from collections import deque
from dataclasses import dataclass


@dataclass
class MapRules:
    infantry_type: str | None
    infantry_cost: int | None
    infantry_rule_by_player: dict[str, str]
    targets_by_player: dict[str, list[str]]
    impassable: set[str]


def _options(element: ET.Element) -> dict[str, str]:
    return {
        option.get("name", ""): option.get("value", "")
        for option in element.findall("option")
    }


def parse_map_rules(xml_text: str) -> MapRules:
    """Read the map facts used by this policy from the player's supplied game XML."""
    root = ET.fromstring(xml_text)
    unit_is_infantry: set[str] = set()
    for attachment in root.findall(".//attachmentList/attachment"):
        if attachment.get("name") != "unitAttachment":
            continue
        options = _options(attachment)
        if options.get("isInfantry", "false").lower() == "true":
            unit_is_infantry.add(attachment.get("attachTo", ""))

    rules_by_name: dict[str, tuple[str, int]] = {}
    for rule in root.findall(".//productionRule"):
        name = rule.get("name", "")
        result = next(
            (
                node
                for node in rule.findall("result")
                if node.get("resourceOrUnit", "") in unit_is_infantry
            ),
            None,
        )
        if result is None:
            continue
        cost = sum(
            int(node.get("quantity", "0"))
            for node in rule.findall("cost")
            if node.get("resource") == "PUs"
        )
        if cost > 0:
            rules_by_name[name] = (result.get("resourceOrUnit", ""), cost)

    frontier_rules: dict[str, set[str]] = {}
    for frontier in root.findall(".//productionFrontier"):
        frontier_rules[frontier.get("name", "")] = {
            node.get("name", "") for node in frontier.findall("frontierRules")
        }
    player_frontiers = {
        node.get("player", ""): node.get("frontier", "")
        for node in root.findall(".//playerProduction")
    }
    infantry_type = None
    infantry_cost = None
    infantry_rule_by_player: dict[str, str] = {}
    for player, frontier_name in player_frontiers.items():
        candidates = [
            (name, result_type, cost)
            for name, (result_type, cost) in rules_by_name.items()
            if name in frontier_rules.get(frontier_name, set())
        ]
        if candidates:
            rule_name, result_type, cost = min(candidates, key=lambda item: (item[2], item[0]))
            infantry_rule_by_player[player] = rule_name
            infantry_type = infantry_type or result_type
            infantry_cost = infantry_cost or cost

    targets: dict[str, list[str]] = {}
    impassable: set[str] = set()
    for attachment in root.findall(".//attachmentList/attachment"):
        if attachment.get("name") != "territoryAttachment":
            continue
        territory = attachment.get("attachTo", "")
        options = _options(attachment)
        if options.get("isImpassable", "false").lower() == "true":
            impassable.add(territory)
        capital_owner = options.get("capital")
        victory_value = int(options.get("victoryCity", "0") or "0")
        if capital_owner:
            targets.setdefault(capital_owner, []).append(territory)
        elif victory_value > 0:
            # Used when the map has victory cities but no capital attachments.
            targets.setdefault("*", []).append(territory)

    return MapRules(
        infantry_type,
        infantry_cost,
        infantry_rule_by_player,
        targets,
        impassable,
    )


class SimpleInfantryAgent:
    def __init__(self) -> None:
        self.player_name: str | None = None
        self.rules: MapRules | None = None
        self.state: dict | None = None
        self.observation_count = 0
        self.game_number: int | None = None
        self.log_path: Path | None = None

    def log(self, event: str, **details: object) -> None:
        timestamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        fields = " ".join(
            f"{key}={json.dumps(value, ensure_ascii=False, separators=(',', ':'))}"
            for key, value in {"game_number": self.game_number, **details}.items()
        )
        line = f"{timestamp} {event} {fields}\n"
        if self.log_path is None:
            sys.stderr.write(line)
            sys.stderr.flush()
            return
        try:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            with self.log_path.open("a", encoding="utf-8") as log_file:
                log_file.write(line)
        except OSError as error:
            sys.stderr.write(f"Could not write agent log {self.log_path}: {error}\n{line}")
            sys.stderr.flush()

    def start_game(self, request: dict) -> dict:
        self.player_name = request["playerName"]
        self.game_number = int(request.get("gameNumber", 1))
        safe_player_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", self.player_name)
        self.log_path = (
            Path.cwd()
            / "planning-agent-testbed"
            / "logs"
            / f"agent-{safe_player_name}-game-{self.game_number}.txt"
        )
        log_already_exists = self.log_path.exists()
        if log_already_exists:
            self.log("agent_process_restarted", player=self.player_name)
        else:
            self.log_path.parent.mkdir(parents=True, exist_ok=True)
            self.log_path.write_text(f"Game number: {self.game_number}\n", encoding="utf-8")
        self.rules = parse_map_rules(request["gameXml"])
        self.state = request["initialState"]
        self.observation_count = 0
        self.log(
            "game_start",
            game=self.state.get("gameName"),
            map=self.state.get("mapName"),
            player=self.player_name,
            round=self.state.get("round"),
            rules_loaded=True,
        )
        return {"ready": True, "protocolVersion": request["schemaVersion"]}

    def handle_turn(self, request: dict) -> dict:
        if self.rules is None or self.player_name is None:
            raise RuntimeError("game_start must be received before turn_request")
        self.state = request["state"]
        self.observation_count += 1
        phase = request["phase"]
        self.log(
            "turn_request",
            game=self.state.get("gameName"),
            player=self.player_name,
            round=self.state.get("round"),
            phase=phase,
            request_id=request.get("requestId"),
        )
        if phase == "purchase":
            action = self.purchase()
        elif phase == "combatMove":
            action = self.combat_moves()
        elif phase == "battle":
            # Delegate combat resolution to TripleA's normal battle UI/rules.
            action = {"fightAll": True}
        elif phase == "place":
            action = self.place()
        else:
            action = {}
        self.log(
            "action_sent",
            game=self.state.get("gameName"),
            player=self.player_name,
            round=self.state.get("round"),
            phase=phase,
            action=action,
        )
        return action

    def purchase(self) -> dict:
        assert self.state is not None and self.rules is not None
        player = next(
            (p for p in self.state["players"] if p["name"] == self.player_name), None
        )
        rule = self.rules.infantry_rule_by_player.get(self.player_name or "")
        if player is None or rule is None or not self.rules.infantry_cost:
            return {"purchaseCount": 0, "rule": None, "unitType": self.rules.infantry_type}
        money = player["resources"].get("PUs", 0)
        return {
            "purchaseCount": max(0, money // self.rules.infantry_cost),
            "rule": rule,
            "unitType": self.rules.infantry_type,
        }

    def place(self) -> dict:
        assert self.state is not None and self.rules is not None
        factories = {
            territory["name"]
            for territory in self.state["territories"]
            if territory.get("owner") == self.player_name
            and not territory["water"]
            and any(
                unit["owner"] == self.player_name and unit["infrastructure"]
                for unit in territory["units"]
            )
        }
        capitals = self.rules.targets_by_player.get(self.player_name or "", [])
        placement_targets = sorted(factories.intersection(capitals)) or sorted(factories)
        return {
            "placeAt": placement_targets[0] if placement_targets else None,
            "unitType": self.rules.infantry_type,
        }

    def combat_moves(self) -> dict:
        assert self.state is not None and self.rules is not None
        territories = self.state["territories"]
        by_name = {territory["name"]: territory for territory in territories}
        targets = self.enemy_targets()
        if not targets:
            return {"moves": []}
        distance, next_step = self.paths_to_targets(by_name, targets)
        proposals: dict[tuple[str, str], list[str]] = {}

        for territory in territories:
            own_infantry = [
                unit
                for unit in territory["units"]
                if unit["owner"] == self.player_name
                and unit["type"] == self.rules.infantry_type
                and float(unit["movementLeft"]) > 0
            ]
            if not own_infantry:
                continue
            target = next_step.get(territory["name"])
            if target is None or target not in by_name:
                continue
            target_state = by_name[target]
            if target_state["water"] or target_state["impassable"]:
                continue
            defenders = [
                unit
                for unit in target_state["units"]
                if unit["owner"] != self.player_name
                and unit["type"] == self.rules.infantry_type
            ]
            if defenders and len(own_infantry) < len(defenders):
                continue
            proposals[(territory["name"], target)] = [unit["id"] for unit in own_infantry]

        return {
            "moves": [
                {"from": source, "to": target, "unitIds": ids}
                for (source, target), ids in sorted(proposals.items())
            ]
        }

    def enemy_targets(self) -> list[str]:
        assert self.rules is not None and self.player_name is not None
        targets = [
            territory
            for owner, territories in self.rules.targets_by_player.items()
            if owner not in (self.player_name, "*")
            for territory in territories
        ]
        if not targets:
            targets = self.rules.targets_by_player.get("*", [])
        return targets

    def paths_to_targets(self, by_name: dict, targets: list[str]) -> tuple[dict, dict]:
        assert self.rules is not None
        distance: dict[str, int] = {}
        next_step: dict[str, str] = {}
        pending = deque()
        for target in sorted(targets):
            if target in by_name and target not in self.rules.impassable:
                distance[target] = 0
                pending.append(target)
        while pending:
            current = pending.popleft()
            for neighbor in sorted(by_name[current].get("neighbors", [])):
                if neighbor not in by_name or neighbor in self.rules.impassable:
                    continue
                if neighbor not in distance:
                    distance[neighbor] = distance[current] + 1
                    next_step[neighbor] = current
                    pending.append(neighbor)
        return distance, next_step


def main() -> None:
    agent = SimpleInfantryAgent()
    for line in sys.stdin:
        try:
            request = json.loads(line)
            if request.get("schemaVersion") != 1:
                raise ValueError("Unsupported protocol version")
            if request.get("type") == "game_start":
                response = agent.start_game(request)
            elif request.get("type") == "turn_request":
                response = agent.handle_turn(request)
            else:
                raise ValueError(f"Unknown message type: {request.get('type')}")
        except Exception as error:  # Send protocol errors back in-band for engine diagnostics.
            agent.log(
                "protocol_error",
                message=f"{type(error).__name__}: {error}",
                request_type=(request.get("type") if "request" in locals() else None),
            )
            response = {"error": f"{type(error).__name__}: {error}"}
        json.dump(response, sys.stdout, separators=(",", ":"))
        sys.stdout.write("\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
