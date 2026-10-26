#!/usr/bin/env python3
"""Tournament: runs testbench JSON configurations one after another and reports win rates.

Each configuration is played by the real engine: the headed client, autostarted, exactly as
`./gradlew :game-headed:run --args="triplea.testbench.config=... triplea.testbench.autostart=true"`,
or, with --no-gui or when there is no display (e.g. on a cluster node), the windowless runner
`./gradlew :game-headed:runTestbenchHeadless --args="triplea.testbench.config=..."`, which opens no
window at all.
The engine appends one JSON line per finished game to the tournament's games.jsonl
(triplea.testbench.results); the report is computed from those lines and written to results.txt:

  - per configuration: wins of each faction / agent and games without a winner
  - head-to-head: every pair of agents that played against each other, over all configurations and
    both seats (so evaluate-Russians.json + evaluate-Germans.json are combined), with per-seat rates
  - per agent: games, wins and win rate over the whole tournament

Agent metrics are written to the tournament's folder, not the agents' own: the engine runs with
triplea.testbench.outputs=<tournament>/agents and gives every agent process
TRIPLEA_TESTBENCH_OUTPUT_DIR=<tournament>/agents/<agent id>, under which the agents write
metrics/<game>/vs-<opponent>/ (helper.metrics_dir, helper.output_folder). A configuration's
clearAgentOutputs is turned off during a tournament (it would wipe what earlier configurations
wrote); use --clear-outputs to also empty the agents' outputDirectories (records, logs, ...).

Before any game, the configurations are renumbered in place: each one's firstGameNumber follows on
from the previous one's games (1, 1 + N, 1 + 2N, ...), and with --games N every games count is set
to N first. A folder argument runs all *.json in it, in name order (numbers compared as numbers,
so 2.x runs before 10.x).

Parallel batches: --batch-size B cuts the tournament's games, in play order, into batches of B
games, and plays all batches at once, each its games one after another (so 72 configurations of 10
games with --batch-size 80 run as 9 engines in parallel; a configuration may be split between two
batches, its games keeping their numbers). Without it, all games are one batch, played one at a
time as above. Each batch writes its own results and engine output (batches/<k>.jsonl, .log), which
are merged into games.jsonl after every configuration; agents write their shared metric CSVs under
a file lock (helper.locked), and their other files are named by game number, which is unique. Each
batch plays a configuration under a copy of its own (.tournament-b<k>-*.json), so suspended games
are checkpointed per batch. Searches limited by wall-clock time get fewer iterations when the
machine is overloaded, so keep the batches (two agent processes each) well under the CPU count.

Instead of configurations, a tournament spec (--spec) can name the agents: every pair of them is
played on the spec's map, once in each seat, `games` games each. The configurations are generated
into games/<game>/config/generated/<spec name>/ (replacing what an earlier run generated there) and
then run as above. A spec looks like:

  {
    "map": "Capture The Flag",            # or a configuration's map object {name, gameName, ...}
    "gameFolder": "capture-the-flag-diag", # games/<folder>; default: the folder holding the spec
    "games": 10,                           # per configuration (one pair in one seat order)
    "agents": ["tripleMind-mcts", "tripleMind-mcts-no-reward-heuristic", "Hard (AI)"],
    "roundLimit": 100                      # any other configuration setting is copied as is
  }

Several maps: replace "map"/"gameFolder" by a "maps" list, e.g.
  "maps": [{"gameFolder": "capture-the-flag-diag", "map": {...}},
           {"gameFolder": "minimap", "map": {...}, "roundLimit": 50}]
Every pairing is played on every map; an entry may override any shared key. The report then has a
section per map and one over all maps.

Optional: "factions" (the seats; default the map's players), "otherAgents" (faction -> agent for
the factions that are not seats), "selfPlay" (also pair each agent with itself; default false) and
"seating", how the agents are put on the seats (a map with two seats is always pairwise):
  "pairwise"  (default) two seats; every ordered pair of agents, one agent per seat.
  "cyclic"    any number of seats and at least as many agents; configuration i puts agent i on the
              first seat, agent i+1 on the second, ... (wrapping around), so with N agents there
              are N configurations and every agent plays every seat once.
  "diagonal"  four seats, in order around the board; every ordered pair of agents, the first on
              seats 1 and 3, the second on seats 2 and 4 (on Capture The Flag the map's order,
              Russians, Italians, Germans, Chinese, already goes around the board). The two
              factions of one agent count as one side in the report (the configuration's "teams").

Agent variants: the same agent with different settings (environment variables) is played as a
separate agent, with its own ID, metrics folder and vs-<opponent> folders, and its own line in the
report. Write it in "agents" (or --agents) as an agent ID followed by KEY=VALUE settings, or as an
object to choose the ID:
  "agents": ["tripleMind-mcts TRIPLEA_MCTS_ROLLOUT=2",
             "tripleMind-mcts TRIPLEA_MCTS_ROLLOUT=10",
             {"agent": "tripleMind-mcts", "id": "tm-long", "env": {"TRIPLEA_MCTS_SECONDS": "5"}},
             "Hard (AI)"]
Without "id" the ID is the agent's followed by its settings, e.g.
tripleMind-mcts-MCTS_ROLLOUT-10 (a leading TRIPLEA_ is dropped). Each variant gets a generated
agent.json in agents/generated-variants/<id>/ whose command sets the variables and runs the agent's
own command; the agent's run script must take the value from the environment (VAR="${VAR:-default}").

Usage (from the repository root or planning-agent-testbed/):
  python3 planning-agent-testbed/tools/tournament.py \\
      --spec planning-agent-testbed/games/capture-the-flag-diag/tournament.json
  python3 planning-agent-testbed/tools/tournament.py \\
      planning-agent-testbed/games/capture-the-flag-diag/config/tournament --games 10
  python3 planning-agent-testbed/tools/tournament.py --spec <spec> --games 10 --batch-size 80
  python3 planning-agent-testbed/tools/tournament.py \\
      planning-agent-testbed/games/capture-the-flag-diag/config/evaluate-Russians.json \\
      planning-agent-testbed/games/capture-the-flag-diag/config/evaluate-Germans.json \\
      planning-agent-testbed/games/capture-the-flag-diag/config/capture-the-flag.json
  python3 planning-agent-testbed/tools/tournament.py --report-only planning-agent-testbed/tournaments/<run>/games.jsonl
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import xml.etree.ElementTree as ET
import shutil
import subprocess
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
TESTBED = HERE.parent
REPO = TESTBED.parent
AGENTS_DIR = TESTBED / "agents"
TOURNAMENTS_DIR = TESTBED / "tournaments"
SAVES_DIR = TESTBED / "saves"
NO_WINNER = "no winner"
BATCH_STAGGER_SECONDS = 15  # between the starts of parallel batches (run_batches)


# ---- running ---------------------------------------------------------------------------------

# Spec keys that describe the tournament rather than a configuration setting.
SPEC_KEYS = {"map", "maps", "gameFolder", "games", "agents", "factions", "otherAgents", "selfPlay",
             "seating"}
SEATINGS = ("pairwise", "cyclic", "diagonal")


def generate_configs(spec_path: Path, agents_override, games_override):
    """Writes one configuration per ordered pair of the spec's agents (both seats), on each of its
    maps, and returns their paths in play order (map by map)."""
    spec = json.loads(spec_path.read_text())
    if "maps" not in spec:
        return generate_map_configs(spec_path, spec, agents_override, games_override, True)
    shared = {k: v for k, v in spec.items() if k != "maps"}
    configs = []
    for entry in spec["maps"]:
        if not entry.get("gameFolder"):
            raise SystemExit(f"every entry of \"maps\" needs a gameFolder: {entry}")
        # A map entry may override any shared key (games, roundLimit, factions, ...).
        configs += generate_map_configs(spec_path, {**shared, **entry}, agents_override,
                                        games_override, False)
    return configs


def generate_map_configs(spec_path: Path, spec, agents_override, games_override, single_map):
    known = agent_manifests()
    agents = resolve_agents(agents_override or spec.get("agents") or [], known)
    if len(agents) < 2 and not (agents and spec.get("selfPlay")):
        raise SystemExit("the spec needs at least two agents")
    for agent in agents:
        if agent not in known and not agent.endswith("(AI)"):
            raise SystemExit(f"unknown agent {agent!r}: not an agent.json id or a built-in (AI) player")
    game_dir = spec_game_dir(spec_path if single_map else None, spec)
    if not game_dir.is_dir():
        raise SystemExit(f"no game folder {game_dir}")
    map_ref = spec.get("map")
    if isinstance(map_ref, str):
        map_ref = {"name": map_ref, "gameName": map_ref}
    if not map_ref:
        raise SystemExit("the spec needs a map")
    if not map_ref.get("aliases"):
        # The engine finds the downloaded map folder (e.g. map/capture_the_flag) by these; without
        # them it plays nothing.
        map_ref = dict(map_ref, aliases=map_aliases(game_dir, map_ref["name"]))
    factions = spec.get("factions") or map_players(game_dir, map_ref.get("gameName") or map_ref["name"])
    others = spec.get("otherAgents") or {}
    seats = [f for f in factions if f not in others]
    seating = spec.get("seating", "pairwise")
    if seating not in SEATINGS:
        raise SystemExit(f"unknown seating {seating!r}; use one of {', '.join(SEATINGS)}")
    if len(seats) == 2:  # seating only matters with more seats; two seats are always pairwise
        seating = "pairwise"
    if seating == "pairwise" and len(seats) != 2:
        raise SystemExit(f"pairwise play needs two seats, got {seats}; set \"factions\" to the two "
                         f"seats and \"otherAgents\" for the remaining factions, or set \"seating\"")
    if seating == "diagonal" and len(seats) != 4:
        raise SystemExit(f"diagonal seating needs four seats, got {seats}; set \"factions\" to four "
                         f"in order around the board and \"otherAgents\" for the remaining factions")
    if seating == "cyclic" and len(agents) < len(seats):
        raise SystemExit(f"cyclic seating needs at least as many agents as seats ({len(seats)}: "
                         f"{', '.join(seats)}), got {len(agents)}")
    if seating == "cyclic" and spec.get("selfPlay"):
        raise SystemExit("selfPlay does not apply to cyclic seating")
    games = games_override or int(spec.get("games", 1))
    settings = {k: v for k, v in spec.items() if k not in SPEC_KEYS}

    if seating == "cyclic":
        lineups = [[agents[(i + k) % len(agents)] for k in range(len(seats))]
                   for i in range(len(agents))]
    else:
        pairs = [(a, b) for a, b in itertools.combinations(agents, 2) for a, b in ((a, b), (b, a))]
        if spec.get("selfPlay"):
            pairs += [(a, a) for a in agents]
        lineups = [[a, b] if seating == "pairwise" else [a, b, a, b] for a, b in pairs]
    out = game_dir / "config" / "generated" / spec_path.stem  # per map: in that map's game folder
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("*.json"):
        old.unlink()
    short = lambda agent: re.sub(r"[^A-Za-z0-9._-]+", "-", agent).strip("-")
    configs = []
    for i, lineup in enumerate(lineups, 1):
        assignment = {**dict(zip(seats, lineup)), **others}
        config = {"map": map_ref, "games": games, "roundLimit": 100, "aiMovePauseMs": 0,
                  "aiCombatStepPauseMs": 0, **settings, "agents": assignment}
        if seating == "diagonal":  # read by the report (opposing); the engine ignores it
            config["teams"] = [[seats[0], seats[2]], [seats[1], seats[3]]]
        config.setdefault("showObservations", False)
        config.setdefault("observationPlayers", {f: False for f in assignment})
        named = lineup if seating == "cyclic" else lineup[:2]
        path = out / f"{i}.{'-vs-'.join(short(a) for a in named)}.json"
        path.write_text(json.dumps(config, indent=2) + "\n")
        configs.append(path)
    if seating == "cyclic":
        sides = f"cyclic over {', '.join(seats)}"
    elif seating == "diagonal":
        sides = f"{seats[0]}+{seats[2]} vs {seats[1]}+{seats[3]}"
    else:
        sides = f"{seats[0]} vs {seats[1]}"
    print(f"Generated {len(configs)} configurations for {game_dir.name} in {out.relative_to(REPO)} "
          f"({sides}, {games} games each)")
    return configs


def map_aliases(game_dir: Path, name: str):
    folders = [d.name for d in sorted((game_dir / "map").glob("*")) if d.is_dir()]
    snake = re.sub(r"[^a-z0-9]+", "_", name.lower()).strip("_")
    return list(dict.fromkeys(folders + [snake]))


def spec_game_dir(spec_path, spec) -> Path:
    if spec.get("gameFolder"):
        return TESTBED / "games" / spec["gameFolder"]
    for d in (spec_path.resolve().parents if spec_path else []):
        if d.parent == TESTBED / "games":
            return d
    name = spec["map"] if isinstance(spec.get("map"), str) else (spec.get("map") or {}).get("name", "")
    return TESTBED / "games" / re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def map_players(game_dir: Path, game_name: str):
    """The players of the game named game_name among the XMLs in game_dir/map."""
    for xml in sorted((game_dir / "map").glob("**/games/*.xml")):
        try:
            root = ET.parse(xml).getroot()
        except ET.ParseError:
            continue
        info = root.find("info")
        if info is not None and info.get("name") == game_name:
            return [p.get("name") for p in root.iter("player") if p.get("name")]
    raise SystemExit(f"no map XML for {game_name!r} in {game_dir / 'map'} (it is downloaded by the "
                     f"first run); set \"factions\" in the spec")


def collect_configs(paths):
    """The configurations named, a folder standing for all *.json in it (not .tournament-* copies),
    in natural name order."""
    natural = lambda p: [int(t) if t.isdigit() else t.lower() for t in re.split(r"(\d+)", p.name)]
    configs = []
    for path in paths:
        path = path.resolve()
        if path.is_dir():
            configs += sorted((c for c in path.glob("*.json") if not c.name.startswith(".")), key=natural)
        else:
            configs.append(path)
    return configs


def renumber_configs(configs, games, first_game):
    """Writes games (when given) and consecutive firstGameNumbers into the configurations, so the
    tournament's games are numbered first_game, first_game + 1, ... across all of them."""
    number = first_game
    for config in configs:
        data = json.loads(config.read_text())
        if games is not None:
            data["games"] = games
        updated = {}
        for key, value in data.items():  # firstGameNumber goes right after games
            if key != "firstGameNumber":
                updated[key] = value
            if key == "games":
                updated["firstGameNumber"] = number
        updated.setdefault("firstGameNumber", number)
        config.write_text(json.dumps(updated, indent=2) + "\n")
        count = int(updated.get("games", 1))
        print(f"  {config.name}: games {number}-{number + count - 1}")
        number += count


def run_config(config: Path, results: Path, max_resumes, gradle_args, outputs: Path, gui=None,
               batch=None, segment=None, log=None) -> Path:
    """Plays one configuration to the end; returns the path the engine knows it by (the
    configuration itself, or a sibling .tournament-*.json copy when settings had to change).
    In a batch (see run_batches) it always plays a copy of its own, .tournament-b<batch>-*.json, so
    its checkpoint folder is the batch's own; segment (first game number, games) then limits it to
    those of its games, and the engine's output goes to the file log instead of the terminal."""
    data = json.loads(config.read_text())
    changed = batch is not None
    say = print if batch is None else (lambda message: print(f"  [batch {batch}] {message.strip()}",
                                                            flush=True))
    if data.get("clearAgentOutputs"):
        data["clearAgentOutputs"] = False
        changed = True
        say(f"  clearAgentOutputs turned off for the tournament (use --clear-outputs)")
    if segment:
        data["firstGameNumber"], data["games"] = segment
    # Map lookup and mapXml are relative to the configuration's folder, so a copy stays beside it.
    prefix = ".tournament-" if batch is None else f".tournament-b{batch}-"
    run_path = config.with_name(f"{prefix}{config.name}") if changed else config
    if changed:
        run_path.write_text(json.dumps(data, indent=2) + "\n")
    planned = int(data.get("games", 1))
    checkpoint = checkpoint_dir(run_path) / "checkpoint.json"
    if checkpoint.exists():
        say(f"  note: {checkpoint} exists, so the engine resumes that suspended run first")
    try:
        for attempt in range(max_resumes + 1):
            args = (f"triplea.testbench.config={run_path} triplea.testbench.autostart=true "
                    f"triplea.testbench.results={results} triplea.testbench.outputs={outputs}")
            if gui is not None:  # overrides each configuration's showGame
                args += f" triplea.testbench.gui={str(gui).lower()}"
            if batch is not None:  # the other batches' games are still writing their logs
                args += " triplea.testbench.keepLogs=true"
            command = [str(REPO / "gradlew"), *gradle_args, engine_task(gui), f"--args={args}"]
            if log:
                log.write(f"\n=== {run_path.name}: {' '.join(command)}\n")
                log.flush()
            code = subprocess.call(command, cwd=REPO, stdout=log, stderr=log and subprocess.STDOUT)
            done = sum(1 for r in read_results(results) if r["configuration"] == str(run_path))
            if code != 0:
                say(f"  engine exited with status {code} after {done}/{planned} games")
                break
            if done == 0 and not checkpoint.exists():
                say(f"  WARNING: the engine played no games of {config.name}. Check its map "
                    f"(name, gameName, aliases) and agents by loading it in the testbench UI.")
                break
            if done >= planned or not checkpoint.exists():
                break
            # An agent asked to suspend (e.g. a time limit): running it again resumes the game.
            say(f"  suspended after {done}/{planned} games; resuming ({attempt + 1}/{max_resumes})")
    finally:
        if changed:
            run_path.unlink(missing_ok=True)
    return run_path


def plan_batches(configs, batch_size):
    """The tournament's games, in play order, cut into batches of batch_size games: per batch a list
    of (configuration, first game number, games) segments. A configuration may be split between two
    batches; its games keep their numbers."""
    batches, segments, room = [], [], batch_size
    for config in configs:
        data = json.loads(config.read_text())
        first, left = int(data["firstGameNumber"]), int(data.get("games", 1))
        while left:
            games = min(left, room)
            segments.append((config, first, games))
            first, left, room = first + games, left - games, room - games
            if not room:
                batches.append(segments)
                segments, room = [], batch_size
    return batches + [segments] if segments else batches


def missing_maps(configs):
    """Game folders of configurations whose map is not downloaded yet: the engine downloads it into
    games/<game>/map on the first run, and engines in parallel would all download it at once."""
    folders = {next((d.parent for d in c.parents if d.name == "config"
                     and d.parent.parent == TESTBED / "games"), None) for c in configs}
    return sorted(f.name for f in folders if f and not any((f / "map").glob("**/games/*.xml")))


def merge_results(out: Path, files):
    """Writes the batches' results into the tournament's games.jsonl, in game order, each line
    naming the configuration it was copied from, as in a tournament without batches. Called under
    run_batches' lock; the file is replaced in one step, so a reader never sees half of it."""
    games = [dict(g, configuration=original_config(g["configuration"]))
             for f in files for g in read_results(f)]
    games.sort(key=lambda g: g.get("game", 0))
    temporary = out / ".games.jsonl.tmp"
    temporary.write_text("".join(json.dumps(g, separators=(",", ":")) + "\n" for g in games))
    os.replace(temporary, out / "games.jsonl")


def run_batches(batches, out: Path, args, labels):
    """Plays all batches at once, each its segments one after another (a thread per batch, each
    running its own engine). A batch has its own results file and engine log, batches/<k>.jsonl and
    batches/<k>.log, so engines never write the same file; after every segment the batches' results
    are merged into games.jsonl and the report is rewritten, one thread at a time."""
    folder = out / "batches"
    folder.mkdir(exist_ok=True)
    files = [folder / f"{k}.jsonl" for k in range(1, len(batches) + 1)]
    lock = threading.Lock()
    stop = threading.Event()
    # The engines keep their agents' logs (triplea.testbench.keepLogs), so the earlier runs' logs
    # are deleted here once, as each engine run would otherwise.
    for old in (TESTBED / "logs").glob("agent-*.txt"):
        old.unlink(missing_ok=True)

    def play(k, segments):
        # Engines start BATCH_STAGGER_SECONDS apart: one at a time checks the Gradle build and
        # unzips maps.
        if stop.wait((k - 1) * BATCH_STAGGER_SECONDS):
            return
        with open(folder / f"{k}.log", "a") as log:
            for i, (config, first, games) in enumerate(segments, 1):
                if stop.is_set():
                    return
                name = config.relative_to(REPO) if REPO in config.parents else config
                print(f"[batch {k}] ({i}/{len(segments)}) {name}: games {first}-{first + games - 1}",
                      flush=True)
                run_config(config, files[k - 1], args.max_resumes, args.gradle_arg, out / "agents",
                           args.gui, k, (first, games), log)
                with lock:
                    labels[str(config)] = str(
                        config.relative_to(TESTBED) if TESTBED in config.parents else config)
                    merge_results(out, files)
                    text = report(read_results(out / "games.jsonl"), labels,
                                  f"Tournament {out.name}", read_timing(out))
                    (out / "results.txt").write_text(text)
        print(f"[batch {k}] done", flush=True)

    with ThreadPoolExecutor(len(batches)) as pool:
        futures = [pool.submit(play, k, segments) for k, segments in enumerate(batches, 1)]
        try:
            for future in futures:
                future.result()
        except BaseException:  # e.g. Ctrl-C: the engines get it too; start no further segments
            stop.set()
            raise
    merge_results(out, files)


def original_config(configuration: str) -> str:
    """The configuration a .tournament-*.json or .tournament-b<batch>-*.json copy was made from;
    any other configuration as it is."""
    path = Path(configuration)
    match = re.fullmatch(r"\.tournament-(?:b\d+-)?(.+)", path.name)
    return str(path.with_name(match.group(1))) if match else configuration


def has_display() -> bool:
    """False on Linux and other X11 systems without DISPLAY / WAYLAND_DISPLAY (cluster nodes, ssh)."""
    if sys.platform in ("darwin", "win32"):
        return True
    return bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))


def engine_task(gui) -> str:
    """The windowless runner for --no-gui or without a display, else the headed client, which
    follows each configuration's showGame (or --gui)."""
    if gui is False:
        return ":game-headed:runTestbenchHeadless"
    if not has_display():
        if gui:
            print("  no display: --gui ignored, playing without a window")
        return ":game-headed:runTestbenchHeadless"
    return ":game-headed:run"


def checkpoint_dir(config: Path) -> Path:
    """As TestbenchSetupPanel.checkpointDirectory: saves/<game folder>/<configuration name>, the
    game folder being games/<game> for configurations in games/<game>/config or a subfolder."""
    config = config.resolve()
    game = next((d.parent for d in config.parents
                 if d.name == "config" and d.parent.parent == TESTBED / "games"), config.parent.parent)
    return SAVES_DIR / game.name / config.stem


def agent_manifests():
    """Agent ID -> (agent folder, outputDirectories) for every agents/**/agent.json. A generated
    variant maps to the folder and outputDirectories of the agent it runs (variantOf)."""
    found, variants = {}, {}
    for manifest in sorted(AGENTS_DIR.glob("**/agent.json")):
        if len(manifest.relative_to(AGENTS_DIR).parts) > 3:  # as the engine: depth 3
            continue
        try:
            m = json.loads(manifest.read_text())
        except ValueError:
            continue
        if m.get("variantOf"):
            variants[m.get("id")] = m["variantOf"]
        elif m.get("id"):
            found[m["id"]] = (manifest.parent, m.get("outputDirectories") or [])
    found.update({v: found[base] for v, base in variants.items() if v and base in found})
    return found


# ---- agent variants --------------------------------------------------------------------------

VARIANTS_DIR = AGENTS_DIR / "generated-variants"
_written = set()  # variants already written by this run (resolve_agents runs once per map)


def parse_agent(entry):
    """A spec / --agents entry -> (agent ID, settings, chosen variant ID): "id", "id KEY=VALUE ...",
    or {"agent": id, "env": {KEY: VALUE}, "id": variant id}. Settings are None without any."""
    if isinstance(entry, dict):
        if not entry.get("agent"):
            raise SystemExit(f"an agent object needs \"agent\": {entry}")
        env = {str(k): str(v) for k, v in (entry.get("env") or {}).items()}
        return entry["agent"], env or None, entry.get("id")
    if not isinstance(entry, str) or not entry.strip():
        raise SystemExit(f"bad agent entry {entry!r}")
    # "Hard (AI)" has a space but no settings: only split off trailing KEY=VALUE words.
    words = entry.split()
    settings = []
    while words and re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*=\S*", words[-1]):
        settings.insert(0, words.pop())
    if not words:
        raise SystemExit(f"no agent in {entry!r}")
    env = dict(s.split("=", 1) for s in settings)
    return " ".join(words), env or None, None


def variant_id(agent, env):
    parts = [f"{re.sub(r'^TRIPLEA_', '', k)}-{v}" for k, v in env.items()]
    return re.sub(r"[^A-Za-z0-9._-]+", "-", "-".join([agent, *parts])).strip("-")


def resolve_agents(entries, known):
    """Agent IDs to play, writing a manifest for every variant (see the module docstring)."""
    ids, envs = [], {}
    for entry in entries:
        agent, env, chosen = parse_agent(entry)
        if env is None and not chosen:
            ids.append(agent)
            continue
        if agent not in known:
            raise SystemExit(f"unknown agent {agent!r} in {entry!r}: variants need an agent.json agent")
        vid = chosen or variant_id(agent, env or {})
        if vid in envs and envs[vid] != (agent, env):
            raise SystemExit(f"two different variants named {vid!r}; give one an \"id\"")
        if vid in known and vid not in envs and not (VARIANTS_DIR / vid / "agent.json").exists():
            raise SystemExit(f"variant ID {vid!r} is already an agent's ID; choose another \"id\"")
        envs[vid] = (agent, env)
        if vid not in _written:
            write_variant(vid, agent, env or {})
            _written.add(vid)
        known[vid] = known[agent]
        ids.append(vid)
    return ids


def write_variant(vid, agent, env):
    folder, _ = agent_manifests()[agent]
    base = json.loads((folder / "agent.json").read_text())
    entrypoint = str(folder / base["entrypoint"]) if base.get("entrypoint") else ""
    # The engine runs the command in the manifest's folder and fills ${agentDir} / ${entrypoint}
    # from it, so they are filled here with the agent's own folder.
    command = [part.replace("${agentDir}", str(folder)).replace("${entrypoint}", entrypoint)
               for part in base["command"]]
    settings = " ".join(f"{k}={v}" for k, v in env.items())
    manifest = {
        "id": vid,
        "name": f"{base.get('name', agent)} [{settings or vid}]",
        "language": base.get("language"),
        "entrypoint": entrypoint or base.get("entrypoint"),
        "command": ["env", *(f"{k}={v}" for k, v in env.items()), *command],
        "description": f"{agent} with {settings or 'its own settings'} (generated by tournament.py).",
        "outputDirectories": [],
        "variantOf": agent,
        "env": env,
    }
    target = VARIANTS_DIR / vid
    target.mkdir(parents=True, exist_ok=True)
    (target / "agent.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Variant {vid}: {agent} with {settings or 'no settings'}")
    for key in env:
        for script in folder.glob("*.sh"):
            if re.search(rf"^\s*export\s+{re.escape(key)}=(?!\"?\$\{{{re.escape(key)}:-)", script.read_text(), re.M):
                print(f"  warning: {script.relative_to(REPO)} sets {key} itself; the variant's value "
                      f"is overridden")


def clear_outputs(agent_ids):
    manifests = agent_manifests()
    for agent_id in sorted(agent_ids):
        if agent_id not in manifests:
            continue  # built-in TripleA players (e.g. "Hard (AI)") have no outputs
        folder, outputs = manifests[agent_id]
        for name in outputs:
            target = (folder / name).resolve()
            if target == folder.resolve() or folder.resolve() not in target.parents or not target.is_dir():
                continue
            for child in target.iterdir():
                shutil.rmtree(child) if child.is_dir() and not child.is_symlink() else child.unlink()
            print(f"  emptied {target.relative_to(REPO)}")


# ---- report ----------------------------------------------------------------------------------

def read_results(path: Path):
    """The result lines of a games.jsonl; a last line without its newline is one the engine is
    still writing (a parallel batch's), so it is left for the next read."""
    if not path.exists():
        return []
    lines = path.read_text().split("\n")[:-1]
    return [json.loads(line) for line in lines if line.strip()]


_teams = {}


def config_teams(configuration):
    """Faction -> team number from a configuration's "teams" (written by diagonal seating), {} when
    it has none or is gone. A .tournament-*.json copy is looked up by its original's name."""
    if configuration not in _teams:
        path = Path(configuration)
        if not path.exists():
            path = Path(original_config(configuration))
        try:
            teams = json.loads(path.read_text()).get("teams") or []
        except (OSError, ValueError):
            teams = []
        _teams[configuration] = {f: i for i, team in enumerate(teams) for f in team}
    return _teams[configuration]


def opposing(game, a, b):
    """True when factions a and b are on different sides (a faction without an alliance is alone);
    factions of one team of the configuration are on the same side."""
    alliances = game.get("alliances") or {}
    teams = config_teams(game["configuration"])
    if a in teams and teams.get(a) == teams.get(b):
        return False
    return a != b and alliances.get(a, a) != alliances.get(b, b)


def pct(n, d):
    return f"{100.0 * n / d:5.1f}%" if d else "    -"


def table(rows, header):
    widths = [max(len(str(r[i])) for r in [header, *rows]) for i in range(len(header))]
    numeric = lambda c: isinstance(c, (int, float)) or str(c).endswith("%") or str(c).strip() == "-"
    fmt = lambda r: "  ".join(str(c).rjust(w) if numeric(c) else str(c).ljust(w)
                              for c, w in zip(r, widths)).rstrip()
    return [fmt(header), fmt(["-" * w for w in widths]), *map(fmt, rows)]


def duration(seconds):
    seconds = int(round(seconds))
    return f"{seconds // 3600}h {seconds % 3600 // 60:02d}m {seconds % 60:02d}s"


def read_timing(out: Path):
    path = out / "timing.json"
    return json.loads(path.read_text()) if path.exists() else {}


def write_timing(out: Path, **values):
    (out / "timing.json").write_text(json.dumps({**read_timing(out), **values}, indent=2) + "\n")


def game_folder(g):
    """The games/<game> folder a result was played in (recorded by the engine; for older lines
    taken from the configuration's path)."""
    if g.get("gameFolder"):
        return g["gameFolder"]
    path = Path(g["configuration"])
    return next((d.parent.name for d in path.parents if d.name == "config"), "unknown")


def head_to_head(finished, heading):
    """Every pair of agents that played against each other, over both seats."""
    lines = [heading, "-" * len(heading)]
    pairs = defaultdict(lambda: {"games": 0, "wins": defaultdict(int), "draws": 0,
                                 "seats": defaultdict(lambda: [0, 0])})
    for g in finished:
        agents = g["agents"]
        seen = {}
        for fa, a in agents.items():
            for fb, b in agents.items():
                if fa < fb and opposing(g, fa, fb):
                    key = tuple(sorted((a, b)))
                    seen.setdefault(key, set()).update(((fa, a), (fb, b)))
        for key, members in seen.items():
            p = pairs[key]
            p["games"] += 1
            winning = {agent for faction, agent in members if faction in g["winners"]}
            if key[0] == key[1]:  # mirror match: only the seats tell the sides apart
                for faction, _ in members:
                    p["seats"][faction][0] += 1
                    p["seats"][faction][1] += faction in g["winners"]
                if not winning:
                    p["draws"] += 1
                continue
            for agent in key:
                seat = "+".join(sorted(f for f, a in members if a == agent))
                p["seats"][(agent, seat)][0] += 1
                p["seats"][(agent, seat)][1] += agent in winning
            if len(winning) == 1:
                p["wins"][next(iter(winning))] += 1
            else:
                p["draws"] += 1
    for (a, b), p in sorted(pairs.items()):
        n = p["games"]
        if a == b:
            lines.append(f"{a} vs itself  ({n} games)")
            rows = [[f"as {seat}", w, pct(w, gn)] for seat, (gn, w) in sorted(p["seats"].items())]
        else:
            lines.append(f"{a} vs {b}  ({n} games)")
            rows = []
            for agent in (a, b):
                seats = ", ".join(f"as {seat} {w}/{gn}" for (ag, seat), (gn, w)
                                  in sorted(p["seats"].items()) if ag == agent)
                rows.append([agent, p["wins"][agent], pct(p["wins"][agent], n), seats])
        rows.append([NO_WINNER, p["draws"], pct(p["draws"], n)] + ([""] if a != b else []))
        header = ["seat", "wins", "win rate"] if a == b else ["agent", "wins", "win rate", "by seat"]
        lines += ["  " + l for l in table(rows, header)] + [""]
    return lines


def per_agent(finished, heading):
    lines = [heading, "-" * len(heading)]
    totals = defaultdict(lambda: [0, 0])
    for g in finished:
        for agent in set(g["agents"].values()):
            totals[agent][0] += 1
            totals[agent][1] += any(f in g["winners"] for f, a in g["agents"].items() if a == agent)
    rows = [[agent, n, w, pct(w, n)] for agent, (n, w) in sorted(totals.items())]
    lines += table(rows, ["agent", "games", "wins", "win rate"])
    return lines


def report(games, labels, title, timing=None):
    finished = [g for g in games if g.get("status")]
    lines = [title, "=" * len(title), ""]
    unfinished = len(games) - len(finished)
    if unfinished:
        lines += [f"{unfinished} game(s) stopped without a result are left out.", ""]

    # Per configuration.
    lines += ["Per configuration", "-----------------"]
    by_config = defaultdict(list)
    for g in finished:
        by_config[g["configuration"]].append(g)
    for config, gs in by_config.items():
        agents = gs[0]["agents"]
        path = Path(config)
        default = str(path.relative_to(TESTBED)) if TESTBED in path.parents else config
        lines.append(f"{labels.get(config, default)}  ({len(gs)} games)")
        lines.append("  " + ", ".join(f"{f} = {a}" for f, a in agents.items()))
        rows = []
        for faction, agent in agents.items():
            wins = sum(faction in g["winners"] for g in gs)
            rows.append([faction, agent, wins, pct(wins, len(gs))])
        draws = sum(not g["winners"] for g in gs)
        rows.append([NO_WINNER, "", draws, pct(draws, len(gs))])
        rounds = sum(g.get("roundsPlayed", 0) for g in gs) / len(gs)
        seconds = sum(g.get("seconds") or 0 for g in gs)
        lines += ["  " + l for l in table(rows, ["faction", "agent", "wins", "win rate"])]
        lines += [f"  mean rounds {rounds:.1f}, time in games {duration(seconds)} "
                  f"({seconds / len(gs):.1f} s per game)", ""]

    maps = defaultdict(list)
    for g in finished:
        maps[game_folder(g)].append(g)
    if len(maps) <= 1:
        lines += head_to_head(finished, "Head-to-head (all configurations, both seats)")
        lines += per_agent(finished, "Per agent (all games)")
    else:
        for name, gs in maps.items():
            heading = f"Map: {name}  ({len(gs)} games)"
            lines += [heading, "=" * len(heading), ""]
            lines += head_to_head(gs, f"Head-to-head on {name} (both seats)")
            lines += per_agent(gs, f"Per agent on {name}") + [""]
        lines += ["All maps", "========", ""]
        lines += head_to_head(finished, "Head-to-head (all maps, both seats)")
        lines += per_agent(finished, "Per agent (all maps)")

    # Time.
    lines += ["", "Time", "----"]
    timing = timing or {}
    fmt = lambda t: time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t))
    if timing.get("started"):
        lines.append(f"started     {fmt(timing['started'])}")
        if timing.get("finished"):
            lines.append(f"finished    {fmt(timing['finished'])}")
            lines.append(f"total       {duration(timing['finished'] - timing['started'])}")
        else:
            lines.append(f"so far      {duration(time.time() - timing['started'])} (still running)")
    in_games = sum(g.get("seconds") or 0 for g in games)
    if (timing.get("batches") or 1) > 1:
        lines.append(f"in games    {duration(in_games)} over {len(games)} games (summed over "
                     f"{timing['batches']} batches of up to {timing['batchSize']} games in parallel)")
    else:
        lines.append(f"in games    {duration(in_games)} over {len(games)} games "
                     f"(the rest is engine start-up and map loading between configurations)")
    return "\n".join(lines) + "\n"


# ---- main ------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("configs", nargs="*", type=Path,
                    help="testbench JSON configurations, run in order; a folder runs all *.json in it")
    ap.add_argument("--out", type=Path,
                    help="output folder (default planning-agent-testbed/tournaments/<timestamp>)")
    ap.add_argument("--spec", type=Path,
                    help="tournament spec JSON: generate a configuration per pair of its agents and run them")
    ap.add_argument("--agents", nargs="+",
                    help="with --spec: play these agents instead of the spec's; a variant is one quoted "
                         "argument, e.g. \"tripleMind-mcts TRIPLEA_MCTS_ROLLOUT=10\"")
    ap.add_argument("--generate-only", action="store_true",
                    help="with --spec: write the configurations and stop")
    ap.add_argument("--games", type=int,
                    help="games per configuration, written into every configuration before starting")
    ap.add_argument("--first-game", type=int, default=1,
                    help="game number of the tournament's first game (default 1)")
    ap.add_argument("--clear-outputs", action="store_true",
                    help="empty the outputDirectories of all agents in the tournament once, first")
    ap.add_argument("--batch-size", type=int,
                    help="games per batch: the games are cut, in order, into batches of this many, "
                         "which run in parallel, each its games one after another (default: all "
                         "games in one batch, i.e. one game at a time)")
    ap.add_argument("--max-resumes", type=int, default=20,
                    help="times a suspended configuration is restarted to resume it (default 20)")
    ap.add_argument("--gradle-arg", action="append", default=[],
                    help="extra argument for gradlew, e.g. --gradle-arg=--offline (repeatable)")
    ap.add_argument("--gui", action=argparse.BooleanOptionalAction, default=None,
                    help="show (--gui) the game window, or play without any window (--no-gui, "
                         "the default without a display), overriding the configurations' showGame")
    ap.add_argument("--report-only", type=Path, metavar="GAMES_JSONL",
                    help="only write the report for an existing games.jsonl (next to it)")
    args = ap.parse_args()

    if args.report_only:
        results = args.report_only.resolve()
        out = results.parent
        labels = {}
    else:
        if args.spec and args.configs:
            ap.error("give either --spec or configurations, not both")
        if not args.spec and not args.configs:
            ap.error("give configurations, a folder of them, --spec or --report-only")
        if args.agents and not args.spec:
            ap.error("--agents needs --spec")
        if args.spec:
            configs = generate_configs(args.spec.resolve(), args.agents, args.games)
            if args.generate_only:
                return
        else:
            configs = collect_configs(args.configs)
        if not configs:
            ap.error("no configurations found")
        for c in configs:
            if not c.is_file():
                ap.error(f"no such configuration: {c}")
            # checked before renumbering, which would write into the file
            if not args.spec and isinstance(json.loads(c.read_text()).get("agents"), list):
                ap.error(f"{c.name} is a tournament spec, not a configuration: run it with --spec "
                         f"{c}, or several specs with tools/run_tournament.sh <spec> <spec> ...")
        if args.games is not None and args.games < 1:
            ap.error("--games must be at least 1")
        if args.batch_size is not None and args.batch_size < 1:
            ap.error("--batch-size must be at least 1")
        print("Numbering the configurations' games")
        renumber_configs(configs, args.games, args.first_game)
        total = sum(int(json.loads(c.read_text()).get("games", 1)) for c in configs)
        batches = plan_batches(configs, args.batch_size or total)
        if len(batches) > 1 and missing_maps(configs):
            ap.error(f"the map of {', '.join(missing_maps(configs))} is not downloaded yet, and parallel "
                     f"batches would all download it at once: play one game of it first (e.g. this "
                     f"tournament with --games 1 and without --batch-size)")
        out = (args.out or TOURNAMENTS_DIR / time.strftime("%Y-%m-%d_%H-%M-%S")).resolve()
        out.mkdir(parents=True, exist_ok=True)
        results = out / "games.jsonl"
        (out / "configs.txt").write_text("".join(f"{c}\n" for c in configs))
        agent_ids = {a for c in configs for a in json.loads(c.read_text()).get("agents", {}).values()}
        if args.clear_outputs:
            print("Clearing agent outputs")
            clear_outputs(agent_ids)
        labels = {}
        started = time.time()
        write_timing(out, started=started, finished=None, batches=len(batches),
                     batchSize=args.batch_size or total)
        if len(batches) > 1:
            print(f"\n{total} games in {len(batches)} batches of up to {args.batch_size}, played in "
                  f"parallel; engine output in {out / 'batches'}/<batch>.log")
            run_batches(batches, out, args, labels)
        else:
            for i, config in enumerate(configs, 1):
                print(f"\n[{i}/{len(configs)}] {config.relative_to(REPO) if REPO in config.parents else config}",
                      flush=True)
                run_path = run_config(config, results, args.max_resumes, args.gradle_arg, out / "agents", args.gui)
                labels[str(run_path)] = labels[str(config)] = str(
                    config.relative_to(TESTBED) if TESTBED in config.parents else config)
                # The report is rewritten after every configuration, so a stopped tournament keeps it.
                text = report(read_results(results), labels, f"Tournament {out.name}", read_timing(out))
                (out / "results.txt").write_text(text)
        finished = time.time()
        write_timing(out, finished=finished)
        print(f"\nTournament finished in {duration(finished - started)}")

    games = read_results(results)
    for g in games:  # configurations run under a .tournament-*.json copy are shown by their name
        if g["configuration"] not in labels and original_config(g["configuration"]) != g["configuration"]:
            labels[g["configuration"]] = original_config(g["configuration"])
    text = report(games, labels, f"Tournament {out.name}", read_timing(out))
    (out / "results.txt").write_text(text)
    print(text)
    print(f"Report: {out / 'results.txt'}\nGames:  {results}\nAgent metrics: {out / 'agents'}")


if __name__ == "__main__":
    sys.exit(main())
