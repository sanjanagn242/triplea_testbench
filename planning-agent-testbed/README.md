# Planning Agent Testbed

This directory is the home for the planning agent testbed's game maps, experiment configurations,
and notes. Each game has one map folder and one config folder, so you can keep multiple experiment
configurations for that map together:

```text
planning-agent-testbed/games/
  capture-the-flag/
    map/       # Downloaded map package folders go here
    config/    # One or more JSON experiment configurations
  minimap/
    map/
    config/
```

The maintained technical notes are [the end-to-end testbench flow](docs/TESTBENCH_FLOW.md) and
[the agent developer guide](docs/AGENT_DEVELOPER_GUIDE.md). Python dependencies are aggregated in
`planning-agent-testbed/requirements.txt`; each Python agent keeps its own manifest alongside its
code. The Simple Infantry C++ agent has no Python package dependencies.

The headed client setup UI lives in
`game-app/game-headed/src/main/java/games/strategy/engine/framework/startup/ui/planningagenttestbed`.

## Requirements

- **Java Development Kit 25** is required to build and run this repository. A JRE alone is not
  sufficient. Check the active Java with `java --version`; Gradle should also report Java 25 with
  `./gradlew --version`. If multiple Java versions are installed, set `JAVA_HOME` and `PATH` to the
  JDK 25 installation before running Gradle.
- Use the repository's Gradle wrapper (`./gradlew`). It downloads the configured Gradle 9.7.1
  distribution and project dependencies as needed; a separate Gradle installation is not required.
  The first build needs network access to download these dependencies.
- The Simple Infantry agent requires **g++ with C++20 support**. If the compiler is not on the
  headed client's `PATH`, set `TRIPLEA_TESTBENCH_CXX`, for example
  `TRIPLEA_TESTBENCH_CXX=/usr/bin/g++ ./gradlew :game-headed:run`.
- Java dependencies are managed by Gradle build files and the version catalog; there is no Java
  `requirements.txt`. Python dependencies are listed in this folder's `requirements.txt` and in
  each Python agent's own manifest. Simple Infantry is C++20 and has no Python dependency.

Install any Python agent dependencies with:

```bash
python3 -m pip install -r planning-agent-testbed/requirements.txt
```

## Start with Capture The Flag

Run this configuration from the repository root:

```bash
./gradlew :game-headed:run --args="triplea.testbench.config=$PWD/planning-agent-testbed/games/capture-the-flag/config/capture-the-flag.json"
```

The client first shows **Planning Agent Testbench** and **Use the game engine**. Choose the
testbench option, then **Run Simulations**, to load the supplied JSON in its editable setup window.
**Evaluate Simulations** is a placeholder for the future metrics viewer. Choose the game engine
option to continue to the regular TripleA setup.

The configuration gives the map's canonical name, game name, and common aliases. The testbench
checks the selected game's `map` folder first. Place manually downloaded map packages inside it,
for example `planning-agent-testbed/games/capture-the-flag/map/capture_the_flag/`. If the map is
missing, it looks for a matching entry in TripleA's map download listing, downloads it, and extracts
it into that game's repository-local map folder.
When `downloadUrl` is set in the `map` object, it downloads directly from that ZIP URL without
contacting the map listing service. The starter Capture The Flag configuration uses this, so it
does not depend on a local map-listing server. Without a `downloadUrl`, it looks up a matching alias
in the map listing. The `mapXml` field is also supported for an XML path relative to the JSON file;
this is useful for maps stored within the repository.

Downloaded map files are kept under `planning-agent-testbed/games/<game>/map` and ignored by Git.
They are not saved in the per-user `~/Documents/triplea` map folder. JSON configurations for that
map belong under `planning-agent-testbed/games/<game>/config`; the UI opens that folder when saving
and opens the `games` folder when loading configurations.

The `agents` object maps faction names from the map to stable agent IDs (for example,
`"Russians": "simple-infantry"`). The run window shows the display name from each agent's manifest.
Change those IDs to compare agents. `roundLimit` ends each game at that round without assigning
a winner. When a game ends, the testbench reloads a fresh copy of the map and automatically starts
the next game until it has run the configured `games` count.
The testbench skips TripleA's end-of-game "continue playing?" prompt so each completed game can
advance directly to the next one. Results are appended to `planning-agent-testbed/logs/game-results.txt`
with the game number, faction-to-agent assignments, winner (or no winner), and elapsed time. The
log directory is present in the repository; generated text logs are ignored by Git. Generated logs
are refreshed at the start of each simulation batch, and each C++ faction writes a separate
`agent-<faction>-game-<number>.txt` file whose first line gives its game number.

Use **Save Configuration…** and **Load Configuration…** in the run setup to save or reuse JSON
configurations.
Changes in the window are used in memory when you start the game. They do not modify the loaded
JSON automatically; choose **Save JSON…** to persist them.

Live JSON observations appear together in one window with a tab for each enabled faction. Set
`showObservations` to `false` to hide the observation window entirely; use `observationPlayers` to
hide selected faction tabs while leaving the others visible. The same global and per-faction
controls are available in the run setup window. Older configurations default to showing all tabs.

`aiMovePauseMs` controls the delay after AI movement and `aiCombatStepPauseMs` controls the delay
between AI combat steps. Set both to `0` for faster runs, or increase them to watch the game. They
are applied before the game starts, so the values stay fixed throughout that game. Omitting either
field uses the engine default (300 ms for moves, 1000 ms for combat steps).

## Planning agents

External agents live in `planning-agent-testbed/agents/<agent-id>/`. Each folder's `agent.json`
defines its stable ID, display name, language, entrypoint, and launch command. The registry discovers
these files and routes every choice through the single generic `ExternalAgentPlayer`; adding an
agent does not require an algorithm-specific Java class or engine label. The first agent,
**Simple Infantry (C++)**, is available in the faction assignment dropdown. The engine starts one
persistent process for each assigned faction using the manifest command. At game start it sends a `game_start`
message containing the map XML for that faction and its initial JSON state. Before each decision
phase for that faction, it sends a `turn_request` containing the current observation; the process
returns one JSON action line. This uses newline-delimited JSON over standard input/output, so an
agent in another language can implement the same protocol without Python bindings or engine
dependencies. A C++20 compiler must be installed when starting the client; its launcher compiles the
agent source when it changes.

The simple agent reads infantry production rules, costs, movement, target capitals, and impassable
territories from the supplied XML and observation. It buys infantry, moves toward enemy objectives,
and asks TripleA to resolve its available battles. Placement currently uses TripleA's built-in
`WeakAi` behavior without sending a placement request to the agent. Proposed moves and purchases are
still checked by the engine delegates before application. Agents should write only protocol
responses to standard output; diagnostics belong on standard error.

`PlayerGameXmlProvider` is the boundary for player-specific rules. The current provider supplies
the common XML unchanged to each faction process, which is appropriate for the current fully
observable setup. A filtered provider can return a different XML string for each player when the
map's visibility rules define how rule and property visibility is represented. The JSON observation
provider is also per-player and can be replaced with a filtered implementation as partial
observation is introduced.

Two ready-to-load configurations are provided:

- `games/capture-the-flag/config/capture-the-flag-simple-vs-easy.json` runs the C++ agent as
  Russians against the Easy AI as Italians; Germans and Chinese use Does Nothing.
- `games/capture-the-flag/config/capture-the-flag-four-simple-agents.json` assigns the C++ agent
  to all four factions.
