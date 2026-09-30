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

The headed client setup UI lives in
`game-app/game-headed/src/main/java/games/strategy/engine/framework/startup/ui/planningagenttestbed`.

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

The `agents` object maps faction names from the map to player type labels available in the client.
Change those values to compare agents. `roundLimit` ends each game at that round without assigning
a winner. When a game ends, the testbench reloads a fresh copy of the map and automatically starts
the next game until it has run the configured `games` count.
The testbench skips TripleA's end-of-game "continue playing?" prompt so each completed game can
advance directly to the next one. Results are appended to `planning-agent-testbed/logs/game-results.txt`
with the game number, faction-to-agent assignments, winner (or no winner), and elapsed time. The
log directory is present in the repository; generated text logs are ignored by Git.

Use **Save Configuration…** and **Load Configuration…** in the run setup to save or reuse JSON
configurations.
Changes in the window are used in memory when you start the game. They do not modify the loaded
JSON automatically; choose **Save JSON…** to persist them.

`aiMovePauseMs` controls the delay after AI movement and `aiCombatStepPauseMs` controls the delay
between AI combat steps. Set both to `0` for faster runs, or increase them to watch the game. They
are applied before the game starts, so the values stay fixed throughout that game. Omitting either
field uses the engine default (300 ms for moves, 1000 ms for combat steps).
