# Planning Agent Testbench Flow

This is the maintained description of the order of operations from client startup through one
simulation batch. The testbench reuses TripleA's game data, turn sequence, AI hooks, and delegates;
the experiment UI and external-agent bridge are additions in the main repository.

## Startup and configuration

1. Start the headed client from the repository root with `./gradlew :game-headed:run`.
2. The client presents **Planning Agent Testbench** and **Use the game engine**. The first enters
   the testbench menu; the second opens the unchanged regular setup flow.
3. **Run Simulations** opens the editable experiment setup. It loads a supplied JSON profile when
   configured, otherwise the default Capture The Flag profile. The profile selects a map, game XML,
   aliases/download URL, game count, round limit, AI pause durations, faction-to-agent assignments,
   and observation-display preferences.
4. `TestbenchMapRepository` looks for the XML in the selected game's repository-local map folder.
   It can use a `mapXml` relative path, find an installed map by configured name/aliases, download
   the configured ZIP, or resolve a download URL through TripleA's map listing. Downloaded maps stay
   under `planning-agent-testbed/games/<game>/map`.
5. Loading the XML populates the faction assignment rows. Each faction also gets a **Visualize
   observation** checkbox. The global **Visualize observations** option controls whether to open the
   observation window at all. Older JSON profiles default to showing all factions.

## Starting a batch and running turns

1. On the first launcher request in a batch, the testbench clears its generated agent logs and
   resets `planning-agent-testbed/logs/game-results.txt`. It sets the configured round cap and AI
   pause durations, then asks `LocalLauncher` to start the selected map and faction players.
2. The simulation number is `completedGames + 1`, starting at 1. It is passed into the selected
   player's Java AI adapter and included in each agent's `game_start` message.
3. If enabled, `StateObservationWindow` opens one window with a tab per selected faction. Its JSON
   panes refresh on engine state changes. The current provider marks snapshots `FULL` and supplies
   the same full map view to every faction.
4. `TestbenchAgentRegistry` discovers `agent.json` manifests under `agents/` and exposes each
   manifest as a faction choice. Every choice instantiates the same `ExternalAgentPlayer`; the
   selected manifest supplies its process launch command. For example, the Simple Infantry
   manifest starts its persistent C++ process through `run-agent.sh`. At game
   initialization it sends a newline-delimited JSON `game_start` message containing the common map
   XML, initial state, faction, protocol version, request ID, and simulation number. The current
   `FullPlayerGameXmlProvider` gives each faction the common XML unchanged; this is the hook for a
   future per-faction XML visibility filter.
5. On the assigned faction's decision phases, the Java adapter sends a `turn_request` with the
   current JSON observation. The external process returns one JSON action line. The generic Java
   adapter translates purchase and movement responses into TripleA delegate calls. The delegates validate
   the submitted actions. The battle response asks TripleA to fight available battles. The place
   phase currently makes no agent request and falls back to TripleA's built-in `WeakAi` placement.
6. The observation snapshot includes game/map/round/step, active player, game properties, player
   resources and technologies, territory ownership/connections/production/objectives, and unit
   ownership, damage, movement, combat stats, and capabilities. It is a fixed v1 DTO rather than an
   automatic serialization of every internal engine object.

## Game completion and source locations

1. The testbench round limit ends a game without a winner when reached. The testbench also bypasses
   the normal continue-playing prompt for completed batch games.
2. It appends result, faction assignments, and elapsed time to `game-results.txt`, hides the
   observation window, reloads a fresh copy of the map, and launches the next numbered game. When
   the batch finishes, it returns to the testbench menu.
3. Each C++ faction writes `agent-<faction>-game-<number>.txt`. Its first line is the simulation
   number, followed by received phase requests, rounds, actions, and protocol errors.

The related main-repository additions are in:

- `game-app/game-headed/.../MetaSetupPanel.java` and `.../planningagenttestbed/TestbenchModePanel.java`:
  initial mode choices.
- `.../planningagenttestbed/TestbenchSetupPanel.java` and `TestbenchMapRepository.java`: JSON,
  map resolution, launch setup, observations preferences, repeated games, and logs.
- `.../planningagenttestbed/TestbenchAgentRegistry.java`, `ExternalAgentPlayer.java`, and each
  `agents/<id>/agent.json`: dynamic player selection plus the shared process/action bridge and
  per-agent launch metadata.
- `.../planningagenttestbed/StateObservation.java`, `StateObservationProvider.java`,
  `FullStateObservationProvider.java`, and `StateObservationWindow.java`: observation model,
  generation, and visualization.
- `game-app/game-core/.../EndRoundDelegate.java`: configured round-limit handling.

For how to implement an agent, see [AGENT_DEVELOPER_GUIDE.md](AGENT_DEVELOPER_GUIDE.md).
