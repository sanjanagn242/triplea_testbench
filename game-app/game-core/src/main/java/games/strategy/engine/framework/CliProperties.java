package games.strategy.engine.framework;

import org.jetbrains.annotations.NonNls;

/** A collection of all CLI related constants. */
public class CliProperties {
  @NonNls public static final String TRIPLEA_GAME = "triplea.game";
  @NonNls public static final String TRIPLEA_TESTBENCH_CONFIG = "triplea.testbench.config";
  // true/false: skip the testbench menus and start the batch in triplea.testbench.config
  @NonNls public static final String TRIPLEA_TESTBENCH_AUTOSTART = "triplea.testbench.autostart";
  // file path: append one JSON line per completed testbench game (used by tools/tournament.py)
  @NonNls public static final String TRIPLEA_TESTBENCH_RESULTS = "triplea.testbench.results";
  // true/false: show the game window during testbench games; overrides the config's showGame
  @NonNls public static final String TRIPLEA_TESTBENCH_GUI = "triplea.testbench.gui";
  // folder path: agents write their outputs (metrics) to <folder>/<agent id>/ instead of their own
  // folder; passed to each agent process as TRIPLEA_TESTBENCH_OUTPUT_DIR (used by tournament.py)
  @NonNls public static final String TRIPLEA_TESTBENCH_OUTPUTS = "triplea.testbench.outputs";
  // true/false: keep earlier agent logs (planning-agent-testbed/logs/agent-*.txt) when a run starts,
  // so runs in parallel do not delete each other's (used by tournament.py with several batches)
  @NonNls public static final String TRIPLEA_TESTBENCH_KEEP_LOGS = "triplea.testbench.keepLogs";
  @NonNls public static final String TRIPLEA_SERVER = "triplea.server"; // true/false
  @NonNls public static final String TRIPLEA_CLIENT = "triplea.client"; // true/false
  @NonNls public static final String TRIPLEA_HOST = "triplea.host";
  @NonNls public static final String TRIPLEA_PORT = "triplea.port";
  @NonNls public static final String TRIPLEA_NAME = "triplea.name";
  @NonNls public static final String TRIPLEA_START = "triplea.start";
  @NonNls public static final String SERVER_PASSWORD = "triplea.server.password";
  @NonNls public static final String LOBBY_URI = "triplea.lobby.uri";
  @NonNls public static final String LOBBY_GAME_COMMENTS = "triplea.lobby.game.comments";
  @NonNls public static final String TRIPLEA_MAP_DOWNLOAD = "triplea.map.download";
  @NonNls public static final String TRIPLEA_MAP_DOWNLOAD_PREFIX = "triplea:";

  @NonNls public static final String TRIPLEA_START_LOCAL = "local";
  @NonNls public static final String TRIPLEA_START_PBF = "pbf";
  @NonNls public static final String TRIPLEA_START_PBEM = "pbem";
  @NonNls public static final String TRIPLEA_START_LOBBY = "lobby";

  private CliProperties() {}
}
