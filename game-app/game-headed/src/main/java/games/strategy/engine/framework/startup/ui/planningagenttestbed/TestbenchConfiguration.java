package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;

/** A testbench JSON configuration: the map, the agent of each faction and the run's length. */
record TestbenchConfiguration(
    MapReference map,
    String mapXml,
    int games,
    Integer firstGameNumber,
    int roundLimit,
    Integer aiMovePauseMs,
    Integer aiCombatStepPauseMs,
    Map<String, String> agents,
    Boolean showGame,
    Boolean showObservations,
    Map<String, Boolean> observationPlayers,
    Boolean clearAgentOutputs) {
  static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  static final int DEFAULT_AI_MOVE_PAUSE_MS = 300;
  static final int DEFAULT_AI_COMBAT_STEP_PAUSE_MS = 1000;
  static final int MAX_AI_PAUSE_MS = 3000;

  /** Reads the configuration in path; it must name a map or a mapXml path. */
  static TestbenchConfiguration read(final Path path) throws IOException {
    final TestbenchConfiguration config =
        GSON.fromJson(Files.readString(path), TestbenchConfiguration.class);
    if (config == null
        || (config.map == null && (config.mapXml == null || config.mapXml.isBlank()))) {
      throw new IllegalArgumentException("JSON must contain a map name or mapXml path.");
    }
    return config;
  }

  /** The game XML this configuration (stored at path) plays, downloading its map if needed. */
  Path resolveMap(final Path path) throws IOException {
    return TestbenchMapRepository.resolveMap(
        path,
        mapXml,
        map == null ? null : map.name,
        map == null ? null : map.gameName,
        map == null ? List.of() : map.aliases,
        map == null ? null : map.downloadUrl);
  }

  int gameCount() {
    return Math.max(1, games);
  }

  int firstGame() {
    return firstGameNumber == null ? 1 : Math.max(1, firstGameNumber);
  }

  int rounds() {
    return Math.max(1, roundLimit);
  }

  int movePauseMs() {
    return clampPause(aiMovePauseMs, DEFAULT_AI_MOVE_PAUSE_MS);
  }

  int combatStepPauseMs() {
    return clampPause(aiCombatStepPauseMs, DEFAULT_AI_COMBAT_STEP_PAUSE_MS);
  }

  private static int clampPause(final Integer value, final int defaultValue) {
    return value == null ? defaultValue : Math.max(0, Math.min(MAX_AI_PAUSE_MS, value));
  }

  record MapReference(String name, String gameName, List<String> aliases, String downloadUrl) {}
}
