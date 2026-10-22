package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import com.google.gson.Gson;
import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.framework.CliProperties;
import games.strategy.engine.framework.GameShutdownRegistry;
import games.strategy.engine.framework.IGame;
import games.strategy.engine.framework.ServerGame;
import games.strategy.engine.framework.map.file.system.loader.InstalledMapsListing;
import games.strategy.engine.framework.startup.launcher.ILauncher;
import games.strategy.engine.framework.startup.launcher.LaunchAction;
import games.strategy.engine.framework.startup.launcher.LocalLauncher;
import games.strategy.engine.framework.startup.launcher.MapNotFoundException;
import games.strategy.engine.framework.startup.launcher.local.PlayerCountrySelection;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.framework.startup.ui.panels.main.game.selector.GameSelectorModel;
import games.strategy.triplea.ResourceLoader;
import games.strategy.triplea.delegate.EndRoundDelegate;
import games.strategy.triplea.settings.ClientSetting;
import games.strategy.triplea.ui.display.HeadlessDisplay;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.Duration;
import java.time.Instant;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import org.triplea.java.ThreadRunner;
import org.triplea.sound.HeadlessSoundChannel;

/**
 * One run of testbench games: numbering, checkpoints of suspended games, the results file and the
 * launch of each game. Free of Swing, so the setup panel and the headless runner share it; a {@link
 * Host} supplies what differs between them (the game window, threads and error reporting).
 */
final class TestbenchRun {
  static final String CHECKPOINT_SAVE = "checkpoint.tsvg";
  static final String CHECKPOINT_INFO = "checkpoint.json";

  /** What the setup panel or the headless runner does around the games of a run. */
  interface Host {
    /**
     * Launch action of one game with these player types. Its startGame must pass the game to
     * gameStarted and give the game a display, sound channel and resource loader (for example with
     * {@link TestbenchRun#setUpWithoutWindow}).
     */
    LaunchAction launchAction(
        Collection<PlayerTypes.Type> playerTypes, Consumer<IGame> gameStarted);

    /** Called on the thread that ended a game; runs nextStep there or on another thread. */
    void afterGame(Runnable nextStep);

    /** All planned games have been played. */
    void onRunFinished();

    /** An agent suspended the run; message tells where its checkpoint is. */
    void onRunSuspended(String message);

    /** The run cannot continue. */
    void onRunFailed(String message);
  }

  /**
   * The run's settings; agentLabels maps every faction to the label of its player type (as shown in
   * the setup panel, see {@link TestbenchAgentRegistry#selectionForId}).
   */
  record Settings(
      int games,
      int firstGameNumber,
      int roundLimit,
      int aiMovePauseMs,
      int aiCombatStepPauseMs,
      Map<String, String> agentLabels,
      boolean clearAgentOutputs) {}

  /** A suspended game: the run's progress when it was saved to checkpoint.tsvg. */
  record Checkpoint(
      String configuration,
      int completedGames,
      int plannedGames,
      Integer firstGameNumber,
      int round,
      long elapsedMillis,
      String savedAt) {
    /** Number of the run's first game; 1 for checkpoints saved before it was recorded. */
    int firstGame() {
      return firstGameNumber == null ? 1 : firstGameNumber;
    }
  }

  /**
   * One line of the triplea.testbench.results file; agents and alliances map each faction to its
   * agent ID and (first) alliance.
   */
  private record GameResultLine(
      String configuration,
      String gameFolder,
      int game,
      Map<String, String> agents,
      Map<String, String> alliances,
      List<String> winners,
      int roundsPlayed,
      String status,
      double seconds) {}

  private final GameSelectorModel gameSelectorModel;
  @Nullable private final Path configurationFile;
  private final Path mapXml;
  private final Settings settings;
  private final Host host;
  private final int plannedGames;
  // Number of the run's first game; game n of the run is numbered firstGame + n - 1.
  private final int firstGame;
  // Start of the run (also of a resumed one); agents measure their run time limit from it.
  private final Instant runStartedAt = Instant.now();
  private int completedGames;
  private Instant currentGameStartedAt;
  // Time the current game had already run before it was last suspended.
  private long previouslyElapsedMillis;
  private volatile IGame currentGame;
  private volatile boolean currentGameSuspended;

  /**
   * A run of the map in mapXml (already loaded into gameSelectorModel). With a checkpoint, whose
   * save game must already be loaded instead, the run continues the suspended one: it keeps its
   * logs and agent outputs and continues its numbering. Otherwise earlier logs and checkpoints are
   * deleted (and the agents' outputs, when settings ask for it).
   */
  TestbenchRun(
      final GameSelectorModel gameSelectorModel,
      @Nullable final Path configurationFile,
      final Path mapXml,
      final Settings settings,
      final Host host,
      @Nullable final Checkpoint resumed) {
    this.gameSelectorModel = gameSelectorModel;
    this.configurationFile = configurationFile;
    this.mapXml = mapXml;
    this.settings = settings;
    this.host = host;
    if (resumed != null) {
      completedGames = resumed.completedGames();
      plannedGames = Math.max(settings.games(), completedGames + 1);
      firstGame = resumed.firstGame();
      previouslyElapsedMillis = resumed.elapsedMillis();
    } else {
      deleteAgentLogs();
      if (settings.clearAgentOutputs()) {
        clearAssignedAgentOutputs();
      }
      deleteCheckpoint(configurationFile);
      plannedGames = settings.games();
      firstGame = settings.firstGameNumber();
      completedGames = 0;
      previouslyElapsedMillis = 0;
    }
  }

  /** Label of the player type a faction gets when a configuration does not assign one. */
  static String defaultAgent(final GamePlayer player) {
    return player.isDefaultTypeDoesNothing()
        ? PlayerTypes.DOES_NOTHING_PLAYER_LABEL
        : PlayerTypes.PRO_AI.getLabel();
  }

  /**
   * What the game window would set up, minus the window: the map's resources (the engine reads
   * them, e.g. for politics texts), no display and no sound.
   */
  static void setUpWithoutWindow(final IGame game) {
    final String mapName = game.getData().getMapName();
    game.setResourceLoader(
        new ResourceLoader(
            InstalledMapsListing.searchAllMapsForMapName(mapName)
                .orElseThrow(() -> new MapNotFoundException(mapName))));
    game.setDisplay(new HeadlessDisplay());
    game.setSoundChannel(new HeadlessSoundChannel());
  }

  /** Launcher of the run's next game, for the map or save game currently loaded. */
  ILauncher nextLauncher() {
    gameSelectorModel
        .getGameData()
        .getProperties()
        .set(EndRoundDelegate.TESTBENCH_ROUND_LIMIT_PROPERTY, settings.roundLimit());
    currentGameStartedAt = Instant.now();
    final int previousMovePause = ClientSetting.aiMovePauseDuration.getValueOrThrow();
    final int previousCombatPause = ClientSetting.aiCombatStepPauseDuration.getValueOrThrow();
    ClientSetting.aiMovePauseDuration.setValue(settings.aiMovePauseMs());
    ClientSetting.aiCombatStepPauseDuration.setValue(settings.aiCombatStepPauseMs());
    GameShutdownRegistry.registerShutdownAction(
        () -> {
          ClientSetting.aiMovePauseDuration.setValue(previousMovePause);
          ClientSetting.aiCombatStepPauseDuration.setValue(previousCombatPause);
        });
    final List<PlayerTypes.Type> playerTypes =
        TestbenchAgentRegistry.getPlayerTypes(
            mapXml,
            firstGame + completedGames,
            runStartedAt.toEpochMilli(),
            this::suspendCurrentGame,
            Map.of(
                "TRIPLEA_TESTBENCH_AGENTS",
                new Gson().toJson(agentIds()),
                "TRIPLEA_TESTBENCH_GAME",
                gameFolderName()));
    final PlayerTypes types = new PlayerTypes(playerTypes);
    final List<PlayerCountrySelection> players =
        settings.agentLabels().entrySet().stream()
            .map(
                entry ->
                    (PlayerCountrySelection)
                        new PlayerCountrySelection() {
                          @Override
                          public String getPlayerName() {
                            return entry.getKey();
                          }

                          @Override
                          public PlayerTypes.Type getPlayerType() {
                            return types.fromLabel(entry.getValue());
                          }

                          @Override
                          public boolean isPlayerEnabled() {
                            return true;
                          }
                        })
            .toList();
    return LocalLauncher.create(
        gameSelectorModel,
        players,
        null,
        host.launchAction(playerTypes, game -> currentGame = game),
        this::onGameCompleted);
  }

  /**
   * Called on the game thread when an agent asks to suspend: saves the game as this configuration's
   * checkpoint and stops it. Returns false, leaving the game running, if the save failed.
   */
  private boolean suspendCurrentGame() {
    final IGame game = currentGame;
    if (!(game instanceof ServerGame serverGame)) {
      return false;
    }
    final Path directory = checkpointDirectory(configurationFile);
    final Path saveFile = directory.resolve(CHECKPOINT_SAVE);
    final Path newSaveFile = directory.resolve("checkpoint-new.tsvg");
    try {
      Files.createDirectories(directory);
      Files.deleteIfExists(newSaveFile);
      // The engine logs save failures instead of throwing; a missing file means it failed.
      serverGame.saveGame(newSaveFile);
      if (!Files.isRegularFile(newSaveFile)) {
        return false;
      }
      Files.move(newSaveFile, saveFile, StandardCopyOption.REPLACE_EXISTING);
      final Checkpoint checkpoint =
          new Checkpoint(
              configurationKey(configurationFile),
              completedGames,
              plannedGames,
              firstGame,
              game.getData().getSequence().getRound(),
              currentGameElapsedMillis(),
              Instant.now().toString());
      Files.writeString(
          directory.resolve(CHECKPOINT_INFO), TestbenchConfiguration.GSON.toJson(checkpoint));
    } catch (final IOException | RuntimeException e) {
      System.err.println("Could not save the suspended testbench game: " + e.getMessage());
      return false;
    }
    currentGameSuspended = true;
    serverGame.stopGame();
    return true;
  }

  private void onGameCompleted() {
    GameShutdownRegistry.runShutdownActions();
    final IGame finishedGame = currentGame;
    currentGame = null;
    if (currentGameSuspended) {
      currentGameSuspended = false;
      host.onRunSuspended(
          String.format(
              java.util.Locale.ROOT,
              "Game %d/%d suspended after %.1f seconds; run the same configuration again to"
                  + " resume.%nCheckpoint: %s%n",
              firstGame + completedGames,
              firstGame + plannedGames - 1,
              currentGameElapsedMillis() / 1000.0,
              checkpointDirectory(configurationFile).resolve(CHECKPOINT_SAVE)));
      return;
    }
    if (finishedGame != null) {
      appendGameResult(finishedGame.getData(), firstGame + completedGames);
    }
    completedGames++;
    previouslyElapsedMillis = 0;
    deleteCheckpoint(configurationFile);
    host.afterGame(this::startNextGameOrFinish);
  }

  private void startNextGameOrFinish() {
    if (!Files.isRegularFile(mapXml) || !gameSelectorModel.loadMapForTestbench(mapXml)) {
      host.onRunFailed("Could not reload the testbench map after the game ended: " + mapXml);
    } else if (completedGames < plannedGames) {
      final ILauncher launcher = nextLauncher();
      ThreadRunner.runInNewThread(launcher::launch);
    } else {
      host.onRunFinished();
    }
  }

  /**
   * Appends one JSON line describing the finished game to the file named by the {@code
   * triplea.testbench.results} property, when it is set. A blank status means the game stopped
   * without reaching a result.
   */
  private void appendGameResult(final GameData data, final int gameNumber) {
    final String resultsFile = System.getProperty(CliProperties.TRIPLEA_TESTBENCH_RESULTS, "");
    if (resultsFile.isBlank() || data == null) {
      return;
    }
    final var properties = data.getProperties();
    final List<String> winners =
        java.util.Arrays.stream(
                properties.get(EndRoundDelegate.TESTBENCH_WINNERS_PROPERTY, "").split(","))
            .map(String::strip)
            .filter(name -> !name.isEmpty())
            .toList();
    final Map<String, String> alliances = new LinkedHashMap<>();
    for (final GamePlayer player : data.getPlayerList().getPlayers()) {
      data.getAllianceTracker().getAlliancesPlayerIsIn(player).stream()
          .sorted()
          .findFirst()
          .ifPresent(alliance -> alliances.put(player.getName(), alliance));
    }
    final GameResultLine line =
        new GameResultLine(
            configurationKey(configurationFile),
            gameFolderName(),
            gameNumber,
            agentIds(),
            alliances,
            winners,
            properties.get(
                EndRoundDelegate.TESTBENCH_ROUNDS_PLAYED_PROPERTY, data.getSequence().getRound()),
            properties.get(EndRoundDelegate.TESTBENCH_STATUS_PROPERTY, ""),
            currentGameElapsedMillis() / 1000.0);
    try {
      final Path path = Path.of(resultsFile);
      if (path.getParent() != null) {
        Files.createDirectories(path.getParent());
      }
      Files.writeString(
          path,
          new Gson().toJson(line) + System.lineSeparator(),
          java.nio.file.StandardOpenOption.CREATE,
          java.nio.file.StandardOpenOption.APPEND);
    } catch (final IOException | RuntimeException e) {
      System.err.println("Could not write the testbench game result: " + e.getMessage());
    }
  }

  private long currentGameElapsedMillis() {
    return previouslyElapsedMillis
        + Duration.between(currentGameStartedAt, Instant.now()).toMillis();
  }

  /** Faction -> stable agent ID (as written in configurations) for this run's assignments. */
  private Map<String, String> agentIds() {
    final Map<String, String> result = new LinkedHashMap<>();
    settings
        .agentLabels()
        .forEach((name, label) -> result.put(name, TestbenchAgentRegistry.idForSelection(label)));
    return result;
  }

  private String gameFolderName() {
    return gameFolderName(configurationFile, mapXml);
  }

  /**
   * Name of the games/(game) folder being played (e.g. capture-the-flag-diag): the configuration's
   * game folder, else the one holding the map; empty when neither is under games/.
   */
  static String gameFolderName(@Nullable final Path configurationFile, @Nullable final Path map) {
    final Path gamesDirectory =
        TestbenchMapRepository.gamesDirectory().toAbsolutePath().normalize();
    final Optional<Path> fromMap =
        Optional.ofNullable(map)
            .map(path -> path.toAbsolutePath().normalize())
            .flatMap(
                path -> {
                  for (Path p = path; p != null; p = p.getParent()) {
                    if (gamesDirectory.equals(p.getParent())) {
                      return Optional.of(p);
                    }
                  }
                  return Optional.empty();
                });
    return TestbenchMapRepository.gameDirectoryOf(configurationFile)
        .or(() -> fromMap)
        .map(directory -> directory.getFileName().toString())
        .orElse("");
  }

  /**
   * Folder of a configuration's suspended game: planning-agent-testbed/saves/(game)/(config), where
   * (game) is the folder holding the configuration's config folder.
   */
  static Path checkpointDirectory(@Nullable final Path configurationFile) {
    final Path saves =
        TestbenchMapRepository.repositoryRoot().resolve("planning-agent-testbed/saves");
    if (configurationFile == null) {
      return saves.resolve("unsaved-configuration");
    }
    final String configName =
        configurationFile.getFileName().toString().replaceFirst("\\.json$", "");
    final Path gameDirectory =
        TestbenchMapRepository.gameDirectoryOf(configurationFile)
            .orElse(configurationFile.getParent().getParent());
    return gameDirectory == null || gameDirectory.getFileName() == null
        ? saves.resolve(configName)
        : saves.resolve(gameDirectory.getFileName().toString()).resolve(configName);
  }

  static String configurationKey(@Nullable final Path configurationFile) {
    return configurationFile == null
        ? ""
        : configurationFile.toAbsolutePath().normalize().toString();
  }

  /** The suspended game of a configuration, if its checkpoint files are present. */
  static Optional<Checkpoint> readCheckpoint(@Nullable final Path configurationFile) {
    final Path directory = checkpointDirectory(configurationFile);
    final Path info = directory.resolve(CHECKPOINT_INFO);
    if (!Files.isRegularFile(info) || !Files.isRegularFile(directory.resolve(CHECKPOINT_SAVE))) {
      return Optional.empty();
    }
    try {
      return Optional.ofNullable(
              TestbenchConfiguration.GSON.fromJson(Files.readString(info), Checkpoint.class))
          .filter(
              checkpoint -> configurationKey(configurationFile).equals(checkpoint.configuration()));
    } catch (IOException | RuntimeException e) {
      System.err.println("Could not read testbench checkpoint " + info + ": " + e.getMessage());
      return Optional.empty();
    }
  }

  private static void deleteCheckpoint(@Nullable final Path configurationFile) {
    final Path directory = checkpointDirectory(configurationFile);
    try {
      Files.deleteIfExists(directory.resolve(CHECKPOINT_INFO));
      Files.deleteIfExists(directory.resolve(CHECKPOINT_SAVE));
    } catch (IOException e) {
      System.err.println("Could not delete testbench checkpoint in " + directory + ": " + e);
    }
  }

  /**
   * Deletes the agents' per-game logs (planning-agent-testbed/logs/agent-*.txt) of earlier runs.
   */
  private static void deleteAgentLogs() {
    final Path logsDirectory =
        TestbenchMapRepository.repositoryRoot().resolve("planning-agent-testbed/logs");
    if (!Files.isDirectory(logsDirectory)) return;
    try (var existingLogs = Files.newDirectoryStream(logsDirectory, "agent-*.txt")) {
      for (final Path existingLog : existingLogs) {
        Files.deleteIfExists(existingLog);
      }
    } catch (final IOException e) {
      System.err.println("Could not delete earlier planning-agent logs: " + e.getMessage());
    }
  }

  /** Empties the declared output folders of every external agent assigned in this run. */
  private void clearAssignedAgentOutputs() {
    settings.agentLabels().values().stream()
        .distinct()
        .map(TestbenchAgentRegistry::definitionForSelection)
        .flatMap(Optional::stream)
        .flatMap(definition -> definition.resolvedOutputDirectories().stream())
        .forEach(TestbenchRun::emptyDirectory);
  }

  private static void emptyDirectory(final Path directory) {
    if (!Files.isDirectory(directory)) return;
    try (var paths = Files.walk(directory)) {
      for (final Path path :
          paths
              .filter(path -> !path.equals(directory))
              .sorted(java.util.Comparator.reverseOrder())
              .toList()) {
        Files.deleteIfExists(path);
      }
    } catch (final IOException e) {
      System.err.println("Could not clear agent outputs in " + directory + ": " + e.getMessage());
    }
  }
}
