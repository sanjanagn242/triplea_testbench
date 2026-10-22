package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import static games.strategy.engine.framework.CliProperties.TRIPLEA_TESTBENCH_AUTOSTART;
import static games.strategy.engine.framework.CliProperties.TRIPLEA_TESTBENCH_CONFIG;
import static games.strategy.engine.framework.CliProperties.TRIPLEA_TESTBENCH_GUI;
import static games.strategy.engine.framework.CliProperties.TRIPLEA_TESTBENCH_OUTPUTS;
import static games.strategy.engine.framework.CliProperties.TRIPLEA_TESTBENCH_RESULTS;

import games.strategy.engine.ClientFileSystemHelper;
import games.strategy.engine.chat.Chat;
import games.strategy.engine.chat.HeadlessChat;
import games.strategy.engine.chat.MessengersChatTransmitter;
import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.framework.AutoSaveFileUtils;
import games.strategy.engine.framework.I18nResourceBundle;
import games.strategy.engine.framework.IGame;
import games.strategy.engine.framework.LocalPlayers;
import games.strategy.engine.framework.ServerGame;
import games.strategy.engine.framework.map.file.system.loader.ZippedMapsExtractor;
import games.strategy.engine.framework.startup.WatcherThreadMessaging;
import games.strategy.engine.framework.startup.launcher.LaunchAction;
import games.strategy.engine.framework.startup.mc.IServerStartupRemote;
import games.strategy.engine.framework.startup.mc.ServerConnectionProps;
import games.strategy.engine.framework.startup.mc.ServerModel;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.framework.startup.ui.panels.main.game.selector.GameSelectorModel;
import games.strategy.engine.player.Player;
import games.strategy.net.Messengers;
import games.strategy.net.websocket.ClientNetworkBridge;
import games.strategy.triplea.settings.ClientSetting;
import java.io.IOException;
import java.nio.file.Path;
import java.util.Arrays;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Locale;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.function.Consumer;
import javax.annotation.Nullable;
import lombok.extern.slf4j.Slf4j;
import org.slf4j.LoggerFactory;
import org.triplea.game.chat.ChatModel;
import org.triplea.map.description.file.MapDescriptionYamlGeneratorRunner;
import org.triplea.util.ExitStatus;

/**
 * Plays the testbench configuration named by triplea.testbench.config to the end without creating
 * any window, so it runs on machines without a display (e.g. cluster nodes), and exits: with status
 * 0 when every game was played or an agent suspended the run, else 1. Accepts the
 * triplea.testbench.* properties as name=value arguments, like the game client; agents see the same
 * environment, files and results as in the setup panel's runs. The configuration's showGame and
 * showObservations, and triplea.testbench.gui, are ignored.
 */
@Slf4j
public final class TestbenchHeadlessRunner implements TestbenchRun.Host {
  private static final List<String> PROPERTIES =
      List.of(
          TRIPLEA_TESTBENCH_CONFIG,
          TRIPLEA_TESTBENCH_AUTOSTART,
          TRIPLEA_TESTBENCH_GUI,
          TRIPLEA_TESTBENCH_OUTPUTS,
          TRIPLEA_TESTBENCH_RESULTS);

  private final GameSelectorModel gameSelectorModel = new GameSelectorModel();
  private final CompletableFuture<ExitStatus> finished = new CompletableFuture<>();

  private TestbenchHeadlessRunner() {}

  public static void main(final String[] args) {
    // Before any AWT class loads: no window can be opened, even by mistake.
    System.setProperty("java.awt.headless", "true");
    detachErrorPopupLogAppender();
    Thread.setDefaultUncaughtExceptionHandler(
        (t, e) -> log.error("Uncaught exception in thread {}: {}", t.getName(), e.toString(), e));
    setPropertiesFromArgs(args);
    if (!I18nResourceBundle.getMapSupportedLocales().contains(Locale.getDefault())) {
      Locale.setDefault(Locale.US);
    }
    ClientSetting.initialize();

    final String configFile = System.getProperty(TRIPLEA_TESTBENCH_CONFIG, "");
    if (configFile.isBlank()) {
      System.err.println(
          "Usage: TestbenchHeadlessRunner "
              + TRIPLEA_TESTBENCH_CONFIG
              + "=<config.json> ["
              + TRIPLEA_TESTBENCH_RESULTS
              + "=<games.jsonl>] ["
              + TRIPLEA_TESTBENCH_OUTPUTS
              + "=<folder>]");
      ExitStatus.FAILURE.exit();
    }
    if (Boolean.parseBoolean(System.getProperty(TRIPLEA_TESTBENCH_GUI, "false"))) {
      System.err.println(
          "Note: " + TRIPLEA_TESTBENCH_GUI + "=true is ignored, games run without a window");
    }
    final TestbenchHeadlessRunner runner = new TestbenchHeadlessRunner();
    final ExitStatus status = runner.play(Path.of(configFile).toAbsolutePath().normalize());
    status.exit();
  }

  /**
   * The game client's logback.xml shows WARN and ERROR logs in a Swing dialog, which cannot be
   * created without a display; the console and log file appenders still get every message.
   */
  private static void detachErrorPopupLogAppender() {
    if (LoggerFactory.getLogger(org.slf4j.Logger.ROOT_LOGGER_NAME)
        instanceof ch.qos.logback.classic.Logger root) {
      root.detachAppender("swingMessage");
    }
  }

  private static void setPropertiesFromArgs(final String[] args) {
    for (final String arg : args) {
      final String[] nameValuePair = arg.split("=", 2);
      if (nameValuePair.length != 2 || !PROPERTIES.contains(nameValuePair[0])) {
        System.err.println("Ignoring argument '" + arg + "'; expected one of " + PROPERTIES);
        continue;
      }
      System.setProperty(nameValuePair[0], nameValuePair[1]);
    }
  }

  /** Plays the configuration's run (resuming its suspended game, if any) and waits for its end. */
  private ExitStatus play(final Path configFile) {
    // Unzips downloaded map zips and writes missing map.yaml files, as the game client does.
    ZippedMapsExtractor.builder()
        .downloadedMapsFolder(ClientFileSystemHelper.getUserMapsFolder())
        .progressIndicator(Runnable::run)
        .build()
        .unzipMapFiles();
    MapDescriptionYamlGeneratorRunner.builder()
        .downloadedMapsFolder(ClientFileSystemHelper.getUserMapsFolder())
        .progressIndicator(Runnable::run)
        .build()
        .generateYamlFiles();
    gameSelectorModel.setReadyForSaveLoad();

    final TestbenchRun run;
    try {
      final TestbenchConfiguration config = TestbenchConfiguration.read(configFile);
      final Path mapXml = config.resolveMap(configFile).toAbsolutePath();
      if (!gameSelectorModel.loadMapForTestbench(mapXml)) {
        return fail("Could not load a valid TripleA map from " + mapXml);
      }
      final Map<String, String> agentLabels =
          agentLabels(config.agents(), gameSelectorModel.getGameData());
      final Optional<TestbenchRun.Checkpoint> checkpoint = TestbenchRun.readCheckpoint(configFile);
      if (checkpoint.isPresent()) {
        final Path saveFile =
            TestbenchRun.checkpointDirectory(configFile).resolve(TestbenchRun.CHECKPOINT_SAVE);
        if (!gameSelectorModel.loadSave(saveFile)) {
          return fail("Could not load the suspended game " + saveFile);
        }
        final TestbenchRun.Checkpoint suspended = checkpoint.get();
        System.out.printf(
            "Resuming game %d/%d at round %d from %s%n",
            suspended.firstGame() + suspended.completedGames(),
            suspended.firstGame() + suspended.plannedGames() - 1,
            suspended.round(),
            saveFile);
      }
      run =
          new TestbenchRun(
              gameSelectorModel,
              configFile,
              mapXml,
              new TestbenchRun.Settings(
                  config.gameCount(),
                  config.firstGame(),
                  config.rounds(),
                  config.movePauseMs(),
                  config.combatStepPauseMs(),
                  agentLabels,
                  Boolean.TRUE.equals(config.clearAgentOutputs())),
              this,
              checkpoint.orElse(null));
    } catch (final IOException | RuntimeException e) {
      log.error("Could not prepare the testbench run of {}", configFile, e);
      return fail("Could not prepare the testbench run of " + configFile + ": " + e.getMessage());
    }
    run.nextLauncher().launch();
    return finished.join();
  }

  /**
   * Faction -> player type label for every faction of the map: the configured agent, else the setup
   * panel's default. Unknown agents and human players are errors, as no one would notice the
   * stand-in or answer the human player's prompts.
   */
  private static Map<String, String> agentLabels(
      @Nullable final Map<String, String> configuredAgents, final GameData data) {
    final Map<String, String> configured =
        configuredAgents == null ? Map.of() : Map.copyOf(configuredAgents);
    final List<String> available =
        Arrays.asList(
            new PlayerTypes(TestbenchAgentRegistry.getPlayerTypes()).getAvailablePlayerLabels());
    final Map<String, String> labels = new LinkedHashMap<>();
    for (final GamePlayer player : data.getPlayerList().getPlayers()) {
      final String agent = configured.get(player.getName());
      final String label =
          agent == null
              ? TestbenchRun.defaultAgent(player)
              : TestbenchAgentRegistry.selectionForId(agent);
      if (!available.contains(label)) {
        throw new IllegalArgumentException(
            "Unknown agent '" + agent + "' for " + player.getName() + "; available: " + available);
      }
      if (label.equals(PlayerTypes.PLAYER_TYPE_HUMAN_LABEL)) {
        throw new IllegalArgumentException(
            player.getName() + " is a human player, which needs the game window");
      }
      labels.put(player.getName(), label);
    }
    configured.keySet().stream()
        .filter(name -> !labels.containsKey(name))
        .forEach(name -> System.err.println("Note: the map has no faction " + name + "; ignored"));
    return labels;
  }

  private ExitStatus fail(final String message) {
    System.err.println(message);
    return ExitStatus.FAILURE;
  }

  @Override
  public LaunchAction launchAction(
      final Collection<PlayerTypes.Type> playerTypes, final Consumer<IGame> gameStarted) {
    return new WindowlessLaunchAction(playerTypes, gameStarted);
  }

  @Override
  public void afterGame(final Runnable nextStep) {
    nextStep.run();
  }

  @Override
  public void onRunFinished() {
    finished.complete(ExitStatus.SUCCESS);
  }

  @Override
  public void onRunSuspended(final String message) {
    System.err.print(message);
    finished.complete(ExitStatus.SUCCESS);
  }

  @Override
  public void onRunFailed(final String message) {
    finished.complete(fail(message + "; stopping."));
  }

  /** Starts local games with no window, display, sound or dialogs. */
  private static final class WindowlessLaunchAction implements LaunchAction {
    private final Collection<PlayerTypes.Type> playerTypes;
    private final Consumer<IGame> gameStarted;

    WindowlessLaunchAction(
        final Collection<PlayerTypes.Type> playerTypes, final Consumer<IGame> gameStarted) {
      this.playerTypes = playerTypes;
      this.gameStarted = gameStarted;
    }

    @Override
    public void handleGameInterruption(
        final GameSelectorModel gameSelectorModel, final ServerModel serverModel) {}

    @Override
    public void onGameInterrupt() {}

    @Override
    public void onEnd(final String message) {
      log.info(message);
    }

    @Override
    public Collection<PlayerTypes.Type> getPlayerTypes() {
      return playerTypes;
    }

    @Override
    public void startGame(
        final LocalPlayers localPlayers,
        final IGame game,
        final Set<Player> players,
        final Chat chat) {
      gameStarted.accept(game);
      TestbenchRun.setUpWithoutWindow(game);
    }

    @Override
    public Path getAutoSaveFile() {
      return getAutoSaveFileUtils()
          .getLostConnectionAutoSaveFile(
              java.time.LocalDateTime.now(java.time.ZoneId.systemDefault()));
    }

    @Override
    public void onLaunch(final ServerGame serverGame) {}

    @Override
    public AutoSaveFileUtils getAutoSaveFileUtils() {
      return new AutoSaveFileUtils();
    }

    @Override
    public ChatModel createChatModel(
        final String chatName,
        final Messengers messengers,
        final ClientNetworkBridge clientNetworkBridge) {
      return new HeadlessChat(
          new Chat(new MessengersChatTransmitter(chatName, messengers, clientNetworkBridge)));
    }

    @Override
    public boolean shouldMinimizeExpensiveAiUse() {
      return false;
    }

    @Override
    public WatcherThreadMessaging createThreadMessaging() {
      return new WatcherThreadMessaging.HeadlessWatcherThreadMessaging();
    }

    @Override
    public Optional<ServerConnectionProps> getFallbackConnection(final Runnable cancelAction) {
      return Optional.empty();
    }

    @Override
    public void handleError(final String error) {
      log.error(error);
    }

    @Override
    public IServerStartupRemote getStartupRemote(
        final IServerStartupRemote.ServerModelView serverModelView) {
      throw new UnsupportedOperationException("Testbench games are local, not network games");
    }

    @Override
    public boolean promptGameStop(
        final String status, final String title, @Nullable final Path mapLocation) {
      return true;
    }

    @Override
    public PlayerTypes.Type getDefaultLocalPlayerType() {
      return PlayerTypes.WEAK_AI;
    }
  }
}
