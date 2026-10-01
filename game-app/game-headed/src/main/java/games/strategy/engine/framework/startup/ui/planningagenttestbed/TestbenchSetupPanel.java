package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import games.strategy.engine.chat.Chat;
import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.framework.GameShutdownRegistry;
import games.strategy.engine.framework.IGame;
import games.strategy.engine.framework.LocalPlayers;
import games.strategy.engine.framework.map.file.system.loader.InstalledMapsListing;
import games.strategy.engine.framework.startup.launcher.ILauncher;
import games.strategy.engine.framework.startup.launcher.LocalLauncher;
import games.strategy.engine.framework.startup.launcher.local.PlayerCountrySelection;
import games.strategy.engine.framework.startup.mc.HeadedLaunchAction;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.framework.startup.ui.SetupPanel;
import games.strategy.engine.framework.startup.ui.panels.main.HeadedServerSetupModel;
import games.strategy.engine.framework.startup.ui.panels.main.game.selector.GameSelectorModel;
import games.strategy.engine.player.Player;
import games.strategy.triplea.delegate.EndRoundDelegate;
import games.strategy.triplea.settings.ClientSetting;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardOpenOption;
import java.time.Duration;
import java.time.Instant;
import java.util.ArrayList;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import javax.swing.Action;
import javax.swing.JButton;
import javax.swing.JCheckBox;
import javax.swing.JComboBox;
import javax.swing.JFileChooser;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.JScrollPane;
import javax.swing.JSpinner;
import javax.swing.JTextField;
import javax.swing.SpinnerNumberModel;
import javax.swing.SwingUtilities;
import javax.swing.filechooser.FileNameExtensionFilter;
import org.triplea.game.startup.SetupModel;
import org.triplea.java.ThreadRunner;
import org.triplea.swing.SwingComponents;

/** Interactive configuration for repeatable local planning-agent experiments. */
public final class TestbenchSetupPanel extends SetupPanel {
  private static final long serialVersionUID = 1L;
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final int DEFAULT_AI_MOVE_PAUSE_MS = 300;
  private static final int DEFAULT_AI_COMBAT_STEP_PAUSE_MS = 1000;
  private static final int MAX_AI_PAUSE_MS = 3000;

  private final HeadedServerSetupModel setupModel;
  private final GameSelectorModel gameSelectorModel;
  private final JTextField mapPath = new JTextField(22);
  private final JLabel mapStatus =
      new JLabel("Each game's maps are stored in planning-agent-testbed/games/<game>/map");
  private final JSpinner gamesCount = new JSpinner(new SpinnerNumberModel(2, 1, 100_000, 1));
  private final JSpinner roundLimit = new JSpinner(new SpinnerNumberModel(100, 1, 100_000, 1));
  private final JSpinner aiMovePauseMs =
      new JSpinner(new SpinnerNumberModel(DEFAULT_AI_MOVE_PAUSE_MS, 0, MAX_AI_PAUSE_MS, 10));
  private final JSpinner aiCombatStepPauseMs =
      new JSpinner(new SpinnerNumberModel(DEFAULT_AI_COMBAT_STEP_PAUSE_MS, 0, MAX_AI_PAUSE_MS, 10));
  private final JCheckBox showObservationWindow = new JCheckBox("Visualize observations", true);
  private final JPanel factionRows = new JPanel(new GridBagLayout());
  private final Map<String, JComboBox<String>> assignments = new LinkedHashMap<>();
  private final Map<String, JCheckBox> observationAssignments = new LinkedHashMap<>();
  private final Path initialConfig;
  private Path activeConfigurationFile;
  private MapReference activeMapReference;
  private boolean experimentRunning;
  private int plannedGames;
  private int completedGames;
  private Instant currentGameStartedAt;
  private final StateObservationWindow stateObservationWindow = new StateObservationWindow();

  public TestbenchSetupPanel(final HeadedServerSetupModel setupModel, final String configFile) {
    this.setupModel = setupModel;
    this.gameSelectorModel = setupModel.getGameSelectorModel();
    this.initialConfig = configFile == null ? null : Path.of(configFile);
    this.activeConfigurationFile = initialConfig;
    buildUi();
    ensureResultsLogExists();
    if (initialConfig != null) {
      readConfiguration(initialConfig);
    } else if (Files.isRegularFile(defaultCaptureTheFlagConfiguration())) {
      readConfiguration(defaultCaptureTheFlagConfiguration());
    } else {
      findCaptureTheFlagMap()
          .ifPresent(
              path -> {
                mapPath.setText(path.toAbsolutePath().toString());
                loadMap(path);
              });
    }
  }

  private void buildUi() {
    setLayout(new BorderLayout(8, 8));
    setBorder(javax.swing.BorderFactory.createEmptyBorder(10, 12, 6, 12));

    final JPanel header = new JPanel(new BorderLayout(0, 2));
    final JLabel title = new JLabel("Run Simulations");
    title.setFont(title.getFont().deriveFont(java.awt.Font.BOLD, 18f));
    header.add(title, BorderLayout.NORTH);
    header.add(
        new JLabel("Choose the map, assign agents, and set the length of each run."),
        BorderLayout.CENTER);
    add(header, BorderLayout.NORTH);

    final JPanel content = new JPanel(new GridBagLayout());
    final JPanel experiment = new JPanel(new GridBagLayout());
    experiment.setBorder(javax.swing.BorderFactory.createTitledBorder("Experiment settings"));
    int row = 0;
    experiment.add(new JLabel("Map XML"), constraints(0, row));
    experiment.add(mapPath, constraints(1, row));
    final JButton browse = new JButton("Browse…");
    browse.addActionListener(e -> chooseMap());
    experiment.add(browse, constraints(2, row++));
    experiment.add(
        mapStatus,
        new GridBagConstraints(
            1,
            row++,
            3,
            1,
            1,
            0,
            GridBagConstraints.WEST,
            GridBagConstraints.HORIZONTAL,
            new Insets(0, 4, 2, 4),
            0,
            0));
    experiment.add(new JLabel("Games"), constraints(0, row));
    experiment.add(gamesCount, constraints(1, row));
    experiment.add(new JLabel("Round limit"), constraints(2, row));
    experiment.add(roundLimit, constraints(3, row++));
    experiment.add(new JLabel("Move pause (ms)"), constraints(0, row));
    experiment.add(aiMovePauseMs, constraints(1, row));
    experiment.add(new JLabel("Combat pause (ms)"), constraints(2, row));
    experiment.add(aiCombatStepPauseMs, constraints(3, row++));
    experiment.add(
        new JLabel("Round cap ends without a winner. Pauses apply at launch."),
        new GridBagConstraints(
            0,
            row++,
            4,
            1,
            1,
            0,
            GridBagConstraints.WEST,
            GridBagConstraints.HORIZONTAL,
            new Insets(3, 4, 4, 4),
            0,
            0));
    experiment.add(
        showObservationWindow,
        new GridBagConstraints(
            0,
            row++,
            4,
            1,
            1,
            0,
            GridBagConstraints.WEST,
            GridBagConstraints.HORIZONTAL,
            new Insets(3, 4, 4, 4),
            0,
            0));
    content.add(
        experiment,
        new GridBagConstraints(
            0,
            0,
            1,
            1,
            1,
            0,
            GridBagConstraints.NORTHWEST,
            GridBagConstraints.HORIZONTAL,
            new Insets(0, 0, 6, 0),
            0,
            0));

    final JPanel players = new JPanel(new BorderLayout(0, 6));
    players.setBorder(javax.swing.BorderFactory.createTitledBorder("Faction agent assignments"));
    final JScrollPane assignmentsScroll = new JScrollPane(factionRows);
    assignmentsScroll.setBorder(javax.swing.BorderFactory.createEmptyBorder());
    assignmentsScroll.setPreferredSize(new java.awt.Dimension(420, 170));
    players.add(assignmentsScroll, BorderLayout.CENTER);
    content.add(
        players,
        new GridBagConstraints(
            0,
            1,
            1,
            1,
            1,
            1,
            GridBagConstraints.NORTHWEST,
            GridBagConstraints.BOTH,
            new Insets(0, 0, 0, 0),
            0,
            0));
    add(content, BorderLayout.CENTER);

    final JPanel controls = new JPanel(new FlowLayout(FlowLayout.RIGHT));
    final JButton importButton = new JButton("Load Configuration…");
    importButton.addActionListener(e -> chooseConfiguration(false));
    final JButton exportButton = new JButton("Save Configuration…");
    exportButton.addActionListener(e -> chooseConfiguration(true));
    controls.add(importButton);
    controls.add(exportButton);
    add(controls, BorderLayout.SOUTH);
  }

  private static GridBagConstraints constraints(int x, int y) {
    return new GridBagConstraints(
        x,
        y,
        1,
        1,
        x == 1 || x == 3 ? 1 : 0,
        0,
        GridBagConstraints.WEST,
        GridBagConstraints.HORIZONTAL,
        new Insets(2, 4, 2, 4),
        0,
        0);
  }

  private void chooseMap() {
    final JFileChooser chooser =
        new JFileChooser(
            TestbenchMapRepository.mapsDirectory(activeConfigurationFile, currentMapName())
                .toFile());
    chooser.setFileFilter(new FileNameExtensionFilter("TripleA game XML (*.xml)", "xml"));
    if (chooser.showOpenDialog(this) == JFileChooser.APPROVE_OPTION) {
      final Path selected = chooser.getSelectedFile().toPath();
      mapPath.setText(selected.toAbsolutePath().toString());
      mapStatus.setText("Map selected: " + selected.getFileName());
      mapStatus.setToolTipText(selected.toAbsolutePath().toString());
      TestbenchMapRepository.configurationFileForMapXml(selected)
          .ifPresent(configurationFile -> activeConfigurationFile = configurationFile);
      activeMapReference = null;
      loadMap(selected);
    }
  }

  private void loadMap(final Path path) {
    if (!Files.isRegularFile(path) || !gameSelectorModel.loadMapForTestbench(path)) {
      mapStatus.setText("Map could not be loaded");
      mapStatus.setToolTipText(path.toAbsolutePath().toString());
      showError("Could not load a valid TripleA map from " + path);
      assignments.clear();
      observationAssignments.clear();
      factionRows.removeAll();
      factionRows.revalidate();
      factionRows.repaint();
      fireListener();
      refreshWindowLayout();
      return;
    }
    mapStatus.setText("Map ready: " + path.getFileName());
    mapStatus.setToolTipText(path.toAbsolutePath().toString());
    final GameData data = gameSelectorModel.getGameData();
    final String[] labels =
        new PlayerTypes(TestbenchAgentRegistry.getPlayerTypes()).getAvailablePlayerLabels();
    final Map<String, String> previous = selectedAssignments();
    final Map<String, Boolean> previousObservationAssignments = selectedObservationAssignments();
    assignments.clear();
    observationAssignments.clear();
    factionRows.removeAll();
    int row = 0;
    factionRows.add(new JLabel("Faction"), constraints(0, row));
    factionRows.add(new JLabel("Agent"), constraints(1, row));
    factionRows.add(new JLabel("Visualize observation"), constraints(2, row++));
    for (GamePlayer player : data.getPlayerList().getPlayers()) {
      factionRows.add(new JLabel(player.getName()), constraints(0, row));
      final JComboBox<String> agent = new JComboBox<>(labels);
      final String selection = previous.getOrDefault(player.getName(), defaultAgent(player));
      agent.setSelectedItem(selection);
      assignments.put(player.getName(), agent);
      factionRows.add(agent, constraints(1, row));
      final JCheckBox showState =
          new JCheckBox("", previousObservationAssignments.getOrDefault(player.getName(), true));
      observationAssignments.put(player.getName(), showState);
      factionRows.add(showState, constraints(2, row++));
    }
    factionRows.revalidate();
    factionRows.repaint();
    fireListener();
    refreshWindowLayout();
  }

  private void refreshWindowLayout() {
    revalidate();
    repaint();
    if (setupModel.getUi() != null) {
      setupModel.getUi().invalidate();
      setupModel.getUi().validate();
      setupModel.getUi().repaint();
    }
  }

  private static String defaultAgent(final GamePlayer player) {
    return player.isDefaultTypeDoesNothing()
        ? PlayerTypes.DOES_NOTHING_PLAYER_LABEL
        : PlayerTypes.PRO_AI.getLabel();
  }

  private Map<String, String> selectedAssignments() {
    final Map<String, String> result = new LinkedHashMap<>();
    assignments.forEach((name, combo) -> result.put(name, (String) combo.getSelectedItem()));
    return result;
  }

  private Map<String, Boolean> selectedObservationAssignments() {
    final Map<String, Boolean> result = new LinkedHashMap<>();
    observationAssignments.forEach((name, checkbox) -> result.put(name, checkbox.isSelected()));
    return result;
  }

  private void chooseConfiguration(final boolean save) {
    final Path initialDirectory =
        save
            ? TestbenchMapRepository.configurationDirectory(
                activeConfigurationFile, currentMapName())
            : TestbenchMapRepository.gamesDirectory();
    final JFileChooser chooser = new JFileChooser(initialDirectory.toFile());
    chooser.setFileFilter(new FileNameExtensionFilter("Testbench JSON (*.json)", "json"));
    if (chooser.showDialog(this, save ? "Save" : "Load") != JFileChooser.APPROVE_OPTION) return;
    final Path path = chooser.getSelectedFile().toPath();
    if (save) writeConfiguration(path);
    else readConfiguration(path);
  }

  private void readConfiguration(final Path path) {
    try {
      final TestbenchConfiguration config =
          GSON.fromJson(Files.readString(path), TestbenchConfiguration.class);
      if (config == null
          || (config.map == null && (config.mapXml == null || config.mapXml.isBlank()))) {
        throw new IllegalArgumentException("JSON must contain a map name or mapXml path.");
      }
      activeConfigurationFile = path.toAbsolutePath().normalize();
      gamesCount.setValue(Math.max(1, config.games));
      roundLimit.setValue(Math.max(1, config.roundLimit));
      aiMovePauseMs.setValue(clampPause(config.aiMovePauseMs, DEFAULT_AI_MOVE_PAUSE_MS));
      aiCombatStepPauseMs.setValue(
          clampPause(config.aiCombatStepPauseMs, DEFAULT_AI_COMBAT_STEP_PAUSE_MS));
      showObservationWindow.setSelected(
          config.showObservations == null || config.showObservations);
      activeMapReference = config.map;
      mapStatus.setText("Resolving map…");
      ThreadRunner.runInNewThread(
          () -> {
            try {
              final Path resolvedMap =
                  TestbenchMapRepository.resolveMap(
                      path,
                      config.mapXml,
                      config.map == null ? null : config.map.name,
                      config.map == null ? null : config.map.gameName,
                      config.map == null ? List.of() : config.map.aliases,
                      config.map == null ? null : config.map.downloadUrl);
              SwingUtilities.invokeLater(
                  () -> {
                    mapPath.setText(resolvedMap.toAbsolutePath().toString());
                    mapStatus.setText("Map ready: " + resolvedMap.getFileName());
                    mapStatus.setToolTipText(resolvedMap.toAbsolutePath().toString());
                    loadMap(resolvedMap);
                    applyAgentAssignments(config.agents);
                    applyObservationAssignments(config.observationPlayers);
                  });
            } catch (IOException | RuntimeException e) {
              SwingUtilities.invokeLater(
                  () -> {
                    mapStatus.setText("Map resolution failed");
                    mapStatus.setToolTipText(e.getMessage());
                    showError("Could not resolve map from testbench config: " + e.getMessage());
                  });
            }
          });
    } catch (IOException | RuntimeException e) {
      showError("Could not read testbench configuration: " + e.getMessage());
    }
  }

  private static int clampPause(final Integer value, final int defaultValue) {
    return value == null ? defaultValue : Math.max(0, Math.min(MAX_AI_PAUSE_MS, value));
  }

  private void applyAgentAssignments(final Map<String, String> configuredAgents) {
    Optional.ofNullable(configuredAgents)
        .orElse(Map.of())
        .forEach(
            (name, agent) -> {
              final JComboBox<String> combo = assignments.get(name);
              final String selection = TestbenchAgentRegistry.selectionForId(agent);
              if (combo != null
                  && java.util.Arrays.asList(
                          new PlayerTypes(TestbenchAgentRegistry.getPlayerTypes())
                              .getAvailablePlayerLabels())
                      .contains(selection)) {
                combo.setSelectedItem(selection);
              }
            });
  }

  private void applyObservationAssignments(final Map<String, Boolean> configuredPlayers) {
    Optional.ofNullable(configuredPlayers)
        .orElse(Map.of())
        .forEach(
            (name, enabled) -> {
              final JCheckBox checkbox = observationAssignments.get(name);
              if (checkbox != null) {
                checkbox.setSelected(enabled == null || enabled);
              }
            });
  }

  private void writeConfiguration(final Path path) {
    try {
      final Path destination = path.toString().endsWith(".json") ? path : Path.of(path + ".json");
      final Path absoluteDestination = destination.toAbsolutePath();
      final Path expectedConfigurationDirectory =
          TestbenchMapRepository.configurationDirectory(activeConfigurationFile, currentMapName())
              .toAbsolutePath()
              .normalize();
      if (!expectedConfigurationDirectory.equals(absoluteDestination.getParent().normalize())) {
        throw new IOException(
            "Save configurations for this map in " + expectedConfigurationDirectory);
      }
      final String mapXml;
      if (activeMapReference != null) {
        mapXml = null;
      } else {
        final Path selectedMap = Path.of(mapPath.getText()).toAbsolutePath();
        if (!selectedMap.startsWith(TestbenchMapRepository.repositoryRoot())) {
          throw new IOException(
              "For a portable testbed JSON, first place the map inside this repository checkout.");
        }
        mapXml = absoluteDestination.getParent().relativize(selectedMap).toString();
      }
      Files.writeString(
          absoluteDestination,
          GSON.toJson(
              new TestbenchConfiguration(
                  activeMapReference,
                  mapXml,
                  (int) gamesCount.getValue(),
                  (int) roundLimit.getValue(),
                  (int) aiMovePauseMs.getValue(),
                  (int) aiCombatStepPauseMs.getValue(),
                  selectedAssignments().entrySet().stream()
                      .collect(
                          java.util.stream.Collectors.toMap(
                              Map.Entry::getKey,
                              entry -> TestbenchAgentRegistry.idForSelection(entry.getValue()),
                              (first, second) -> first,
                              LinkedHashMap::new)),
                  showObservationWindow.isSelected(),
                  selectedObservationAssignments())));
    } catch (IOException e) {
      showError("Could not save testbench configuration: " + e.getMessage());
    }
  }

  private Optional<Path> findCaptureTheFlagMap() {
    return InstalledMapsListing.parseMapFiles(
            TestbenchMapRepository.mapsDirectory(
                defaultCaptureTheFlagConfiguration(), "Capture The Flag"))
        .findGameXmlPathByGameName("Capture The Flag");
  }

  private Path defaultCaptureTheFlagConfiguration() {
    return TestbenchMapRepository.repositoryRoot()
        .resolve("planning-agent-testbed/games/capture-the-flag/config/capture-the-flag.json");
  }

  private String currentMapName() {
    if (activeMapReference != null && activeMapReference.name != null) {
      return activeMapReference.name;
    }
    final GameData data = gameSelectorModel.getGameData();
    return data == null ? "Capture The Flag" : data.getMapName();
  }

  private void showError(final String message) {
    JOptionPane.showMessageDialog(
        this, message, "Planning agent testbench", JOptionPane.ERROR_MESSAGE);
  }

  @Override
  public List<Action> getUserActions() {
    return List.of();
  }

  @Override
  public boolean isCancelButtonVisible() {
    return true;
  }

  @Override
  public boolean canGameStart() {
    return gameSelectorModel.getGameData() != null && !assignments.isEmpty();
  }

  @Override
  public void postStartGame() {
    SetupModel.clearPbfPbemInformation(gameSelectorModel.getGameData().getProperties());
  }

  @Override
  public void cancel() {}

  @Override
  public Optional<ILauncher> getLauncher() {
    if (!experimentRunning) {
      refreshSimulationLogs();
      experimentRunning = true;
      plannedGames = (int) gamesCount.getValue();
      completedGames = 0;
    }
    gameSelectorModel
        .getGameData()
        .getProperties()
        .set(EndRoundDelegate.TESTBENCH_ROUND_LIMIT_PROPERTY, (int) roundLimit.getValue());
    currentGameStartedAt = Instant.now();
    final int previousMovePause = ClientSetting.aiMovePauseDuration.getValueOrThrow();
    final int previousCombatPause = ClientSetting.aiCombatStepPauseDuration.getValueOrThrow();
    ClientSetting.aiMovePauseDuration.setValue((int) aiMovePauseMs.getValue());
    ClientSetting.aiCombatStepPauseDuration.setValue((int) aiCombatStepPauseMs.getValue());
    GameShutdownRegistry.registerShutdownAction(
        () -> {
          ClientSetting.aiMovePauseDuration.setValue(previousMovePause);
          ClientSetting.aiCombatStepPauseDuration.setValue(previousCombatPause);
        });
    final List<PlayerCountrySelection> players = new ArrayList<>();
    final boolean showAllObservations = showObservationWindow.isSelected();
    final Map<String, Boolean> observationsByPlayer = selectedObservationAssignments();
    final int simulationNumber = completedGames + 1;
    final List<PlayerTypes.Type> selectedGamePlayerTypes =
        TestbenchAgentRegistry.getPlayerTypes(Path.of(mapPath.getText()), simulationNumber);
    assignments.forEach(
        (name, combo) ->
            players.add(
                new PlayerCountrySelection() {
                  @Override
                  public String getPlayerName() {
                    return name;
                  }

                  @Override
                  public PlayerTypes.Type getPlayerType() {
                    return new PlayerTypes(selectedGamePlayerTypes)
                        .fromLabel((String) combo.getSelectedItem());
                  }

                  @Override
                  public boolean isPlayerEnabled() {
                    return true;
                  }
                }));
    return Optional.of(
        LocalLauncher.create(
            gameSelectorModel,
            players,
            this,
            new HeadedLaunchAction(setupModel.getUi()) {
              @Override
              public java.util.Collection<PlayerTypes.Type> getPlayerTypes() {
                return selectedGamePlayerTypes;
              }

              @Override
              public void startGame(
                  final LocalPlayers localPlayers,
                  final IGame game,
                  final Set<Player> localGamePlayers,
                final Chat chat) {
                super.startGame(localPlayers, game, localGamePlayers, chat);
                SwingUtilities.invokeLater(
                    () ->
                        stateObservationWindow.observe(
                            game.getData(), showAllObservations, observationsByPlayer));
              }

              @Override
              public boolean promptGameStop(
                  final String status, final String title, final Path mapLocation) {
                return true;
              }
            },
            this::onGameCompleted));
  }

  private void onGameCompleted() {
    stateObservationWindow.stopObserving();
    GameShutdownRegistry.runShutdownActions();
    completedGames++;
    appendGameLog();
    SwingUtilities.invokeLater(
        () -> {
          final Path selectedMap = Path.of(mapPath.getText());
          if (!Files.isRegularFile(selectedMap)
              || !gameSelectorModel.loadMapForTestbench(selectedMap)) {
            experimentRunning = false;
            showError("Could not reload the testbench map after the game ended: " + selectedMap);
            SwingComponents.setFrameFromComponentVisible(this);
            return;
          }
          if (completedGames < plannedGames) {
            getLauncher().ifPresent(launcher -> ThreadRunner.runInNewThread(launcher::launch));
          } else {
            experimentRunning = false;
            plannedGames = 0;
            completedGames = 0;
            setupModel.showTestbenchMenu();
            games.strategy.engine.framework.ui.MainFrame.show();
          }
        });
  }

  private void appendGameLog() {
    try {
      final String winners =
          gameSelectorModel
              .getGameData()
              .getProperties()
              .get(EndRoundDelegate.TESTBENCH_WINNERS_PROPERTY, "");
      final String result = winners.isBlank() ? "No winner" : "Winner(s): " + winners;
      final String players =
          selectedAssignments().entrySet().stream()
              .map(entry -> entry.getKey() + " = " + entry.getValue())
              .collect(java.util.stream.Collectors.joining(", "));
      final long elapsedMillis = Duration.between(currentGameStartedAt, Instant.now()).toMillis();
      final Path logFile = resultsLogPath();
      Files.createDirectories(logFile.getParent());
      Files.writeString(
          logFile,
          String.format(
              java.util.Locale.ROOT,
              "Game %d/%d completed%nPlayers: %s%nResult: %s%nElapsed: %.1f seconds%n%n",
              completedGames,
              plannedGames,
              players,
              result,
              elapsedMillis / 1000.0),
          StandardOpenOption.CREATE,
          StandardOpenOption.APPEND);
    } catch (IOException | RuntimeException e) {
      System.err.println("Could not write planning-agent testbench log: " + e.getMessage());
    }
  }

  private void ensureResultsLogExists() {
    final Path logFile = resultsLogPath();
    try {
      Files.createDirectories(logFile.getParent());
      if (Files.notExists(logFile)) {
        Files.writeString(logFile, "Planning agent testbench results\n\n");
      }
    } catch (IOException e) {
      System.err.println("Could not create planning-agent testbench log: " + e.getMessage());
    }
  }

  private static Path resultsLogPath() {
    return TestbenchMapRepository.repositoryRoot()
        .resolve("planning-agent-testbed/logs/game-results.txt");
  }

  private static void refreshSimulationLogs() {
    final Path logsDirectory = resultsLogPath().getParent();
    try {
      Files.createDirectories(logsDirectory);
      try (var existingLogs = Files.newDirectoryStream(logsDirectory, "agent-*.txt")) {
        for (final Path existingLog : existingLogs) {
          Files.deleteIfExists(existingLog);
        }
      }
      Files.writeString(
          resultsLogPath(),
          "Planning agent testbench results\n\n",
          StandardOpenOption.CREATE,
          StandardOpenOption.TRUNCATE_EXISTING);
    } catch (final IOException e) {
      System.err.println("Could not refresh planning-agent testbench logs: " + e.getMessage());
    }
  }

  private record TestbenchConfiguration(
      MapReference map,
      String mapXml,
      int games,
      int roundLimit,
      Integer aiMovePauseMs,
      Integer aiCombatStepPauseMs,
      Map<String, String> agents,
      Boolean showObservations,
      Map<String, Boolean> observationPlayers) {}

  private record MapReference(
      String name, String gameName, List<String> aliases, String downloadUrl) {}
}
