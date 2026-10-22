package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import com.google.gson.Gson;
import games.strategy.engine.chat.Chat;
import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.framework.CliProperties;
import games.strategy.engine.framework.IGame;
import games.strategy.engine.framework.LocalPlayers;
import games.strategy.engine.framework.map.file.system.loader.InstalledMapsListing;
import games.strategy.engine.framework.startup.launcher.ILauncher;
import games.strategy.engine.framework.startup.launcher.LaunchAction;
import games.strategy.engine.framework.startup.mc.HeadedLaunchAction;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.framework.startup.ui.SetupPanel;
import games.strategy.engine.framework.startup.ui.panels.main.HeadedServerSetupModel;
import games.strategy.engine.framework.startup.ui.panels.main.game.selector.GameSelectorModel;
import games.strategy.engine.player.Player;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.Collection;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.function.Consumer;
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
import org.triplea.util.ExitStatus;

/** Interactive configuration for repeatable local planning-agent experiments. */
public final class TestbenchSetupPanel extends SetupPanel {
  private static final long serialVersionUID = 1L;
  private static final Gson GSON = TestbenchConfiguration.GSON;

  private final HeadedServerSetupModel setupModel;
  private final GameSelectorModel gameSelectorModel;
  private final JTextField mapPath = new JTextField(22);
  private final JLabel mapStatus =
      new JLabel("Each game's maps are stored in planning-agent-testbed/games/<game>/map");
  private final JSpinner gamesCount = new JSpinner(new SpinnerNumberModel(2, 1, 100_000, 1));
  private final JSpinner firstGameNumber = new JSpinner(new SpinnerNumberModel(1, 1, 1_000_000, 1));
  private final JSpinner roundLimit = new JSpinner(new SpinnerNumberModel(100, 1, 100_000, 1));
  private final JSpinner aiMovePauseMs =
      new JSpinner(
          new SpinnerNumberModel(
              TestbenchConfiguration.DEFAULT_AI_MOVE_PAUSE_MS,
              0,
              TestbenchConfiguration.MAX_AI_PAUSE_MS,
              10));
  private final JSpinner aiCombatStepPauseMs =
      new JSpinner(
          new SpinnerNumberModel(
              TestbenchConfiguration.DEFAULT_AI_COMBAT_STEP_PAUSE_MS,
              0,
              TestbenchConfiguration.MAX_AI_PAUSE_MS,
              10));
  private final JCheckBox showGameWindow = new JCheckBox("Show game window", true);
  private final JCheckBox showObservationWindow = new JCheckBox("Visualize observations", true);
  private final JCheckBox clearAgentOutputs =
      new JCheckBox("Clear previous agent outputs (metrics, logs, …) when the run starts", false);
  private final JPanel factionRows = new JPanel(new GridBagLayout());
  private final Map<String, JComboBox<String>> assignments = new LinkedHashMap<>();
  private final Map<String, JCheckBox> observationAssignments = new LinkedHashMap<>();
  private final Path initialConfig;
  private Path activeConfigurationFile;
  private TestbenchConfiguration.MapReference activeMapReference;
  // The run being played, from Play until its last game; null in between.
  private TestbenchRun run;
  // Set when a suspended game has been loaded and the next launch resumes it instead of a new run.
  private TestbenchRun.Checkpoint pendingResume;
  private boolean autoStartPending;
  private final boolean exitWhenBatchDone; // autostarted from the command line
  private final StateObservationWindow stateObservationWindow = new StateObservationWindow();

  public TestbenchSetupPanel(final HeadedServerSetupModel setupModel, final String configFile) {
    this(setupModel, configFile, false);
  }

  /**
   * With autoStart, the batch in configFile starts as soon as its map has loaded, as if Play had
   * been pressed, and the client exits when the batch is done.
   */
  public TestbenchSetupPanel(
      final HeadedServerSetupModel setupModel, final String configFile, final boolean autoStart) {
    this.setupModel = setupModel;
    this.autoStartPending = autoStart && configFile != null;
    this.exitWhenBatchDone = autoStartPending;
    this.gameSelectorModel = setupModel.getGameSelectorModel();
    this.initialConfig = configFile == null ? null : Path.of(configFile);
    this.activeConfigurationFile = initialConfig;
    buildUi();
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
    experiment.add(new JLabel("First game number"), constraints(0, row));
    experiment.add(firstGameNumber, constraints(1, row++));
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
    showGameWindow.setToolTipText(
        "When off, games run without the map window, battle window and sounds, which is faster."
            + " Overridden by -Dtriplea.testbench.gui=true|false.");
    experiment.add(
        showGameWindow,
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
    clearAgentOutputs.setToolTipText(
        "Empties the outputDirectories listed in each assigned agent's agent.json, once, before"
            + " the first game of the run.");
    experiment.add(
        clearAgentOutputs,
        new GridBagConstraints(
            0,
            row++,
            4,
            1,
            1,
            0,
            GridBagConstraints.WEST,
            GridBagConstraints.HORIZONTAL,
            new Insets(0, 4, 4, 4),
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
    pendingResume = null;
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
      final String selection =
          previous.getOrDefault(player.getName(), TestbenchRun.defaultAgent(player));
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
      final TestbenchConfiguration config = TestbenchConfiguration.read(path);
      activeConfigurationFile = path.toAbsolutePath().normalize();
      gamesCount.setValue(config.gameCount());
      firstGameNumber.setValue(config.firstGame());
      roundLimit.setValue(config.rounds());
      aiMovePauseMs.setValue(config.movePauseMs());
      aiCombatStepPauseMs.setValue(config.combatStepPauseMs());
      final String guiProperty = System.getProperty(CliProperties.TRIPLEA_TESTBENCH_GUI, "");
      showGameWindow.setSelected(
          guiProperty.isBlank()
              ? config.showGame() == null || config.showGame()
              : Boolean.parseBoolean(guiProperty));
      showObservationWindow.setSelected(
          config.showObservations() == null || config.showObservations());
      clearAgentOutputs.setSelected(Boolean.TRUE.equals(config.clearAgentOutputs()));
      activeMapReference = config.map();
      mapStatus.setText("Resolving map…");
      ThreadRunner.runInNewThread(
          () -> {
            try {
              final Path resolvedMap = config.resolveMap(path);
              SwingUtilities.invokeLater(
                  () -> {
                    mapPath.setText(resolvedMap.toAbsolutePath().toString());
                    mapStatus.setText("Map ready: " + resolvedMap.getFileName());
                    mapStatus.setToolTipText(resolvedMap.toAbsolutePath().toString());
                    loadMap(resolvedMap);
                    applyAgentAssignments(config.agents());
                    applyObservationAssignments(config.observationPlayers());
                    resumeOrLaunchIfAutoStartPending();
                  });
            } catch (IOException | RuntimeException e) {
              SwingUtilities.invokeLater(
                  () -> {
                    autoStartPending = false;
                    mapStatus.setText("Map resolution failed");
                    mapStatus.setToolTipText(e.getMessage());
                    showError("Could not resolve map from testbench config: " + e.getMessage());
                  });
            }
          });
    } catch (IOException | RuntimeException e) {
      autoStartPending = false;
      showError("Could not read testbench configuration: " + e.getMessage());
    }
  }

  /**
   * When this configuration has a suspended game, loads it so the next launch continues from it
   * (asking first unless autostarted), then autostarts if requested.
   */
  private void resumeOrLaunchIfAutoStartPending() {
    final Optional<TestbenchRun.Checkpoint> checkpoint =
        TestbenchRun.readCheckpoint(activeConfigurationFile);
    if (checkpoint.isEmpty()) {
      launchIfAutoStartPending();
      return;
    }
    final TestbenchRun.Checkpoint suspended = checkpoint.get();
    if (!autoStartPending
        && JOptionPane.showConfirmDialog(
                this,
                String.format(
                    "Game %d of %d was suspended at round %d (%s).%nResume it?%n%n"
                        + "Choose No to start a new run; the suspended game is deleted when it"
                        + " starts.",
                    suspended.firstGame() + suspended.completedGames(),
                    suspended.firstGame() + suspended.plannedGames() - 1,
                    suspended.round(),
                    suspended.savedAt()),
                "Planning agent testbench",
                JOptionPane.YES_NO_OPTION)
            != JOptionPane.YES_OPTION) {
      return;
    }
    mapStatus.setText("Loading the suspended game…");
    final Path saveFile =
        TestbenchRun.checkpointDirectory(activeConfigurationFile)
            .resolve(TestbenchRun.CHECKPOINT_SAVE);
    ThreadRunner.runInNewThread(
        () -> {
          final boolean loaded = gameSelectorModel.loadSave(saveFile);
          SwingUtilities.invokeLater(
              () -> {
                if (!loaded) {
                  autoStartPending = false;
                  mapStatus.setText("Suspended game could not be loaded");
                  showError("Could not load the suspended game " + saveFile);
                  return;
                }
                pendingResume = suspended;
                mapStatus.setText(
                    String.format(
                        "Resuming game %d/%d at round %d",
                        suspended.firstGame() + suspended.completedGames(),
                        suspended.firstGame() + suspended.plannedGames() - 1,
                        suspended.round()));
                mapStatus.setToolTipText(saveFile.toAbsolutePath().toString());
                fireListener();
                launchIfAutoStartPending();
              });
        });
  }

  /**
   * Starts the batch once, for the autostart command-line option; stays on this screen on error.
   */
  private void launchIfAutoStartPending() {
    if (!autoStartPending) {
      return;
    }
    autoStartPending = false;
    if (canGameStart()) { // otherwise loadMap has already reported the error
      games.strategy.engine.framework.ui.MainFrame.startGameDirectly(setupModel);
    }
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
                  (int) firstGameNumber.getValue(),
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
                  showGameWindow.isSelected(),
                  showObservationWindow.isSelected(),
                  selectedObservationAssignments(),
                  clearAgentOutputs.isSelected())));
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
    if (activeMapReference != null && activeMapReference.name() != null) {
      return activeMapReference.name();
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
    if (run == null) {
      final Map<String, String> agentLabels = selectedAssignments();
      final TestbenchRun.Settings settings =
          new TestbenchRun.Settings(
              (int) gamesCount.getValue(),
              (int) firstGameNumber.getValue(),
              (int) roundLimit.getValue(),
              (int) aiMovePauseMs.getValue(),
              (int) aiCombatStepPauseMs.getValue(),
              agentLabels,
              clearAgentOutputs.isSelected());
      run =
          new TestbenchRun(
              gameSelectorModel,
              activeConfigurationFile,
              Path.of(mapPath.getText()),
              settings,
              new PanelHost(
                  showGameWindow.isSelected(),
                  showObservationWindow.isSelected(),
                  selectedObservationAssignments(),
                  agentLabels,
                  Path.of(mapPath.getText())),
              pendingResume);
      pendingResume = null;
    }
    return Optional.of(run.nextLauncher());
  }

  /** Runs the games of a run in the game window or without it, and returns here afterwards. */
  private final class PanelHost implements TestbenchRun.Host {
    private final boolean showGame;
    private final boolean showAllObservations;
    private final Map<String, Boolean> observationsByPlayer;
    private final Map<String, String> agentLabels;
    private final Path observedMapXml;

    PanelHost(
        final boolean showGame,
        final boolean showAllObservations,
        final Map<String, Boolean> observationsByPlayer,
        final Map<String, String> agentLabels,
        final Path observedMapXml) {
      this.showGame = showGame;
      this.showAllObservations = showAllObservations;
      this.observationsByPlayer = observationsByPlayer;
      this.agentLabels = agentLabels;
      this.observedMapXml = observedMapXml;
    }

    @Override
    public LaunchAction launchAction(
        final Collection<PlayerTypes.Type> playerTypes, final Consumer<IGame> gameStarted) {
      return new HeadedLaunchAction(setupModel.getUi()) {
        @Override
        public Collection<PlayerTypes.Type> getPlayerTypes() {
          return playerTypes;
        }

        @Override
        public void startGame(
            final LocalPlayers localPlayers,
            final IGame game,
            final Set<Player> localGamePlayers,
            final Chat chat) {
          gameStarted.accept(game);
          if (showGame) {
            super.startGame(localPlayers, game, localGamePlayers, chat);
          } else {
            TestbenchRun.setUpWithoutWindow(game);
          }
          SwingUtilities.invokeLater(
              () ->
                  stateObservationWindow.observe(
                      game.getData(),
                      showAllObservations,
                      observationsByPlayer,
                      agentLabels,
                      observedMapXml));
        }

        @Override
        public boolean promptGameStop(
            final String status, final String title, final Path mapLocation) {
          return true;
        }
      };
    }

    @Override
    public void afterGame(final Runnable nextStep) {
      stateObservationWindow.stopObserving();
      SwingUtilities.invokeLater(nextStep);
    }

    @Override
    public void onRunFinished() {
      if (exitWhenBatchDone) {
        ExitStatus.SUCCESS.exit();
      }
      run = null;
      setupModel.showTestbenchMenu();
      games.strategy.engine.framework.ui.MainFrame.show();
    }

    @Override
    public void onRunSuspended(final String message) {
      stateObservationWindow.stopObserving();
      run = null;
      if (exitWhenBatchDone) {
        System.err.print(message);
        ExitStatus.SUCCESS.exit();
      }
      SwingUtilities.invokeLater(
          () -> {
            final Path selectedMap = Path.of(mapPath.getText());
            if (Files.isRegularFile(selectedMap)) {
              gameSelectorModel.loadMapForTestbench(selectedMap);
            }
            JOptionPane.showMessageDialog(
                TestbenchSetupPanel.this,
                message,
                "Planning agent testbench",
                JOptionPane.INFORMATION_MESSAGE);
            setupModel.showTestbenchMenu();
            games.strategy.engine.framework.ui.MainFrame.show();
          });
    }

    @Override
    public void onRunFailed(final String message) {
      run = null;
      if (exitWhenBatchDone) {
        System.err.println(message + "; stopping.");
        ExitStatus.FAILURE.exit();
      }
      showError(message);
      SwingComponents.setFrameFromComponentVisible(TestbenchSetupPanel.this);
    }
  }
}
