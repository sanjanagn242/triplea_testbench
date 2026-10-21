package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import games.strategy.engine.chat.Chat;
import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.framework.CliProperties;
import games.strategy.engine.framework.GameShutdownRegistry;
import games.strategy.engine.framework.IGame;
import games.strategy.engine.framework.LocalPlayers;
import games.strategy.engine.framework.ServerGame;
import games.strategy.engine.framework.map.file.system.loader.InstalledMapsListing;
import games.strategy.engine.framework.startup.launcher.ILauncher;
import games.strategy.engine.framework.startup.launcher.LocalLauncher;
import games.strategy.engine.framework.startup.launcher.MapNotFoundException;
import games.strategy.engine.framework.startup.launcher.local.PlayerCountrySelection;
import games.strategy.engine.framework.startup.mc.HeadedLaunchAction;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.framework.startup.ui.SetupPanel;
import games.strategy.engine.framework.startup.ui.panels.main.HeadedServerSetupModel;
import games.strategy.engine.framework.startup.ui.panels.main.game.selector.GameSelectorModel;
import games.strategy.engine.player.Player;
import games.strategy.triplea.ResourceLoader;
import games.strategy.triplea.delegate.EndRoundDelegate;
import games.strategy.triplea.settings.ClientSetting;
import games.strategy.triplea.ui.display.HeadlessDisplay;
import java.awt.BorderLayout;
import java.awt.FlowLayout;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
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
import org.triplea.sound.HeadlessSoundChannel;
import org.triplea.swing.SwingComponents;
import org.triplea.util.ExitStatus;

/** Interactive configuration for repeatable local planning-agent experiments. */
public final class TestbenchSetupPanel extends SetupPanel {
  private static final long serialVersionUID = 1L;
  private static final Gson GSON = new GsonBuilder().setPrettyPrinting().create();
  private static final int DEFAULT_AI_MOVE_PAUSE_MS = 300;
  private static final int DEFAULT_AI_COMBAT_STEP_PAUSE_MS = 1000;
  private static final int MAX_AI_PAUSE_MS = 3000;
  private static final String CHECKPOINT_SAVE = "checkpoint.tsvg";
  private static final String CHECKPOINT_INFO = "checkpoint.json";

  private final HeadedServerSetupModel setupModel;
  private final GameSelectorModel gameSelectorModel;
  private final JTextField mapPath = new JTextField(22);
  private final JLabel mapStatus =
      new JLabel("Each game's maps are stored in planning-agent-testbed/games/<game>/map");
  private final JSpinner gamesCount = new JSpinner(new SpinnerNumberModel(2, 1, 100_000, 1));
  private final JSpinner firstGameNumber = new JSpinner(new SpinnerNumberModel(1, 1, 1_000_000, 1));
  private final JSpinner roundLimit = new JSpinner(new SpinnerNumberModel(100, 1, 100_000, 1));
  private final JSpinner aiMovePauseMs =
      new JSpinner(new SpinnerNumberModel(DEFAULT_AI_MOVE_PAUSE_MS, 0, MAX_AI_PAUSE_MS, 10));
  private final JSpinner aiCombatStepPauseMs =
      new JSpinner(new SpinnerNumberModel(DEFAULT_AI_COMBAT_STEP_PAUSE_MS, 0, MAX_AI_PAUSE_MS, 10));
  private final JCheckBox showGameWindow = new JCheckBox("Show game window", true);
  private final JCheckBox showObservationWindow = new JCheckBox("Visualize observations", true);
  private final JCheckBox clearAgentOutputs =
      new JCheckBox("Clear previous agent outputs (metrics, logs, …) when the run starts", false);
  private final JPanel factionRows = new JPanel(new GridBagLayout());
  private final Map<String, JComboBox<String>> assignments = new LinkedHashMap<>();
  private final Map<String, JCheckBox> observationAssignments = new LinkedHashMap<>();
  private final Path initialConfig;
  private Path activeConfigurationFile;
  private MapReference activeMapReference;
  private boolean experimentRunning;
  private int plannedGames;
  private int completedGames;
  // Number of the run's first game; game n of the run is numbered firstGame + n - 1.
  private int firstGame = 1;
  private Instant currentGameStartedAt;
  // Start of the current run (also of a resumed one); agents measure their run time limit from it.
  private Instant runStartedAt = Instant.now();
  // Time the current game had already run before it was last suspended.
  private long previouslyElapsedMillis;
  // Set when a suspended game has been loaded and the next launch resumes it instead of a new run.
  private Checkpoint pendingResume;
  private volatile IGame currentGame;
  private volatile boolean currentGameSuspended;
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

  /**
   * Name of the games/(game) folder being played (e.g. capture-the-flag-diag): the configuration's
   * game folder, else the one holding the map; empty when neither is under games/.
   */
  private String gameFolderName() {
    final Path gamesDirectory =
        TestbenchMapRepository.gamesDirectory().toAbsolutePath().normalize();
    final Optional<Path> fromMap =
        mapPath.getText().isBlank()
            ? Optional.empty()
            : Optional.ofNullable(Path.of(mapPath.getText()).toAbsolutePath().normalize())
                .flatMap(
                    map -> {
                      for (Path p = map; p != null; p = p.getParent()) {
                        if (gamesDirectory.equals(p.getParent())) {
                          return Optional.of(p);
                        }
                      }
                      return Optional.empty();
                    });
    return TestbenchMapRepository.gameDirectoryOf(activeConfigurationFile)
        .or(() -> fromMap)
        .map(directory -> directory.getFileName().toString())
        .orElse("");
  }

  /** Faction -> stable agent ID (as written in configurations) for the current assignments. */
  private Map<String, String> selectedAgentIds() {
    final Map<String, String> result = new LinkedHashMap<>();
    selectedAssignments()
        .forEach((name, label) -> result.put(name, TestbenchAgentRegistry.idForSelection(label)));
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
      firstGameNumber.setValue(
          config.firstGameNumber == null ? 1 : Math.max(1, config.firstGameNumber));
      roundLimit.setValue(Math.max(1, config.roundLimit));
      aiMovePauseMs.setValue(clampPause(config.aiMovePauseMs, DEFAULT_AI_MOVE_PAUSE_MS));
      aiCombatStepPauseMs.setValue(
          clampPause(config.aiCombatStepPauseMs, DEFAULT_AI_COMBAT_STEP_PAUSE_MS));
      final String guiProperty = System.getProperty(CliProperties.TRIPLEA_TESTBENCH_GUI, "");
      showGameWindow.setSelected(
          guiProperty.isBlank()
              ? config.showGame == null || config.showGame
              : Boolean.parseBoolean(guiProperty));
      showObservationWindow.setSelected(config.showObservations == null || config.showObservations);
      clearAgentOutputs.setSelected(Boolean.TRUE.equals(config.clearAgentOutputs));
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
    final Optional<Checkpoint> checkpoint = readCheckpoint();
    if (checkpoint.isEmpty()) {
      launchIfAutoStartPending();
      return;
    }
    final Checkpoint suspended = checkpoint.get();
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
    final Path saveFile = checkpointDirectory().resolve(CHECKPOINT_SAVE);
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
      experimentRunning = true;
      runStartedAt = Instant.now();
      if (pendingResume != null) {
        // Keep the logs and agent outputs of the suspended run and continue its numbering.
        completedGames = pendingResume.completedGames();
        plannedGames = Math.max((int) gamesCount.getValue(), completedGames + 1);
        firstGame = pendingResume.firstGame();
        previouslyElapsedMillis = pendingResume.elapsedMillis();
        pendingResume = null;
      } else {
        deleteAgentLogs();
        if (clearAgentOutputs.isSelected()) {
          clearAssignedAgentOutputs();
        }
        deleteCheckpoint();
        plannedGames = (int) gamesCount.getValue();
        firstGame = (int) firstGameNumber.getValue();
        completedGames = 0;
        previouslyElapsedMillis = 0;
      }
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
    final boolean showGame = showGameWindow.isSelected();
    final boolean showAllObservations = showObservationWindow.isSelected();
    final Map<String, Boolean> observationsByPlayer = selectedObservationAssignments();
    final Map<String, String> agentLabels = selectedAssignments();
    final Path observedMapXml = Path.of(mapPath.getText());
    final int simulationNumber = firstGame + completedGames;
    final List<PlayerTypes.Type> selectedGamePlayerTypes =
        TestbenchAgentRegistry.getPlayerTypes(
            Path.of(mapPath.getText()),
            simulationNumber,
            runStartedAt.toEpochMilli(),
            this::suspendCurrentGame,
            Map.of(
                "TRIPLEA_TESTBENCH_AGENTS",
                new Gson().toJson(selectedAgentIds()),
                "TRIPLEA_TESTBENCH_GAME",
                gameFolderName()));
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
                currentGame = game;
                if (showGame) {
                  super.startGame(localPlayers, game, localGamePlayers, chat);
                } else {
                  // What the game window would set up, minus the window: the map's resources
                  // (the engine reads them, e.g. for politics texts), no display and no sound.
                  final String mapName = game.getData().getMapName();
                  game.setResourceLoader(
                      new ResourceLoader(
                          InstalledMapsListing.searchAllMapsForMapName(mapName)
                              .orElseThrow(() -> new MapNotFoundException(mapName))));
                  game.setDisplay(new HeadlessDisplay());
                  game.setSoundChannel(new HeadlessSoundChannel());
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
            },
            this::onGameCompleted));
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
    final Path directory = checkpointDirectory();
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
              activeConfigurationKey(),
              completedGames,
              plannedGames,
              firstGame,
              game.getData().getSequence().getRound(),
              currentGameElapsedMillis(),
              Instant.now().toString());
      Files.writeString(directory.resolve(CHECKPOINT_INFO), GSON.toJson(checkpoint));
    } catch (final IOException | RuntimeException e) {
      System.err.println("Could not save the suspended testbench game: " + e.getMessage());
      return false;
    }
    currentGameSuspended = true;
    serverGame.stopGame();
    return true;
  }

  private void onGameCompleted() {
    stateObservationWindow.stopObserving();
    GameShutdownRegistry.runShutdownActions();
    final IGame finishedGame = currentGame;
    currentGame = null;
    if (currentGameSuspended) {
      currentGameSuspended = false;
      onGameSuspended();
      return;
    }
    if (finishedGame != null) {
      appendGameResult(finishedGame.getData(), firstGame + completedGames);
    }
    completedGames++;
    previouslyElapsedMillis = 0;
    deleteCheckpoint();
    SwingUtilities.invokeLater(
        () -> {
          final Path selectedMap = Path.of(mapPath.getText());
          if (!Files.isRegularFile(selectedMap)
              || !gameSelectorModel.loadMapForTestbench(selectedMap)) {
            experimentRunning = false;
            if (exitWhenBatchDone) {
              System.err.println("Could not reload the testbench map; stopping: " + selectedMap);
              ExitStatus.FAILURE.exit();
            }
            showError("Could not reload the testbench map after the game ended: " + selectedMap);
            SwingComponents.setFrameFromComponentVisible(this);
            return;
          }
          if (completedGames < plannedGames) {
            getLauncher().ifPresent(launcher -> ThreadRunner.runInNewThread(launcher::launch));
          } else if (exitWhenBatchDone) {
            ExitStatus.SUCCESS.exit();
          } else {
            experimentRunning = false;
            plannedGames = 0;
            completedGames = 0;
            setupModel.showTestbenchMenu();
            games.strategy.engine.framework.ui.MainFrame.show();
          }
        });
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
            activeConfigurationKey(),
            gameFolderName(),
            gameNumber,
            selectedAgentIds(),
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

  /** Ends the run after a suspension; starting the same configuration again resumes the game. */
  private void onGameSuspended() {
    final Path checkpoint = checkpointDirectory().resolve(CHECKPOINT_SAVE);
    final String message =
        String.format(
            java.util.Locale.ROOT,
            "Game %d/%d suspended after %.1f seconds; run the same configuration again to resume."
                + "%nCheckpoint: %s%n",
            firstGame + completedGames,
            lastGameNumber(),
            currentGameElapsedMillis() / 1000.0,
            checkpoint);
    experimentRunning = false;
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
              this, message, "Planning agent testbench", JOptionPane.INFORMATION_MESSAGE);
          plannedGames = 0;
          completedGames = 0;
          setupModel.showTestbenchMenu();
          games.strategy.engine.framework.ui.MainFrame.show();
        });
  }

  /** Number of the run's last game. */
  private int lastGameNumber() {
    return firstGame + plannedGames - 1;
  }

  private long currentGameElapsedMillis() {
    return previouslyElapsedMillis
        + Duration.between(currentGameStartedAt, Instant.now()).toMillis();
  }

  /**
   * Folder of this configuration's suspended game: planning-agent-testbed/saves/(game)/(config),
   * where (game) is the folder holding the configuration's config folder.
   */
  private Path checkpointDirectory() {
    final Path saves =
        TestbenchMapRepository.repositoryRoot().resolve("planning-agent-testbed/saves");
    if (activeConfigurationFile == null) {
      return saves.resolve("unsaved-configuration");
    }
    final String configName =
        activeConfigurationFile.getFileName().toString().replaceFirst("\\.json$", "");
    final Path gameDirectory =
        TestbenchMapRepository.gameDirectoryOf(activeConfigurationFile)
            .orElse(activeConfigurationFile.getParent().getParent());
    return gameDirectory == null || gameDirectory.getFileName() == null
        ? saves.resolve(configName)
        : saves.resolve(gameDirectory.getFileName().toString()).resolve(configName);
  }

  private String activeConfigurationKey() {
    return activeConfigurationFile == null
        ? ""
        : activeConfigurationFile.toAbsolutePath().normalize().toString();
  }

  /** The suspended game of the active configuration, if its checkpoint files are present. */
  private Optional<Checkpoint> readCheckpoint() {
    final Path directory = checkpointDirectory();
    final Path info = directory.resolve(CHECKPOINT_INFO);
    if (!Files.isRegularFile(info) || !Files.isRegularFile(directory.resolve(CHECKPOINT_SAVE))) {
      return Optional.empty();
    }
    try {
      return Optional.ofNullable(GSON.fromJson(Files.readString(info), Checkpoint.class))
          .filter(checkpoint -> activeConfigurationKey().equals(checkpoint.configuration()));
    } catch (IOException | RuntimeException e) {
      System.err.println("Could not read testbench checkpoint " + info + ": " + e.getMessage());
      return Optional.empty();
    }
  }

  private void deleteCheckpoint() {
    final Path directory = checkpointDirectory();
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
    selectedAssignments().values().stream()
        .distinct()
        .map(TestbenchAgentRegistry::definitionForSelection)
        .flatMap(Optional::stream)
        .flatMap(definition -> definition.resolvedOutputDirectories().stream())
        .forEach(TestbenchSetupPanel::emptyDirectory);
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

  private record TestbenchConfiguration(
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
      Boolean clearAgentOutputs) {}

  /** A suspended game: the run's progress when it was saved to checkpoint.tsvg. */
  private record Checkpoint(
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

  private record MapReference(
      String name, String gameName, List<String> aliases, String downloadUrl) {}

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
}
