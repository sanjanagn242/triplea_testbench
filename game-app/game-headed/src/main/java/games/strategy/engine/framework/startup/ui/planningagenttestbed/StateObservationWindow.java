package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.data.events.GameDataChangeListener;
import java.awt.BorderLayout;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.geom.Point2D;
import java.nio.file.Path;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.concurrent.atomic.AtomicBoolean;
import java.util.concurrent.atomic.AtomicLong;
import javax.annotation.Nullable;
import javax.swing.BorderFactory;
import javax.swing.JFrame;
import javax.swing.JLabel;
import javax.swing.JPanel;
import javax.swing.JTabbedPane;
import javax.swing.SwingUtilities;

/**
 * Shows selected players' live observations in tabs in one read-only window. Each tab draws the map
 * from exactly what that player's observation contains, and names the agent playing it.
 */
public final class StateObservationWindow {
  private final StateObservationProvider provider = new FullStateObservationProvider();
  private final JFrame frame = new JFrame("State Observations");
  private final JTabbedPane playerTabs = new JTabbedPane();
  private final Map<String, PlayerPane> playerPanes = new LinkedHashMap<>();
  private final AtomicLong revision = new AtomicLong();
  private GameData gameData;
  private GameDataChangeListener listener;

  public StateObservationWindow() {
    frame.setDefaultCloseOperation(JFrame.HIDE_ON_CLOSE);
    frame.setLayout(new BorderLayout());
    frame.add(playerTabs, BorderLayout.CENTER);
    frame.setMinimumSize(new Dimension(800, 560));
    frame.setSize(1100, 780);
    frame.setLocationByPlatform(true);
  }

  /**
   * Starts showing observations for the enabled players of {@code data}.
   *
   * @param agentNames faction name -> label of the agent assigned to it (shown in each tab)
   * @param gameXml the map's game XML, used only to find the map's centers.txt for node positions
   */
  public void observe(
      final GameData data,
      final boolean enabled,
      final Map<String, Boolean> enabledPlayers,
      final Map<String, String> agentNames,
      @Nullable final Path gameXml) {
    detachListener();
    gameData = data;
    revision.set(0);
    if (!enabled) {
      hideWindow();
      return;
    }
    final Map<String, GamePlayer> players = new LinkedHashMap<>();
    try (GameData.Unlocker ignored = data.acquireReadLock()) {
      data.getPlayerList().getPlayers().stream()
          .filter(player -> !player.isNull())
          .sorted(java.util.Comparator.comparing(GamePlayer::getName))
          .filter(player -> enabledPlayers.getOrDefault(player.getName(), true))
          .forEach(player -> players.put(player.getName(), player));
    }
    playerTabs.removeAll();
    playerPanes.clear();
    final Map<String, Point2D.Double> centers = ObservationMapPanel.loadCenters(gameXml);
    players.forEach(
        (name, player) -> {
          final String agent = agentNames.getOrDefault(name, "unassigned");
          final PlayerPane pane = new PlayerPane(name, agent, centers);
          pane.setObservedPlayer(data, player);
          playerPanes.put(name, pane);
          playerTabs.addTab(name + " — " + agent, pane.panel);
        });
    if (playerPanes.isEmpty()) {
      hideWindow();
      return;
    }
    frame.setTitle("State Observations — " + data.getGameName());
    listener =
        change -> {
          revision.incrementAndGet();
          SwingUtilities.invokeLater(() -> playerPanes.values().forEach(PlayerPane::refresh));
        };
    data.addDataChangeListener(listener);
    frame.setVisible(true);
    frame.toFront();
    playerPanes.values().forEach(PlayerPane::refresh);
  }

  public void stopObserving() {
    detachListener();
    playerPanes.values().forEach(PlayerPane::clear);
    SwingUtilities.invokeLater(() -> frame.setVisible(false));
  }

  private void detachListener() {
    if (gameData != null && listener != null) {
      gameData.removeDataChangeListener(listener);
    }
    gameData = null;
    listener = null;
  }

  private void hideWindow() {
    playerTabs.removeAll();
    playerPanes.clear();
    SwingUtilities.invokeLater(() -> frame.setVisible(false));
  }

  private final class PlayerPane {
    private final JPanel panel = new JPanel();
    private final ObservationMapPanel map;
    private final JLabel status = new JLabel("Preparing initial state snapshot…");
    private final AtomicBoolean refreshQueued = new AtomicBoolean();
    private GameData observedData;
    private GamePlayer player;

    private PlayerPane(
        final String faction, final String agent, final Map<String, Point2D.Double> centers) {
      map = new ObservationMapPanel(centers);
      final JLabel header = new JLabel(faction + " · agent: " + agent);
      header.setFont(header.getFont().deriveFont(Font.BOLD, 15f));
      header.setBorder(BorderFactory.createEmptyBorder(6, 8, 0, 8));
      status.setBorder(BorderFactory.createEmptyBorder(0, 8, 4, 8));
      panel.setLayout(new BorderLayout(6, 6));
      panel.add(header, BorderLayout.NORTH);
      panel.add(map, BorderLayout.CENTER);
      panel.add(status, BorderLayout.SOUTH);
    }

    private void setObservedPlayer(final GameData data, final GamePlayer observingPlayer) {
      observedData = data;
      player = observingPlayer;
    }

    private void clear() {
      observedData = null;
      player = null;
    }

    private void refresh() {
      if (observedData == null || !refreshQueued.compareAndSet(false, true)) {
        return;
      }
      SwingUtilities.invokeLater(
          () -> {
            final long renderedRevision = revision.get();
            try {
              final GameData currentData = observedData;
              final GamePlayer currentPlayer = player;
              if (currentData == null || currentPlayer == null) {
                return;
              }
              final StateObservation observation = provider.observe(currentData, currentPlayer);
              map.setObservation(observation);
              status.setText(
                  "Revision "
                      + revision.get()
                      + " · FULL observation · refreshed on each engine state change");
            } catch (final RuntimeException exception) {
              status.setText(
                  "Unable to create state snapshot for "
                      + (player == null ? "player" : player.getName())
                      + ": "
                      + exception.getClass().getSimpleName()
                      + ": "
                      + exception.getMessage());
            } finally {
              refreshQueued.set(false);
              if (observedData != null && revision.get() != renderedRevision) {
                refresh();
              }
            }
          });
    }
  }
}
