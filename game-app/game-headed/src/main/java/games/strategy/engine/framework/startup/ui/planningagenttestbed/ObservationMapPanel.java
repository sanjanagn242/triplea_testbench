package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import java.awt.BasicStroke;
import java.awt.Color;
import java.awt.Dimension;
import java.awt.Font;
import java.awt.FontMetrics;
import java.awt.Graphics;
import java.awt.Graphics2D;
import java.awt.RenderingHints;
import java.awt.event.MouseEvent;
import java.awt.geom.Ellipse2D;
import java.awt.geom.Point2D;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.LinkedHashMap;
import java.util.List;
import java.util.Map;
import java.util.Set;
import java.util.TreeMap;
import javax.annotation.Nullable;
import javax.swing.JPanel;
import javax.swing.ToolTipManager;

/**
 * Draws one player's {@link StateObservation} as a territory graph. Everything shown comes from the
 * observation itself (owners, neighbors, units, production, capitals, victory cities, resources,
 * reserve units); only node positions come from the map's centers.txt when available, which carries
 * no game state. Without it, a deterministic spring layout of the observation's neighbor graph is
 * used.
 */
final class ObservationMapPanel extends JPanel {
  private static final long serialVersionUID = 1L;
  private static final Color[] PALETTE = {
    new Color(0x1f77b4), new Color(0xd62728), new Color(0x2ca02c), new Color(0xff7f0e),
    new Color(0x9467bd), new Color(0x8c564b), new Color(0xe377c2), new Color(0x17becf),
    new Color(0xbcbd22), new Color(0x7f7f7f)
  };
  private static final Color NEUTRAL = new Color(0xd3d3d3);
  private static final Color WATER = new Color(0xcfe8ff);
  private static final Color IMPASSABLE = new Color(0x555555);
  private static final Color EDGE = new Color(0x999999);
  private static final Color OWN_BORDER = new Color(0xd4a017);
  private static final int NODE_RADIUS = 14;
  private static final int SIDE_PANEL_WIDTH = 300;
  private static final int MARGIN = 40;
  // Room around the graph for territory names (left/bottom) and unit stacks (right).
  private static final int LEFT_PAD = 70;
  private static final int BOTTOM_PAD = 70;
  private static final int UNIT_LABEL_PAD = 170;

  private final Map<String, Point2D.Double> centers;
  private transient StateObservation observation;
  private Map<String, Point2D.Double> layout = Map.of();
  private final Map<String, Point2D.Double> screen = new HashMap<>();

  /** {@code centers} maps territory names to layout positions; empty means compute a layout. */
  ObservationMapPanel(final Map<String, Point2D.Double> centers) {
    this.centers = centers;
    setBackground(Color.WHITE);
    setPreferredSize(new Dimension(1000, 720));
    ToolTipManager.sharedInstance().registerComponent(this);
  }

  void setObservation(final StateObservation observation) {
    this.observation = observation;
    if (observation != null && layout.isEmpty()) {
      layout = positionsFor(observation);
    }
    repaint();
  }

  /** Territory centers from a map's centers.txt near the game XML, or empty if none is found. */
  static Map<String, Point2D.Double> loadCenters(@Nullable final Path gameXml) {
    if (gameXml == null) {
      return Map.of();
    }
    Path directory = gameXml.toAbsolutePath().getParent();
    for (int level = 0; level < 4 && directory != null; level++) {
      final Path centers = directory.resolve("centers.txt");
      if (Files.isRegularFile(centers)) {
        return parseCenters(centers);
      }
      directory = directory.getParent();
    }
    return Map.of();
  }

  private static Map<String, Point2D.Double> parseCenters(final Path file) {
    final Map<String, Point2D.Double> result = new HashMap<>();
    try {
      for (final String line : Files.readAllLines(file)) {
        final int open = line.lastIndexOf('(');
        final int comma = line.indexOf(',', open);
        final int close = line.indexOf(')', open);
        if (open <= 0 || comma < 0 || close < comma) {
          continue;
        }
        result.put(
            line.substring(0, open).trim(),
            new Point2D.Double(
                Double.parseDouble(line.substring(open + 1, comma).trim()),
                Double.parseDouble(line.substring(comma + 1, close).trim())));
      }
    } catch (final IOException | NumberFormatException e) {
      return Map.of();
    }
    return result;
  }

  private Map<String, Point2D.Double> positionsFor(final StateObservation obs) {
    final boolean allCentered =
        obs.territories().stream().allMatch(t -> centers.containsKey(t.name()));
    return allCentered ? centers : springLayout(obs);
  }

  /** Deterministic Fruchterman-Reingold layout of the observation's neighbor graph. */
  private static Map<String, Point2D.Double> springLayout(final StateObservation obs) {
    final List<String> names =
        obs.territories().stream().map(StateObservation.TerritoryState::name).sorted().toList();
    final int n = names.size();
    final Map<String, Integer> index = new HashMap<>();
    for (int i = 0; i < n; i++) {
      index.put(names.get(i), i);
    }
    final double[] x = new double[n];
    final double[] y = new double[n];
    for (int i = 0; i < n; i++) {
      x[i] = Math.cos(2 * Math.PI * i / Math.max(1, n));
      y[i] = Math.sin(2 * Math.PI * i / Math.max(1, n));
    }
    final List<int[]> edges = new ArrayList<>();
    for (final StateObservation.TerritoryState t : obs.territories()) {
      for (final String neighbor : t.neighbors()) {
        final Integer a = index.get(t.name());
        final Integer b = index.get(neighbor);
        if (a != null && b != null && a < b) {
          edges.add(new int[] {a, b});
        }
      }
    }
    final double k = Math.sqrt(4.0 / Math.max(1, n));
    double temperature = 0.2;
    for (int iteration = 0; iteration < 300; iteration++) {
      final double[] dx = new double[n];
      final double[] dy = new double[n];
      for (int i = 0; i < n; i++) {
        for (int j = i + 1; j < n; j++) {
          final double ddx = x[i] - x[j];
          final double ddy = y[i] - y[j];
          final double dist = Math.max(1e-6, Math.hypot(ddx, ddy));
          final double force = k * k / dist;
          dx[i] += ddx / dist * force;
          dy[i] += ddy / dist * force;
          dx[j] -= ddx / dist * force;
          dy[j] -= ddy / dist * force;
        }
      }
      for (final int[] edge : edges) {
        final double ddx = x[edge[0]] - x[edge[1]];
        final double ddy = y[edge[0]] - y[edge[1]];
        final double dist = Math.max(1e-6, Math.hypot(ddx, ddy));
        final double force = dist * dist / k;
        dx[edge[0]] -= ddx / dist * force;
        dy[edge[0]] -= ddy / dist * force;
        dx[edge[1]] += ddx / dist * force;
        dy[edge[1]] += ddy / dist * force;
      }
      for (int i = 0; i < n; i++) {
        final double length = Math.max(1e-6, Math.hypot(dx[i], dy[i]));
        x[i] += dx[i] / length * Math.min(length, temperature);
        y[i] += dy[i] / length * Math.min(length, temperature);
      }
      temperature *= 0.985;
    }
    final Map<String, Point2D.Double> result = new HashMap<>();
    for (int i = 0; i < n; i++) {
      result.put(names.get(i), new Point2D.Double(x[i], y[i]));
    }
    return result;
  }

  @Override
  protected void paintComponent(final Graphics graphics) {
    super.paintComponent(graphics);
    final Graphics2D g = (Graphics2D) graphics.create();
    try {
      g.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
      g.setRenderingHint(
          RenderingHints.KEY_TEXT_ANTIALIASING, RenderingHints.VALUE_TEXT_ANTIALIAS_ON);
      if (observation == null) {
        g.setColor(Color.DARK_GRAY);
        g.drawString("Waiting for the first observation…", MARGIN, MARGIN);
        return;
      }
      fitToPanel();
      final Map<String, Color> ownerColors = ownerColors();
      drawEdges(g);
      for (final StateObservation.TerritoryState t : observation.territories()) {
        drawTerritory(g, t, ownerColors);
      }
      drawSidePanel(g, ownerColors);
    } finally {
      g.dispose();
    }
  }

  private void fitToPanel() {
    screen.clear();
    double minX = Double.MAX_VALUE;
    double minY = Double.MAX_VALUE;
    double maxX = -Double.MAX_VALUE;
    double maxY = -Double.MAX_VALUE;
    for (final Point2D.Double p : layout.values()) {
      minX = Math.min(minX, p.x);
      minY = Math.min(minY, p.y);
      maxX = Math.max(maxX, p.x);
      maxY = Math.max(maxY, p.y);
    }
    final double width =
        Math.max(1, getWidth() - SIDE_PANEL_WIDTH - LEFT_PAD - UNIT_LABEL_PAD - MARGIN);
    final double height = Math.max(1, getHeight() - MARGIN - BOTTOM_PAD);
    final double scale =
        Math.min(width / Math.max(1e-6, maxX - minX), height / Math.max(1e-6, maxY - minY));
    for (final StateObservation.TerritoryState t : observation.territories()) {
      final Point2D.Double p = layout.get(t.name());
      if (p != null) {
        screen.put(
            t.name(),
            new Point2D.Double(LEFT_PAD + (p.x - minX) * scale, MARGIN + (p.y - minY) * scale));
      }
    }
  }

  private Map<String, Color> ownerColors() {
    final Map<String, Color> colors = new HashMap<>();
    int i = 0;
    for (final StateObservation.PlayerState player : observation.players()) {
      colors.put(player.name(), PALETTE[i++ % PALETTE.length]);
    }
    return colors;
  }

  private void drawEdges(final Graphics2D g) {
    g.setColor(EDGE);
    g.setStroke(new BasicStroke(1.2f));
    final Set<String> drawn = new HashSet<>();
    for (final StateObservation.TerritoryState t : observation.territories()) {
      final Point2D.Double a = screen.get(t.name());
      for (final String neighbor : t.neighbors()) {
        final Point2D.Double b = screen.get(neighbor);
        final String key =
            t.name().compareTo(neighbor) < 0
                ? t.name() + "|" + neighbor
                : neighbor + "|" + t.name();
        if (a != null && b != null && drawn.add(key)) {
          g.drawLine((int) a.x, (int) a.y, (int) b.x, (int) b.y);
        }
      }
    }
  }

  private void drawTerritory(
      final Graphics2D g,
      final StateObservation.TerritoryState t,
      final Map<String, Color> ownerColors) {
    final Point2D.Double p = screen.get(t.name());
    if (p == null) {
      return;
    }
    final Color fill;
    if (t.impassable()) {
      fill = IMPASSABLE;
    } else if (t.owner() != null && ownerColors.containsKey(t.owner())) {
      fill = ownerColors.get(t.owner());
    } else {
      fill = t.water() ? WATER : NEUTRAL;
    }
    final Ellipse2D node =
        new Ellipse2D.Double(
            p.x - NODE_RADIUS, p.y - NODE_RADIUS, 2 * NODE_RADIUS, 2 * NODE_RADIUS);
    g.setColor(fill);
    g.fill(node);
    final boolean own = observation.observationPlayer().equals(t.owner());
    g.setColor(own ? OWN_BORDER : Color.BLACK);
    g.setStroke(new BasicStroke(own ? 3f : 1.4f));
    g.draw(node);

    g.setFont(getFont().deriveFont(Font.PLAIN, 10f));
    final FontMetrics metrics = g.getFontMetrics();
    final StringBuilder title = new StringBuilder(t.name());
    if (t.capitalFor() != null) {
      title.append(" ★");
    }
    if (t.victoryCityValue() > 0) {
      title.append(" VC");
    }
    g.setColor(Color.BLACK);
    final String label = title.toString();
    g.drawString(
        label, (int) (p.x - metrics.stringWidth(label) / 2.0), (int) (p.y + NODE_RADIUS + 12));
    if (t.production() > 0) {
      final String production = String.valueOf(t.production());
      g.setColor(contrast(fill));
      g.drawString(
          production, (int) (p.x - metrics.stringWidth(production) / 2.0), (int) (p.y + 4));
    }

    int line = 0;
    for (final Map.Entry<String, Integer> entry : unitCounts(t.units()).entrySet()) {
      g.setColor(Color.BLACK);
      g.drawString(
          entry.getValue() + " " + entry.getKey(),
          (int) (p.x + NODE_RADIUS + 4),
          (int) (p.y - 4 + line * 12));
      line++;
    }
  }

  private static Color contrast(final Color color) {
    final double luminance =
        0.299 * color.getRed() + 0.587 * color.getGreen() + 0.114 * color.getBlue();
    return luminance > 140 ? Color.BLACK : Color.WHITE;
  }

  /** "type (owner)" -> count, in a stable order. */
  private static Map<String, Integer> unitCounts(final List<StateObservation.UnitState> units) {
    final Map<String, Integer> counts = new TreeMap<>();
    for (final StateObservation.UnitState unit : units) {
      counts.merge(unit.type() + " (" + unit.owner() + ")", 1, Integer::sum);
    }
    return counts;
  }

  private void drawSidePanel(final Graphics2D g, final Map<String, Color> ownerColors) {
    final int x = getWidth() - SIDE_PANEL_WIDTH - MARGIN / 2;
    int y = MARGIN / 2 + 14;
    g.setColor(new Color(0xfffbe6));
    g.fillRoundRect(x - 10, MARGIN / 2, SIDE_PANEL_WIDTH, getHeight() - MARGIN, 12, 12);
    g.setColor(Color.BLACK);
    g.setFont(getFont().deriveFont(Font.BOLD, 13f));
    g.drawString("Round " + observation.round() + " · " + observation.step(), x, y);
    y += 18;
    g.setFont(getFont().deriveFont(Font.PLAIN, 12f));
    g.drawString("Active player: " + observation.activePlayer(), x, y);
    y += 16;
    g.drawString(
        observation.observationMode() + " observation for " + observation.observationPlayer(),
        x,
        y);
    y += 24;
    for (final StateObservation.PlayerState player : observation.players()) {
      g.setColor(ownerColors.getOrDefault(player.name(), NEUTRAL));
      g.fillOval(x, y - 10, 11, 11);
      g.setColor(Color.BLACK);
      g.setFont(getFont().deriveFont(Font.BOLD, 12f));
      g.drawString(player.name(), x + 16, y);
      y += 15;
      g.setFont(getFont().deriveFont(Font.PLAIN, 11f));
      final StringBuilder resources = new StringBuilder();
      player
          .resources()
          .forEach(
              (name, amount) -> resources.append(name).append(' ').append(amount).append("  "));
      g.drawString(resources.toString().trim(), x + 16, y);
      y += 14;
      if (!player.technologies().isEmpty()) {
        g.drawString("Tech: " + String.join(", ", player.technologies()), x + 16, y);
        y += 14;
      }
      final Map<String, Integer> reserve = new LinkedHashMap<>();
      for (final StateObservation.UnitState unit : player.unitsInReserve()) {
        reserve.merge(unit.type(), 1, Integer::sum);
      }
      if (!reserve.isEmpty()) {
        final StringBuilder text = new StringBuilder("To place:");
        reserve.forEach((type, count) -> text.append(' ').append(count).append(' ').append(type));
        g.drawString(text.toString(), x + 16, y);
        y += 14;
      }
      y += 8;
    }
    g.setColor(Color.DARK_GRAY);
    g.drawString("Gold border = owned by " + observation.observationPlayer(), x, y + 6);
    g.drawString("★ capital · VC victory city · number = production", x, y + 20);
  }

  @Override
  public String getToolTipText(final MouseEvent event) {
    if (observation == null) {
      return null;
    }
    for (final StateObservation.TerritoryState t : observation.territories()) {
      final Point2D.Double p = screen.get(t.name());
      if (p != null && p.distance(event.getX(), event.getY()) <= NODE_RADIUS + 2) {
        final StringBuilder html = new StringBuilder("<html><b>").append(t.name()).append("</b>");
        html.append("<br>owner: ").append(t.owner() == null ? "none" : t.owner());
        html.append("<br>production: ").append(t.production());
        if (t.capitalFor() != null) {
          html.append("<br>capital of ").append(t.capitalFor());
        }
        if (t.victoryCityValue() > 0) {
          html.append("<br>victory city value: ").append(t.victoryCityValue());
        }
        if (t.water()) {
          html.append("<br>water");
        }
        if (t.impassable()) {
          html.append("<br>impassable");
        }
        html.append("<br>neighbors: ").append(String.join(", ", t.neighbors()));
        for (final StateObservation.UnitState unit : t.units()) {
          html.append("<br>")
              .append(unit.type())
              .append(" (")
              .append(unit.owner())
              .append(") hits ")
              .append(unit.hits())
              .append(", movement left ")
              .append(unit.movementLeft());
        }
        return html.append("</html>").toString();
      }
    }
    return null;
  }
}
