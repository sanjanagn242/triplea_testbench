package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import static org.assertj.core.api.Assertions.assertThat;

import java.awt.event.MouseEvent;
import java.awt.geom.Point2D;
import java.awt.image.BufferedImage;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.List;
import java.util.Map;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.io.TempDir;

class ObservationMapPanelTest {
  private static StateObservation.UnitState unit(final String type, final String owner) {
    return new StateObservation.UnitState(
        type + owner, type, owner, 0, 0, "0", "1", 1, 1, 2, false, false, false);
  }

  private static StateObservation observation() {
    return new StateObservation(
        "FULL",
        "Germans",
        "Test",
        "test_map",
        3,
        "Combat Move",
        "Germans",
        6,
        Map.of(),
        List.of(
            new StateObservation.PlayerState(
                "Germans", Map.of("PUs", 12), List.of(), List.of(unit("infantry", "Germans"))),
            new StateObservation.PlayerState("Russians", Map.of("PUs", 9), List.of(), List.of())),
        List.of(
            new StateObservation.TerritoryState(
                "Berlin",
                false,
                "Germans",
                List.of("Flag"),
                10,
                10,
                "Germans",
                1,
                false,
                List.of(unit("infantry", "Germans"), unit("infantry", "Germans"))),
            new StateObservation.TerritoryState(
                "Flag",
                false,
                null,
                List.of("Berlin", "Moscow", "Sea"),
                15,
                0,
                null,
                0,
                false,
                List.of(unit("infantry", "Neutral"))),
            new StateObservation.TerritoryState(
                "Moscow",
                false,
                "Russians",
                List.of("Flag"),
                10,
                10,
                "Russians",
                1,
                false,
                List.of()),
            new StateObservation.TerritoryState(
                "Sea", true, null, List.of("Flag"), 0, 0, null, 0, false, List.of())));
  }

  @Test
  void findsCentersNextToTheGameXml(@TempDir final Path dir) throws Exception {
    final Path map = Files.createDirectories(dir.resolve("my_map/map/games"));
    Files.writeString(
        map.getParent().resolve("centers.txt"),
"""
red north 1  (12,34)
Flag  (5,6)
""");
    final Path xml = Files.writeString(map.resolve("my_map.xml"), "<game/>");

    assertThat(ObservationMapPanel.loadCenters(xml))
        .containsEntry("red north 1", new Point2D.Double(12, 34))
        .containsEntry("Flag", new Point2D.Double(5, 6));
    assertThat(ObservationMapPanel.loadCenters(null)).isEmpty();
  }

  @Test
  void paintsAnObservationAndDescribesTerritoriesOnHover() {
    final ObservationMapPanel panel = new ObservationMapPanel(Map.of()); // spring layout fallback
    panel.setSize(900, 600);
    panel.setObservation(observation());

    final BufferedImage image = new BufferedImage(900, 600, BufferedImage.TYPE_INT_RGB);
    panel.paint(image.getGraphics());

    // Hover every pixel row/column on a coarse grid until a territory reports its tooltip.
    String tooltip = null;
    for (int x = 0; x < 900 && tooltip == null; x += 4) {
      for (int y = 0; y < 600 && tooltip == null; y += 4) {
        tooltip =
            panel.getToolTipText(
                new MouseEvent(panel, MouseEvent.MOUSE_MOVED, 0, 0, x, y, 0, false));
      }
    }
    assertThat(tooltip).isNotNull().contains("owner:").contains("neighbors:");
  }
}
