package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import games.strategy.engine.framework.startup.mc.HeadedPlayerTypes;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.player.Player;
import java.util.ArrayList;
import java.util.List;
import java.nio.file.Path;

/** Agent types implemented specifically for the planning-agent testbench. */
public final class TestbenchAgentRegistry {
  private TestbenchAgentRegistry() {}

  public static List<PlayerTypes.Type> getPlayerTypes() {
    return getPlayerTypes(null, 1);
  }

  public static List<PlayerTypes.Type> getPlayerTypes(final Path commonGameXml) {
    return getPlayerTypes(commonGameXml, 1);
  }

  public static List<PlayerTypes.Type> getPlayerTypes(
      final Path commonGameXml, final int simulationNumber) {
    final List<PlayerTypes.Type> playerTypes = new ArrayList<>(HeadedPlayerTypes.getPlayerTypes());
    playerTypes.add(
        new PlayerTypes.Type(SimplePlanningAgentAi.PLAYER_LABEL) {
          @Override
          public Player newPlayerWithName(final String name) {
            return new SimplePlanningAgentAi(name, commonGameXml, simulationNumber);
          }
        });
    return List.copyOf(playerTypes);
  }
}
