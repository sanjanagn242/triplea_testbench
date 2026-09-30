package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import games.strategy.engine.data.GamePlayer;
import java.io.IOException;
import java.nio.file.Path;

/** Supplies the game-rule XML visible to one agent player. */
public interface PlayerGameXmlProvider {
  String getXmlForPlayer(Path commonGameXml, GamePlayer player) throws IOException;
}
