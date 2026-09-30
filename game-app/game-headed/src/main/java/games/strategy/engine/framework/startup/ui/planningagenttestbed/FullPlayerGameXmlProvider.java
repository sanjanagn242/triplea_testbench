package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import games.strategy.engine.data.GamePlayer;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;

/** Current full-observation rule XML provider; partial-visibility filters can replace this. */
public final class FullPlayerGameXmlProvider implements PlayerGameXmlProvider {
  @Override
  public String getXmlForPlayer(final Path commonGameXml, final GamePlayer player)
      throws IOException {
    return Files.readString(commonGameXml);
  }
}
