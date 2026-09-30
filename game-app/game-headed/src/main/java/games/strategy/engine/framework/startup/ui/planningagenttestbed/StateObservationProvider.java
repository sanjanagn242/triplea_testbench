package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;

/** Builds the state view that will eventually be delivered to a planning agent. */
public interface StateObservationProvider {
  StateObservation observe(GameData gameData, GamePlayer player);
}
