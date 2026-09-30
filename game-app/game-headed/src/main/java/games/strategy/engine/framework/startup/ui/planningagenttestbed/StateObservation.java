package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import java.util.List;
import java.util.Map;

/** JSON-friendly snapshot of the state currently visible to one player. */
public record StateObservation(
    String observationMode,
    String observationPlayer,
    String gameName,
    String mapName,
    int round,
    String step,
    String activePlayer,
    int diceSides,
    Map<String, String> gameProperties,
    List<PlayerState> players,
    List<TerritoryState> territories) {
  public record PlayerState(
      String name,
      Map<String, Integer> resources,
      List<String> technologies,
      List<UnitState> unitsInReserve) {}

  public record TerritoryState(
      String name,
      boolean water,
      String owner,
      List<String> neighbors,
      int production,
      int unitProduction,
      String capitalFor,
      int victoryCityValue,
      boolean impassable,
      List<UnitState> units) {}

  public record UnitState(
      String id,
      String type,
      String owner,
      int hits,
      int damage,
      String movementUsed,
      String movementLeft,
      int movement,
      int attack,
      int defense,
      boolean air,
      boolean sea,
      boolean infrastructure) {}
}
