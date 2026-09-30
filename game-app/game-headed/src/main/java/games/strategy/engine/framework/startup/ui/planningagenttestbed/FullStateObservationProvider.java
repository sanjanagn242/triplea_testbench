package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.data.Territory;
import games.strategy.engine.data.Unit;
import games.strategy.engine.data.Resource;
import games.strategy.engine.data.properties.IEditableProperty;
import games.strategy.triplea.attachments.TerritoryAttachment;
import games.strategy.triplea.attachments.UnitAttachment;
import games.strategy.triplea.delegate.TechAdvance;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;

/** Fully observable snapshot implementation; future visibility policies can replace this provider. */
public final class FullStateObservationProvider implements StateObservationProvider {
  @Override
  public StateObservation observe(final GameData gameData, final GamePlayer player) {
    try (GameData.Unlocker ignored = gameData.acquireReadLock()) {
      final var step = gameData.getSequence().getStep();
      final List<StateObservation.PlayerState> players =
          gameData.getPlayerList().getPlayers().stream()
              .filter(candidate -> !candidate.isNull())
              .sorted(Comparator.comparing(GamePlayer::getName))
              .map(
                  candidate -> {
                    final Map<String, Integer> resources = new TreeMap<>();
                    gameData.getResourceList().getResources().stream()
                        .sorted(Comparator.comparing(Resource::getName))
                        .forEach(
                            resource ->
                                resources.put(
                                    resource.getName(),
                                    candidate.getResources().getQuantity(resource)));
                    final List<String> technologies =
                        candidate.getTechnologyFrontierList().getFrontiers().stream()
                            .flatMap(frontier -> frontier.getTechs().stream())
                            .filter(advance -> advance.hasTech(candidate.getTechAttachment()))
                            .map(TechAdvance::getName)
                            .sorted()
                            .toList();
                    final List<StateObservation.UnitState> unitsInReserve =
                        candidate.getUnits().stream().map(FullStateObservationProvider::unitState).toList();
                    return new StateObservation.PlayerState(
                        candidate.getName(), resources, technologies, unitsInReserve);
                  })
              .toList();
      final List<StateObservation.TerritoryState> territories =
          gameData.getMap().getTerritories().stream()
              .sorted(Comparator.comparing(Territory::getName))
              .map(
                  territory ->
                      {
                        final TerritoryAttachment attachment =
                            TerritoryAttachment.get(territory).orElse(null);
                        return new StateObservation.TerritoryState(
                            territory.getName(),
                            territory.isWater(),
                            territory.getOwner().isNull() ? null : territory.getOwner().getName(),
                            gameData.getMap().getNeighbors(territory).stream()
                                .map(Territory::getName)
                                .sorted()
                                .toList(),
                            attachment == null ? 0 : attachment.getProduction(),
                            attachment == null ? 0 : attachment.getUnitProduction(),
                            attachment == null ? null : attachment.getCapital().orElse(null),
                            attachment == null ? 0 : attachment.getVictoryCity(),
                            attachment != null && attachment.getIsImpassable(),
                            territory.getUnitCollection().getUnitCount() == 0
                                ? List.of()
                                : territory.getUnitCollection().getUnits().stream()
                                    .sorted(
                                        Comparator.comparing(
                                                (Unit unit) -> unit.getType().getName())
                                            .thenComparing(unit -> unit.getOwner().getName()))
                                    .map(FullStateObservationProvider::unitState)
                                    .toList());
                      })
              .toList();
      final GamePlayer activePlayer = step.getPlayerId();
      return new StateObservation(
          "FULL",
          player.getName(),
          gameData.getGameName(),
          gameData.getMapName(),
          gameData.getSequence().getRound(),
          step.getDisplayName(),
          activePlayer == null ? null : activePlayer.getName(),
          gameData.getDiceSides(),
          gameProperties(gameData),
          players,
          territories);
    }
  }

  private static StateObservation.UnitState unitState(final Unit unit) {
    final UnitAttachment attachment = unit.getUnitAttachment();
    return new StateObservation.UnitState(
        unit.getId().toString(),
        unit.getType().getName(),
        unit.getOwner().getName(),
        unit.getHits(),
        unit.getUnitDamage(),
        unit.getAlreadyMoved().toPlainString(),
        unit.getMovementLeft().toPlainString(),
        attachment.getMovement(unit.getOwner()),
        attachment.getAttack(unit.getOwner()),
        attachment.getDefense(unit.getOwner()),
        attachment.isAir(),
        attachment.isSea(),
        attachment.isInfrastructure());
  }

  private static Map<String, String> gameProperties(final GameData gameData) {
    final Map<String, String> properties = new TreeMap<>();
    gameData
        .getProperties()
        .getConstantPropertiesByName()
        .forEach((name, value) -> properties.put(name, String.valueOf(value)));
    for (final IEditableProperty<?> property : gameData.getProperties().getEditableProperties()) {
      properties.put(property.getName(), String.valueOf(property.getValue()));
    }
    return properties;
  }
}
