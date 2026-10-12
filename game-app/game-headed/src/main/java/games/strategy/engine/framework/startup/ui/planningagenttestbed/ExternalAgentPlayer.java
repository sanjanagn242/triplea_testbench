package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import games.strategy.engine.data.GameData;
import games.strategy.engine.data.GamePlayer;
import games.strategy.engine.data.GameState;
import games.strategy.engine.data.MoveDescription;
import games.strategy.engine.data.ProductionRule;
import games.strategy.engine.data.Route;
import games.strategy.engine.data.Territory;
import games.strategy.engine.data.Unit;
import games.strategy.engine.data.UnitType;
import games.strategy.engine.player.PlayerBridge;
import games.strategy.triplea.ai.weak.WeakAi;
import games.strategy.triplea.delegate.EndRoundDelegate;
import games.strategy.triplea.delegate.data.BattleListing;
import games.strategy.triplea.delegate.remote.IAbstractPlaceDelegate;
import games.strategy.triplea.delegate.remote.IBattleDelegate;
import games.strategy.triplea.delegate.remote.IMoveDelegate;
import games.strategy.triplea.delegate.remote.IPurchaseDelegate;
import games.strategy.triplea.delegate.remote.ITechDelegate;
import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.IOException;
import java.io.InputStreamReader;
import java.io.OutputStreamWriter;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.HashSet;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import java.util.Set;
import java.util.concurrent.CompletableFuture;
import java.util.concurrent.ExecutionException;
import java.util.concurrent.TimeUnit;
import java.util.concurrent.TimeoutException;
import lombok.extern.slf4j.Slf4j;
import org.triplea.java.collections.IntegerMap;

/**
 * Generic process adapter: exchanges protocol messages and applies returned actions via delegates.
 */
@Slf4j
public final class ExternalAgentPlayer extends WeakAi {
  private static final Gson GSON = new GsonBuilder().serializeNulls().create();
  private static final int PROTOCOL_VERSION = 1;
  // How long a finished game waits for the agent to acknowledge game_end before stopping it.
  private static final long GAME_END_TIMEOUT_SECONDS = 10;
  private final ExternalAgentDefinition definition;
  private final Path commonGameXml;
  private final int simulationNumber;
  private final PlayerGameXmlProvider gameXmlProvider = new FullPlayerGameXmlProvider();
  private Process agentProcess;
  private BufferedReader agentOutput;
  private BufferedWriter agentInput;
  private long nextRequestId;
  // Territories the agent explicitly targeted with a raid or a normal attack this move phase;
  // consulted when the engine asks whether bombers should bomb.
  private final Set<Territory> declaredRaids = new HashSet<>();
  private final Set<Territory> declaredAttacks = new HashSet<>();

  public ExternalAgentPlayer(
      final String playerName,
      final ExternalAgentDefinition definition,
      final Path commonGameXml,
      final int simulationNumber) {
    super(playerName, definition.name());
    this.definition = definition;
    this.commonGameXml = commonGameXml;
    this.simulationNumber = simulationNumber;
  }

  @Override
  public void initialize(final PlayerBridge playerBridge, final GamePlayer gamePlayer) {
    super.initialize(playerBridge, gamePlayer);
    initializeAgentProcess(gamePlayer);
  }

  @Override
  public void stopGame() {
    notifyGameEnd();
    stopAgentProcess();
    super.stopGame();
  }

  @Override
  public void purchase(
      final boolean purchaseForBid,
      final int pusToSpend,
      final IPurchaseDelegate purchaseDelegate,
      final GameData data,
      final GamePlayer player) {
    if (purchaseForBid) {
      return;
    }
    final Decision decision = askAgent("purchase", data, player, false);
    if (decision == null) {
      return;
    }
    final IntegerMap<ProductionRule> purchases = new IntegerMap<>();
    if (decision.purchases() != null) {
      for (final PurchaseOrder order : decision.purchases()) {
        if (order == null || order.rule() == null || order.count() <= 0) continue;
        player.getProductionFrontier().getRules().stream()
            .filter(rule -> rule.getName().equals(order.rule()))
            .findFirst()
            .ifPresent(rule -> purchases.put(rule, order.count()));
      }
    } else if (decision.purchaseCount() > 0 && decision.rule() != null) {
      // Support the original action shape; new agents should send the generic purchases[] list.
      player.getProductionFrontier().getRules().stream()
          .filter(
              rule ->
                  rule.getName().equals(decision.rule())
                      && rule.getResults().keySet().stream()
                          .anyMatch(
                              result ->
                                  result instanceof UnitType unitType
                                      && unitType.getName().equals(decision.unitType())))
          .findFirst()
          .ifPresent(rule -> purchases.put(rule, Math.min(decision.purchaseCount(), pusToSpend)));
    }
    if (!purchases.isEmpty()) {
      final String error = purchaseDelegate.purchase(purchases);
      if (error != null) log.warn("{} purchase was rejected: {}", player.getName(), error);
    }
  }

  @Override
  protected void tech(
      final ITechDelegate techDelegate, final GameData data, final GamePlayer player) {
    askAgent("tech", data, player, false);
  }

  @Override
  protected void move(
      final boolean nonCombat,
      final IMoveDelegate moveDelegate,
      final GameData data,
      final GamePlayer player) {
    final Decision decision =
        askAgent(nonCombat ? "nonCombatMove" : "combatMove", data, player, nonCombat);
    if (decision == null || decision.moves() == null) {
      return;
    }
    final Map<String, Territory> territories = new HashMap<>();
    data.getMap()
        .getTerritories()
        .forEach(territory -> territories.put(territory.getName(), territory));
    final Map<String, Unit> units = new HashMap<>();
    data.getUnits().getUnits().forEach(unit -> units.put(unit.getId().toString(), unit));
    declaredRaids.clear();
    declaredAttacks.clear();
    for (final MoveOrder order : decision.moves()) {
      // An optional full route (start..end) lets agents submit multi-step moves such as blitzes.
      final List<Territory> path = new ArrayList<>();
      if (order.route() != null && order.route().size() >= 2) {
        order.route().forEach(name -> path.add(territories.get(name)));
      } else {
        path.add(territories.get(order.from()));
        path.add(territories.get(order.to()));
      }
      if (path.contains(null) || order.unitIds() == null || order.unitIds().isEmpty()) {
        continue;
      }
      final Territory from = path.get(0);
      final Territory to = path.get(path.size() - 1);
      if ("RAID".equals(order.intent())) {
        declaredRaids.add(to);
      } else if ("NORMAL".equals(order.intent())) {
        declaredAttacks.add(to);
      }
      final List<Unit> moving = new ArrayList<>();
      for (final String id : order.unitIds()) {
        final Unit unit = units.get(id);
        if (unit != null && unit.getOwner().equals(player) && from.getUnits().contains(unit)) {
          moving.add(unit);
        }
      }
      if (!moving.isEmpty()) {
        final Optional<String> error =
            moveDelegate.performMove(new MoveDescription(moving, new Route(path)));
        error.ifPresent(message -> log.warn("{} move rejected: {}", player.getName(), message));
      }
    }
  }

  @Override
  public void place(
      final boolean placeForBid,
      final IAbstractPlaceDelegate placeDelegate,
      final GameState data,
      final GamePlayer player) {
    if (placeForBid) {
      super.place(placeForBid, placeDelegate, data, player);
      return;
    }
    final Decision decision = askAgent("place", getGameData(), player, false);
    if (decision == null || decision.placements() == null) {
      // Agents that do not return placements[] keep the built-in WeakAi placement.
      super.place(placeForBid, placeDelegate, data, player);
      return;
    }
    final Map<String, Unit> held = new HashMap<>();
    player.getUnits().forEach(unit -> held.put(unit.getId().toString(), unit));
    for (final PlacementOrder order : decision.placements()) {
      final Territory at = getGameData().getMap().getTerritoryOrNull(order.territory());
      if (at == null || order.unitIds() == null) continue;
      final List<Unit> units = new ArrayList<>();
      for (final String id : order.unitIds()) {
        final Unit unit = held.remove(id);
        if (unit != null) units.add(unit);
      }
      if (units.isEmpty()) continue;
      placeDelegate
          // The default two-argument overload is not callable through the remote proxy.
          .placeUnits(units, at, IAbstractPlaceDelegate.BidMode.NOT_BID)
          .ifPresent(error -> log.warn("{} placement rejected: {}", player.getName(), error));
    }
  }

  @Override
  public boolean shouldBomberBomb(final Territory territory) {
    if (declaredRaids.contains(territory)) return true;
    if (declaredAttacks.contains(territory)) return false;
    return super.shouldBomberBomb(territory);
  }

  @Override
  protected void battle(final IBattleDelegate battleDelegate) {
    final Decision decision = askAgent("battle", getGameData(), getGamePlayer(), false);
    if (decision == null) {
      super.battle(battleDelegate);
      return;
    }
    if (!decision.fightAll()) {
      return;
    }
    while (true) {
      final BattleListing listing = battleDelegate.getBattleListing();
      if (listing.isEmpty()) {
        return;
      }
      listing.forEachBattle(
          (type, territory) -> {
            final String error = battleDelegate.fightBattle(territory, type.isBombingRun(), type);
            if (error != null) {
              log.debug("{} battle request rejected: {}", getGamePlayer().getName(), error);
            }
          });
    }
  }

  private synchronized Decision askAgent(
      final String phase, final GameData data, final GamePlayer player, final boolean nonCombat) {
    try {
      final StateObservation observation = new FullStateObservationProvider().observe(data, player);
      if (agentProcess == null || !agentProcess.isAlive()) {
        initializeAgentProcess(player);
      }
      if (agentProcess == null || !agentProcess.isAlive()) {
        return null;
      }
      final AgentRequest request =
          new AgentRequest(
              "turn_request",
              PROTOCOL_VERSION,
              nextRequestId++,
              player.getName(),
              phase,
              null,
              null,
              observation,
              nonCombat,
              simulationNumber,
              null);
      final String response = exchange(request);
      if (response == null || response.isBlank()) {
        log.error(
            "External testbench agent returned no decision during {} for {}",
            phase,
            player.getName());
        return null;
      }
      final com.google.gson.JsonObject responseObject =
          GSON.fromJson(response, com.google.gson.JsonObject.class);
      if (responseObject.has("error")) {
        throw new IOException("Agent error: " + responseObject.get("error").getAsString());
      }
      return GSON.fromJson(response, Decision.class);
    } catch (final IOException e) {
      log.error(
          "Communication with the external testbench agent failed during {} for {}: {}",
          phase,
          player.getName(),
          e.toString(),
          e);
    } catch (final RuntimeException e) {
      log.error("Invalid response from external testbench agent", e);
    }
    return null;
  }

  private void initializeAgentProcess(final GamePlayer player) {
    startAgentProcess();
    if (agentProcess == null || !agentProcess.isAlive()) {
      return;
    }
    try {
      final AgentRequest request =
          new AgentRequest(
              "game_start",
              PROTOCOL_VERSION,
              nextRequestId++,
              player.getName(),
              null,
              gameXmlProvider.getXmlForPlayer(commonGameXml, player),
              new FullStateObservationProvider().observe(getGameData(), player),
              null,
              false,
              simulationNumber,
              null);
      final String response = exchange(request);
      final GameStartResponse acknowledgement = GSON.fromJson(response, GameStartResponse.class);
      if (acknowledgement == null || !acknowledgement.ready()) {
        throw new IOException("Agent did not accept game initialization: " + response);
      }
    } catch (final IOException | RuntimeException e) {
      log.error("Could not initialize external testbench agent", e);
      stopAgentProcess();
    }
  }

  /**
   * Sends {@code game_end} with the result and the final observation when the game has ended
   * (winners or the testbench round limit). A game stopped before that has no result and sends
   * nothing. The agent's acknowledgement is awaited for at most {@link #GAME_END_TIMEOUT_SECONDS}.
   */
  private void notifyGameEnd() {
    final GameData data = getGameData();
    final Process process = agentProcess;
    if (data == null || process == null || !process.isAlive()) {
      return;
    }
    final String status = data.getProperties().get(EndRoundDelegate.TESTBENCH_STATUS_PROPERTY, "");
    if (status.isBlank()) {
      return;
    }
    final GamePlayer player = getGamePlayer();
    final List<String> winners =
        java.util.Arrays.stream(
                data.getProperties()
                    .get(EndRoundDelegate.TESTBENCH_WINNERS_PROPERTY, "")
                    .split(","))
            .map(String::strip)
            .filter(name -> !name.isEmpty())
            .toList();
    final GameResult result =
        new GameResult(
            winners,
            winners.contains(player.getName()),
            data.getProperties()
                .get(
                    EndRoundDelegate.TESTBENCH_ROUNDS_PLAYED_PROPERTY,
                    data.getSequence().getRound()),
            status);
    try {
      final AgentRequest request =
          new AgentRequest(
              "game_end",
              PROTOCOL_VERSION,
              nextRequestId++,
              player.getName(),
              null,
              null,
              null,
              new FullStateObservationProvider().observe(data, player),
              false,
              simulationNumber,
              result);
      CompletableFuture.supplyAsync(
              () -> {
                try {
                  return exchange(request);
                } catch (final IOException e) {
                  throw new java.io.UncheckedIOException(e);
                }
              })
          .get(GAME_END_TIMEOUT_SECONDS, TimeUnit.SECONDS);
    } catch (final TimeoutException e) {
      log.warn("{} agent did not acknowledge game_end in time", player.getName());
      // The exchange still holds this player's lock while it reads; ending the process ends that
      // read.
      process.destroy();
    } catch (final ExecutionException | RuntimeException e) {
      log.warn("Could not send game_end to the {} agent: {}", player.getName(), e.toString());
    } catch (final InterruptedException e) {
      Thread.currentThread().interrupt();
    }
  }

  private void startAgentProcess() {
    stopAgentProcess();
    try {
      final Path root = TestbenchAgentRegistry.findRepositoryRoot();
      final Path agentDirectory = definition.agentDirectory();
      final List<String> command =
          definition.command().stream()
              .map(
                  part ->
                      part.replace("${repoRoot}", root.toString())
                          .replace("${agentDir}", agentDirectory.toString())
                          .replace(
                              "${entrypoint}",
                              agentDirectory.resolve(definition.entrypoint()).toString()))
              .toList();
      final ProcessBuilder processBuilder =
          new ProcessBuilder(command)
              .directory(agentDirectory.toFile())
              .redirectError(ProcessBuilder.Redirect.INHERIT);
      agentProcess = processBuilder.start();
      agentInput =
          new BufferedWriter(
              new OutputStreamWriter(
                  agentProcess.getOutputStream(), java.nio.charset.StandardCharsets.UTF_8));
      agentOutput =
          new BufferedReader(
              new InputStreamReader(
                  agentProcess.getInputStream(), java.nio.charset.StandardCharsets.UTF_8));
    } catch (final IOException e) {
      log.error("Could not start external agent '{}'", definition.id(), e);
      stopAgentProcess();
    }
  }

  private synchronized String exchange(final AgentRequest request) throws IOException {
    agentInput.write(GSON.toJson(request));
    agentInput.newLine();
    agentInput.flush();
    final String response = agentOutput.readLine();
    if (response == null) {
      throw new IOException("Agent closed its response stream");
    }
    return response;
  }

  private synchronized void stopAgentProcess() {
    try {
      if (agentInput != null) {
        agentInput.close();
      }
      if (agentProcess != null) {
        agentProcess.destroy();
      }
    } catch (final IOException e) {
      log.debug("Error closing external testbench agent", e);
    } finally {
      agentInput = null;
      agentOutput = null;
      agentProcess = null;
    }
  }

  private record AgentRequest(
      String type,
      int schemaVersion,
      long requestId,
      String playerName,
      String phase,
      String gameXml,
      StateObservation initialState,
      StateObservation state,
      boolean nonCombat,
      int gameNumber,
      GameResult result) {}

  private record GameResult(List<String> winners, boolean won, int roundsPlayed, String status) {}

  private record GameStartResponse(boolean ready) {}

  private record Decision(
      List<PurchaseOrder> purchases,
      int purchaseCount,
      String rule,
      String unitType,
      List<MoveOrder> moves,
      String placeAt,
      List<PlacementOrder> placements,
      boolean fightAll) {}

  private record PlacementOrder(String territory, List<String> unitIds) {}

  private record PurchaseOrder(String rule, int count) {}

  private record MoveOrder(
      String from, String to, List<String> unitIds, List<String> route, String intent) {}
}
