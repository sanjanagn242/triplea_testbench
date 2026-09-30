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
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;
import java.util.Optional;
import lombok.extern.slf4j.Slf4j;
import org.triplea.java.collections.IntegerMap;

/** Bridges the external Python testbench agent to TripleA's validated game delegates. */
@Slf4j
public final class SimplePlanningAgentAi extends WeakAi {
  public static final String PLAYER_LABEL = "Simple Infantry (Python)";
  private static final Gson GSON = new GsonBuilder().serializeNulls().create();
  private static final String SCRIPT = "planning-agent-testbed/agents/simple-infantry/agent.py";
  private static final int PROTOCOL_VERSION = 1;
  private static final String PYTHON_ENV = "TRIPLEA_TESTBENCH_PYTHON";
  private static final String PYTHON_PROPERTY = "triplea.testbench.python";
  private final Path commonGameXml;
  private final int simulationNumber;
  private final PlayerGameXmlProvider gameXmlProvider = new FullPlayerGameXmlProvider();
  private Process agentProcess;
  private BufferedReader agentOutput;
  private BufferedWriter agentInput;
  private long nextRequestId;

  public SimplePlanningAgentAi(
      final String playerName, final Path commonGameXml, final int simulationNumber) {
    super(playerName, PLAYER_LABEL);
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
    if (decision == null || decision.purchaseCount() <= 0) {
      return;
    }
    final ProductionRule infantryRule =
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
            .orElse(null);
    if (infantryRule == null) {
      log.warn("{} has no production rule for infantry", player.getName());
      return;
    }
    final int infantryCost =
        infantryRule.getCosts().getInt(data.getResourceList().getResourceOrThrow("PUs"));
    if (infantryCost <= 0) {
      log.warn("Infantry production rule has no positive PUs cost");
      return;
    }
    final int count = Math.min(decision.purchaseCount(), pusToSpend / infantryCost);
    if (count > 0) {
      final IntegerMap<ProductionRule> purchases = new IntegerMap<>();
      purchases.put(infantryRule, count);
      final String error = purchaseDelegate.purchase(purchases);
      if (error != null) {
        log.warn("{} could not buy infantry: {}", player.getName(), error);
      }
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
    data.getMap().getTerritories().forEach(territory -> territories.put(territory.getName(), territory));
    final Map<String, Unit> units = new HashMap<>();
    data.getUnits().getUnits().forEach(unit -> units.put(unit.getId().toString(), unit));
    for (final MoveOrder order : decision.moves()) {
      final Territory from = territories.get(order.from());
      final Territory to = territories.get(order.to());
      if (from == null || to == null || order.unitIds() == null || order.unitIds().isEmpty()) {
        continue;
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
            moveDelegate.performMove(new MoveDescription(moving, new Route(from, to)));
        error.ifPresent(message -> log.debug("{} move rejected: {}", player.getName(), message));
      }
    }
  }

  @Override
  public void place(
      final boolean placeForBid,
      final IAbstractPlaceDelegate placeDelegate,
      final GameState data,
      final GamePlayer player) {
    // Placement is temporarily delegated to TripleA's built-in AI. Do not request a Python action.
    super.place(placeForBid, placeDelegate, data, player);
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
            final String error =
                battleDelegate.fightBattle(territory, type.isBombingRun(), type);
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
              simulationNumber);
      final String response = exchange(request);
      if (response == null || response.isBlank()) {
        log.error(
            "External testbench agent returned no decision during {} for {}", phase, player.getName());
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
              simulationNumber);
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

  private void startAgentProcess() {
    stopAgentProcess();
    try {
      final Path root = findRepositoryRoot();
      final Path script = root.resolve(SCRIPT);
      if (!Files.isRegularFile(script)) {
        throw new IOException("External testbench agent script was not found: " + script);
      }
      final String pythonCommand = pythonCommand();
      agentProcess =
          new ProcessBuilder(pythonCommand, script.toString()).directory(root.toFile()).start();
      agentInput =
          new BufferedWriter(
              new OutputStreamWriter(
                  agentProcess.getOutputStream(), java.nio.charset.StandardCharsets.UTF_8));
      agentOutput =
          new BufferedReader(
              new InputStreamReader(
                  agentProcess.getInputStream(), java.nio.charset.StandardCharsets.UTF_8));
    } catch (final IOException e) {
      log.error(
          "Could not start external testbench agent using '{}'. Set {} to the Python 3 executable path.",
          pythonCommand(),
          PYTHON_ENV,
          e);
      stopAgentProcess();
    }
  }

  private static String pythonCommand() {
    return System.getProperty(
        PYTHON_PROPERTY, System.getenv().getOrDefault(PYTHON_ENV, "python3"));
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

  private static Path findRepositoryRoot() {
    Path directory = Path.of(System.getProperty("user.dir")).toAbsolutePath();
    while (directory != null) {
      if (Files.isRegularFile(directory.resolve(SCRIPT))) {
        return directory;
      }
      directory = directory.getParent();
    }
    throw new IllegalStateException("Could not locate " + SCRIPT + " from the current directory");
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
      int gameNumber) {}

  private record GameStartResponse(boolean ready) {}

  private record Decision(
      int purchaseCount,
      String rule,
      String unitType,
      List<MoveOrder> moves,
      String placeAt,
      boolean fightAll) {}

  private record MoveOrder(String from, String to, List<String> unitIds) {}
}
