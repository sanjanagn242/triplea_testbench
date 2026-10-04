package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import static org.assertj.core.api.Assertions.assertThat;
import static org.mockito.Mockito.mock;

import games.strategy.engine.ClientFileSystemHelper;
import games.strategy.engine.data.GameData;
import games.strategy.engine.data.gameparser.GameParser;
import games.strategy.engine.framework.GameRunner;
import games.strategy.engine.framework.ServerGame;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.player.Player;
import games.strategy.net.LocalNoOpMessenger;
import games.strategy.net.Messengers;
import games.strategy.net.websocket.ClientNetworkBridge;
import games.strategy.triplea.delegate.EndRoundDelegate;
import games.strategy.triplea.settings.ClientSetting;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.HashMap;
import java.util.Map;
import java.util.Set;
import lombok.extern.slf4j.Slf4j;
import org.junit.jupiter.api.Test;
import org.junit.jupiter.api.condition.EnabledIfEnvironmentVariable;
import org.sonatype.goodies.prefs.memory.MemoryPreferences;
import org.triplea.game.server.HeadlessGameServer;
import org.triplea.game.server.HeadlessLaunchAction;
import org.triplea.io.FileUtils;

/**
 * Plays a game (Capture the Flag by default) in the real engine with an external testbench agent
 * controlling one faction and TripleA's Easy AI controlling the rest. Opt-in because it launches
 * the agent's process: set TRIPLEA_LIVE_AGENT_TEST=1 (optionally TRIPLEA_LIVE_AGENT=python-mcts,
 * TRIPLEA_LIVE_FACTION=Russians, TRIPLEA_LIVE_MAX_ROUNDS=60, and TRIPLEA_LIVE_XML=game XML path
 * relative to the repository root).
 */
@Slf4j
@EnabledIfEnvironmentVariable(named = "TRIPLEA_LIVE_AGENT_TEST", matches = "1")
class ExternalAgentLiveGameTest {
  @Test
  void externalAgentPlaysCaptureTheFlag() throws Exception {
    HeadlessLaunchAction.setSkipMapResourceLoading(true);
    ClientSetting.setPreferences(new MemoryPreferences());
    ClientSetting.aiMovePauseDuration.setValue(0);
    ClientSetting.aiCombatStepPauseDuration.setValue(0);
    final Path tempRoot = FileUtils.newTempFolder();
    FileUtils.writeToFile(tempRoot.resolve(".triplea-root"), "");
    Files.createDirectory(tempRoot.resolve("assets"));
    ClientFileSystemHelper.setCodeSourceFolder(tempRoot);
    System.setProperty(GameRunner.TRIPLEA_HEADLESS, "true");

    final String agentId = System.getenv().getOrDefault("TRIPLEA_LIVE_AGENT", "python-mcts");
    final String faction = System.getenv().getOrDefault("TRIPLEA_LIVE_FACTION", "Russians");
    final int maxRounds =
        Integer.parseInt(System.getenv().getOrDefault("TRIPLEA_LIVE_MAX_ROUNDS", "60"));
    final Path xml =
        TestbenchAgentRegistry.findRepositoryRoot()
            .resolve(
                System.getenv()
                    .getOrDefault(
                        "TRIPLEA_LIVE_XML",
                        "planning-agent-testbed/games/capture-the-flag/map/capture_the_flag/map/"
                            + "games/capture_the_flag.xml"));
    final GameData gameData = GameParser.parse(xml, false).orElseThrow();
    final String agentLabel = TestbenchAgentRegistry.selectionForId(agentId);
    final PlayerTypes.Type agentType =
        TestbenchAgentRegistry.getPlayerTypes(xml, 1).stream()
            .filter(type -> type.getLabel().equals(agentLabel))
            .findFirst()
            .orElseThrow();
    final Map<String, PlayerTypes.Type> playerTypes = new HashMap<>();
    for (var player : gameData.getPlayerList().getPlayers()) {
      playerTypes.put(
          player.getName(), player.getName().equals(faction) ? agentType : PlayerTypes.WEAK_AI);
    }
    final Set<Player> players = gameData.getGameLoader().newPlayers(playerTypes);
    final HeadlessLaunchAction launchAction =
        new HeadlessLaunchAction(mock(HeadlessGameServer.class));
    final ServerGame game =
        new ServerGame(
            gameData,
            players,
            new HashMap<>(),
            new Messengers(new LocalNoOpMessenger()),
            ClientNetworkBridge.NO_OP_SENDER,
            launchAction);
    game.setDelegateAutosavesEnabled(false);
    gameData.getGameLoader().startGame(game, players, launchAction, null);
    game.setStopGameOnDelegateExecutionStop(true);
    while (!game.isGameOver() && game.getData().getSequence().getRound() <= maxRounds) {
      game.runNextStep();
    }
    final EndRoundDelegate endRound = (EndRoundDelegate) game.getData().getEndRoundDelegate();
    log.info(
        "LIVE RESULT agent={} faction={} round={} gameOver={} winners={}",
        agentId,
        faction,
        game.getData().getSequence().getRound(),
        game.isGameOver(),
        endRound.getWinners());
    game.stopGame();
    assertThat(game.getData().getSequence().getRound()).isGreaterThan(2);
  }
}
