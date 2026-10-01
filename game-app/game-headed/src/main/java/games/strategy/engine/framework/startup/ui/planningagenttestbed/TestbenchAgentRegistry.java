package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import com.google.gson.Gson;
import com.google.gson.GsonBuilder;
import games.strategy.engine.framework.startup.mc.HeadedPlayerTypes;
import games.strategy.engine.framework.startup.ui.PlayerTypes;
import games.strategy.engine.player.Player;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.Optional;

/** Discovers external agent definitions and creates player types backed by the shared adapter. */
public final class TestbenchAgentRegistry {
  private static final Gson GSON = new GsonBuilder().create();
  private static final String AGENT_DIRECTORY = "planning-agent-testbed/agents";

  private TestbenchAgentRegistry() {}

  public static List<PlayerTypes.Type> getPlayerTypes() {
    return getPlayerTypes(null, 1);
  }

  public static List<PlayerTypes.Type> getPlayerTypes(final Path commonGameXml) {
    return getPlayerTypes(commonGameXml, 1);
  }

  public static List<PlayerTypes.Type> getPlayerTypes(
      final Path commonGameXml, final int simulationNumber) {
    final List<PlayerTypes.Type> types = new ArrayList<>(HeadedPlayerTypes.getPlayerTypes());
    discoverAgents().stream()
        .map(
            definition ->
                new PlayerTypes.Type(definition.name()) {
                  @Override
                  public Player newPlayerWithName(final String name) {
                    return new ExternalAgentPlayer(
                        name, definition, commonGameXml, simulationNumber);
                  }
                })
        .forEach(types::add);
    return List.copyOf(types);
  }

  /** Stable manifest ID for a visible label; built-in TripleA player labels are returned as-is. */
  public static String idForSelection(final String selection) {
    if (selection == null) return null;
    return discoverAgents().stream()
        .filter(definition -> definition.name().equals(selection))
        .map(ExternalAgentDefinition::id)
        .findFirst()
        .orElse(selection);
  }

  /** Resolves a stable manifest ID from a config to the label shown in the assignment list. */
  public static String selectionForId(final String id) {
    if (id == null) return null;
    return discoverAgents().stream()
        .filter(definition -> definition.id().equals(id))
        .map(ExternalAgentDefinition::name)
        .findFirst()
        .orElse(id);
  }

  private static List<ExternalAgentDefinition> discoverAgents() {
    final Path agentRoot = findRepositoryRoot().resolve(AGENT_DIRECTORY);
    if (!Files.isDirectory(agentRoot)) return List.of();
    try (var manifests = Files.find(agentRoot, 3, (path, attrs) ->
        attrs.isRegularFile() && path.getFileName().toString().equals("agent.json"))) {
      return manifests
          .sorted(Comparator.comparing(Path::toString))
          .map(TestbenchAgentRegistry::readDefinition)
          .flatMap(Optional::stream)
          .toList();
    } catch (IOException e) {
      throw new IllegalStateException("Could not scan external agent manifests in " + agentRoot, e);
    }
  }

  private static Optional<ExternalAgentDefinition> readDefinition(final Path manifest) {
    try {
      final AgentManifest json = GSON.fromJson(Files.readString(manifest), AgentManifest.class);
      if (json == null
          || blank(json.id)
          || blank(json.name)
          || json.command == null
          || json.command.isEmpty()) {
        throw new IllegalArgumentException("Manifest must specify id, name, and command");
      }
      return Optional.of(
          new ExternalAgentDefinition(
              json.id,
              json.name,
              json.language,
              json.entrypoint,
              List.copyOf(json.command),
              json.description,
              manifest.toAbsolutePath().normalize()));
    } catch (IOException | RuntimeException e) {
      throw new IllegalStateException("Invalid external agent manifest: " + manifest, e);
    }
  }

  private static boolean blank(final String value) {
    return value == null || value.isBlank();
  }

  static Path findRepositoryRoot() {
    Path directory = Path.of(System.getProperty("user.dir")).toAbsolutePath().normalize();
    while (directory != null) {
      if (Files.isDirectory(directory.resolve(AGENT_DIRECTORY))) return directory;
      directory = directory.getParent();
    }
    throw new IllegalStateException("Could not locate " + AGENT_DIRECTORY + " from user.dir");
  }

  private static final class AgentManifest {
    String id;
    String name;
    String language;
    String entrypoint;
    List<String> command;
    String description;
  }
}
