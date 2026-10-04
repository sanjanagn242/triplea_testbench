package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import static org.assertj.core.api.Assertions.assertThat;

import java.nio.file.Path;
import java.util.List;
import org.junit.jupiter.api.Test;

class ExternalAgentDefinitionTest {
  private static final Path AGENT_DIR = Path.of("/repo/planning-agent-testbed/agents/demo");

  private static ExternalAgentDefinition definition(final List<String> outputDirectories) {
    return new ExternalAgentDefinition(
        "demo",
        "Demo",
        "Python",
        "agent.py",
        List.of("python3", "agent.py"),
        "",
        outputDirectories,
        AGENT_DIR.resolve("agent.json"));
  }

  @Test
  void outputDirectoriesResolveInsideTheAgentFolder() {
    assertThat(definition(List.of("metrics", "logs", "trees/run")).resolvedOutputDirectories())
        .containsExactly(
            AGENT_DIR.resolve("metrics"), AGENT_DIR.resolve("logs"), AGENT_DIR.resolve("trees/run"));
  }

  @Test
  void outputDirectoriesOutsideTheAgentFolderOrTheFolderItselfAreIgnored() {
    assertThat(
            definition(List.of("..", "../other-agent", "/tmp", ".", "metrics/../.."))
                .resolvedOutputDirectories())
        .isEmpty();
  }

  @Test
  void noOutputDirectoriesDeclared() {
    assertThat(definition(List.of()).resolvedOutputDirectories()).isEmpty();
  }
}
