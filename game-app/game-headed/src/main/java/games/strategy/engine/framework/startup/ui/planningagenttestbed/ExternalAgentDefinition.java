package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import java.nio.file.Path;
import java.util.List;

/** Metadata describing how the testbench starts an external agent process. */
record ExternalAgentDefinition(
    String id,
    String name,
    String language,
    String entrypoint,
    List<String> command,
    String description,
    Path manifestPath) {
  Path agentDirectory() {
    return manifestPath.getParent();
  }
}
