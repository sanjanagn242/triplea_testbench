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
    List<String> outputDirectories,
    Path manifestPath) {
  Path agentDirectory() {
    return manifestPath.getParent();
  }

  /**
   * Folders (relative to the agent directory) holding the agent's own run outputs, such as metrics
   * or logs; the testbench may empty them when an experiment starts.
   */
  List<Path> resolvedOutputDirectories() {
    final Path root = agentDirectory().toAbsolutePath().normalize();
    return outputDirectories.stream()
        .map(directory -> root.resolve(directory).normalize())
        .filter(directory -> directory.startsWith(root) && !directory.equals(root))
        .toList();
  }
}
