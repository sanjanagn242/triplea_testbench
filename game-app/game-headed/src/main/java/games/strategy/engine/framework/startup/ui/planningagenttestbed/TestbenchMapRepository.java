package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import games.strategy.engine.ClientFileSystemHelper;
import games.strategy.engine.framework.map.download.DownloadConfiguration;
import games.strategy.engine.framework.map.file.system.loader.InstalledMap;
import games.strategy.engine.framework.map.file.system.loader.InstalledMapsListing;
import games.strategy.engine.framework.map.file.system.loader.ZippedMapsExtractor;
import games.strategy.engine.framework.map.listing.MapListingFetcher;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.util.ArrayList;
import java.util.List;
import java.util.Locale;
import java.util.Optional;
import org.triplea.http.client.lobby.maps.listing.MapDownloadItem;
import org.triplea.io.FileUtils;
import org.triplea.map.description.file.MapDescriptionYaml;

/** Locates and downloads maps into the testbed's repository-local map store. */
final class TestbenchMapRepository {
  private static final String TESTBED_DIRECTORY = "planning-agent-testbed";
  private static final String GAMES_DIRECTORY = "games";
  private static final String CONFIG_DIRECTORY = "config";
  private static final String MAP_DIRECTORY = "map";

  private TestbenchMapRepository() {}

  static Path gamesDirectory() {
    return repositoryRoot().resolve(TESTBED_DIRECTORY).resolve(GAMES_DIRECTORY);
  }

  static Path configurationDirectory(final Path configurationFile, final String mapName) {
    final Path configurationDirectory =
        gameDirectory(configurationFile, mapName).resolve(CONFIG_DIRECTORY);
    try {
      return Files.createDirectories(configurationDirectory);
    } catch (IOException e) {
      throw new IllegalStateException(
          "Unable to create testbed configuration directory: " + configurationDirectory, e);
    }
  }

  static Optional<Path> configurationFileForMapXml(final Path mapXml) {
    final Path gamesDirectory = gamesDirectory().toAbsolutePath().normalize();
    Path current = mapXml.toAbsolutePath().normalize().getParent();
    while (current != null) {
      if (MAP_DIRECTORY.equals(current.getFileName().toString())) {
        final Path gameDirectory = current.getParent();
        if (gameDirectory != null && gamesDirectory.equals(gameDirectory.getParent())) {
          return Optional.of(gameDirectory.resolve(CONFIG_DIRECTORY).resolve("experiment.json"));
        }
      }
      current = current.getParent();
    }
    return Optional.empty();
  }

  static Path mapsDirectory(final Path configurationFile, final String mapName) {
    final Path mapsDirectory = gameDirectory(configurationFile, mapName).resolve(MAP_DIRECTORY);
    try {
      final Path createdMapsDirectory = Files.createDirectories(mapsDirectory);
      System.setProperty(
          ClientFileSystemHelper.ADDITIONAL_MAPS_FOLDER_PROPERTY,
          createdMapsDirectory.toAbsolutePath().toString());
      return createdMapsDirectory;
    } catch (IOException e) {
      throw new IllegalStateException(
          "Unable to create testbed map directory: " + mapsDirectory, e);
    }
  }

  private static Path gameDirectory(final Path configurationFile, final String mapName) {
    final Path gamesDirectory = gamesDirectory().toAbsolutePath().normalize();
    if (configurationFile != null) {
      final Path configurationDirectory =
          configurationFile.toAbsolutePath().normalize().getParent();
      if (configurationDirectory != null
          && CONFIG_DIRECTORY.equals(configurationDirectory.getFileName().toString())) {
        final Path configuredGameDirectory = configurationDirectory.getParent();
        if (gamesDirectory.equals(configuredGameDirectory.getParent())) {
          return configuredGameDirectory;
        }
      }
    }
    if (mapName == null || mapName.isBlank()) {
      throw new IllegalArgumentException("A map name is required to locate its testbed folder.");
    }
    final String directoryName =
        mapName.toLowerCase(Locale.ROOT).replaceAll("[^a-z0-9]+", "-").replaceAll("^-|-$", "");
    return gamesDirectory.resolve(directoryName);
  }

  static Path repositoryRoot() {
    Path current = Path.of(System.getProperty("user.dir")).toAbsolutePath();
    while (current != null) {
      final Path testbedDirectory = current.resolve(TESTBED_DIRECTORY);
      if (Files.isDirectory(testbedDirectory)) {
        return current;
      }
      current = current.getParent();
    }
    throw new IllegalStateException(
        "Unable to locate the repository's planning-agent-testbed directory from "
            + System.getProperty("user.dir"));
  }

  static Path resolveMap(
      final Path configurationFile,
      final String mapXml,
      final String mapName,
      final String gameName,
      final List<String> aliases,
      final String downloadUrl)
      throws IOException {
    final List<String> mapAliases = aliases == null ? List.of() : new ArrayList<>(aliases);
    final String canonicalMapName = firstNonBlank(mapName, mapXml);
    if (canonicalMapName == null) {
      throw new IllegalArgumentException("Configuration must contain a map name or mapXml path.");
    }
    // Make repository-local map artwork discoverable by the game UI after the map is launched.
    final Path mapsDirectory = mapsDirectory(configurationFile, canonicalMapName);
    mapAliases.add(canonicalMapName);

    final Path configuredFile = resolveConfiguredFile(configurationFile, mapXml);
    if (configuredFile != null) {
      return configuredFile;
    }

    final Optional<Path> installedMap =
        findInstalledMap(mapsDirectory, canonicalMapName, gameName, mapAliases);
    if (installedMap.isPresent()) {
      return installedMap.get();
    }

    final String resolvedDownloadUrl;
    if (downloadUrl != null && !downloadUrl.isBlank()) {
      resolvedDownloadUrl = downloadUrl;
    } else {
      resolvedDownloadUrl =
          MapListingFetcher.getMapDownloadList().stream()
              .filter(item -> matches(item.getMapName(), mapAliases))
              .map(MapDownloadItem::getDownloadUrl)
              .findFirst()
              .orElse(null);
    }
    if (resolvedDownloadUrl == null || resolvedDownloadUrl.isBlank()) {
      throw new IllegalArgumentException(
          "Map '"
              + canonicalMapName
              + "' is not in the testbed map folder or the download listing. "
              + "Install it in "
              + mapsDirectory
              + " or provide downloadUrl in the JSON map entry.");
    }

    downloadMap(resolvedDownloadUrl, canonicalMapName, mapsDirectory);
    return findInstalledMap(mapsDirectory, canonicalMapName, gameName, mapAliases)
        .orElseThrow(
            () ->
                new IllegalArgumentException(
                    "Downloaded map '"
                        + canonicalMapName
                        + "' but could not find game XML '"
                        + gameName
                        + "'. Check gameName and aliases in the JSON."));
  }

  private static Path resolveConfiguredFile(final Path configurationFile, final String mapXml) {
    if (mapXml == null || mapXml.isBlank()) {
      return null;
    }
    final Path requested = Path.of(mapXml);
    if (requested.isAbsolute() && Files.isRegularFile(requested)) {
      return requested.normalize().startsWith(repositoryRoot()) ? requested : null;
    }
    if (!requested.isAbsolute()) {
      final Path relativeToConfig =
          configurationFile.toAbsolutePath().getParent().resolve(requested);
      if (Files.isRegularFile(relativeToConfig)
          && relativeToConfig.toAbsolutePath().normalize().startsWith(repositoryRoot())) {
        return relativeToConfig;
      }
      if (Files.isRegularFile(requested)
          && requested.toAbsolutePath().normalize().startsWith(repositoryRoot())) {
        return requested.toAbsolutePath();
      }
    }
    return null;
  }

  private static Optional<Path> findInstalledMap(
      final Path mapsDirectory,
      final String mapName,
      final String gameName,
      final List<String> aliases) {
    final InstalledMapsListing listing = InstalledMapsListing.parseMapFiles(mapsDirectory);
    return aliases.stream()
        .map(listing::findInstalledMapByName)
        .flatMap(Optional::stream)
        .distinct()
        .map(installedMap -> xmlPathForGame(installedMap, gameName, mapName))
        .flatMap(Optional::stream)
        .findFirst();
  }

  private static Optional<Path> xmlPathForGame(
      final InstalledMap installedMap, final String gameName, final String mapName) {
    final String requestedGameName = firstNonBlank(gameName, mapName);
    final Optional<String> matchingGame =
        installedMap.getGameNames().stream()
            .filter(name -> normalize(name).equals(normalize(requestedGameName)))
            .findFirst();
    if (matchingGame.isPresent()) {
      return installedMap.getGameXmlFilePath(matchingGame.get());
    }
    if (gameName == null || gameName.isBlank()) {
      final List<String> gameNames = List.copyOf(installedMap.getGameNames());
      if (gameNames.size() == 1) {
        return installedMap.getGameXmlFilePath(gameNames.getFirst());
      }
    }
    return Optional.empty();
  }

  private static void downloadMap(
      final String downloadUrl, final String mapName, final Path mapsDirectory) throws IOException {
    final Path temporaryDirectory = Files.createTempDirectory("triplea-testbed-map-");
    final String safeFileName = mapName.replaceAll("[^A-Za-z0-9_-]", "_");
    final Path zipFile = temporaryDirectory.resolve(safeFileName + ".zip");
    try {
      DownloadConfiguration.contentReader()
          .download(
              downloadUrl,
              inputStream -> {
                Files.copy(inputStream, zipFile, StandardCopyOption.REPLACE_EXISTING);
                return zipFile;
              })
          .orElseThrow(() -> new IOException("Failed to download map from " + downloadUrl));
      final Path extracted =
          ZippedMapsExtractor.unzipMap(zipFile, mapsDirectory)
              .orElseThrow(() -> new IOException("Failed to extract downloaded map " + mapName));
      moveUpSingleDirectory(extracted);
      if (MapDescriptionYaml.fromMap(extracted).isEmpty()) {
        MapDescriptionYaml.generateForMap(extracted);
      }
    } finally {
      FileUtils.deleteDirectory(temporaryDirectory);
    }
  }

  private static void moveUpSingleDirectory(final Path extractedMap) throws IOException {
    final List<Path> children = FileUtils.listFiles(extractedMap).stream().toList();
    if (children.size() != 1 || !Files.isDirectory(children.getFirst())) {
      return;
    }
    final Path nestedMap = children.getFirst();
    for (final Path child : FileUtils.listFiles(nestedMap)) {
      Files.move(child, extractedMap.resolve(child.getFileName()));
    }
    Files.delete(nestedMap);
  }

  private static boolean matches(final String mapName, final List<String> aliases) {
    return aliases.stream().anyMatch(alias -> normalize(alias).equals(normalize(mapName)));
  }

  private static String normalize(final String value) {
    return InstalledMapsListing.normalizeName(value).toLowerCase(Locale.ROOT);
  }

  private static String firstNonBlank(final String first, final String second) {
    if (first != null && !first.isBlank()) {
      return first;
    }
    return second == null || second.isBlank() ? null : second;
  }
}
