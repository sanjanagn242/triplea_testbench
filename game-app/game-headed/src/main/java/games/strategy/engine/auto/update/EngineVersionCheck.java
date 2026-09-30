package games.strategy.engine.auto.update;

import games.strategy.triplea.settings.ClientSetting;
import java.awt.Component;
import java.net.URI;
import java.time.Instant;
import lombok.experimental.UtilityClass;
import org.triplea.config.product.ProductVersionReader;
import org.triplea.http.client.latest.version.LatestVersionClient;
import org.triplea.http.client.latest.version.LatestVersionResponse;

@UtilityClass
final class EngineVersionCheck {
  static void checkForLatestEngineVersionOut(final Component parentComponent) {
    ClientSetting.lastCheckForEngineUpdate.setValueAndFlush(Instant.now().toEpochMilli());

    final URI lobbyUri = ClientSetting.lobbyUri.getValueOrThrow();
    if (isLoopbackHost(lobbyUri.getHost())) {
      // Local test servers usually do not implement the production update endpoint.
      return;
    }

    LatestVersionClient.fetchLatestVersion(
            lobbyUri, ProductVersionReader.getCurrentVersion())
        .filter(
            response ->
                !LatestVersionResponse.RecommendedAction.NO_UPDATE
                    .toString()
                    .equals(response.getRecommendedUpdateAction()))
        .ifPresent(response -> OutOfDateDialog.showOutOfDateComponent(parentComponent, response));
  }

  private static boolean isLoopbackHost(final String host) {
    return host != null
        && (host.equalsIgnoreCase("localhost")
            || host.endsWith(".localhost")
            || host.startsWith("127.")
            || host.equalsIgnoreCase("::1")
            || host.equalsIgnoreCase("[::1]"));
  }
}
