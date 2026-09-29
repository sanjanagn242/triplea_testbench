package games.strategy.engine.framework.startup.ui.planningagenttestbed;

import games.strategy.engine.framework.startup.launcher.ILauncher;
import games.strategy.engine.framework.startup.ui.SetupPanel;
import games.strategy.engine.framework.startup.ui.panels.main.HeadedServerSetupModel;
import java.awt.BorderLayout;
import java.awt.Font;
import java.awt.GridBagConstraints;
import java.awt.GridBagLayout;
import java.awt.Insets;
import java.util.List;
import java.util.Optional;
import javax.swing.Action;
import javax.swing.BorderFactory;
import javax.swing.JButton;
import javax.swing.JLabel;
import javax.swing.JOptionPane;
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.UIManager;
import org.triplea.swing.JButtonBuilder;

/** Mode selection screen for the planning-agent testbench. */
public final class TestbenchModePanel extends SetupPanel {
  private static final long serialVersionUID = 1L;

  public TestbenchModePanel(final HeadedServerSetupModel model) {
    setLayout(new BorderLayout(0, 20));
    setBorder(BorderFactory.createEmptyBorder(28, 30, 28, 30));

    final JPanel header = new JPanel(new GridBagLayout());
    header.add(title("Planning Agent Testbench"), rowConstraints(0));
    header.add(
        subtitle("Run repeatable games or review results from earlier simulations."),
        rowConstraints(1));
    add(header, BorderLayout.NORTH);

    final JButton runSimulations =
        new JButtonBuilder("Run Simulations")
            .biggerFont()
            .toolTipText("Choose a map, agents, run count, and round limit")
            .actionListener(() -> model.showTestbench())
            .build();
    final JButton evaluateSimulations =
        new JButtonBuilder("Evaluate Simulations")
            .biggerFont()
            .toolTipText("Open logged game metrics")
            .actionListener(
                () ->
                    JOptionPane.showMessageDialog(
                        this,
                        "The simulation results viewer will be available in a later update.",
                        "Evaluate Simulations",
                        JOptionPane.INFORMATION_MESSAGE))
            .build();

    final JPanel choices = new JPanel(new GridBagLayout());
    choices.add(
        choiceCard(runSimulations, "Configure and launch a batch of planning-agent games."),
        wideRowConstraints(0));
    choices.add(
        choiceCard(evaluateSimulations, "Explore the metrics recorded during completed games."),
        wideRowConstraints(1));
    add(choices, BorderLayout.CENTER);
  }

  private static JPanel choiceCard(final JButton button, final String description) {
    final JPanel card = new JPanel(new BorderLayout(0, 8));
    card.setBorder(
        BorderFactory.createCompoundBorder(
            BorderFactory.createEtchedBorder(), BorderFactory.createEmptyBorder(14, 16, 14, 16)));
    card.add(button, BorderLayout.NORTH);
    card.add(subtitle(description), BorderLayout.CENTER);
    return card;
  }

  private static JLabel title(final String text) {
    final JLabel label = new JLabel(text, SwingConstants.CENTER);
    label.setFont(UIManager.getFont("Label.font").deriveFont(Font.BOLD, 21f));
    return label;
  }

  private static JLabel subtitle(final String text) {
    final JLabel label = new JLabel(text, SwingConstants.CENTER);
    label.setFont(UIManager.getFont("Label.font").deriveFont(13f));
    return label;
  }

  private static GridBagConstraints rowConstraints(final int row) {
    return new GridBagConstraints(
        0,
        row,
        1,
        1,
        1,
        0,
        GridBagConstraints.CENTER,
        GridBagConstraints.HORIZONTAL,
        new Insets(5, 0, 5, 0),
        0,
        0);
  }

  private static GridBagConstraints wideRowConstraints(final int row) {
    return new GridBagConstraints(
        0,
        row,
        1,
        1,
        1,
        0,
        GridBagConstraints.CENTER,
        GridBagConstraints.HORIZONTAL,
        new Insets(8, 0, 8, 0),
        0,
        0);
  }

  @Override
  public boolean canGameStart() {
    return false;
  }

  @Override
  public List<Action> getUserActions() {
    return List.of();
  }

  @Override
  public boolean isCancelButtonVisible() {
    return true;
  }

  @Override
  public void cancel() {}

  @Override
  public Optional<ILauncher> getLauncher() {
    throw new UnsupportedOperationException();
  }

  @Override
  public void postStartGame() {}
}
