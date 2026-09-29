package games.strategy.engine.framework.startup.ui;

import games.strategy.engine.framework.startup.launcher.ILauncher;
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
import javax.swing.JPanel;
import javax.swing.SwingConstants;
import javax.swing.UIManager;
import org.triplea.swing.JButtonBuilder;
import org.triplea.swing.jpanel.GridBagConstraintsAnchor;
import org.triplea.swing.jpanel.GridBagConstraintsBuilder;
import org.triplea.swing.jpanel.GridBagConstraintsFill;

/** First choice screen for selecting the planning testbench or the regular game engine. */
public final class MetaSetupPanel extends SetupPanel {
  private static final long serialVersionUID = 1L;

  public MetaSetupPanel(final HeadedServerSetupModel model) {
    setLayout(new BorderLayout(0, 20));
    setBorder(BorderFactory.createEmptyBorder(28, 30, 28, 30));

    final JPanel header = new JPanel(new GridBagLayout());
    header.add(title("Welcome to TripleA"), rowConstraints(0));
    header.add(
        subtitle("Choose a planning experiment or open the standard game setup."),
        rowConstraints(1));
    add(header, BorderLayout.NORTH);

    final JButton planningTestbench =
        new JButtonBuilder("Planning Agent Testbench")
            .biggerFont()
            .toolTipText("Configure and run a planning agent experiment")
            .actionListener(model::showTestbenchMenu)
            .build();
    final JButton gameEngine =
        new JButtonBuilder("Use the game engine")
            .biggerFont()
            .toolTipText("Open the regular TripleA game setup")
            .actionListener(model::showGameEngine)
            .build();

    final JPanel options = new JPanel(new GridBagLayout());
    options.add(
        choiceCard(planningTestbench, "Set agents, maps, rounds, and run controls."),
        wideRowConstraints(0));
    options.add(
        choiceCard(gameEngine, "Play TripleA with the regular game setup."), wideRowConstraints(1));
    add(options, BorderLayout.CENTER);
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

  static JLabel title(final String text) {
    final JLabel label = new JLabel(text, SwingConstants.CENTER);
    label.setFont(UIManager.getFont("Label.font").deriveFont(Font.BOLD, 21f));
    return label;
  }

  static JLabel subtitle(final String text) {
    final JLabel label = new JLabel(text, SwingConstants.CENTER);
    label.setFont(UIManager.getFont("Label.font").deriveFont(13f));
    return label;
  }

  private static GridBagConstraints wideRowConstraints(final int row) {
    return new GridBagConstraintsBuilder(0, row)
        .anchor(GridBagConstraintsAnchor.CENTER)
        .fill(GridBagConstraintsFill.HORIZONTAL)
        .weightX(1)
        .insets(new Insets(8, 0, 8, 0))
        .build();
  }

  private static GridBagConstraints rowConstraints(final int row) {
    return new GridBagConstraintsBuilder(0, row)
        .anchor(GridBagConstraintsAnchor.CENTER)
        .fill(GridBagConstraintsFill.NONE)
        .insets(new Insets(12, 0, 0, 0))
        .build();
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
    return false;
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
