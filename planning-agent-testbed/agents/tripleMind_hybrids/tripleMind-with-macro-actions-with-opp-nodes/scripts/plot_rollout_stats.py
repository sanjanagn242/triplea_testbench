"""When do rollouts reach a terminal state?

Reads metrics/rollout_stats.csv (one row per rollout: player, game, round, iteration,
rollout_depth_reached, reward, is_terminal) and writes rollout_stats.png and rollout_stats_summary.txt
to the same folder.

  round                  game round of the decision the rollout was run for (where you are in the game)
  rollout_depth_reached  rounds the rollout advanced before it stopped
  is_terminal            1 = the rollout stopped in a terminal state: a win if reward > 0, else a loss

Top panel: share of rollouts per decision round that ended in a terminal win / loss.
Bottom panel: for terminal rollouts, the game round (round + rollout_depth_reached) where the terminal state was reached.
"""
import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

WIN_COLOR = "#2a78d6"
LOSS_COLOR = "#eb6834"
TEXT = "#52514e"
GRID = "#e4e3df"

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()
FOLDER = args.folder

df = pd.read_csv(f"{FOLDER}/rollout_stats.csv")
df["terminal"] = df["is_terminal"] == 1
df["outcome"] = np.select([df["terminal"] & (df["reward"] > 0), df["terminal"]], ["win", "loss"], "none")
df["terminal_round"] = df["round"] + df["rollout_depth_reached"]

rounds = np.arange(df["round"].min(), df["round"].max() + 1)
per_round = (df.groupby("round")["outcome"].value_counts().unstack(fill_value=0)
             .reindex(index=rounds, columns=["win", "loss", "none"], fill_value=0))
total = per_round.sum(axis=1).replace(0, np.nan)
win_pct = 100 * per_round["win"] / total
loss_pct = 100 * per_round["loss"] / total

term = df[df["terminal"]]
if len(term):
    end_rounds = np.arange(term["terminal_round"].min(), term["terminal_round"].max() + 1)
else:
    end_rounds = rounds
heat = (term.groupby(["terminal_round", "round"]).size().unstack(fill_value=0)
        .reindex(index=end_rounds, columns=rounds, fill_value=0))
heat_pct = 100 * heat / total.values  # share of all rollouts at that decision round

# -----------------------------
# Figure
# -----------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True,
                               gridspec_kw={"height_ratios": [3, 2]})

ax1.bar(rounds, win_pct.fillna(0), width=0.7, color=WIN_COLOR, label="Terminal win",
        edgecolor="white", linewidth=1)
ax1.bar(rounds, loss_pct.fillna(0), width=0.7, bottom=win_pct.fillna(0), color=LOSS_COLOR,
        label="Terminal loss", edgecolor="white", linewidth=1)
for r in rounds:
    n = per_round.loc[r].sum()
    if n:
        top = win_pct.fillna(0)[r] + loss_pct.fillna(0)[r]
        ax1.annotate(f"{top:.0f}%\nn={n}", (r, top), xytext=(0, 4), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8, color=TEXT)
ax1.set_ylim(0, max(10, float(np.nanmax((win_pct + loss_pct).values, initial=0)) * 1.25))
ax1.set_ylabel("Rollouts reaching terminal (%)", fontsize=12)
ax1.set_title("How often rollouts reach a terminal state, by decision round", fontsize=13)
ax1.legend(frameon=False, loc="upper left")

im = ax2.imshow(heat_pct.where(heat > 0).values, aspect="auto",  # empty cells stay blank
                origin="lower", cmap="Blues", vmin=0,
                extent=(rounds[0] - 0.5, rounds[-1] + 0.5, end_rounds[0] - 0.5, end_rounds[-1] + 0.5))
ax2.plot(rounds, rounds, color=TEXT, linewidth=1, linestyle="--", label="Decision round")
ax2.legend(frameon=False, loc="upper left", fontsize=9)
ax2.set_ylabel("Game round where\nterminal reached", fontsize=12)
ax2.set_xlabel("Decision round (where you are in the game)", fontsize=12)
ax2.set_title("Where terminal states are found (share of that round's rollouts)", fontsize=13)
fig.colorbar(im, ax=ax2, label="% of rollouts", pad=0.01)

ax2.set_xticks(rounds)
for ax in (ax1, ax2):
    ax.tick_params(colors=TEXT)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
ax1.yaxis.grid(True, color=GRID)
ax1.set_axisbelow(True)

plt.tight_layout()
plt.savefig(f"{FOLDER}/rollout_stats.png", dpi=150)
plt.close()

# -----------------------------
# Summary
# -----------------------------
lines = ["Rollouts Reaching Terminal\n", "=" * 40 + "\n\n"]
lines.append(f"Games: {df['game'].nunique()}   Rollouts: {len(df)}   "
             f"Terminal: {df['terminal'].sum()} ({100 * df['terminal'].mean():.1f}%)\n\n")
lines.append(f"{'round':>5} {'rollouts':>9} {'win%':>6} {'loss%':>6} {'term%':>6} {'mean ahead':>11}\n")
for r in rounds:
    n = int(per_round.loc[r].sum())
    if not n:
        continue
    t = term[term["round"] == r]
    ahead = f"{t['rollout_depth_reached'].mean():.2f}" if len(t) else "-"
    lines.append(f"{r:>5} {n:>9} {win_pct[r]:>6.1f} {loss_pct[r]:>6.1f} "
                 f"{win_pct[r] + loss_pct[r]:>6.1f} {ahead:>11}\n")
if len(term):
    lines.append(f"\nGame round where terminal was reached (round + rollout_depth_reached):\n")
    for tr, c in term["terminal_round"].value_counts().sort_index().items():
        lines.append(f"  round {tr}: {c}\n")

summary_path = f"{FOLDER}/rollout_stats_summary.txt"
with open(summary_path, "w") as f:
    f.writelines(lines)

print(f"Plot written to {FOLDER}/rollout_stats.png")
print(f"Summary written to {summary_path}")
