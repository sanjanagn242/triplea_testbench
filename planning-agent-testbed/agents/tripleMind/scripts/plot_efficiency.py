import pandas as pd
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--folder", type=str, required=True)
args = parser.parse_args()

main_folder = args.folder
FOLDER = f"{main_folder}"

# Load data
df = pd.read_csv(f"{FOLDER}/efficiency.csv")



import numpy as np

# =============================
# Aggregate analysis across games
# =============================

# Aggregate by round position across games
agg = df.groupby("round")

mean_iters = agg["num_iterations"].mean()
p25_iters = agg["num_iterations"].quantile(0.25)
p75_iters = agg["num_iterations"].quantile(0.75)


# -----------------------------
# Aggregate summary figure
# -----------------------------
fig, ax = plt.subplots(figsize=(10, 6))

# --- 1. Iterations per round (mean + IQR) ---
ax.plot(mean_iters.index, mean_iters.values, label="Mean", marker='o')
ax.fill_between(
    mean_iters.index,
    p25_iters.values,
    p75_iters.values,
    alpha=0.3,
    label="25–75 percentile"
)
ax.set_title("Iterations per Round (Across Games)", fontsize=13)
ax.set_ylabel("Iterations", fontsize=13)
ax.set_xlabel("Round", fontsize=13)
ax.legend()


plt.tight_layout()
plt.savefig(f"{FOLDER}/aggregate_game_summary.png")
plt.close()

# print(f"Aggregate summary plot saved to {FOLDER}/aggregate_game_summary.png")

# -----------------------------
# Game-wise summary generation
# -----------------------------
summary_lines = []
summary_lines.append("MCTS Efficiency Summary (Game-wise)\n")
summary_lines.append("=" * 40 + "\n\n")

per_game_stats = []
for game_id, gdf in df.groupby("game"):
    gdf = gdf.copy()
#     gdf["exploration_coverage"] = gdf["explored"] / gdf["total_actions"]

    summary_lines.append(f"Game {game_id}\n")
    summary_lines.append("-" * 20 + "\n")

    summary_lines.append(f"Total rounds: {gdf['round'].nunique()}\n")

    summary_lines.append(
        f"Iterations per round:\n"
        f"  Mean: {gdf['num_iterations'].mean():.2f}\n"
        f"  Median: {gdf['num_iterations'].median()}\n"
        f"  Min : {gdf['num_iterations'].min()}\n"
        f"  Max : {gdf['num_iterations'].max()}\n"
    )

    per_game_stats.append({
        "game": game_id,
        "iterations": gdf["num_iterations"]
        # "avg_exploration_coverage": gdf["exploration_coverage"].mean(),
        # "avg_depth": gdf["avg_depth"].mean()
    })


overall_df = pd.DataFrame(per_game_stats)
all_iters = pd.concat(overall_df['iterations'].tolist())

summary_lines.append("\nOVERALL AVERAGES (Across Games)\n")
summary_lines.append("=" * 30 + "\n")

summary_lines.append(
    f"Iterations per round (game-averaged):\n"
    f"  Median: {all_iters.median():.2f}\n"
    f"  Min: {df['num_iterations'].min()}\n"
    f"  Max: {df['num_iterations'].max()}\n"
)


# -----------------------------
# Write summary
# -----------------------------
summary_path = f"{FOLDER}/efficiency_summary.txt"
with open(summary_path, "w") as f:
    f.writelines(summary_lines)

print(f"Summary written to {summary_path}")


fig, ax = plt.subplots(figsize=(10, 6))
mean_value = agg["value"].mean()
p25_value = agg["value"].quantile(0.25)
p75_value = agg["value"].quantile(0.75)
ax.plot(mean_value.index, mean_value.values, label="Mean", marker='o')
ax.fill_between(
    mean_value.index,
    p25_value.values,
    p75_value.values,
    alpha=0.3,
    label="25–75 percentile"
)
ax.set_title("Root node value per Round (Across Games)", fontsize=13)
ax.set_ylabel("Value", fontsize=13)
ax.set_xlabel("Round", fontsize=13)
ax.legend()

plt.tight_layout()
plt.savefig(f"{FOLDER}/agg_root_node_value.png")
plt.close()


