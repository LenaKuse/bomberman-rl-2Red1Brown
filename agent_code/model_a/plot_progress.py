import pandas as pd
import matplotlib.pyplot as plt
import sys

run_name = sys.argv[1] if len(sys.argv) > 1 else "baseline" # Stores the name of the run for which we want to plot the progress. Default is "baseline" if no argument is provided.

train_data = pd.read_csv(f"agent_code/model_a/experiments/{run_name}_training_progress.csv",
                          names=["round", "reward", "avg_td_error"]) # Read the training progress data from the CSV file corresponding to the specified run name. The CSV file is expected to have three columns: "round", "reward", and "avg_td_error".
eval_data = pd.read_csv(f"agent_code/model_a/experiments/{run_name}_eval_progress.csv",
                         names=["round", "reward"]) # Read the evaluation progress data from the CSV file corresponding to the specified run name. The CSV file is expected to have two columns: "round" and "reward".

window = 11

MIN_ROUNDS_FOR_PLOT = window + 10
if len(train_data) < MIN_ROUNDS_FOR_PLOT:
    print(f"Not enough training data ({len(train_data)} rounds) to produce a meaningful plot "
          f"(need at least {MIN_ROUNDS_FOR_PLOT}). Skipping plot.")
    sys.exit(0)


train_data["reward_rolling"] = train_data["reward"].rolling(window=window, center=True).mean()
train_data["td_error_rolling"] = train_data["avg_td_error"].rolling(window=window, center=True).mean()
eval_stats = eval_data.groupby("round")["reward"].agg(["mean", "std"])

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(train_data["round"], train_data["reward"], alpha=0.2, label="Training reward (per round)")
ax1.plot(train_data["round"], train_data["reward_rolling"], linewidth=2, label="Training reward (rolling avg)")
ax1.errorbar(eval_stats.index, eval_stats["mean"], yerr=eval_stats["std"],
             fmt="o-", linewidth=2, color="red", capsize=4, label="Eval reward (mean ± std)")
ax1.set_ylabel("Reward")
ax1.legend()
ax1.set_title(f"Training progress: {run_name}")

ax2.plot(train_data["round"], train_data["avg_td_error"], alpha=0.2, label="Avg. TD-error (per round)")
ax2.plot(train_data["round"], train_data["td_error_rolling"], linewidth=2, label="Avg. TD-error (rolling avg)")
ax2.set_xlabel("Training round")  
ax2.set_ylabel("Avg. |TD-error|")
ax2.legend()

plt.savefig(f"agent_code/model_a/experiments/{run_name}_progress.png")
plt.show()