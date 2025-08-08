import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("optimization_log.csv")

# Sort iterations
df = df.sort_values("iteration")

# Line plot for objective over iterations
plt.figure(figsize=(10, 6))
plt.plot(df["iteration"], df["target"], marker='o', color='blue')
plt.title("Objective Function Over Iterations")
plt.xlabel("Iteration")
plt.ylabel("Negative Cost (Higher is Better)")
plt.grid(True)
plt.tight_layout()
plt.savefig("target_over_iterations.png")
plt.show()

# Correlation heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(df[["mips", "pes", "ram", "target"]].corr(),
            annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation between Parameters and Objective")
plt.tight_layout()
plt.savefig("correlation_heatmap.png")
plt.show()

plt.scatter(df['mips'], df['target'], label='MIPS')
plt.scatter(df['pes'], df['target'], label='PEs')
plt.scatter(df['ram'], df['target'], label='RAM')
plt.xlabel("Configuration Parameter")
plt.ylabel("Target")
plt.legend()
plt.title("Parameter vs Objective Value")
plt.grid()
plt.show()
