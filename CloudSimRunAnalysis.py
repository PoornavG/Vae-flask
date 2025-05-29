import pandas as pd
import matplotlib.pyplot as plt
import os

# === CONFIGURATION ===
BASE_DIR = os.path.expanduser("~/cloudsim-simulator/cloudsim/modules/cloudsim-examples")
metrics_file = os.path.join(BASE_DIR, "simulation_metrics.csv")
host_file   = os.path.join(BASE_DIR, "host_details.csv")
OUTPUT_DIR  = os.path.expanduser("~/cloudsim-simulator/vae-api/plots")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df = pd.read_csv(metrics_file)
host_df = pd.read_csv(host_file)

# === PROCESS TIMINGS ===
df['runtime']       = df['finish'] - df['start']
df['waiting']       = df['start'] - df['arrival']
df['arrival_hours'] = df['arrival'] / 3600.0  # convert seconds to hours

# === PLOT 1: HISTOGRAM OF JOB RUNTIMES ===
plt.figure(figsize=(8, 5))
plt.hist(df['runtime'], bins=30, edgecolor='black')
plt.title("Histogram of Job Runtimes")
plt.xlabel("Runtime (s)")
plt.ylabel("Number of Jobs")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_job_runtimes.png"))
plt.close()

# === PLOT 2: HISTOGRAM OF JOB WAITING TIMES ===
plt.figure(figsize=(8, 5))
plt.hist(df['waiting'], bins=30, edgecolor='black')
plt.title("Histogram of Job Waiting Times")
plt.xlabel("Waiting Time (s)")
plt.ylabel("Number of Jobs")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "hist_job_waiting_times.png"))
plt.close()

# === PLOT 3: SCATTER - ARRIVAL TIME VS. RUNTIME ===
plt.figure(figsize=(8, 5))
plt.scatter(df['arrival_hours'], df['runtime'], alpha=0.5)
plt.title("Arrival Time vs Runtime")
plt.xlabel("Arrival Time (hours)")
plt.ylabel("Runtime (s)")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "scatter_arrival_vs_runtime.png"))
plt.close()

# === PLOT 4: JOB THROUGHPUT OVER TIME ===
bin_width = 600  # 10-minute intervals in seconds
df['arrival_bin'] = (df['arrival'] // bin_width) * bin_width
throughput = df.groupby('arrival_bin').size()

plt.figure(figsize=(10, 5))
plt.plot(throughput.index / 3600.0, throughput.values, marker='o')
plt.title("Job Throughput Over Time")
plt.xlabel("Time (hours)")
plt.ylabel("Jobs per 10-minute Interval")
plt.grid(True)
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "throughput_over_time.png"))
plt.close()

# === PLOT 5: HOST RESOURCE BAR CHART ===
host = host_df.iloc[0]
resources = {
    'RAM (MB)':         host['ram'],
    'Bandwidth (MBps)': host['bw'],
    'PEs':              host['numPes'],
    'MIPS':             host['peMips']
}

plt.figure(figsize=(8, 5))
plt.bar(resources.keys(), resources.values())
plt.title("Host Resource Configuration")
plt.ylabel("Value")
plt.grid(axis='y')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "host_resource_configuration.png"))
plt.close()

# === PLOT 6: CORRELATION MATRIX ===
# Note: your 'df' must contain the 'pes' column; if not, replace with 'numPes'
numerical_cols = ['length', 'cpuTime', 'runtime', 'waiting', 'pes']
df_corr = df[numerical_cols].corr()

plt.figure(figsize=(8, 6))
plt.imshow(df_corr, cmap='coolwarm', interpolation='nearest')
plt.colorbar()
plt.xticks(range(len(df_corr.columns)), df_corr.columns, rotation=45)
plt.yticks(range(len(df_corr.index)), df_corr.index)
plt.title("Correlation Matrix of Job Metrics")
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, "correlation_matrix.png"))
plt.close()

print(f"All plots saved to {OUTPUT_DIR}")
