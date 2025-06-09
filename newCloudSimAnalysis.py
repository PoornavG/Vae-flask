#!/usr/bin/env python3
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.stats as st
from pandas.plotting import scatter_matrix

# === CONFIGURATION ===
# Update these paths to where your CSVs are located and where you want outputs saved:
BASE_DIR    = os.path.expanduser("~/cloudsim-simulator/cloudsim/modules/cloudsim-examples")
OUTPUT_DIR  = os.path.expanduser("~/cloudsim-simulator/vae-api/deep_analysis")

metrics_file = os.path.join(BASE_DIR, "simulation_metrics.csv")
host_file    = os.path.join(BASE_DIR, "host_details.csv")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# === LOAD DATA ===
df      = pd.read_csv(metrics_file)
host_df = pd.read_csv(host_file)

# === DERIVED METRICS ===
df['runtime']       = df['finish'] - df['start']
df['waiting']       = df['start'] - df['arrival']
df['arrival_hours'] = df['arrival'] / 3600.0  # convert seconds to hours

# === 1) DESCRIPTIVE STATISTICS ===
stats = df[['length', 'cpuTime', 'runtime', 'waiting', 'pes']].describe().T
stats['skewness']  = df[['length', 'cpuTime', 'runtime', 'waiting', 'pes']].skew().values
stats['kurtosis']  = df[['length', 'cpuTime', 'runtime', 'waiting', 'pes']].kurtosis().values
stats.to_csv(os.path.join(OUTPUT_DIR, "descriptive_statistics.csv"))
print("Saved descriptive statistics to CSV.")

# === 2) GROUP-BY PEs STATISTICS ===
group_stats = df.groupby('pes')[['runtime', 'waiting']].agg(['mean', 'median', 'std', 'count'])
group_stats.to_csv(os.path.join(OUTPUT_DIR, "group_by_pes_stats.csv"))
print("Saved group-by-pes statistics to CSV.")

# === 3) PAIRWISE SCATTER MATRIX ===
fig = plt.figure(figsize=(8, 8))
scatter_matrix(
    df[['length', 'cpuTime', 'runtime', 'waiting']],
    alpha=0.6, diagonal='hist', ax=fig.add_subplot(111)
)
plt.suptitle("Pairwise Scatter Matrix of Job Metrics")
plt.tight_layout()
fn = os.path.join(OUTPUT_DIR, "pairwise_scatter_matrix.png")
plt.savefig(fn)
plt.close()
print(f"Saved pairwise scatter matrix to {fn}")

# === 4) BOXPLOTS OF RUNTIME AND WAITING ===
plt.figure(figsize=(8, 5))
plt.boxplot([df['runtime'], df['waiting']], labels=['Runtime', 'Waiting'])
plt.title("Boxplots of Runtime and Waiting Times")
plt.ylabel("Seconds")
plt.grid(True, axis='y')
plt.tight_layout()
fn = os.path.join(OUTPUT_DIR, "boxplots_runtime_waiting.png")
plt.savefig(fn)
plt.close()
print(f"Saved boxplots to {fn}")

# === 5) QQ-PLOTS FOR NORMALITY CHECK ===
for col in ['runtime', 'waiting']:
    plt.figure(figsize=(6, 6))
    st.probplot(df[col], dist="norm", plot=plt)
    plt.title(f"QQ-Plot of {col.capitalize()}")
    plt.tight_layout()
    fn = os.path.join(OUTPUT_DIR, f"qqplot_{col}.png")
    plt.savefig(fn)
    plt.close()
    print(f"Saved QQ-plot for {col} to {fn}")

# === 6) DISTRIBUTION FITTING (Lognormal) ===
runtime_data = df['runtime'][df['runtime'] > 0]
shape, loc, scale = st.lognorm.fit(runtime_data, floc=0)
x = np.linspace(runtime_data.min(), runtime_data.max(), 1000)
pdf_fitted = st.lognorm.pdf(x, shape, loc, scale)

plt.figure(figsize=(8, 5))
plt.hist(runtime_data, bins=50, density=True, edgecolor='black', alpha=0.7)
plt.plot(x, pdf_fitted, linewidth=2)
plt.title("Runtime Histogram with Lognormal Fit")
plt.xlabel("Runtime (s)")
plt.ylabel("Density")
plt.grid(True)
plt.tight_layout()
fn = os.path.join(OUTPUT_DIR, "runtime_lognorm_fit.png")
plt.savefig(fn)
plt.close()
print(f"Saved runtime lognormal fit plot to {fn}")

# === 7) AUTOCORRELATION OF JOB THROUGHPUT ===
bin_width = 600  # seconds (10-minute intervals)
df['arrival_bin'] = (df['arrival'] // bin_width) * bin_width
throughput = df.groupby('arrival_bin').size()

autocorr_values = [throughput.autocorr(lag=i) for i in range(1, 25)]
plt.figure(figsize=(8, 5))
plt.stem(range(1, 25), autocorr_values, use_line_collection=True)
plt.title("Autocorrelation of Job Throughput (lags in 10-min intervals)")
plt.xlabel("Lag (intervals)")
plt.ylabel("Autocorrelation")
plt.grid(True)
plt.tight_layout()
fn = os.path.join(OUTPUT_DIR, "throughput_autocorrelation.png")
plt.savefig(fn)
plt.close()
print(f"Saved throughput autocorrelation plot to {fn}")

print(f"All analysis outputs saved under: {OUTPUT_DIR}")
