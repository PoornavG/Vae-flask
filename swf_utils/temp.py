import sys
import os
import pandas as pd

# Ensure your swf_utils directory (with both swf_categorizer.py and compute_job_count_edges) is on the path
script_dir = os.path.abspath(os.path.join(os.getcwd(), 'swf_utils'))
sys.path.insert(0, script_dir)

from swf_categorizer import parse_sdsc_sp2_log, compute_job_count_edges,detect_and_remove_anomalies

# ───────── CONFIG ─────────
swf_path    = '/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf'
output_path = '/home/poornav/cloudsim-simulator/swf_categories_with_stats.xlsx'
# ──────────────────────────

# 1) Parse, detect anomalies, label jobs & summarize
df = parse_sdsc_sp2_log(swf_path)
anoms, clean = detect_and_remove_anomalies(df, 0.01)
# 2) Compute job‐count edges for Low/Mid/High activity
job_count_edges = compute_job_count_edges(clean)

print("Activity bin edges:", job_count_edges)

# 3) Aggregate per‐user job counts
user_counts_df = (
    df['UserID']
    .value_counts()
    .rename('job_count')
    .reset_index()
    .rename(columns={'index': 'UserID'})
)

# 4) Bin users into ActivityBin using the edges
user_counts_df['ActivityBin'] = pd.cut(
    user_counts_df['job_count'],
    bins=job_count_edges,
    labels=['Low', 'Mid', 'High'],
    include_lowest=True
)

# 5) Build a summary of how many users are in each bin
activity_counts = (
    user_counts_df['ActivityBin']
    .value_counts()
    .rename_axis('ActivityBin')
    .reset_index(name='UserCount')
)
print("User distribution across activity bins:")
print(activity_counts)

