import sys
import os
import pandas as pd

# make sure your swf_utils directory is on the path
script_dir = os.path.abspath(os.path.join(os.getcwd(), 'swf_utils'))
sys.path.insert(0, script_dir)

from swf_categorizer import (
    parse_sdsc_sp2_log,
    detect_and_remove_anomalies,
    compute_job_count_edges,
    compute_bin_edges
)

# ───────── CONFIG ─────────
SWF_PATH    = "/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf"
OUTPUT_PATH = "/home/poornav/cloudsim-simulator/swf_user_27_categories.xlsx"
ANOMALY_PCT = 0.01    # fraction contamination
# ──────────────────────────

if __name__ == "__main__":
    # 1) parse & anomaly‐filter
    df = parse_sdsc_sp2_log(SWF_PATH)
    _, clean = detect_and_remove_anomalies(df, ANOMALY_PCT)

    # 2) compute bin edges
    jc_edges   = compute_job_count_edges(clean)
    rt_edges, cpu_edges = compute_bin_edges(clean)

    print("Job-count edges:", jc_edges)
    print("Runtime edges :", rt_edges)
    print("CPU-util edges:", cpu_edges)

    # 3) aggregate per-user counts & runtime
    user_counts = (
        clean
        .groupby('UserID')
        .agg(
            job_count   = ('JobID', 'count'),
            avg_runtime = ('RunTime', 'mean')
        )
        .reset_index()
    )

    # 4) compute per-user CPU utilization first
    tmp = clean.copy()
    tmp['CPUUtil'] = (
        tmp['AverageCPUTimeUsed']
        / (tmp['RunTime'] * tmp['AllocatedProcessors'])
    ).replace([float('inf'), -float('inf')], 0).fillna(0)

    user_cpu = (
        tmp
        .groupby('UserID')['CPUUtil']
        .mean()
        .reset_index(name='avg_cpu_util')
    )

    # 5) merge into single DataFrame
    user_df = user_counts.merge(user_cpu, on='UserID')

    # 6) bin each metric into Low/Mid/High
    user_df['JobCountBin'] = pd.cut(
        user_df['job_count'], bins=jc_edges,
        labels=['Low','Mid','High'], include_lowest=True
    )
    user_df['RuntimeBin'] = pd.cut(
        user_df['avg_runtime'], bins=rt_edges,
        labels=['Low','Mid','High'], include_lowest=True
    )
    user_df['CPUUtilBin'] = pd.cut(
        user_df['avg_cpu_util'], bins=cpu_edges,
        labels=['Low','Mid','High'], include_lowest=True
    )

    # 7) combine into 27 categories
    user_df['UserCategory'] = (
        user_df['JobCountBin'].astype(str) + '_' +
        user_df['RuntimeBin'].astype(str)  + '_' +
        user_df['CPUUtilBin'].astype(str)
    )

    # 8) summarise
    summary = (
        user_df['UserCategory']
        .value_counts()
        .rename_axis('UserCategory')
        .reset_index(name='NumUsers')
        .sort_values('UserCategory')
    )

    print("\nUsers per 27-category:")
    print(summary)

    # 9) save to Excel with two sheets: PerUser & Summary
    with pd.ExcelWriter(OUTPUT_PATH) as writer:
        user_df.to_excel(writer, sheet_name='PerUser', index=False)
        summary.to_excel(writer, sheet_name='Summary', index=False)

    print(f"\nResults written to {OUTPUT_PATH}")
