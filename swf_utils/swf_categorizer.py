import os
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

# ───────── CONFIG ─────────
SWF_PATH     = "/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf"
ANOMALY_PCT  = 1.0    # percent contamination (percent as a whole number)
# ──────────────────────────


def parse_sdsc_sp2_log(path):
    """Parse SWF, reconstruct timestamps & interarrival."""
    cols = [
        "JobID","SubmitTime","WaitTime","RunTime","AllocatedProcessors",
        "AverageCPUTimeUsed","UsedMemory","RequestedProcessors","RequestedTime",
        "RequestedMemory","Status","UserID","GroupID","ExecutableID",
        "QueueNumber","PartitionNumber","PrecedingJobNumber","ThinkTimeOfPrecedingJob"
    ]
    header_info = {}
    unix_start_time = None

    with open(path) as f:
        lines = f.readlines()

    for line in lines:
        if not line.startswith(';'):
            break
        if ': ' in line:
            k, v = line[1:].split(': ', 1)
            header_info[k.strip()] = v.strip()
            if k.strip() == 'UnixStartTime':
                unix_start_time = int(v)

    rows = []
    for line in lines:
        if line.startswith(';') or not line.strip():
            continue
        parts = line.split()
        if len(parts) == len(cols):
            rows.append(parts)

    df = pd.DataFrame(rows, columns=cols).apply(pd.to_numeric, errors='coerce')
    if unix_start_time is not None:
        base = pd.to_datetime(unix_start_time, unit='s')
        df['SubmitDateTime'] = base + pd.to_timedelta(df['SubmitTime'], unit='s')
    else:
        df['SubmitDateTime'] = pd.to_datetime(df['SubmitTime'], unit='s', origin='unix')

    df['SubmitTimeOfDay'] = df['SubmitDateTime'].dt.strftime('%H:%M:%S')
    df = df.sort_values('SubmitDateTime').reset_index(drop=True)
    df['Interarrival'] = df['SubmitDateTime'].diff().dt.total_seconds().fillna(0)
    return df


def detect_and_remove_anomalies(df, contamination):
    """Flag and remove anomalies using IsolationForest."""
    feats = ['Interarrival','RunTime','AllocatedProcessors','AverageCPUTimeUsed','UsedMemory']
    X = df[feats].fillna(0)
    iso = IsolationForest(n_estimators=100, contamination=contamination, random_state=0, n_jobs=-1)
    df2 = df.copy()
    df2['Anomaly'] = iso.fit_predict(X) == -1
    anomalies = df2[df2['Anomaly']].copy()
    clean = df2[~df2['Anomaly']].copy()
    return anomalies, clean


def head_tail_breaks(series, minority_frac=0.4):
    """Compute head/tail breakpoints for heavy-tailed data."""
    vals = series.dropna().values
    if vals.size == 0:
        return [0.0, 1.0, 2.0]
    breaks = [vals.min()]
    head = vals.copy()
    while True:
        m = head.mean()
        breaks.append(m)
        new_head = head[head > m]
        if new_head.size == 0 or (new_head.size / vals.size > minority_frac):
            break
        head = new_head
    breaks.append(vals.max())
    return sorted(set(breaks))


def normalize_breaks(breaks):
    """Ensure exactly 4 edges for 3 bins (Low/Mid/High)."""
    uniq = sorted(set(breaks))
    if len(uniq) < 4:
        mn, mx = uniq[0], uniq[-1]
        uniq = list(np.linspace(mn, mx, num=4))
    elif len(uniq) > 4:
        third = len(uniq) // 3
        uniq = [uniq[0], uniq[third], uniq[2*third], uniq[-1]]
    return uniq


def label_categories(df):
    """Filter invalid jobs, compute CPUUtil, bin into 3×3 categories."""
    df = df.copy()
    df = df[(df['RunTime'] > 0) & (df['AverageCPUTimeUsed'] >= 0) & (df['AllocatedProcessors'] > 0)]
    df['CPUUtil'] = (df['AverageCPUTimeUsed'] / (df['RunTime'] * df['AllocatedProcessors'])).fillna(0)
    raw_rt = head_tail_breaks(df['RunTime'])
    rt_edges = normalize_breaks(raw_rt)
    df['RuntimeBin'] = pd.cut(df['RunTime'], bins=rt_edges, labels=['Low','Mid','High'], include_lowest=True)
    df['IntensityBin'] = pd.qcut(df['CPUUtil'], q=3, labels=['Low','Mid','High'])
    df['Category'] = df['RuntimeBin'].astype(str) + '_' + df['IntensityBin'].astype(str)
    return df


def split_by_weekday(df):
    """Split labeled DataFrame into buckets for each weekday."""
    df = df.copy()
    df['Weekday'] = df['SubmitDateTime'].dt.weekday
    df['SubmitTimeOfDay'] = df['SubmitDateTime'].dt.strftime('%H:%M:%S')
    return {wd: df[df['Weekday'] == wd].copy() for wd in range(7)}


def compute_bin_edges(df):
    """
    Compute edges for runtime (head-tail) and CPUUtil (quantile) bins.
    Returns:
      - rt_edges: list of 4 floats
      - cpu_edges: list of 4 floats
    """
    tmp = df.copy()
    tmp['CPUUtil'] = (tmp['AverageCPUTimeUsed'] / (tmp['RunTime'] * tmp['AllocatedProcessors'])).replace([np.inf, -np.inf], np.nan).fillna(0)
    cpu_vals = tmp['CPUUtil'].dropna().values
    if cpu_vals.size == 0:
        cpu_edges = [0.0, 0.33, 0.67, 1.0]
    else:
        raw = np.quantile(cpu_vals, [0.0, 1/3, 2/3, 1.0])
        cpu_edges = sorted(set(raw))
        if len(cpu_edges) < 4:
            cpu_edges = list(np.linspace(cpu_edges[0], cpu_edges[-1], num=4))
    raw_rt = head_tail_breaks(tmp['RunTime'])
    rt_edges = normalize_breaks(raw_rt)
    return rt_edges, cpu_edges

def compute_job_count_edges(df):
    """
    Compute edges for user job-count (head-tail) bins based on a DataFrame slice.
    Returns a list of 4 floats defining the Low/Mid/High activity thresholds:
      [min_count, edge1, edge2, max_count].
    """
    # Count jobs per user
    user_job_counts = df['UserID'].value_counts().values

    # If there are no users or only one unique count, use default linear bins
    if len(user_job_counts) == 0:
        return [0, 1, 2, 3]
    
    # Use head-tail breaks to find raw thresholds
    raw_breaks = head_tail_breaks(pd.Series(user_job_counts))
    
    # Normalize to exactly 4 edges
    job_count_edges = normalize_breaks(raw_breaks)
    
    return job_count_edges

def classify_job(job, rt_edges, cpu_edges):
    """Classify single job record into category using precomputed edges."""
    cpu_util = (job['AverageCPUTimeUsed'] / (job['RunTime'] * job['AllocatedProcessors'])) if job['RunTime'] > 0 else 0
    rt_bin = pd.cut([job['RunTime']], bins=rt_edges, labels=['Low','Mid','High'], include_lowest=True)[0]
    ci_bin = pd.cut([cpu_util], bins=cpu_edges, labels=['Low','Mid','High'], include_lowest=True)[0]
    return f"{rt_bin}_{ci_bin}"
