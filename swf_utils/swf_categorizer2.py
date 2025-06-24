import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest

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
        # Return a default set of monotonically increasing breaks if series is empty
        return [0.0, 1.0, 2.0, 3.0] # Ensure at least 4 edges for 3 bins
    breaks = [vals.min()]
    head = vals.copy()
    while True:
        m = head.mean()
        breaks.append(m)
        new_head = head[head > m]
        if new_head.size == 0 or (new_head.size / vals.size > minority_frac):
            break
        head = new_head
    # Ensure there are at least 4 unique values and add max if not present
    unique_breaks = sorted(list(set(breaks)))
    if vals.max() not in unique_breaks:
        unique_breaks.append(vals.max())
    return sorted(unique_breaks)


def normalize_breaks(breaks):
    """Ensure exactly 4 edges for 3 bins (Low/Mid/High) and that they are monotonically increasing."""
    uniq = sorted(list(set(breaks)))
    
    # Handle cases where not enough unique breaks are generated
    if len(uniq) < 2: # Need at least 2 points to define an interval
        return [0.0, 1/3, 2/3, 1.0] # Default linear bins

    if len(uniq) < 4:
        # If less than 4 unique breaks, create a linear space between min and max
        mn, mx = uniq[0], uniq[-1]
        # Avoid division by zero if mn == mx
        if mn == mx:
            return [mn, mn + 0.1, mn + 0.2, mn + 0.3] # Add small increments
        return list(np.linspace(mn, mx, num=4))
    elif len(uniq) > 4:
        # If more than 4, select representative points (min, two intermediate, max)
        # This part assumes there are at least 4 points to pick from.
        # If there are exactly 4, this will return uniq itself.
        # If there are more, it will pick the first, two roughly equally spaced, and the last.
        q = np.linspace(0, 1, 4)
        indices = [int(x * (len(uniq) - 1)) for x in q]
        # Ensure unique indices
        indices = sorted(list(set(indices)))
        if len(indices) < 4: # Fallback if indices are not unique enough
            return list(np.linspace(uniq[0], uniq[-1], num=4))
        uniq = [uniq[i] for i in indices]
    
    return sorted(uniq)


def compute_bin_edges(df):
    """
    Compute edges for runtime (head-tail) and CPUUtil (quantile) bins.
    Returns:
      - rt_edges: list of 4 floats
      - cpu_edges: list of 4 floats
    """
    tmp = df.copy()
    # Handle potential division by zero or large numbers leading to inf
    tmp['CPUUtil'] = (tmp['AverageCPUTimeUsed'] / (tmp['RunTime'] * tmp['AllocatedProcessors'])).replace([np.inf, -np.inf], np.nan).fillna(0)
    cpu_vals = tmp['CPUUtil'].dropna().values
    
    # Ensure at least two unique values for quantile calculation
    if len(np.unique(cpu_vals)) < 2:
        # If all values are the same or not enough diversity, use a default range
        if cpu_vals.size > 0:
            val = cpu_vals[0]
            cpu_edges = [val, val + 0.1, val + 0.2, val + 0.3] # Small increments
        else:
            cpu_edges = [0.0, 0.33, 0.67, 1.0]
    else:
        raw = np.quantile(cpu_vals, [0.0, 1/3, 2/3, 1.0])
        cpu_edges = sorted(list(set(raw)))
        if len(cpu_edges) < 4:
            # If after unique and sort, we still don't have 4 edges, linearly space them
            cpu_edges = list(np.linspace(cpu_edges[0], cpu_edges[-1], num=4))
            
    raw_rt = head_tail_breaks(tmp['RunTime'])
    rt_edges = normalize_breaks(raw_rt)
    
    return rt_edges, cpu_edges

def label_and_categorize_jobs(df, rt_edges, cpu_edges):
    """
    Labels jobs with RuntimeBin, IntensityBin, and Category based on precomputed edges.
    Also computes CPUUtil and filters invalid jobs.
    """
    df_filtered = df.copy()
    
    # Filter out invalid jobs before computing CPUUtil and binning
    df_filtered = df_filtered[
        (df_filtered['RunTime'] > 0) & 
        (df_filtered['AllocatedProcessors'] > 0) & 
        (df_filtered['AverageCPUTimeUsed'] >= 0)
    ].copy() # Make a copy to avoid SettingWithCopyWarning

    # Compute CPUUtil, handling potential division by zero or infinite values
    df_filtered['CPUUtil'] = (df_filtered['AverageCPUTimeUsed'] / 
                               (df_filtered['RunTime'] * df_filtered['AllocatedProcessors'])).replace([np.inf, -np.inf], np.nan).fillna(0)

    # Bin 'RunTime' using the provided rt_edges
    # Ensure rt_edges has at least 2 distinct values and is sorted for pd.cut
    if len(np.unique(rt_edges)) < 2:
        raise ValueError(f"Runtime edges must have at least 2 distinct values. Got: {rt_edges}")
    df_filtered['RuntimeBin'] = pd.cut(
        df_filtered['RunTime'], 
        bins=sorted(rt_edges), # Ensure bins are sorted
        labels=['Low','Mid','High'], 
        include_lowest=True, 
        right=False # This means intervals are [a, b) except for the last one which is [a, b]
    )

    # Bin 'CPUUtil' using the provided cpu_edges
    # Ensure cpu_edges has at least 2 distinct values and is sorted for pd.cut
    if len(np.unique(cpu_edges)) < 2:
        raise ValueError(f"CPU Utilization edges must have at least 2 distinct values. Got: {cpu_edges}")
    df_filtered['IntensityBin'] = pd.cut(
        df_filtered['CPUUtil'], 
        bins=sorted(cpu_edges), # Ensure bins are sorted
        labels=['Low','Mid','High'], 
        include_lowest=True, 
        right=False
    )
    
    # Combine the bins into a 'Category' string
    df_filtered['Category'] = df_filtered['RuntimeBin'].astype(str) + '_' + df_filtered['IntensityBin'].astype(str)
    
    return df_filtered

def split_by_weekday(df):
    """Split labeled DataFrame into buckets for each weekday."""
    df = df.copy()
    df['Weekday'] = df['SubmitDateTime'].dt.weekday
    df['SubmitTimeOfDay'] = df['SubmitDateTime'].dt.strftime('%H:%M:%S')
    return {wd: df[df['Weekday'] == wd].copy() for wd in range(7)}

def compute_user_burst_metrics(df, burst_threshold_seconds=60):
    """For each user, find the size of their largest job burst."""
    if df.empty:
        return pd.Series(dtype=int)

    df_sorted = df.sort_values(['UserID', 'SubmitDateTime'])
    df_sorted['UserInterarrival'] = df_sorted.groupby('UserID')['SubmitDateTime'].diff().dt.total_seconds()
    
    # A new burst starts if the interarrival time is greater than the threshold or it's the first job of a user
    is_new_burst = (df_sorted['UserInterarrival'] > burst_threshold_seconds) | (df_sorted['UserInterarrival'].isna())
    # `cumsum()` on boolean creates unique group IDs for each burst
    burst_group_id = is_new_burst.cumsum()
    
    # Group by UserID and the burst_group_id to get burst sizes
    burst_sizes = df_sorted.groupby(['UserID', burst_group_id]).transform('size')
    
    max_bursts_per_user = burst_sizes.groupby(df_sorted['UserID']).max()
    return max_bursts_per_user

def compute_burst_activity_edges(df, burst_threshold_seconds=60):
    """Compute edges for user activity bins based on job burst sizes."""
    user_max_bursts = compute_user_burst_metrics(df, burst_threshold_seconds)

    if user_max_bursts.empty:
        # Default edges for burst activity if no user bursts can be computed
        return [0, 1, 2, 3] # Low, Mid, High based on count of jobs in burst
    
    raw_breaks = head_tail_breaks(user_max_bursts)
    burst_count_edges = normalize_breaks(raw_breaks)
    
    return burst_count_edges

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
        return [0, 1, 2, 3] # Default edges

    # Use head-tail breaks to find raw thresholds
    raw_breaks = head_tail_breaks(pd.Series(user_job_counts))
    
    # Normalize to exactly 4 edges
    job_count_edges = normalize_breaks(raw_breaks)
    
    return job_count_edges

def classify_job(job, rt_edges, cpu_edges):
    """Classify single job record into category using precomputed edges."""
    # Ensure RunTime and AllocatedProcessors are not zero to avoid division by zero
    cpu_util = (job['AverageCPUTimeUsed'] / (job['RunTime'] * job['AllocatedProcessors'])) if (job['RunTime'] > 0 and job['AllocatedProcessors'] > 0) else 0
    
    # Use pd.cut for single value classification, but ensure the value is in a list/array
    # Also ensure the edges are sorted before passing to pd.cut
    rt_bin = pd.cut([job['RunTime']], bins=sorted(rt_edges), labels=['Low','Mid','High'], include_lowest=True, right=False)[0]
    ci_bin = pd.cut([cpu_util], bins=sorted(cpu_edges), labels=['Low','Mid','High'], include_lowest=True, right=False)[0]
    return f"{rt_bin}_{ci_bin}"

def calculate_interval_burstiness(df, interval_minutes):
    """
    Calculate burstiness metrics for intervals of specified length across the entire dataset.
    
    Parameters:
    - df: DataFrame containing job data with 'SubmitDateTime' column.
    - interval_minutes: Length of each interval in minutes (e.g., 10 for 10-minute intervals).
    
    Returns:
    - A DataFrame containing burstiness metrics for each interval.
    """
    # Create interval start and end times
    start_time = df['SubmitDateTime'].min().floor('h')
    end_time = df['SubmitDateTime'].max().ceil('h')
    interval = pd.Timedelta(minutes=interval_minutes)
    
    # Prepare to store metrics
    metrics_list = []
    
    current_time = start_time
    while current_time < end_time:
        next_time = current_time + interval
        interval_data = df[(df['SubmitDateTime'] >= current_time) & (df['SubmitDateTime'] < next_time)]
        
        # Calculate job count
        job_count = len(interval_data)
        
        # Calculate job arrival rate
        interval_duration = interval.total_seconds() / 60  # in minutes
        arrival_rate = job_count / interval_duration if interval_duration > 0 else 0
        
        # Calculate interarrival times
        if job_count < 2:
            interarrival_times = []
        else:
            sorted_times = interval_data['SubmitDateTime'].sort_values()
            interarrival_times = [
                (sorted_times.iloc[i+1] - sorted_times.iloc[i]).total_seconds()
                for i in range(len(sorted_times)-1)
            ]
        
        # Calculate Coefficient of Variation (CV)
        if interarrival_times:
            mean_interarrival = np.mean(interarrival_times)
            std_interarrival = np.std(interarrival_times)
            cv = std_interarrival / mean_interarrival if mean_interarrival != 0 else 0
        else:
            cv = 0
        
        # Identify bursts (consecutive jobs within BURST_THRESHOLD_SECONDS)
        bursts = []
        current_burst = []
        sorted_times = interval_data['SubmitDateTime'].sort_values()
        for i in range(len(sorted_times)):
            if i == 0:
                current_burst.append(sorted_times.iloc[i])
            else:
                if (sorted_times.iloc[i] - sorted_times.iloc[i-1]).total_seconds() <= BURST_THRESHOLD_SECONDS:
                    current_burst.append(sorted_times.iloc[i])
                else:
                    if len(current_burst) >= 2:
                        bursts.append(current_burst)
                    current_burst = [sorted_times.iloc[i]]
        if len(current_burst) >= 2:
            bursts.append(current_burst)
        
        # Calculate max burst duration
        max_burst_duration = max(
            [(burst[-1] - burst[0]).total_seconds() for burst in bursts]
        ) if bursts else 0
        
        metrics_list.append({
            'start_time': current_time,
            'end_time': next_time,
            'job_count': job_count,
            'arrival_rate': arrival_rate,
            'cv': cv,
            'max_burst_duration': max_burst_duration
        })
        
        current_time = next_time
    
    return pd.DataFrame(metrics_list)


def determine_burstiness_thresholds(df_metrics):
    """
    Determine thresholds for classifying burstiness levels based on metrics.
    
    Parameters:
    - df_metrics: DataFrame containing burstiness metrics for intervals.
    
    Returns:
    - A dictionary containing threshold values for each metric.
    """
    # Calculate percentiles for each metric
    job_count_percentiles = np.percentile(df_metrics['job_count'], [33, 66])
    arrival_rate_percentiles = np.percentile(df_metrics['arrival_rate'], [33, 66])
    cv_percentiles = np.percentile(df_metrics['cv'], [33, 66])
    max_burst_duration_percentiles = np.percentile(df_metrics['max_burst_duration'], [33, 66])
    
    return {
        'job_count': {
            'low': job_count_percentiles[0],
            'medium': job_count_percentiles[1],
            'high': df_metrics['job_count'].max()
        },
        'arrival_rate': {
            'low': arrival_rate_percentiles[0],
            'medium': arrival_rate_percentiles[1],
            'high': df_metrics['arrival_rate'].max()
        },
        'cv': {
            'low': cv_percentiles[0],
            'medium': cv_percentiles[1],
            'high': df_metrics['cv'].max()
        },
        'max_burst_duration': {
            'low': max_burst_duration_percentiles[0],
            'medium': max_burst_duration_percentiles[1],
            'high': df_metrics['max_burst_duration'].max()
        }
    }


def calculate_interval_burstiness(df, interval_minutes, burst_threshold_seconds):
    """
    Calculate burstiness metrics for intervals of specified length across the entire dataset.
    
    Parameters:
    - df: DataFrame containing job data with 'SubmitDateTime' column.
    - interval_minutes: Length of each interval in minutes (e.g., 10 for 10-minute intervals).
    - burst_threshold_seconds: Threshold in seconds to determine if jobs are in the same burst.
    
    Returns:
    - A DataFrame containing burstiness metrics for each interval.
    """
    # Create interval start and end times
    start_time = df['SubmitDateTime'].min().floor('h')
    end_time = df['SubmitDateTime'].max().ceil('h')
    interval = pd.Timedelta(minutes=interval_minutes)
    
    # Prepare to store metrics
    metrics_list = []
    
    current_time = start_time
    while current_time < end_time:
        next_time = current_time + interval
        interval_data = df[(df['SubmitDateTime'] >= current_time) & (df['SubmitDateTime'] < next_time)]
        
        # Calculate job count
        job_count = len(interval_data)
        
        # Calculate job arrival rate
        interval_duration = interval.total_seconds() / 60  # in minutes
        arrival_rate = job_count / interval_duration if interval_duration > 0 else 0
        
        # Calculate interarrival times
        if job_count < 2:
            interarrival_times = []
        else:
            sorted_times = interval_data['SubmitDateTime'].sort_values()
            interarrival_times = [
                (sorted_times.iloc[i+1] - sorted_times.iloc[i]).total_seconds()
                for i in range(len(sorted_times)-1)
            ]
        
        # Calculate Coefficient of Variation (CV)
        if interarrival_times:
            mean_interarrival = np.mean(interarrival_times)
            std_interarrival = np.std(interarrival_times)
            cv = std_interarrival / mean_interarrival if mean_interarrival != 0 else 0
        else:
            cv = 0
        
        # Identify bursts (consecutive jobs within burst_threshold_seconds)
        bursts = []
        current_burst = []
        sorted_times = interval_data['SubmitDateTime'].sort_values()
        for i in range(len(sorted_times)):
            if i == 0:
                current_burst.append(sorted_times.iloc[i])
            else:
                if (sorted_times.iloc[i] - sorted_times.iloc[i-1]).total_seconds() <= burst_threshold_seconds:
                    current_burst.append(sorted_times.iloc[i])
                else:
                    if len(current_burst) >= 2:
                        bursts.append(current_burst)
                    current_burst = [sorted_times.iloc[i]]
        if len(current_burst) >= 2:
            bursts.append(current_burst)
        
        # Calculate max burst duration
        max_burst_duration = max(
            [(burst[-1] - burst[0]).total_seconds() for burst in bursts]
        ) if bursts else 0
        
        metrics_list.append({
            'start_time': current_time,
            'end_time': next_time,
            'job_count': job_count,
            'arrival_rate': arrival_rate,
            'cv': cv,
            'max_burst_duration': max_burst_duration
        })
        
        current_time = next_time
    
    return pd.DataFrame(metrics_list)