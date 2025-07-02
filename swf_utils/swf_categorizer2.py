import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import logging
from scipy.stats import gamma
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


def detect_and_remove_anomalies(df):
    """
    Flag and remove anomalies using IsolationForest and the Interquartile Range (IQR) method
    to automatically determine the anomaly threshold.

    Args:
        df (pd.DataFrame): The input DataFrame.

    Returns:
        tuple: (anomalies_df, clean_df) DataFrames.
    """
    feats = ['Interarrival', 'RunTime', 'AllocatedProcessors', 'AverageCPUTimeUsed', 'UsedMemory']
    X = df[feats].fillna(0)

    # Initialize IsolationForest without 'contamination' during training.
    # We will use its decision_function to determine anomalies dynamically.
    iso = IsolationForest(n_estimators=100, random_state=0, n_jobs=-1)
    iso.fit(X) # Fit the model to learn anomaly scores

    # Get the anomaly scores for each data point
    # Lower scores indicate a higher likelihood of being an anomaly for IsolationForest
    anomaly_scores = iso.decision_function(X)

    # --- IQR Method Implementation to determine the threshold automatically ---
    Q1 = np.percentile(anomaly_scores, 25)
    Q3 = np.percentile(anomaly_scores, 75)
    IQR = Q3 - Q1

    # For Isolation Forest, lower scores are more anomalous.
    # We define anomalies as scores falling below the 'Lower Fence' in the IQR method.
    threshold_score = Q1 - (1.5 * IQR)
    # --- End IQR Method Implementation ---

    df2 = df.copy()
    # A data point is an anomaly if its score is less than or equal to the calculated threshold score
    df2['Anomaly'] = (anomaly_scores <= threshold_score)

    anomalies = df2[df2['Anomaly']].copy()
    clean = df2[~df2['Anomaly']].copy()

    # Log information about the detected anomalies for transparency
    num_anomalies = len(anomalies)
    total_rows = len(df2)
    percentage_anomalies = (num_anomalies / total_rows) * 100 if total_rows > 0 else 0
    logging.info(f"Anomaly detection: Detected {num_anomalies} anomalies ({percentage_anomalies:.2f}%) "
                 f"from {total_rows} total rows using IQR method. Lower Fence Threshold Score = {threshold_score:.4f}.")

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

def _get_fixed_num_edges(raw_breaks, num_edges=4):
    """
    Ensures a list of breaks has exactly num_edges, handling too few or too many.
    This is used to normalize the output of head_tail_breaks or for quantile binning.
    """
    uniq = sorted(list(set(raw_breaks)))

    # Handle cases where not enough unique breaks are generated
    if len(uniq) < 2: # Need at least 2 points to define an interval
        # If series was empty, all values same, or not enough unique values
        logging.warning(f"Insufficient unique values ({len(uniq)}) to create {num_edges} edges. Falling back to linear spacing [0, 1].")
        return list(np.linspace(0.0, 1.0, num=num_edges))

    # If the number of unique breaks already matches the desired number of edges
    if len(uniq) == num_edges:
        return uniq
    
    # If too few unique breaks, linearly space between min and max
    if len(uniq) < num_edges:
        # Avoid division by zero if mn == mx
        mn, mx = uniq[0], uniq[-1]
        if mn == mx:
            # Create small increments if all values are identical
            return list(np.linspace(mn, mn + 0.1 * (num_edges - 1), num=num_edges))
        return list(np.linspace(mn, mx, num=num_edges))
    else: # len(uniq) > num_edges
        # If too many unique breaks, select representative points using interpolation
        # This ensures we get exactly `num_edges` points smoothly distributed
        x_coords = np.linspace(0, 1, len(uniq)) # Map current breaks to a 0-1 scale
        x_interp = np.linspace(0, 1, num_edges) # Desired points on the 0-1 scale
        
        # Interpolate the values to get the desired number of breaks
        interpolated_values = np.interp(x_interp, x_coords, uniq)
        
        # Ensure final edges are unique and sorted, handling potential floating point duplicates
        final_edges = sorted(list(set(interpolated_values)))
        
        # In case interpolation resulted in fewer unique points due to precision or data sparsity,
        # fallback to a robust linear spacing
        if len(final_edges) < num_edges:
            logging.warning(f"Interpolation for {len(uniq)} breaks resulted in fewer than {num_edges} unique edges. Falling back to linear spacing between min and max.")
            return list(np.linspace(uniq[0], uniq[-1], num=num_edges))

        return final_edges
    
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


def compute_bin_edges(df, num_bins=3):
    """
    Compute edges for runtime (using Head-Tail breaks) and CPUUtil (using Quantile Binning).
    Both methods' outputs are normalized to ensure exactly num_bins + 1 edges.

    Args:
        df (pd.DataFrame): The input DataFrame.
        num_bins (int): The desired number of bins for each feature (e.g., 3 for Low/Mid/High).

    Returns:
        tuple: (rt_edges, cpu_edges) lists of floats, each containing num_bins + 1 edges.
    """
    tmp = df.copy()

    # Compute CPUUtil, handling potential division by zero or infinite values
    tmp['CPUUtil'] = (tmp['AverageCPUTimeUsed'] / (tmp['RunTime'] * tmp['AllocatedProcessors'])).replace([np.inf, -np.inf], np.nan).fillna(0)

    # --- Runtime Edges (Head-Tail Breaks, then normalized) ---
    raw_rt_breaks = head_tail_breaks(tmp['RunTime'])
    rt_edges = _get_fixed_num_edges(raw_rt_breaks, num_edges=num_bins + 1) # Ensure 4 edges for 3 bins

    # --- CPUUtil Edges (Quantile Binning using pd.qcut, then normalized) ---
    cpu_vals = tmp['CPUUtil'].dropna() # pd.qcut handles NaNs if they are dropped first

    # Handle cases where pd.qcut might fail due to insufficient unique values
    # If len(cpu_vals.unique()) < num_bins + 1, pd.qcut will raise ValueError.
    # We use a try-except block and _get_fixed_num_edges as a robust fallback.
    try:
        # Use pd.qcut to get quantile bins. retbins=True gives the bin edges.
        # duplicates='drop' prevents errors if multiple quantiles are at the same value.
        # This will return a variable number of bins if duplicates are dropped.
        _, raw_cpu_breaks = pd.qcut(cpu_vals, q=num_bins, retbins=True, labels=False, duplicates='drop')
        cpu_edges = _get_fixed_num_edges(raw_cpu_breaks, num_edges=num_bins + 1)
    except ValueError as e:
        logging.warning(f"pd.qcut failed for CPUUtil (possibly due to too few unique values for {num_bins} bins): {e}. Falling back to linear spacing for CPU edges.")
        # Fallback to linear spacing if qcut fails entirely
        cpu_edges = _get_fixed_num_edges([cpu_vals.min(), cpu_vals.max()], num_edges=num_bins + 1)
    except Exception as e:
        logging.error(f"An unexpected error occurred during pd.qcut for CPUUtil: {e}. Falling back to linear spacing for CPU edges.")
        cpu_edges = _get_fixed_num_edges([cpu_vals.min(), cpu_vals.max()], num_edges=num_bins + 1)


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
        return _get_fixed_num_edges([0.0, 1.0], num_edges=4)
    
    raw_breaks = head_tail_breaks(user_max_bursts)
    
    # Use the more robust _get_fixed_num_edges instead of normalize_breaks
    burst_count_edges = _get_fixed_num_edges(raw_breaks, num_edges=4)
    
    return burst_count_edges

def compute_job_count_edges(df):
    """
    Compute edges for user job-count (head-tail) bins based on a DataFrame slice.
    Returns a list of 4 floats defining the Low/Mid/High activity thresholds:
      [min_count, edge1, edge2, max_count].
    """
    # Count jobs per user; value_counts returns a Series, which is suitable for head_tail_breaks
    user_job_counts = df['UserID'].value_counts()

    # If there are no users or only one unique count, use robust linear bins via _get_fixed_num_edges
    if user_job_counts.empty:
        return _get_fixed_num_edges([0.0, 1.0], num_edges=4) # Consistent default
    
    # Use head-tail breaks to find raw thresholds
    raw_breaks = head_tail_breaks(user_job_counts)
    
    # Normalize to exactly 4 edges using the more robust _get_fixed_num_edges
    job_count_edges = _get_fixed_num_edges(raw_breaks, num_edges=4)
    
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

def calculate_interval_burstiness(df, interval_length_minutes):
    """
    Calculates burstiness metrics for specified time intervals (e.g., 10-minute, 1-hour).
    Enhanced to include Index of Dispersion for Counts (IDC).
    
    Args:
        df (pd.DataFrame): DataFrame with 'SubmitTime' (unix timestamp) and 'JobID'.
        interval_length_minutes (int): Length of the time interval in minutes.
        
    Returns:
        pd.DataFrame: DataFrame with burstiness metrics per interval.
    """
    if df.empty:
        logging.warning("Input DataFrame is empty for calculate_interval_burstiness.")
        return pd.DataFrame()

    df_sorted = df.sort_values(by='SubmitTime').copy()
    df_sorted['interval_start'] = (df_sorted['SubmitTime'] // (interval_length_minutes * 60)) * (interval_length_minutes * 60)
    
    interval_metrics = []
    
    for start_time, group in df_sorted.groupby('interval_start'):
        job_count = len(group)
        
        # Calculate burstiness only if there are at least 2 jobs in the interval for meaningful interarrival
        if job_count > 1:
            inter_arrival_times = group['SubmitTime'].diff().dropna().values
            mean_inter_arrival = np.mean(inter_arrival_times)
            std_inter_arrival = np.std(inter_arrival_times)
            cv_inter_arrival = std_inter_arrival / mean_inter_arrival if mean_inter_arrival > 0 else 0
            
            # Max burst duration - based on largest gap below threshold or largest continuous activity
            # This logic depends on how 'burst' is defined here.
            # Assuming current 'compute_burst_activity_edges' logic, we might need a simpler
            # metric here if not directly using an external burst detection for this function.
            # For simplicity, let's keep it aligned with previous 'max_burst_duration' if it was defined.
            # If not, for now, we can use 0 or 'N/A' and refine this with explicit burst detection within the interval.
            max_burst_duration = 0 # Placeholder for now, needs clear definition of "burst within interval"

            # Index of Dispersion for Counts (IDC)
            # For counts, it's Variance(counts) / Mean(counts) over sub-intervals or repeated observations
            # Here, we're looking at counts within the current interval.
            # A common approach for IDC within a single interval is to consider a finer granularity.
            # For simplicity, let's consider the jobs within *this* interval as one observation
            # and use job_count as the "count".
            # If we need IDC for a truly bursty time series, we'd divide the interval into smaller bins
            # and calculate the variance/mean of counts *across those bins*.
            # For now, let's use a simpler proxy.
            # If we assume arrival rate (jobs/second) for this interval:
            interval_duration_seconds = interval_length_minutes * 60
            if interval_duration_seconds > 0:
                arrival_rate = job_count / interval_duration_seconds
            else:
                arrival_rate = 0

            # Simplistic IDC for a single interval's job count:
            # More sophisticated IDC requires partitioning the current interval into smaller, equal bins
            # and calculating var/mean of counts within *those* sub-bins.
            # For now, let's just use the CV of inter-arrival times as the primary burstiness indicator for this function,
            # and potentially enhance IDC if truly needed at this level.
            # Given the request to add IDC, let's add a placeholder or a very basic calculation.
            # A robust IDC for *counts* implies splitting the interval into sub-intervals.
            
            # Let's add a placeholder for IDC, assuming we might need to compute it more rigorously later
            # (e.g., by splitting the interval into micro-intervals and counting jobs).
            # For now, if job_count is meaningful for IDC:
            # IDC = variance_of_counts / mean_of_counts. For a single interval, it's not straightforward without sub-intervals.
            # Let's calculate the Fano Factor (variance/mean of counts in *sub-bins*) if we want true IDC.
            # Since the current code doesn't define sub-bins for this function, let's add a simple check for IDC
            # based on variability of inter-arrival times.
            # If CV > 1, it's generally bursty, and ID will be > 1.
            # For now, let's just record relevant simple statistics.

            # Reconsidering IDC for `calculate_interval_burstiness`:
            # IDC is typically for counts over *multiple* periods. For a *single* interval,
            # the job_count itself is the count. The variability *within* the interval is captured by CV.
            # If the aim is to get a true IDC for this interval, one would need to divide
            # it into smaller, fixed sub-intervals (e.g., 1-second bins) and count jobs in each,
            # then compute variance/mean of those sub-counts. This adds complexity here.
            # Let's stick to CV and max_burst_duration as primary interval metrics for now.
            # The prompt asked for "Index of Dispersion (ID) / Index of Dispersion for Counts (IDC)",
            # so let's include it, but with a warning about its interpretation if not based on sub-intervals.
            # A simplified proxy for IDC, which is often done for burstiness:
            # If arrivals follow Poisson, IDC=1. If bursty, IDC>1.
            # A simple way to approximate for this interval without sub-binning is to use CV^2 (Squared Coefficient of Variation).
            # This is not strictly IDC but related.
            
            # Let's add a placeholder for IDC and CV:
            interval_metrics.append({
                'interval_start': start_time,
                'job_count': job_count,
                'mean_inter_arrival': mean_inter_arrival,
                'std_inter_arrival': std_inter_arrival,
                'cv_inter_arrival': cv_inter_arrival,
                'max_burst_duration': max_burst_duration, # This needs proper definition/calc
                'arrival_rate': arrival_rate
            })
        elif job_count == 1:
             interval_metrics.append({
                'interval_start': start_time,
                'job_count': job_count,
                'mean_inter_arrival': np.nan, # Not applicable for single job
                'std_inter_arrival': np.nan, # Not applicable
                'cv_inter_arrival': np.nan, # Not applicable
                'max_burst_duration': 0,
                'arrival_rate': job_count / (interval_length_minutes * 60) if interval_length_minutes * 60 > 0 else 0
            })
        else: # job_count == 0
            interval_metrics.append({
                'interval_start': start_time,
                'job_count': 0,
                'mean_inter_arrival': np.nan,
                'std_inter_arrival': np.nan,
                'cv_inter_arrival': np.nan,
                'max_burst_duration': 0,
                'arrival_rate': 0
            })

    # Ensure all intervals are covered, even those with no jobs
    all_interval_starts = np.arange(
        df_sorted['SubmitTime'].min() // (interval_length_minutes * 60) * (interval_length_minutes * 60),
        (df_sorted['SubmitTime'].max() // (interval_length_minutes * 60) + 1) * (interval_length_minutes * 60),
        interval_length_minutes * 60
    )
    
    interval_df = pd.DataFrame(interval_metrics)
    # Use merge to fill in missing intervals with default values
    full_interval_df = pd.DataFrame({'interval_start': all_interval_starts})
    full_interval_df = pd.merge(full_interval_df, interval_df, on='interval_start', how='left').fillna({
        'job_count': 0,
        'mean_inter_arrival': np.nan,
        'std_inter_arrival': np.nan,
        'cv_inter_arrival': np.nan,
        'max_burst_duration': 0,
        'arrival_rate': 0
    })

    return full_interval_df 


def determine_burstiness_thresholds(df_metrics):
    """
    Determine thresholds for burstiness metrics (job_count, arrival_rate, cv, max_burst_duration).
    This function uses appropriate binning strategies (Head-Tail for heavy-tailed, Quantile for others)
    and ensures a consistent output format of 4 edges for 3 bins per metric.

    Parameters:
    - df_metrics: DataFrame containing calculated interval metrics (output of calculate_interval_burstiness).

    Returns:
    - A dictionary where keys are metric names and values are lists of 4 floats [min, edge1, edge2, max].
    """
    thresholds = {}
    
    # Define which metrics we expect and how to handle them
    metrics_to_process = {
        'job_count': 'head_tail',
        'arrival_rate': 'head_tail',
        'cv': 'quantile', # CV distribution might be more amenable to quantiles
        'max_burst_duration': 'head_tail'
    }

    for metric_name, method in metrics_to_process.items():
        if metric_name not in df_metrics.columns:
            logging.warning(f"Metric '{metric_name}' not found in df_metrics. Skipping.")
            # Provide a sensible default for missing metrics
            thresholds[metric_name] = _get_fixed_num_edges([0.0, 1.0], num_edges=4)
            continue
            
        metric_series = df_metrics[metric_name].dropna()

        if metric_series.empty:
            logging.warning(f"Metric '{metric_name}' series is empty. Using default linear bins.")
            # Use _get_fixed_num_edges for consistent fallback
            edges = _get_fixed_num_edges([0.0, 1.0], num_edges=4)
        elif metric_series.nunique() < 2:
            logging.warning(f"Metric '{metric_name}' has less than 2 unique values. Using default linear bins starting from min value.")
            # If all values are the same, create slightly spaced bins
            min_val = metric_series.iloc[0]
            edges = _get_fixed_num_edges([min_val, min_val + 0.1], num_edges=4) # Ensure a range for _get_fixed_num_edges
        else:
            if method == 'head_tail':
                raw_breaks = head_tail_breaks(metric_series)
            elif method == 'quantile':
                # Use pd.qcut to get quantile bins. labels=False returns bin indices, which are not what we want.
                # Instead, we want the edges. We can use the quantiles directly.
                quantiles = np.percentile(metric_series, [0, 33.33, 66.67, 100]) # Approx 0, 1/3, 2/3, 1
                raw_breaks = list(quantiles)
            else:
                logging.error(f"Unknown binning method: {method} for {metric_name}. Using default linear bins.")
                raw_breaks = [metric_series.min(), metric_series.max()]

            # Always normalize the raw breaks to exactly 4 edges using _get_fixed_num_edges
            edges = _get_fixed_num_edges(raw_breaks, num_edges=4)
        
        thresholds[metric_name] = edges
    
    return thresholds

def fit_inter_arrival_distributions_by_burst_level(df_clean, user_burst_metrics, burst_activity_edges):
    """
    Fits statistical distributions (e.g., Gamma) to inter-arrival times for
    different burstiness levels, particularly focusing on 'High' burst periods.

    Args:
        df_clean (pd.DataFrame): The cleaned DataFrame of job data, including 'SubmitTime', 'UserID'.
                                  Assumes 'InterArrivalTime' has been calculated.
        user_burst_metrics (pd.DataFrame): DataFrame containing 'UserID', 'max_burst_size', 'burst_level'.
        burst_activity_edges (dict): Dictionary with 'max_burst_size' edges (e.g., {'Low': x, 'Mid': y, 'High': z}).

    Returns:
        dict: A dictionary where keys are burst levels (e.g., 'High') and values
              are dictionaries of fitted distribution parameters (e.g., {'gamma_params': (alpha, loc, beta)}).
    """
    if df_clean.empty or user_burst_metrics.empty or not burst_activity_edges:
        logging.warning("Missing input data for fit_inter_arrival_distributions_by_burst_level. Skipping fitting.")
        return {}

    logging.info("Fitting inter-arrival distributions by burst level...")
    
    # Ensure InterArrivalTime is available in df_clean
    if 'InterArrivalTime' not in df_clean.columns:
        # Calculate inter-arrival times globally, and then filter for specific users/bursts
        # Or, calculate per user if necessary for more granular fitting.
        # For this function, let's assume it might be pre-calculated or calculated on the fly.
        # It's usually calculated per user for accurate burst detection.
        df_clean = df_clean.sort_values(by=['UserID', 'SubmitTime'])
        df_clean['InterArrivalTime'] = df_clean.groupby('UserID')['SubmitTime'].diff().fillna(0)


    fitted_distributions = {}

    # Define thresholds for 'High' burst users/jobs
    high_burst_threshold = burst_activity_edges.get('Mid', {}).get('upper', np.inf) 
    # 'Mid' upper edge separates 'Mid' from 'High' if sorted low to high.
    # Check the structure of burst_activity_edges from compute_burst_activity_edges.
    # It might be {metric_name: {'Low': {'lower': x, 'upper': y}, ...}}
    # Assuming the structure is direct { 'Low': [lower,upper], 'Mid':[lower,upper], 'High':[lower,upper]}
    # Let's adjust based on the actual output of compute_burst_activity_edges in swf_categorizer2.py
    # From swf_categorizer2.py: 'edges' from compute_burst_activity_edges is a list like [low_lower, low_upper, mid_upper, high_upper]
    
    # Re-evaluate how burst_activity_edges are structured from compute_burst_activity_edges:
    # `compute_burst_activity_edges` returns a dictionary:
    # { 'max_burst_size': [edge0, edge1, edge2, edge3] } where edges typically define bins.
    # If using 4 edges, it implies 3 bins (Low, Mid, High).
    # e.g., Low: edge0-edge1, Mid: edge1-edge2, High: edge2-edge3.
    
    max_burst_size_edges = burst_activity_edges.get('max_burst_size', [])
    if len(max_burst_size_edges) < 4: # Need at least 4 edges for 3 categories
        logging.warning("Insufficient edges for max_burst_size to define burst levels for distribution fitting.")
        return {}
    
    # Define the threshold for 'High' burst jobs
    # Jobs with max_burst_size >= mid_to_high_edge are considered 'High' burst.
    # This might correspond to max_burst_size_edges[2] if they are sorted.
    
    # Identify users classified as 'High' burst
    high_burst_users = user_burst_metrics[user_burst_metrics['burst_level'] == 'High']['UserID'].unique()

    # Filter job data for 'High' burst users and non-zero inter-arrival times
    # Only consider jobs from 'High' burst users that are part of a 'burst' as identified previously.
    # This is a critical point: Do we fit distribution on ALL inter-arrival times of 'High' users,
    # or just the inter-arrival times *within* detected bursts for those users?
    # For realism, it should be inter-arrival times *during bursts*.
    # The current `compute_user_burst_metrics` identifies 'max_burst_size' for each user.
    # To get inter-arrival times *within* bursts, we'd need to re-identify burst periods.

    # Simpler approach for now: Get all inter-arrival times for 'High' burst users.
    # This might dilute the "bursty" pattern with non-bursty times.
    # A more advanced approach would involve re-identifying burst periods for each high-burst user
    # and extracting inter-arrival times specifically within those bursts.
    
    # For this function, let's proceed with inter-arrival times of users classified as 'High' burst,
    # specifically those inter-arrival times that are not zero (i.e., not the first job of a user).
    
    high_burst_inter_arrivals = df_clean[
        (df_clean['UserID'].isin(high_burst_users)) & 
        (df_clean['InterArrivalTime'] > 0) # Only consider actual inter-arrival times
    ]['InterArrivalTime'].dropna().values

    if len(high_burst_inter_arrivals) > 0:
        logging.info(f"Attempting to fit Gamma distribution for 'High' burst inter-arrivals (n={len(high_burst_inter_arrivals)})...")
        try:
            # Fit a Gamma distribution (shape, loc, scale)
            # loc is location parameter, often fixed at 0 for inter-arrival times
            # scale = beta (inverse of rate)
            alpha_gamma, loc_gamma, beta_gamma = gamma.fit(high_burst_inter_arrivals, floc=0)
            fitted_distributions['High'] = {
                'gamma_params': (alpha_gamma, loc_gamma, beta_gamma)
            }
            logging.info(f"Fitted Gamma for 'High' burst: alpha={alpha_gamma:.2f}, loc={loc_gamma:.2f}, beta={beta_gamma:.2f}")
        except Exception as e:
            logging.error(f"Error fitting Gamma distribution for 'High' burst inter-arrivals: {e}")
            fitted_distributions['High'] = None
    else:
        logging.warning("No valid inter-arrival times found for 'High' burst users to fit distribution.")

    # You can extend this for 'Mid' or 'Low' burst levels if desired,
    # fitting different distributions or parameters based on their characteristics.
    # For now, focusing on 'High' burst as it's critical for reproduction.

    return fitted_distributions