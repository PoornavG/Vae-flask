import pandas as pd
import numpy as np
import os
import logging
import jenkspy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import gamma
import json
from collections import defaultdict,Counter
from sklearn.ensemble import IsolationForest
import sys
import json
DEFAULT_GRANULARITY='hour'
# Add the project's root directory to the Python path
# This ensures that `config` can be imported regardless of the current working directory
try:
    # Use relative path from the current file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    sys.path.insert(0, parent_dir)
except NameError:
    # Handle cases where __file__ is not defined (e.g., in an interactive environment)
    pass

# Now you can import from the config module
from config import (
    SWF_PATH,
    BURST_LEVEL_MAP,
    DEFAULT_NUM_BINS,
    DEFAULT_BURST_THRESH_SECONDS,
    SEED
)

# ─── LOGGING CONFIG ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SWF_Categorizer')

# ─── PARSING AND PREPROCESSING ──────────────────────────────────────────
def parse_sdsc_sp2_log(path):
    """
    Parse SWF, reconstruct timestamps & interarrival.
    """
    cols = [
        "JobID","SubmitTime","WaitTime","RunTime","AllocatedProcessors",
        "AverageCPUTimeUsed","UsedMemory","RequestedProcessors","RequestedTime",
        "RequestedMemory","Status","UserID","GroupID","ExecutableID",
        "QueueNumber","PartitionNumber","PrecedingJobNumber","ThinkTimeOfPrecedingJob"
    ]
    
    # Read the data, ignoring commented lines
    df = pd.read_csv(
        path,
        sep=r'\s+',
        comment=';',
        header=None,
        names=cols,
        na_values=['-1']
    )
    
    # Convert relevant columns to numeric
    df = df.apply(pd.to_numeric, errors='coerce')
    
    # Reconstruct timestamps and calculate inter-arrival times
    df['SubmitTime'] = pd.to_datetime(df['SubmitTime'], unit='s', origin='unix')
    df['Interarrival'] = df['SubmitTime'].diff().dt.total_seconds().fillna(0)
    
    # Drop first row as it has no inter-arrival time
    df = df.iloc[1:].copy()
    
    # Fill missing values for key features with 0
    fill_cols = ['RunTime', 'AllocatedProcessors', 'AverageCPUTimeUsed', 'UsedMemory', 'Interarrival']
    for col in fill_cols:
        df[col] = df[col].fillna(0)
    
    # Drop rows where essential columns are NaN after conversion
    df.dropna(subset=['SubmitTime', 'RunTime', 'AllocatedProcessors'], inplace=True)

    logger.info(f"Parsed SWF file with {len(df)} records.")
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

# ─── CATEGORIZATION AND BINNING (USING JENKS) ──────────────────────────
def compute_bin_edges(df, num_bins=DEFAULT_NUM_BINS):
    """
    Computes optimal bin edges using the Jenks Natural Breaks algorithm.
    """
    if len(df) < num_bins:
        logger.warning(f"Not enough data to compute Jenks breaks ({len(df)} samples, {num_bins} bins). Using equal-sized bins.")
        # Fallback to qcut for equal-sized bins
        return pd.qcut(df['RunTime'], q=num_bins, retbins=True, duplicates='drop')[1], pd.qcut(df['AverageCPUTimeUsed'], q=num_bins, retbins=True, duplicates='drop')[1]

    # Use Jenks for optimal binning
    rt_edges = jenkspy.jenks_breaks(df['RunTime'].dropna().values, n_classes=num_bins)
    cpu_edges = jenkspy.jenks_breaks(df['AverageCPUTimeUsed'].dropna().values, n_classes=num_bins)
    
    # Ensure breaks start at 0
    if rt_edges[0] > 0: rt_edges.insert(0, 0)
    if cpu_edges[0] > 0: cpu_edges.insert(0, 0)
    
    logger.info(f"Computed Jenks breaks for RunTime: {rt_edges}")
    logger.info(f"Computed Jenks breaks for CPUTime: {cpu_edges}")
    
    return rt_edges, cpu_edges

def calculate_interval_burstiness(df, time_interval_sec=DEFAULT_BURST_THRESH_SECONDS):
    """
    Divides data into time intervals and calculates job count per interval.
    """
    # Create a time-based index
    df['SubmitTime_sec'] = df['SubmitTime'].astype(np.int64) // 10**9
    
    # Group jobs into fixed-size time bins (e.g., 60 seconds)
    df['IntervalID'] = df['SubmitTime_sec'] // time_interval_sec
    
    # Count jobs per interval
    interval_counts = df.groupby('IntervalID').size().reset_index(name='JobCount')
    
    return interval_counts

def compute_burst_activity_edges(df, num_bins=DEFAULT_NUM_BINS):
    """
    Computes optimal bin edges for burstiness using Jenks.
    """
    interval_counts = calculate_interval_burstiness(df)
    
    if len(interval_counts) < num_bins:
        logger.warning(f"Not enough intervals to compute Jenks breaks ({len(interval_counts)} intervals, {num_bins} bins). Using quantiles.")
        return pd.qcut(interval_counts['JobCount'], q=num_bins, retbins=True, duplicates='drop')[1]
        
    burst_count_edges = jenkspy.jenks_breaks(interval_counts['JobCount'].values, n_classes=num_bins)
    
    if burst_count_edges[0] > 0: burst_count_edges.insert(0, 0)
    
    logger.info(f"Computed Jenks breaks for burst activity: {burst_count_edges}")
    return burst_count_edges

def classify_job(row, rt_edges, cpu_edges, burst_count_edges):
    """
    Classifies a single job into a category based on its resource requirements.
    """
    rt_bin = np.digitize(row['RunTime'], rt_edges)
    cpu_bin = np.digitize(row['AverageCPUTimeUsed'], cpu_edges)
    burst_bin = np.digitize(row['BurstSize'], burst_count_edges,right=True)

    # Combine bins to form a category label (e.g., 'rt_bin_1_cpu_bin_2_burst_3')
    return f"RT_{rt_bin}_CPU_{cpu_bin}_BURST_{burst_bin}"

def label_and_categorize_jobs(df_clean, rt_edges, cpu_edges, burst_count_edges):
    """
    Labels each job with its category based on resource usage and burstiness.
    """
    # Add a 'BurstSize' column to each job based on its submit time interval
    interval_counts = calculate_interval_burstiness(df_clean)
    interval_map = interval_counts.set_index('IntervalID')['JobCount'].to_dict()
    
    df_clean['SubmitTime_sec'] = df_clean['SubmitTime'].astype(np.int64) // 10**9
    df_clean['IntervalID'] = df_clean['SubmitTime_sec'] // DEFAULT_BURST_THRESH_SECONDS
    
    # Map the job count back to each job
    df_clean['BurstSize'] = df_clean['IntervalID'].map(interval_map).fillna(0)
    
    # Classify each job into a category
    df_clean['Category'] = df_clean.apply(
        lambda row: classify_job(row, rt_edges, cpu_edges, burst_count_edges),
        axis=1
    )
    return df_clean.drop(columns=['SubmitTime_sec', 'IntervalID', 'BurstSize'])

def compute_user_burst_metrics(df_clean):
    """
    Computes key metrics for each user, including burstiness.
    """
    # Aggregate job count and inter-arrival time stats
    user_profiles = df_clean.groupby('UserID').agg(
        JobCount=('JobID', 'count'),
        AvgRunTime=('RunTime', 'mean'),
        AvgAllocatedProcessors=('AllocatedProcessors', 'mean'),
        Interarrival_mean=('Interarrival', 'mean'),
        Interarrival_std=('Interarrival', 'std')
    ).reset_index()

    # Calculate burst-related metrics
    df_clean['is_burst'] = df_clean['Interarrival'] < DEFAULT_BURST_THRESH_SECONDS
    burst_counts = df_clean[df_clean['is_burst']].groupby('UserID').size().reset_index(name='BurstJobCount')
    user_profiles = user_profiles.merge(burst_counts, on='UserID', how='left').fillna(0)
    user_profiles['BurstinessRatio'] = user_profiles['BurstJobCount'] / user_profiles['JobCount']
    
    # --- FIXED: Correctly calculate MaxBurstSize for each user ---
    # First, calculate BurstSize for each job
    interval_counts = calculate_interval_burstiness(df_clean)
    interval_map = interval_counts.set_index('IntervalID')['JobCount'].to_dict()
    
    # Ensure these columns exist before mapping
    if 'SubmitTime_sec' not in df_clean.columns:
        df_clean['SubmitTime_sec'] = df_clean['SubmitTime'].astype(np.int64) // 10**9
    if 'IntervalID' not in df_clean.columns:
        df_clean['IntervalID'] = df_clean['SubmitTime_sec'] // DEFAULT_BURST_THRESH_SECONDS
        
    df_clean['BurstSize'] = df_clean['IntervalID'].map(interval_map).fillna(0)

    # Then, find the maximum BurstSize for each user
    max_burst_size_per_user = df_clean.groupby('UserID')['BurstSize'].max().reset_index(name='MaxBurstSize')

    # Merge the MaxBurstSize back into the user_profiles dataframe
    user_profiles = user_profiles.merge(max_burst_size_per_user, on='UserID', how='left').fillna(0)

    # Classify burst level
    user_profiles['BurstLevel'] = pd.qcut(
        user_profiles['BurstinessRatio'],
        q=[0, 0.33, 0.66, 1], # Quantiles for Low/Mid/High
        labels=['Low', 'Mid', 'High'],
        duplicates='drop'
    )
    
    logger.info("Computed user profiles and burstiness metrics.")
    return user_profiles

def compute_mmpp_parameters(df_clean):
    """
    Computes the transition matrix and Poisson rates for the MMPP model
    based on user burst levels.
    """
    logger.info("Computing MMPP transition matrix and Poisson rates...")
    
    # 1. Compute transition matrix
    df_clean['Interarrival_cat'] = pd.cut(
        df_clean['Interarrival'],
        bins=[0, DEFAULT_BURST_THRESH_SECONDS, df_clean['Interarrival'].max()],
        labels=['Burst', 'Non_Burst'],
        right=False
    )
    
    # Create a sequence of burst states for each user
    user_sequences = defaultdict(list)
    for user_id, group in df_clean.sort_values('SubmitTime').groupby('UserID'):
        user_sequences[user_id] = group['Interarrival_cat'].values
        
    transitions = {
        'Burst_to_Burst': 0,
        'Burst_to_Non_Burst': 0,
        'Non_Burst_to_Burst': 0,
        'Non_Burst_to_Non_Burst': 0,
    }
    
    for seq in user_sequences.values():
        for i in range(len(seq) - 1):
            if seq[i] == 'Burst' and seq[i+1] == 'Burst':
                transitions['Burst_to_Burst'] += 1
            elif seq[i] == 'Burst' and seq[i+1] == 'Non_Burst':
                transitions['Burst_to_Non_Burst'] += 1
            elif seq[i] == 'Non_Burst' and seq[i+1] == 'Burst':
                transitions['Non_Burst_to_Burst'] += 1
            elif seq[i] == 'Non_Burst' and seq[i+1] == 'Non_Burst':
                transitions['Non_Burst_to_Non_Burst'] += 1
    
    # Compute probabilities
    burst_total = transitions['Burst_to_Burst'] + transitions['Burst_to_Non_Burst']
    non_burst_total = transitions['Non_Burst_to_Burst'] + transitions['Non_Burst_to_Non_Burst']

    p_bb = transitions['Burst_to_Burst'] / burst_total if burst_total > 0 else 0
    p_bn = transitions['Burst_to_Non_Burst'] / burst_total if burst_total > 0 else 0
    p_nb = transitions['Non_Burst_to_Burst'] / non_burst_total if non_burst_total > 0 else 0
    p_nn = transitions['Non_Burst_to_Non_Burst'] / non_burst_total if non_burst_total > 0 else 0

    # P_TRANSITION matrix (Low, Mid, High)
    # The MMPP model in simser10.py has 3 states, so we need to map our 2 states to 3.
    # We can use the 'Low', 'Mid', 'High' burst level categories from the user profiles.
    # The current mapping in simser10.py is hardcoded. It is better to calculate it here.
    
    transition_matrix = np.array([
        [p_bb, p_bn, 0],   # From 'Low' (Burst) to 'Low' or 'Mid'
        [p_nb, p_nn, 0],   # From 'Mid' (Non_Burst) to 'Burst' or 'Non_Burst'
        [0, 0, 1]    # From 'High' (not modeled)
    ])
    
    # 2. Compute Poisson rates
    burst_interarrivals = df_clean[df_clean['Interarrival_cat'] == 'Burst']['Interarrival'].mean()
    non_burst_interarrivals = df_clean[df_clean['Interarrival_cat'] == 'Non_Burst']['Interarrival'].mean()

    # Rate is 1 / mean
    rate_burst = 1 / burst_interarrivals if burst_interarrivals > 0 else 1
    rate_non_burst = 1 / non_burst_interarrivals if non_burst_interarrivals > 0 else 1
    
    poisson_rates = {
        'Low': rate_burst,
        'Mid': rate_non_burst,
        'High': rate_non_burst # Use a fallback for high burst
    }
    
    logger.info(f"Transition Matrix:\n{transition_matrix}")
    logger.info(f"Poisson Rates: {poisson_rates}")
    
    return transition_matrix.tolist(), poisson_rates

def export_subsets_to_excel(df_clean, subsets_dir="subsets"):
    """
    Categorizes jobs and exports them to an Excel file, creating subsets for VAE training.
    """
    os.makedirs(subsets_dir, exist_ok=True)
    
    # 1. Compute Jenks breaks for resource bins
    rt_edges, cpu_edges = compute_bin_edges(df_clean)
    
    # 2. Compute Jenks breaks for burst activity bins
    burst_count_edges = compute_burst_activity_edges(df_clean)
    
    # 3. Label and categorize jobs
    categorized_df = label_and_categorize_jobs(
        df_clean,
        rt_edges=rt_edges,
        cpu_edges=cpu_edges,
        burst_count_edges=burst_count_edges
    )
    category_counts = categorized_df['Category'].value_counts()
    threshold = 500
    sparse_categories = category_counts[category_counts < threshold].index
    valid_categories = category_counts[category_counts >= threshold].index

    def parse_cat(cat):
        # Example: RT_2_CPU_3_BURST_1
        parts = cat.split('_')
        return int(parts[1]), int(parts[3]), int(parts[5])

    def find_nearest_valid(cat, valid_cats):
        rt, cpu, burst = parse_cat(cat)
        min_dist = float('inf')
        best_cat = cat
        for vcat in valid_cats:
            vrt, vcpu, vburst = parse_cat(vcat)
            dist = abs(rt - vrt) + abs(cpu - vcpu) + abs(burst - vburst)
            if dist < min_dist:
                min_dist = dist
                best_cat = vcat
        return best_cat

    # Build reassignment map
    reassignment_map = {
        sparse: find_nearest_valid(sparse, valid_categories)
        for sparse in sparse_categories
    }

    # Apply reassignment
    categorized_df['Category'] = categorized_df['Category'].apply(
        lambda c: reassignment_map.get(c, c)
    )

    # Re-count after reassignment
    final_counts = categorized_df['Category'].value_counts()
    final_valid_cats = final_counts[final_counts >= threshold].index
    filtered_df = categorized_df[categorized_df['Category'].isin(final_valid_cats)]

    # 4. Export subsets to Excel
    writer = pd.ExcelWriter(os.path.join(subsets_dir, "categorized_subsets.xlsx"))
    for category in filtered_df['Category'].unique():
        subset_df = filtered_df[filtered_df['Category'] == category]
        subset_df.to_excel(writer, sheet_name=category, index=False)
        logger.info(f"Exported {len(subset_df)} jobs for category '{category}'")
    writer.close()
    
    logger.info(f"Categorized subsets exported to '{subsets_dir}/categorized_subsets.xlsx'")
    return filtered_df

def main():
    """
    Main function to orchestrate the parsing, cleaning, and categorization.
    """
    # 1. Parse the log and remove anomalies
    df = parse_sdsc_sp2_log(SWF_PATH)
    anomalies, clean_data = detect_and_remove_anomalies(df)
    
    if clean_data.empty:
        logger.error("No clean data available after anomaly removal. Exiting.")
        return
        
    # 2. Compute user profiles and save them for the generator
    user_profiles_df = compute_user_burst_metrics(clean_data)
    user_profiles_df.to_csv("user_profiles.csv", index=False)
    logger.info("User profiles exported to 'user_profiles.csv'")
    
    # 3. Compute MMPP parameters and save them to a JSON file
    transition_matrix, poisson_rates = compute_mmpp_parameters(clean_data)
    mmpp_params = {
        'P_TRANSITION': transition_matrix,
        'POISSON_RATES': poisson_rates
    }
    with open('mmpp_config.json', 'w') as f:
        json.dump(mmpp_params, f, indent=4)
    logger.info("MMPP parameters exported to 'mmpp_config.json'")

    # 4. Export categorized subsets for VAE training
    export_subsets_to_excel(clean_data)
    # ─── Export the time→category distribution ─────────────────────────────────
    from collections import defaultdict
    granularity_minutes=10
    # 1) tag each job with its weekday and bucketed time
    bucket_freq = f"{granularity_minutes}T"   # e.g. "10T" for 10‑minute
    clean_data['Day'] = clean_data['SubmitTime'].dt.day_name()
    clean_data['TimeBucket'] = (
        clean_data['SubmitTime']
        .dt.floor(bucket_freq)
        .dt.strftime('%H:%M')
    )

    # 2) group & count
    grp = (
        clean_data
        .groupby(['Day', 'TimeBucket', 'Category'])
        .size()
        .reset_index(name='Count')
    )

    # 3) assemble nested dict
    category_distribution = defaultdict(lambda: defaultdict(dict))
    for _, row in grp.iterrows():
        day = row['Day']
        tb  = row['TimeBucket']
        cat = row['Category']
        cnt = int(row['Count'])
        category_distribution[day][tb][cat] = cnt

    # ─── Export the time→category distribution ─────────────────────────────────
    from collections import defaultdict, Counter

# ─── Export the time→category distribution (with per-slot user counts) ───
    TIME_DIST_FILE = "time_category_distribution.json"
    granularity_minutes = 10
    bucket_freq = f"{granularity_minutes}T"  # e.g., "10T" for 10-minute intervals

    # 1️⃣ Add Day and TimeBucket columns
    clean_data["Day"] = clean_data["SubmitTime"].dt.day_name()
    clean_data["TimeBucket"] = clean_data["SubmitTime"].dt.floor(bucket_freq).dt.strftime('%H:%M')

    # 2️⃣ Build nested structure with category_counts and user_counts
    category_distribution = defaultdict(lambda: defaultdict(lambda: {
        "category_counts": Counter(),
        "user_counts": defaultdict(Counter)
    }))

    for _, row in clean_data.iterrows():
        day = row["Day"]
        time_bucket = row["TimeBucket"]
        category = row["Category"]
        user_id = row["UserID"]

        slot = category_distribution[day][time_bucket]
        slot["category_counts"][category] += 1
        slot["user_counts"][category][user_id] += 1

    # 3️⃣ Convert all Counters to plain dicts for JSON serialization
    def convert_nested_counters(obj):
        if isinstance(obj, Counter):
            return dict(obj)
        elif isinstance(obj, defaultdict):
            return {k: convert_nested_counters(v) for k, v in obj.items()}
        elif isinstance(obj, dict):
            return {k: convert_nested_counters(v) for k, v in obj.items()}
        return obj

    category_distribution = convert_nested_counters(category_distribution)

    # 4️⃣ Final output JSON
    out = {
        "granularity_minutes": granularity_minutes,
        "category_distribution": category_distribution
    }

    with open(TIME_DIST_FILE, "w") as f:
        json.dump(out, f, indent=4)

    logger.info(f"Time–category distribution (with per-slot user counts) exported to '{TIME_DIST_FILE}'")


    
if __name__ == '__main__':
    main()