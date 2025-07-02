import pandas as pd
import numpy as np
import os
import logging
import jenkspy
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from scipy.stats import gamma
import json
from collections import defaultdict
from sklearn.ensemble import IsolationForest
import sys

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
    SEED,
    ANOMALY_PCT # Import ANOMALY_PCT directly from config
)

# ─── LOGGING CONFIG ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SWF_Categorizer')

# ─── PARSING AND PREPROCESSING ──────────────────────────────────────────
def parse_sdsc_sp2_log(path):
    """
    Parses an SWF log file into a pandas DataFrame.
    Filters out invalid records and renames columns for clarity.
    """
    column_names = [
        "JobId", "SubmitTime", "WaitTime", "RunTime", "AllocatedProcessors",
        "UsedProcessorTime", "UsedMemory", "RequestedProcessors", "RequestedTime",
        "RequestedMemory", "Status", "UserId", "GroupId", "ExecutableNumber",
        "QueueNumber", "PartitionNumber", "PrecedingJobNumber", "ThinkTime"
    ]
    
    # Read the file, skipping comments and setting column names
    # SWF files often have comments starting with ';'
    df = pd.read_csv(path, sep=r'\s+', comment=';', header=None, names=column_names, on_bad_lines='skip')
    
    # Convert relevant columns to numeric, coercing errors to NaN
    numeric_cols = [
        "SubmitTime", "WaitTime", "RunTime", "AllocatedProcessors",
        "UsedProcessorTime", "UsedMemory", "RequestedProcessors", "RequestedTime",
        "RequestedMemory", "Status", "UserId", "GroupId", "ExecutableNumber",
        "QueueNumber", "PartitionNumber", "PrecedingJobNumber", "ThinkTime"
    ]
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
        
    # Drop rows where essential numeric conversions failed
    df.dropna(subset=["SubmitTime", "RunTime", "AllocatedProcessors"], inplace=True)
    
    # Calculate AverageCPUTimeUsed
    # Avoid division by zero: if AllocatedProcessors is 0, AverageCPUTimeUsed should be 0 or NaN
    df['AverageCPUTimeUsed'] = df.apply(
        lambda row: row['UsedProcessorTime'] / row['AllocatedProcessors'] if row['AllocatedProcessors'] > 0 else 0,
        axis=1
    )
    
    # Filter out records with non-positive RunTime or AllocatedProcessors after calculation
    df = df[(df['RunTime'] > 0) & (df['AllocatedProcessors'] > 0)].copy()

    logger.info(f"Parsed SWF file with {len(df)} records.")
    return df

def detect_and_remove_anomalies(df):
    """
    Detects and removes anomalies using Isolation Forest based on 'RunTime', 'AllocatedProcessors',
    and 'AverageCPUTimeUsed'.
    
    Args:
        df (pd.DataFrame): Input DataFrame of job data.
                                              
    Returns:
        tuple: A tuple containing (cleaned_df, anomaly_df)
    """
    features_for_isolation = ['RunTime', 'AllocatedProcessors', 'AverageCPUTimeUsed', 'UsedMemory']
    
    # Ensure features exist and handle NaNs if any
    isolation_data = df[features_for_isolation].copy()
    
    # Fill any NaNs in these features before feeding to IsolationForest
    # Using median for robustness against outliers
    for col in features_for_isolation:
        if isolation_data[col].isnull().any():
            median_val = isolation_data[col].median()
            isolation_data[col].fillna(median_val, inplace=True)
            logger.info(f"Filled NaN in {col} with median: {median_val}")

    # Use ANOMALY_PCT from config as contamination
    anomaly_threshold_percentage = ANOMALY_PCT # Retrieve from imported config
    
    if not (0 < anomaly_threshold_percentage <= 0.5):
        logger.warning(f"Invalid ANOMALY_PCT from config: {anomaly_threshold_percentage}. Clamping to 0.01.")
        anomaly_threshold_percentage = 0.01 # Default to a small value if out of range

    iso_forest = IsolationForest(
        random_state=SEED,
        contamination=anomaly_threshold_percentage,
        n_estimators=100,  # Number of base estimators in the ensemble
        max_features=1.0,  # Number of features to draw from X to train each base estimator
        max_samples='auto', # The number of samples to draw from X to train each base estimator
        bootstrap=True,    # If True, individual trees are drawn with replacement.
        n_jobs=-1          # Use all available CPU cores
    )
    
    # Fit and predict anomalies (-1 for outliers, 1 for inliers)
    df['anomaly_score'] = iso_forest.fit_predict(isolation_data)
    
    cleaned_df = df[df['anomaly_score'] == 1].drop(columns=['anomaly_score']).copy()
    anomaly_df = df[df['anomaly_score'] == -1].drop(columns=['anomaly_score']).copy()
    
    logger.info(f"Anomaly detection complete. Original: {len(df)} records. Cleaned: {len(cleaned_df)}, Anomalies: {len(anomaly_df)}")
    
    # Corrected return order to match train_all_vae2.py expectation
    return cleaned_df, anomaly_df

def compute_bin_edges(series, num_bins=3):
    """
    Computes bin edges using Jenks Natural Breaks Optimization.
    Ensures unique edges and handles small data.
    """
    if series.nunique() <= num_bins:
        # Use quantiles or simple linspace if Jenks fails
        if series.nunique() == 1: # All values are the same
            min_val = series.min()
            max_val = series.max()
            return [min_val, max_val + 1e-6] # A simple range for constant data
        else:
            # Use quantiles for non-constant but few unique values
            return list(series.quantile(np.linspace(0, 1, num_bins + 1)).unique())

    try:
        # CORRECTED: Use series.tolist() directly
        breaks = jenkspy.jenks_breaks(series.tolist(), n_classes=num_bins)
        # Ensure the breaks cover the min and max of the series
        if breaks[0] > series.min():
            breaks.insert(0, series.min())
        if breaks[-1] < series.max():
            breaks.append(series.max())
        
        # Remove duplicates which can happen if breaks are very close
        return sorted(list(dict.fromkeys(breaks)))
    except ValueError as e:
        logger.warning(f"Jenks breaks failed for series (unique values: {series.nunique()}, num_bins: {num_bins}). Falling back to quantiles. Error: {e}")
        # Fallback to quantile-based binning if Jenks fails (e.g., due to identical values)
        return list(series.quantile(np.linspace(0, 1, num_bins + 1)).unique())

def compute_burst_activity_edges(submit_times, run_times, job_ids, burst_threshold_seconds=DEFAULT_BURST_THRESH_SECONDS, num_bins=3):
    """
    Calculates burst activity based on job inter-arrival times and uses Jenks
    natural breaks to categorize burst levels.
    """
    df_temp = pd.DataFrame({
        'SubmitTime': submit_times,
        'RunTime': run_times,
        'JobId': job_ids
    }).sort_values(by='SubmitTime').reset_index(drop=True)
    
    # Calculate inter-arrival times (IAT)
    df_temp['IAT'] = df_temp['SubmitTime'].diff().fillna(0)
    
    # Define burst activity based on IAT relative to RunTime
    # Simplified burst metric: a burst occurs if IAT is less than a small threshold (e.g., 60 seconds)
    # A more sophisticated metric could involve the ratio of IAT to RunTime or other factors.
    df_temp['BurstActivity'] = (df_temp['IAT'] < burst_threshold_seconds).astype(int) # 1 for burst, 0 otherwise

    # We need to categorize the overall 'busyness' or 'inter-arrival pattern'
    # Let's consider inverse IAT (submission rate) as a metric to categorize
    # To avoid division by zero for IAT=0, add a small epsilon
    df_temp['SubmissionRate'] = 1 / (df_temp['IAT'] + 1e-6) # Jobs per second (approx)
    
    # Use Jenks on SubmissionRate to find natural breaks for burst levels
    # Exclude the first job's IAT=0 for this calculation as it's not a true inter-arrival
    submission_rates_for_binning = df_temp['SubmissionRate'][df_temp['IAT'] > 0]

    if submission_rates_for_binning.empty:
        logger.warning("No inter-arrival times > 0 for burst activity binning. Using default edges.")
        return [0, 1e-6, 1e-5, 1e-4] # Placeholder/default if no activity

    edges = compute_bin_edges(submission_rates_for_binning, num_bins=num_bins)
    
    # Adjust edges if they don't encompass all data after calculation
    min_rate = submission_rates_for_binning.min()
    max_rate = submission_rates_for_binning.max()
    if edges[0] > min_rate:
        edges.insert(0, min_rate)
    if edges[-1] < max_rate:
        edges.append(max_rate)
    
    # Ensure unique and sorted edges
    edges = sorted(list(dict.fromkeys(edges)))

    logger.info(f"Computed burst activity edges: {edges}")
    return edges


def label_and_categorize_jobs(df, rt_edges, cpu_edges, burst_edges):
    """
    Labels jobs based on RunTime, CPU utilization, and Burst Activity bins.
    
    Args:
        df (pd.DataFrame): The DataFrame with 'RunTime', 'AverageCPUTimeUsed', 'SubmitTime', 'JobId'.
        rt_edges (list): Bin edges for RunTime.
        cpu_edges (list): Bin edges for AverageCPUTimeUsed.
        burst_edges (list): Bin edges for BurstActivity (or SubmissionRate).
        
    Returns:
        pd.DataFrame: The DataFrame with 'RunTime_Bin', 'CPU_Bin', 'Burst_Bin', and 'Category' columns.
    """
    df_categorized = df.copy()

    # Apply RunTime binning
    df_categorized['RunTime_Bin'] = pd.cut(
        df_categorized['RunTime'], bins=rt_edges, labels=False, include_lowest=True, right=True
    ).fillna(-1).astype(int) # Fillna with -1 for values outside bins, then cast to int
    df_categorized['RunTime_Bin'] = df_categorized['RunTime_Bin'].replace(-1, 0) # Map -1 to 0 or handle as an "unknown" category

    # Apply CPU utilization binning
    df_categorized['CPU_Bin'] = pd.cut(
        df_categorized['AverageCPUTimeUsed'], bins=cpu_edges, labels=False, include_lowest=True, right=True
    ).fillna(-1).astype(int)
    df_categorized['CPU_Bin'] = df_categorized['CPU_Bin'].replace(-1, 0) # Map -1 to 0

    # Calculate 'SubmissionRate' on the categorized df for burst binning
    df_categorized = df_categorized.sort_values(by='SubmitTime').reset_index(drop=True)
    df_categorized['IAT'] = df_categorized['SubmitTime'].diff().fillna(0)
    df_categorized['SubmissionRate'] = 1 / (df_categorized['IAT'] + 1e-6) # Add epsilon to avoid division by zero

    # Apply Burst Activity binning
    df_categorized['Burst_Bin'] = pd.cut(
        df_categorized['SubmissionRate'], bins=burst_edges, labels=False, include_lowest=True, right=True
    ).fillna(-1).astype(int)
    df_categorized['Burst_Bin'] = df_categorized['Burst_Bin'].replace(-1, 0) # Map -1 to 0
    
    # Create the combined category string
    df_categorized['Category'] = df_categorized.apply(
        lambda row: f"RT_{row['RunTime_Bin']}_CPU_{row['CPU_Bin']}_BURST_{row['Burst_Bin']}",
        axis=1
    )
    
    # Export categorized data into a single Excel file with separate sheets for each category
    subsets_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "subsets")
    os.makedirs(subsets_dir, exist_ok=True)
    output_excel_path = os.path.join(subsets_dir, "categorized_subsets.xlsx")
    
    # Filter out categories that are too small before exporting to avoid empty sheets
    min_export_size = 5 # A small threshold for export, can be adjusted
    
    # Get all unique categories present in the DataFrame after categorization
    all_categories_present = df_categorized['Category'].unique()
    
    # Filter for categories that meet the minimum size requirement for export
    categories_to_export = [
        cat for cat in all_categories_present
        if len(df_categorized[df_categorized['Category'] == cat]) >= min_export_size
    ]

    if not categories_to_export:
        logger.warning("No categories meet the minimum export size. No categorized subsets will be exported.")
        # Return the full df_categorized even if not exported, for further processing
        return df_categorized

    # Create a Pandas Excel writer using XlsxWriter as the engine.
    with pd.ExcelWriter(output_excel_path, engine='xlsxwriter') as writer:
        # Write the full categorized DataFrame to a sheet named 'categorized_jobs'
        df_categorized.to_excel(writer, sheet_name='categorized_jobs', index=False)
        logger.info(f"Full categorized data exported to 'categorized_jobs' sheet.")

        # Optionally, write each subset to its own sheet
        filtered_df = df_categorized[df_categorized['Category'].isin(categories_to_export)].copy()
        
        # Group by the 'Category' column and iterate
        for category, subset_df in filtered_df.groupby('Category'):
            # Write each DataFrame to a specific sheet
            subset_df.to_excel(writer, sheet_name=category, index=False)
            logger.info(f"Exported {len(subset_df)} jobs for category '{category}'")
    writer.close() # Ensure the writer is closed if not using 'with' statement

    logger.info(f"Categorized subsets exported to '{output_excel_path}'")
    
    return filtered_df # Return the filtered DataFrame for consistency with previous calls

def main():
    """
    Main function to orchestrate the parsing, cleaning, and categorization.
    """
    # 1. Parse the log and remove anomalies
    df = parse_sdsc_sp2_log(SWF_PATH)
    
    # Corrected call to detect_and_remove_anomalies - no explicit ANOMALY_PCT argument
    clean_data, anomaly_df = detect_and_remove_anomalies(df) 
    
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
    with open('mmpp_parameters.json', 'w') as f:
        json.dump(mmpp_params, f, indent=4)
    logger.info("MMPP parameters exported to 'mmpp_parameters.json'")

    # 4. Categorization
    logger.info("Starting job categorization...")
    rt_edges = compute_bin_edges(clean_data['RunTime'], num_bins=DEFAULT_NUM_BINS)
    cpu_edges = compute_bin_edges(clean_data['AverageCPUTimeUsed'], num_bins=DEFAULT_NUM_BINS)
    burst_edges = compute_burst_activity_edges(
        clean_data['SubmitTime'], 
        clean_data['RunTime'], 
        clean_data['JobId'],
        burst_threshold_seconds=DEFAULT_BURST_THRESH_SECONDS,
        num_bins=DEFAULT_NUM_BINS
    )
    
    df_categorized = label_and_categorize_jobs(clean_data, rt_edges, cpu_edges, burst_edges)
    logger.info("Job categorization complete.")

# Define compute_user_burst_metrics and compute_mmpp_parameters if they exist elsewhere or define as stubs
# If these functions are meant to be imported, ensure their source file is accessible.
# For now, adding stubs to prevent NameError if they are not defined in this file.
def compute_user_burst_metrics(df):
    """Placeholder for user burst metrics computation."""
    logger.warning("`compute_user_burst_metrics` function is a stub. Implement its logic.")
    # Example: group by user and calculate some stats
    return df.groupby('UserId').agg(
        total_runtime=('RunTime', 'sum'),
        avg_runtime=('RunTime', 'mean')
    ).reset_index()

def compute_mmpp_parameters(df):
    """Placeholder for MMPP parameter computation."""
    logger.warning("`compute_mmpp_parameters` function is a stub. Implement its logic.")
    # Example: simple dummy return
    # This would involve more complex statistical modeling
    num_states = len(POISSON_RATES) # Assuming POISSON_RATES is imported from config
    p_transition_dummy = [[1.0/num_states for _ in range(num_states)] for _ in range(num_states)]
    poisson_rates_dummy = list(POISSON_RATES.values()) # Directly use values from config.POISSON_RATES


    return p_transition_dummy, poisson_rates_dummy


if __name__ == "__main__":
    main()