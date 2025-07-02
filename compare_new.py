import os
import glob
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import json 

# --- Import from provided files ---
# Configuration values are loaded from config.py
from config import WEIGHTS_DIR, HIDDEN_DIMS, EXCEL_SUBSET_PATH, DEFAULT_LATENT_DIM

# Attempt to import MMPP specific configurations from config.py
# If not found, default mock values are used.
try:
    from config import BURST_LEVEL_MAP, POISSON_RATES, P_TRANSITION
except ImportError:
    print("MMPP specific configurations (POISSON_RATES, P_TRANSITION, BURST_LEVEL_MAP) not fully found in config.py. Using mock values.")
    POISSON_RATES = {'Low': 0.05, 'Mid': 0.5, 'High': 2.0}
    P_TRANSITION = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]
    # This specific BURST_LEVEL_MAP ensures correct 0-indexed state mapping for P_TRANSITION
    BURST_LEVEL_MAP = {'Low': 0, 'Mid': 1, 'High': 2} 

# Import VAE model and data preprocessing function
from vae_training2 import VAE, load_and_preprocess

# --- Helper function for MMPP (corrected) ---
# Internal mappings to ensure correct 0-indexed state access for P_TRANSITION
_BURST_LEVEL_TO_INDEX = {'Low': 0, 'Mid': 1, 'High': 2}
_INDEX_TO_BURST_LEVEL = {v: k for k, v in _BURST_LEVEL_TO_INDEX.items()}

def generate_mmpp_interarrival_time(current_burst_level_str, num_jobs_to_generate):
    """
    Generates a sequence of inter-arrival times using an MMPP (Markov Modulated Poisson Process) model.
    
    Args:
        current_burst_level_str (str): The initial burst level ('Low', 'Mid', or 'High').
        num_jobs_to_generate (int): The number of inter-arrival times to generate.
        
    Returns:
        list: A list of generated inter-arrival times.
    """
    interarrivals = []
    
    # Get the initial numerical state index from the string label. Defaults to 'Mid' (index 1) if not found.
    current_state_idx = _BURST_LEVEL_TO_INDEX.get(current_burst_level_str, 1) 
    
    for _ in range(num_jobs_to_generate):
        # Retrieve the Poisson rate for the current state.
        state_label = _INDEX_TO_BURST_LEVEL.get(current_state_idx, 'Mid') 
        current_rate = POISSON_RATES[state_label] 

        if current_rate > 0:
            interarrival_time = np.random.exponential(scale=1/current_rate)
        else:
            interarrival_time = 0.0 
        
        interarrivals.append(max(0.0, interarrival_time))
        
        # Transition to the next state based on the P_TRANSITION matrix
        transition_probs = P_TRANSITION[current_state_idx]
        current_state_idx = np.random.choice(len(transition_probs), p=transition_probs)

    return interarrivals

# --- VAE Model Loading ---
def load_all_vaes_for_comparison():
    """
    Loads all trained VAE models, their scalers, and feature names from the designated weights directory.
    
    Returns:
        dict: A dictionary where keys are category names and values are dictionaries
              containing the 'model', 'scaler', 'latent_dim', and 'features' for each VAE.
    """
    print("Loading VAE models and scalers for comparison...")
    models = {}
    for ckpt_path in glob.glob(os.path.join(WEIGHTS_DIR, "*_vae_ckpt.pth")):
        category = os.path.basename(ckpt_path).replace("_vae_ckpt.pth", "")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            state = ckpt["model_state"]
            scaler = ckpt["scaler"]
            feats = ckpt["features"]
            latent_d = ckpt["latent_dim"] 
            
            model = VAE(input_dim=len(feats), hidden_dims=HIDDEN_DIMS, latent_dim=latent_d)
            model.load_state_dict(state)
            model.eval()

            model.features = feats # Attach features to the model for convenience
            
            models[category] = {
                "model": model,
                "scaler": scaler,
                "latent_dim": latent_d,
                "features": feats
            }
            print(f"Loaded VAE for category '{category}'.")
        except Exception as e:
            print(f"Failed to load VAE for category '{category}': {e}")
            
    return models

# --- Data Generation ---
def generate_synthetic_data(model_data, num_samples, burst_level='Mid'):
    """
    Generates synthetic job data for a given VAE model.
    
    Args:
        model_data (dict): Dictionary containing the VAE 'model', 'scaler', 'latent_dim', and 'features'.
        num_samples (int): The number of synthetic samples to generate.
        burst_level (str): The burst level for MMPP inter-arrival time generation (default 'Mid').
        
    Returns:
        pd.DataFrame: A DataFrame containing the generated synthetic data.
    """
    model = model_data['model']
    scaler = model_data['scaler']
    latent_d = model_data['latent_dim']
    features = model_data['features'] 

    generated_data = []
    
    interarrival_sequence = generate_mmpp_interarrival_time(burst_level, num_samples)
    interarrival_iterator = iter(interarrival_sequence)

    batch_size = 256
    num_batches = (num_samples + batch_size - 1) // batch_size

    for _ in range(num_batches):
        current_batch_size = min(batch_size, num_samples - len(generated_data))
        if current_batch_size <= 0:
            break

        z = torch.randn(current_batch_size, latent_d)
        
        with torch.no_grad():
            model.eval()
            outp = model.decode(z)
            scaled_output_numpy = outp.numpy()

        # Inverse transform to bring data back to original scale
        real_vals = scaler.inverse_transform(scaled_output_numpy)
        
        for vals in real_vals:
            job_dict = {}
            for i, feature_name in enumerate(features):
                job_dict[feature_name] = vals[i]
            
            # Add interarrival time, ensuring we don't run out
            try:
                job_dict["interarrival"] = next(interarrival_iterator)
            except StopIteration:
                job_dict["interarrival"] = 0.0 

            # Basic post-processing to ensure valid non-negative integer/float values
            job_dict["RunTime"] = max(0.0, job_dict.get("RunTime", 0.0))
            job_dict["AllocatedProcessors"] = max(1, int(round(job_dict.get("AllocatedProcessors", 1))))
            job_dict["AverageCPUTimeUsed"] = max(0.0, job_dict.get("AverageCPUTimeUsed", 0.0))
            job_dict["UsedMemory"] = max(0.0, job_dict.get("UsedMemory", 0.0))
            job_dict["WaitTime"] = max(0.0, job_dict.get("WaitTime", 0.0)) 
            job_dict["SubmitTime"] = max(0.0, job_dict.get("SubmitTime", 0.0)) 

            generated_data.append(job_dict)

    return pd.DataFrame(generated_data)

# --- Numerical Comparison ---
def numerical_comparison(original_df, generated_df, category, features):
    """
    Performs numerical comparison of statistical properties (mean, std, min, max, quantiles, KS test, correlation).
    
    Args:
        original_df (pd.DataFrame): DataFrame containing the original data.
        generated_df (pd.DataFrame): DataFrame containing the generated data.
        category (str): The name of the category being compared.
        features (list): A list of feature names to compare.
    """
    print(f"\n--- Numerical Comparison for Category: {category} ---")
    
    metrics = ['mean', 'std', 'min', 'max', '25%', '50%', '75%']
    original_desc = original_df[features].describe().loc[metrics]
    generated_desc = generated_df[features].describe().loc[metrics]

    print("\nOriginal Data Statistics:")
    print(original_desc)
    print("\nGenerated Data Statistics:")
    print(generated_desc)

    print("\n--- Feature-wise Comparison (Mean & Std Dev) ---")
    comparison_df = pd.DataFrame(index=features, columns=['Original Mean', 'Generated Mean', 'Mean Diff (%)', 'Original Std', 'Generated Std', 'Std Diff (%)'])
    for feature in features:
        orig_mean = original_desc.loc['mean', feature]
        gen_mean = generated_desc.loc['mean', feature]
        orig_std = original_desc.loc['std', feature]
        gen_std = generated_desc.loc['std', feature]

        mean_diff_pct = (abs(gen_mean - orig_mean) / orig_mean) * 100 if orig_mean != 0 else np.nan
        std_diff_pct = (abs(gen_std - orig_std) / orig_std) * 100 if orig_std != 0 else np.nan

        comparison_df.loc[feature] = [orig_mean, gen_mean, f"{mean_diff_pct:.2f}%", orig_std, gen_std, f"{std_diff_pct:.2f}%"]
    print(comparison_df)

    print("\n--- Kolmogorov-Smirnov (KS) Test (p-value for each feature) ---")
    print("Null Hypothesis (H0): The two samples are drawn from the same continuous distribution.")
    print("A high p-value (e.g., > 0.05) suggests we cannot reject H0, meaning distributions are similar.")
    ks_results = {}
    for feature in features:
        # Filter out NaN values before performing KS test
        orig_vals = original_df[feature].dropna().values
        gen_vals = generated_df[feature].dropna().values
        if len(orig_vals) > 0 and len(gen_vals) > 0:
            statistic, p_value = stats.ks_2samp(orig_vals, gen_vals)
            ks_results[feature] = {'Statistic': f"{statistic:.4f}", 'P-value': f"{p_value:.4f}"}
        else:
            ks_results[feature] = {'Statistic': 'N/A', 'P-value': 'N/A (Insufficient data)'}
    print(pd.DataFrame.from_dict(ks_results, orient='index'))

    print("\n--- Correlation Matrix Comparison ---")
    print("Original Data Correlation:")
    print(original_df[features].corr())
    print("\nGenerated Data Correlation:")
    print(generated_df[features].corr())


# --- Visual Comparison ---
def visual_comparison(original_df, generated_df, category, features):
    """
    Generates plots for visual comparison (histograms and pair plots).
    
    Args:
        original_df (pd.DataFrame): DataFrame containing the original data.
        generated_df (pd.DataFrame): DataFrame containing the generated data.
        category (str): The name of the category being compared.
        features (list): A list of feature names to visualize.
    """
    print(f"\n--- Visual Comparison for Category: {category} ---")

    # Histograms for each feature
    print("Generating Histograms...")
    fig_hist, axes_hist = plt.subplots(nrows=len(features), ncols=1, figsize=(8, 4 * len(features)))
    fig_hist.suptitle(f'Feature Histograms for Category: {category}', fontsize=16)
    if len(features) == 1: 
        axes_hist = [axes_hist] # Ensure axes_hist is iterable even for a single feature
    
    for i, feature in enumerate(features):
        ax = axes_hist[i]
        sns.histplot(original_df[feature], color='blue', label='Original', kde=True, stat='density', alpha=0.5, ax=ax, bins=50)
        sns.histplot(generated_df[feature], color='red', label='Generated', kde=True, stat='density', alpha=0.5, ax=ax, bins=50)
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f'comparison_plots/{category}_histograms.png')
    plt.close(fig_hist)
    print(f"  - Histograms saved to comparison_plots/{category}_histograms.png")

    # Pair Plots (limited to a subset of features for readability and performance)
    if len(features) > 5:
        print(f"Skipping pair plots for {category} due to many features ({len(features)}). Showing first 5.")
        pair_features = features[:5]
    else:
        pair_features = features

    print(f"Generating Pair Plots for features: {pair_features}...")
    combined_df = pd.concat([
        original_df[pair_features].assign(DataSource='Original'),
        generated_df[pair_features].assign(DataSource='Generated')
    ])
    
    g = sns.pairplot(combined_df, hue='DataSource', diag_kind='kde', plot_kws={'alpha':0.6})
    g.fig.suptitle(f'Pair Plots for Category: {category}', y=1.02, fontsize=16)
    plt.savefig(f'comparison_plots/{category}_pairplots.png')
    plt.close(g.fig)
    print(f"  - Pair plots saved to comparison_plots/{category}_pairplots.png")


# --- Main Comparison Script Execution ---
def run_comparison():
    """
    Orchestrates the comparison process: loading models and data, performing numerical
    and visual comparisons for each VAE model category.
    """
    os.makedirs('comparison_plots', exist_ok=True) # Create directory for plots

    # 1. Load all VAE models
    vae_models_data = load_all_vaes_for_comparison()
    if not vae_models_data:
        print("No VAE models loaded. Exiting comparison.")
        return

    # Process each category/model
    for category_name, model_info in vae_models_data.items():
        print(f"\n===== Comparing for Category: '{category_name}' =====")
        
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features']

        # 2. Load original data for this category
        try:
            # The sheet name in the Excel file is derived by removing 'category_' prefix.
            sheet_name_in_excel = category_name.replace("category_", "") 
            original_df_full = pd.read_excel(EXCEL_SUBSET_PATH, sheet_name=sheet_name_in_excel)
            original_df = original_df_full[features].copy()
            # Convert to numeric and fill any NaNs with 0, consistent with training preprocessing
            for col in original_df.columns:
                original_df[col] = pd.to_numeric(original_df[col], errors='coerce').fillna(0)
            
            print(f"Loaded original data for '{category_name}': {len(original_df)} samples.")
            
        except Exception as e:
            print(f"Could not load original data for category '{category_name}' from '{EXCEL_SUBSET_PATH}' (sheet: '{sheet_name_in_excel}'): {e}. Skipping.")
            continue
        
        if original_df.empty:
            print(f"Original data for category '{category_name}' is empty. Skipping comparison.")
            continue

        # 3. Generate synthetic data
        num_samples_to_generate = len(original_df) # Generate same number of samples as original
        generated_df = generate_synthetic_data(model_info, num_samples_to_generate)
        print(f"Generated {len(generated_df)} synthetic samples for '{category_name}'.")

        if generated_df.empty:
            print(f"Generated data for category '{category_name}' is empty. Skipping comparison.")
            continue

        # Ensure both dataframes have the same columns and order for accurate comparison
        original_df = original_df[features]
        generated_df = generated_df[features]

        # 4. Perform Numerical Comparison
        numerical_comparison(original_df, generated_df, category_name, features)

        # 5. Perform Visual Comparison
        visual_comparison(original_df, generated_df, category_name, features)
        
    print("\nComparison complete. Check 'comparison_plots' directory for visualizations.")

# Entry point for the script
if __name__ == '__main__':
    run_comparison()