import os
import glob
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler, QuantileTransformer
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import numpy as np
import json
import torch.serialization # Import torch.serialization

# Import configurations
try:
    from config import (
        WEIGHTS_DIR, DEFAULT_LATENT_DIM, SEQUENCE_LENGTH, EXCEL_SUBSET_PATH, SEED,
        POISSON_RATES, P_TRANSITION # MMPP specific configurations
    )
except ImportError:
    print("Error: config.py not found or incomplete. Exiting.")
    exit()

# Import VAE model and utility functions from vae_training3.py
try:
    from vae_training3 import (
        VAE,
        set_seed,
        load_and_preprocess, # Used for loading data for comparison
    )
except ImportError:
    print("Error: vae_training3.py not found. Ensure it's in the correct path.")
    exit()

# Add StandardScaler, QuantileTransformer, numpy._core.multiarray._reconstruct, numpy.ndarray, numpy.dtype, and numpy.dtypes.ObjectDType to safe globals for torch.load
torch.serialization.add_safe_globals([StandardScaler, QuantileTransformer, np._core.multiarray._reconstruct, np.ndarray, np.dtype, np.dtypes.ObjectDType, np.dtypes.Float64DType])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED) # Ensure reproducibility

# ─── SYNTHETIC DATA GENERATION ──────────────────────────────────────────
def generate_synthetic_data(model_info, num_sequences, sequence_length):
    """
    Generates synthetic job sequences using a trained VAE model.
    Applies inverse transformation and post-processing (clamping, rounding).
    """
    model = model_info['model']
    quantile_scaler, standard_scaler = model_info['scaler']
    features = model_info['features']
    original_min_max = model_info['original_min_max'] # Use this for final clamping

    model.eval()
    synthetic_sequences = []
    
    with torch.no_grad():
        for _ in range(num_sequences):
            # Generate a random latent vector
            z = torch.randn(1, model.latent_dim).to(device)
            # Decode to generate a sequence
            decoded_sequence = model.decode(z, sequence_length).cpu().numpy()
            synthetic_sequences.append(decoded_sequence.squeeze(0)) # Remove batch dimension

    synthetic_data_scaled = np.vstack(synthetic_sequences) # Stack all sequences into a flat array

    # Apply inverse transformations in correct order
    synthetic_data_quantile_scale = standard_scaler.inverse_transform(synthetic_data_scaled)
    synthetic_data_original_scale = quantile_scaler.inverse_transform(synthetic_data_quantile_scale)

    generated_df = pd.DataFrame(synthetic_data_original_scale, columns=features)

    # Apply clamping to original min/max values
    for feature in features:
        min_val = original_min_max[feature]['min']
        max_val = original_min_max[feature]['max']
        generated_df[feature] = np.clip(generated_df[feature], min_val, max_val)
        # Ensure non-negative and potentially round for integer features
        if feature == 'AllocatedProcessors': # Assuming this is an integer feature
             generated_df[feature] = generated_df[feature].round().astype(int)
        else:
            generated_df[feature] = generated_df[feature].apply(lambda x: max(0, x)) # Ensure non-negative for other features

    return generated_df

# ─── COMPARISON FUNCTIONS ──────────────────────────────────────────

def numerical_comparison(original_df, generated_df, category_name, features):
    """Performs and prints numerical comparisons (mean, std, min, max, median, skew, kurtosis, KS test)."""
    print(f"\n--- Numerical Comparison for Category: {category_name} ---")
    
    metrics = {
        'mean': lambda df, col: df[col].mean(),
        'std': lambda df, col: df[col].std(),
        'min': lambda df, col: df[col].min(),
        'max': lambda df, col: df[col].max(),
        'median': lambda df, col: df[col].median(),
        'skew': lambda df, col: df[col].skew(),
        'kurtosis': lambda df, col: df[col].kurtosis(),
    }
    
    for feature in features:
        print(f"\nFeature: {feature}")
        for metric_name, metric_func in metrics.items():
            try:
                original_val = metric_func(original_df, feature)
                generated_val = metric_func(generated_df, feature)
                print(f"  {metric_name.capitalize()}: Original={original_val:.4f}, Generated={generated_val:.4f}")
            except Exception as e:
                print(f"  Error computing {metric_name} for {feature}: {e}")

        # Kolmogorov-Smirnov test (KS test)
        # Only perform if both datasets have enough non-zero data
        # For 'AllocatedProcessors', use integer values for KS test
        if feature == 'AllocatedProcessors':
            # Ensure integer type for KS test on discrete data
            orig_data = original_df[feature].dropna().astype(int)
            gen_data = generated_df[feature].dropna().astype(int)
        else:
            orig_data = original_df[feature].dropna()
            gen_data = generated_df[feature].dropna()

        if len(orig_data) > 1 and len(gen_data) > 1:
            try:
                ks_statistic, p_value = stats.ks_2samp(orig_data, gen_data)
                print(f"  KS Test (Statistic={ks_statistic:.4f}, P-value={p_value:.4f})")
                if p_value < 0.05:
                    print("  -> Distributions are likely different (reject H0 at 5% significance).")
                else:
                    print("  -> Distributions are likely similar (fail to reject H0 at 5% significance).")
            except ValueError as e:
                print(f"  KS Test failed for {feature}: {e}")
        else:
            print(f"  Not enough data for KS Test on feature {feature} (original: {len(orig_data)}, generated: {len(gen_data)})")


def visual_comparison(original_df, generated_df, category_name, features):
    """Generates and saves distribution plots (histograms and KDEs)."""
    plot_dir = 'comparison_plots'
    os.makedirs(plot_dir, exist_ok=True)

    print(f"\n--- Visual Comparison for Category: {category_name} ---")

    for feature in features:
        plt.figure(figsize=(10, 6))
        sns.histplot(original_df[feature], color='blue', label='Original', kde=True, stat='density', alpha=0.6)
        sns.histplot(generated_df[feature], color='red', label='Generated', kde=True, stat='density', alpha=0.6)
        
        plt.title(f'Distribution of {feature} for Category {category_name}')
        plt.xlabel(feature)
        plt.ylabel('Density')
        plt.legend()
        plt.grid(True, linestyle='--', alpha=0.7)
        
        plot_path = os.path.join(plot_dir, f'category_{category_name}_{feature}_histograms_RVAE.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"  - Saved plot for {feature} to {plot_path}")

def compare_synthetic_data():
    """
    Main function to load trained VAE models, generate synthetic data,
    and compare it with the original data.
    """
    print("Starting comparison of synthetic data...")

    # Load paths to all trained VAE checkpoints
    ckpt_files = glob.glob(os.path.join(WEIGHTS_DIR, '*_vae_ckpt.pth'))
    
    if not ckpt_files:
        print(f"No VAE checkpoints found in {WEIGHTS_DIR}. Please train models first.")
        return

    # Assuming EXCEL_SUBSET_PATH is the Excel file containing original categorized subsets
    if not os.path.exists(EXCEL_SUBSET_PATH):
        print(f"Error: Original categorized subsets Excel file not found at {EXCEL_SUBSET_PATH}. Cannot compare.")
        return

    excel_data = pd.ExcelFile(EXCEL_SUBSET_PATH)

    for ckpt_file in ckpt_files:
        category_name = os.path.basename(ckpt_file).replace('_vae_ckpt.pth', '').replace('category_', '')
        
        print(f"\nProcessing Category: {category_name}")

        # 1. Load original data for this category
        try:
            original_df = excel_data.parse(sheet_name=category_name)
        except Exception as e:
            print(f"Failed to load original data for category '{category_name}': {e}")
            continue
        
        if original_df.empty:
            print(f"Original data for category '{category_name}' is empty. Skipping comparison.")
            continue

        # 2. Load trained VAE model and its associated scalers
        try:
            ckpt = torch.load(ckpt_file, map_location=device)
            model = VAE(input_dim=len(ckpt['features']), latent_dim=ckpt['latent_dim']).to(device)
            model.load_state_dict(ckpt['model_state'])
            
            model_info = {
                'model': model,
                'scaler': ckpt['scaler'], # This will be the (quantile_scaler, standard_scaler) tuple
                'features': ckpt['features'],
                'latent_dim': ckpt['latent_dim'],
                'original_min_max': ckpt['original_min_max']
            }
        except Exception as e:
            print(f"Failed to load VAE model or scalers for category '{category_name}': {e}")
            continue

        features = model_info['features'] # Use features from the loaded model info for consistency
        original_df = original_df[features] # Filter original_df to only include trained features


        # 3. Generate synthetic data
        num_original_sequences = max(0, len(original_df) - SEQUENCE_LENGTH + 1)
        
        # Ensure we generate a reasonable number of sequences, at least one if possible.
        if num_original_sequences == 0 and len(original_df) >= 1: # If original data is too short for a full sequence, generate at least one.
            num_sequences_to_generate = 1
        else:
            num_sequences_to_generate = num_original_sequences
            
        if num_sequences_to_generate == 0:
            print(f"Not enough original data ({len(original_df)} jobs) to generate even one sequence of length {SEQUENCE_LENGTH}. Skipping generation for {category_name}.")
            continue

        generated_df = generate_synthetic_data(model_info, num_sequences_to_generate, SEQUENCE_LENGTH)
        print(f"Generated {len(generated_df)} synthetic jobs (from {num_sequences_to_generate} sequences) for '{category_name}'.")

        if generated_df.empty:
            print(f"Generated data for category '{category_name}' is empty. Skipping comparison.")
            continue
        
        # Ensure both dataframes have the same columns for comparison.
        # The 'features' list defines the common columns that were used in VAE training.

        # 4. Perform Numerical Comparison
        numerical_comparison(original_df, generated_df, category_name, features)

        # 5. Perform Visual Comparison
        visual_comparison(original_df, generated_df, category_name, features)
        
    print("\nComparison complete. Check 'comparison_plots' directory for visualizations.")

# Entry point for the script
if __name__ == '__main__':
    compare_synthetic_data()