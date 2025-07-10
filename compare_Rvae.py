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
import torch.serialization # Import torch.serialization
from sklearn.preprocessing import PowerTransformer

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

# Add StandardScaler, PowerTransformer and numpy._core.multiarray._reconstruct to safe globals for torch.load

torch.serialization.add_safe_globals([StandardScaler, np._core.multiarray._reconstruct, np.ndarray])
torch.serialization.add_safe_globals([PowerTransformer])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
set_seed(SEED) # Ensure reproducibility for data loading and model initialization

# --- Helper function for MMPP (corrected from compare_new, adapted for interarrival time generation) ---
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
            
        # Ensure interarrival_time is non-negative
        interarrivals.append(max(0.0, interarrival_time))

        # Transition to the next state based on the P_TRANSITION matrix
        transition_probs = P_TRANSITION[current_state_idx]
        current_state_idx = np.random.choice(len(transition_probs), p=transition_probs)

    return interarrivals

# --- VAE Model Loading (Adapted for RVAE and sequence-aware loading) ---
def load_all_rvaes_for_comparison():
    """
    Loads all trained RVAE models, their scalers, feature names, and original min/max values
    from the designated weights directory.
    
    Returns:
        dict: A dictionary where keys are category names and values are dictionaries
              containing the 'model', 'scaler', 'latent_dim', 'features', and 'original_min_max'
              for each RVAE.
    """
    print("Loading RVAE models and scalers for comparison...")
    models = {}
    for ckpt_path in glob.glob(os.path.join(WEIGHTS_DIR, "*_vae_ckpt.pth")):
        category = os.path.basename(ckpt_path).replace("_vae_ckpt.pth", "")
        try:
            # Load the checkpoint dictionary with weights_only=False to allow custom classes/objects
            checkpoint = torch.load(ckpt_path, map_location=device, weights_only=False)
            
            # Retrieve saved parameters
            saved_latent_dim = checkpoint.get('latent_dim', DEFAULT_LATENT_DIM)
            saved_features = checkpoint.get('features', None)
            saved_scaler = checkpoint.get('scaler', StandardScaler())
            saved_pt       = checkpoint.get('power_transformer', None)
            # FIXED: Load original_min_max
            saved_original_min_max = checkpoint.get('original_min_max', {})
            
            if saved_features is None:
                print(f"Warning: 'features' not found in checkpoint for {category}. Skipping model load.")
                continue

            # FIXED: VAE constructor in vae_training3.py only expects input_dim and latent_dim
            model = VAE(
                input_dim=len(saved_features),
                latent_dim=saved_latent_dim
            ).to(device)
            
            model.load_state_dict(checkpoint['model_state'])
            model.eval()

            models[category] = {
                "model": model,
                "scaler": saved_scaler,
                "power_transformer": saved_pt,
                "latent_dim": saved_latent_dim,
                "features": saved_features,
                "original_min_max": saved_original_min_max # Store original min/max
            }
            print(f"Loaded RVAE for category '{category}'.")
        except Exception as e:
            print(f"Failed to load RVAE for category '{category}': {e}")
            
    return models

# --- Data Generation (Adapted for RVAE's sequence generation) ---
def generate_synthetic_data(model_data, num_sequences_to_generate, sequence_length, burst_level='Mid', temperature=1.0):
    """
    Generates synthetic job sequence data for a given RVAE model.
    
    Args:
        model_data (dict): Dictionary containing the RVAE 'model', 'scaler', 'latent_dim', 'features',
                           and 'original_min_max'.
        num_sequences_to_generate (int): The number of synthetic sequences to generate.
        sequence_length (int): The length of each sequence to generate.
        burst_level (str): The burst level for MMPP inter-arrival time generation (default 'Mid').
        temperature (float): The temperature for sampling latent space.
        
    Returns:
        pd.DataFrame: A DataFrame containing the generated synthetic data, flattened from sequences.
                      Includes 'interarrival' time.
    """
    model = model_data['model']
    scaler = model_data['scaler']
    latent_d = model_data['latent_dim']
    features = model_data['features']
    original_min_max = model_data['original_min_max'] # FIXED: Get original min/max
    
    # Features that were log-transformed during training
    skewed_features = ['RunTime', 'AverageCPUTimeUsed']
    pt = model_data.get('power_transformer', None)
    
    generated_sequences = []
    
    # Generate interarrival times for individual jobs.
    # The total number of jobs will be num_sequences_to_generate * sequence_length
    total_jobs_for_interarrival = num_sequences_to_generate * sequence_length
    interarrival_sequence = generate_mmpp_interarrival_time(burst_level, total_jobs_for_interarrival)
    interarrival_iterator = iter(interarrival_sequence)

    batch_size = 64 # Use a reasonable batch size for generation
    num_batches = (num_sequences_to_generate + batch_size - 1) // batch_size

    for _ in range(num_batches):
        current_batch_size = min(batch_size, num_sequences_to_generate - len(generated_sequences))
        if current_batch_size <= 0:
            break
        
        z = torch.randn(current_batch_size, latent_d).to(device) * temperature # Generate latent vectors on device with temperature
        
        with torch.no_grad():
            model.eval()
            # RVAE's decode method expects (z, sequence_length)
            decoded_sequences_scaled = model.decode(z, sequence_length=sequence_length).cpu().numpy()

        # Inverse transform to bring data back to original scale
        # decoded_sequences_scaled shape: (batch_size, sequence_length, num_features)
        # Apply inverse transform to each feature across all time steps in a sequence
        
        # Reshape for scaler: (num_samples * sequence_length, num_features)
        reshaped_for_inverse = decoded_sequences_scaled.reshape(-1, len(features))
        real_vals_flat = scaler.inverse_transform(reshaped_for_inverse)
        # === NEW: inverse PowerTransform for skewed cols ===
        if pt is not None:
            # find indices of skewed features in your feature list
            skew_idx = [i for i, f in enumerate(features) if f in skewed_features]
            if skew_idx:
                # extract those columns
                sub = real_vals_flat[:, skew_idx]
                # build a DataFrame whose columns match the ones transformer saw
                sub_df = pd.DataFrame(sub, columns=[features[i] for i in skew_idx])
                # inverse_transform returns a NumPy array directly
                inv_arr = pt.inverse_transform(sub_df)
                # plug the inverted values back in
                real_vals_flat[:, skew_idx] = inv_arr
        
        # Reshape back to sequences: (batch_size, sequence_length, num_features)
        real_vals_sequences = real_vals_flat.reshape(current_batch_size, sequence_length, len(features))
        
        generated_sequences.extend(real_vals_sequences)

    # Flatten the list of sequences into a single DataFrame of jobs
    all_generated_jobs = []
    for seq_idx, sequence_data in enumerate(generated_sequences):
        for step_idx, job_vals in enumerate(sequence_data):
            job_dict = {}
            for i, feature_name in enumerate(features):
                # all features are now on raw scale
                val = job_vals[i]
                
                job_dict[feature_name] = val
            
            # Add interarrival time, ensuring we don't run out
            try:
                job_dict["interarrival"] = next(interarrival_iterator)
            except StopIteration:
                job_dict["interarrival"] = 0.0 # Default if interarrival times run out

            # FIXED: Apply clamping to original min/max and handle discrete values
            for feature_name in features:
                if feature_name in original_min_max:
                    min_val = original_min_max[feature_name]['min']
                    max_val = original_min_max[feature_name]['max']
                    
                    # Ensure numeric and clip to original range
                    job_dict[feature_name] = np.clip(job_dict[feature_name], min_val, max_val)
                    
                    # Round specific features to integers if they are inherently discrete
                    if feature_name in ['AllocatedProcessors']: # Add other integer features if any
                        job_dict[feature_name] = int(round(job_dict[feature_name]))
                
                # Ensure non-negativity for values that cannot be negative
                if feature_name in ['RunTime', 'AverageCPUTimeUsed', 'UsedMemory', 'AllocatedProcessors']:
                    job_dict[feature_name] = max(0.0, job_dict[feature_name])
            
            # Handle interarrival non-negativity
            job_dict["interarrival"] = max(0.0, job_dict["interarrival"])

            all_generated_jobs.append(job_dict)

    # Convert to DataFrame and apply nan_to_num for robustness
    generated_df = pd.DataFrame(all_generated_jobs)
    for col in generated_df.columns:
        if generated_df[col].dtype == 'float64' or generated_df[col].dtype == 'float32':
            # Replace NaNs/Infs after all transformations for final cleanliness
            generated_df[col] = np.nan_to_num(generated_df[col], nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)

    return generated_df

# Helper function to load original data for a given category
def load_original_dataframe(category, models):
    """
    Loads the original dataframe for a given category,
    using the features & original_min_max from `models[category]`.
    """
    # read the Excel subset
    sheet = category.replace("category_", "")
    df_full = pd.read_excel(EXCEL_SUBSET_PATH, sheet_name=sheet)

    # pull metadata from the passed-in dict
    feats = models[category]['features']
    orig_mm = models[category]['original_min_max']

    # subset + clean/clamp as before
    orig = df_full[feats].apply(pd.to_numeric, errors='coerce').fillna(0)
    for f in feats:
        if f in orig_mm:
            orig[f] = orig[f].clip(orig_mm[f]['min'], orig_mm[f]['max'])
        if f in ['AllocatedProcessors']:
            orig[f] = orig[f].round().astype(int)
        if f in ['RunTime','AverageCPUTimeUsed','UsedMemory','AllocatedProcessors']:
            orig[f] = orig[f].clip(lower=0.0)
    return orig


def tune_temperatures(models,
                      temps=np.linspace(0.5, 4.0, 15),
                      n_samples=1000,
                      seq_len=SEQUENCE_LENGTH):
    """
    For each category/model in `models`, sample at each temperature in `temps`,
    generate `n_samples` synthetic sequences, compute the generated mean and std
    of each skewed feature vs. the original, and pick the temperature minimizing
    the total mean + std absolute diff (in %).

    Also includes a lightweight std boosting correction if needed.

    Returns: dict mapping category -> best_temperature.
    """
    best_temps = {}
    skewed = ['RunTime', 'AverageCPUTimeUsed']

    for cat, mdata in models.items():
        orig_df = load_original_dataframe(cat, models)
        orig_means = orig_df[skewed].mean()
        orig_stds = orig_df[skewed].std()

        best_score = float('inf')
        best_T = temps[0]

        for T in temps:
            # sample latent with temperature T
            z = torch.randn(n_samples, mdata['latent_dim'], device=device) * T

            mdata['model'].eval()
            with torch.no_grad():
                x_dec = mdata['model'].decode(z, sequence_length=seq_len)

            flat = x_dec.reshape(-1, len(mdata['features']))
            unscaled = mdata['scaler'].inverse_transform(flat.cpu().detach().numpy())
            df_flat = pd.DataFrame(unscaled, columns=mdata['features'])

            # Apply inverse power-transform (if used)
            if mdata.get('power_transformer', None) is not None:
                idx = [i for i, f in enumerate(mdata['features']) if f in skewed]
                sub_df = pd.DataFrame(unscaled[:, idx],
                                      columns=[mdata['features'][i] for i in idx])
                inv_sub = mdata['power_transformer'].inverse_transform(sub_df)
                df_flat.iloc[:, idx] = inv_sub

            # Optional variance boost to underdispersed skewed features
            for feat in skewed:
                gen_std = df_flat[feat].std()
                orig_std = orig_stds[feat]
                std_diff_pct = abs(gen_std - orig_std) / orig_std * 100

                if gen_std < orig_std and std_diff_pct > 40:
                    correction_std = (orig_std - gen_std) * 0.75  # dampen noise
                    print(f"[{cat}] Boosting std of '{feat}' by ~{correction_std:.3f}")
                    df_flat[feat] += np.random.normal(
                        loc=0.0,
                        scale=correction_std,
                        size=df_flat.shape[0]
                    )

            # Average over each full sequence
            gen_df = df_flat.groupby(df_flat.index // seq_len).mean()
            gen_means = gen_df[skewed].mean()
            gen_stds = gen_df[skewed].std()

            # Percent error in mean and std
            mean_diffs = (gen_means - orig_means).abs() / orig_means.abs() * 100
            std_diffs = (gen_stds - orig_stds).abs() / orig_stds.abs() * 100

            score = mean_diffs.mean() + std_diffs.mean()

            if score < best_score:
                best_score = score
                best_T = T

        print(f"Category {cat!r}: best T = {best_T:.2f} (score: {best_score:.2f}%)")
        best_temps[cat] = best_T

    return best_temps


# --- Numerical Comparison ---
def numerical_comparison(original_df, generated_df, category, features):
    """
    Performs numerical comparison of statistical properties (mean, std, min, max, quantiles, KS test, correlation).
    
    Args:
        original_df (pd.DataFrame): DataFrame containing the original data.
        generated_df (pd.DataFrame): DataFrame containing the generated data.
        category (str): The name of the category being compared.
        features (list): A list of feature names to compare (these were used for VAE training).
    """
    print(f"\n--- Numerical Comparison for Category: {category} ---")
    
    # Ensure 'interarrival' is included if present in generated_df (it's not in original_df)
    all_features_to_compare = list(features)
    if 'interarrival' in generated_df.columns and 'interarrival' not in all_features_to_compare:
        all_features_to_compare.append('interarrival')

    metrics = ['mean', 'std', 'min', 'max', '25%', '50%', '75%']
    # Ensure original_df only describes features relevant to it
    original_desc = original_df[features].describe().loc[metrics]    
    generated_desc = generated_df[all_features_to_compare].describe().loc[metrics]

    print("\nOriginal Data Statistics (for VAE-trained features):")
    print(original_desc)
    print("\nGenerated Data Statistics (including interarrival):")
    print(generated_desc)

    print("\n--- Feature-wise Comparison (Mean & Std Dev) ---")
    comparison_df = pd.DataFrame(index=all_features_to_compare, columns=['Original Mean', 'Generated Mean', 'Mean Diff (%)', 'Original Std', 'Generated Std', 'Std Diff (%)'])
    for feature in all_features_to_compare:
        orig_mean = original_desc.loc['mean', feature] if feature in original_desc.columns else np.nan
        gen_mean = generated_desc.loc['mean', feature]
        orig_std = original_desc.loc['std', feature] if feature in original_desc.columns else np.nan
        gen_std = generated_desc.loc['std', feature]

        # Handle division by zero or NaN for percentage difference calculation
        mean_diff_pct = (abs(gen_mean - orig_mean) / orig_mean) * 100 if orig_mean != 0 and not np.isnan(orig_mean) and np.isfinite(orig_mean) else np.nan
        std_diff_pct = (abs(gen_std - orig_std) / orig_std) * 100 if orig_std != 0 and not np.isnan(orig_std) and np.isfinite(orig_std) else np.nan

        comparison_df.loc[feature] = [orig_mean, gen_mean, f"{mean_diff_pct:.2f}%" if not np.isnan(mean_diff_pct) else "N/A",
                                       orig_std, gen_std, f"{std_diff_pct:.2f}%" if not np.isnan(std_diff_pct) else "N/A"]
    print(comparison_df)

    print("\n--- Kolmogorov-Smirnov (KS) Test (p-value for each feature) ---")
    print("Null Hypothesis (H0): The two samples are drawn from the same continuous distribution.")
    print("A high p-value (e.g., > 0.05) suggests we cannot reject H0, meaning distributions are similar.")
    ks_results = {}
    for feature in all_features_to_compare:
        # Filter out NaN/inf values before KS test
        orig_vals = original_df[feature].values if feature in original_df.columns else np.array([])
        gen_vals = generated_df[feature].values
        
        # Ensure only finite values are passed to ks_2samp and there's enough data
        orig_vals = orig_vals[np.isfinite(orig_vals)]
        gen_vals = gen_vals[np.isfinite(gen_vals)]

        if len(orig_vals) > 1 and len(gen_vals) > 1: # ks_2samp needs at least 2 samples
            statistic, p_value = stats.ks_2samp(orig_vals, gen_vals)
            ks_results[feature] = {'Statistic': f"{statistic:.4f}", 'P-value': f"{p_value:.4f}"}
        else:
            ks_results[feature] = {'Statistic': 'N/A', 'P-value': 'N/A (Insufficient data for KS test)'}
    print(pd.DataFrame.from_dict(ks_results, orient='index'))

    print("\n--- Correlation Matrix Comparison ---")
    print("Original Data Correlation (for VAE input features):")
    # Drop NaN/inf before calculating correlation
    print(original_df[features].corr())    
    print("\nGenerated Data Correlation (including interarrival):")
    print(generated_df[all_features_to_compare].corr())


# --- Visual Comparison ---
def visual_comparison(original_df, generated_df, category, features):
    """
    Generates plots for visual comparison (histograms).
    
    Args:
        original_df (pd.DataFrame): DataFrame containing the original data.
        generated_df (pd.DataFrame): DataFrame containing the generated data.
        category (str): The name of the category being compared.
        features (list): A list of feature names to visualize (these were used for VAE training).
    """
    print(f"\n--- Visual Comparison for Category: {category} ---")

    # Ensure 'interarrival' is included if present in generated_df
    all_features_to_compare_visual = list(features)
    if 'interarrival' in generated_df.columns and 'interarrival' not in all_features_to_compare_visual:
        all_features_to_compare_visual.append('interarrival')

    # Clip values to a reasonable range for plotting to prevent visual distortions from extreme outliers
    MAX_PLOT_VAL = 1e6    
    for feat in all_features_to_compare_visual:
        if feat in original_df.columns:
            original_df[feat] = original_df[feat].replace([np.inf, -np.inf], np.nan).fillna(0)
            original_df[feat] = original_df[feat].clip(upper=MAX_PLOT_VAL)
        if feat in generated_df.columns: # Generated data might also have outliers due to expm1 or slight numerical instability
            generated_df[feat] = generated_df[feat].replace([np.inf, -np.inf], np.nan).fillna(0)
            generated_df[feat] = generated_df[feat].clip(upper=MAX_PLOT_VAL)
    
    # Histograms for each feature
    print("Generating Histograms...")
    fig_hist, axes_hist = plt.subplots(nrows=len(all_features_to_compare_visual), ncols=1, figsize=(8, 4 * len(all_features_to_compare_visual)))
    fig_hist.suptitle(f'Feature Histograms for Category: {category}', fontsize=16)
    if len(all_features_to_compare_visual) == 1:    
        axes_hist = [axes_hist] # Ensure axes_hist is iterable even for a single feature
    
    # Determine the smaller dataset size for consistent plotting
    min_samples = min(len(original_df), len(generated_df))

    for i, feature in enumerate(all_features_to_compare_visual):
        ax = axes_hist[i]
        
        # Only plot original if the feature exists in original_df and has data
        if feature in original_df.columns:
            original_data_for_plot = original_df[feature].copy()
            original_data_for_plot = original_data_for_plot[np.isfinite(original_data_for_plot)].sample(n=min_samples, random_state=SEED) # Sample
            if not original_data_for_plot.empty and len(original_data_for_plot.unique()) > 1:
                sns.histplot(original_data_for_plot, color='blue', label='Original', kde=True, stat='density', alpha=0.5, ax=ax, bins=50)
            elif not original_data_for_plot.empty:
                sns.histplot(original_data_for_plot, color='blue', label='Original', stat='density', alpha=0.5, ax=ax, bins=1)

        generated_data_for_plot = generated_df[feature].copy()
        generated_data_for_plot = generated_data_for_plot[np.isfinite(generated_data_for_plot)].sample(n=min_samples, random_state=SEED) # Sample
        if not generated_data_for_plot.empty and len(generated_data_for_plot.unique()) > 1:
            sns.histplot(generated_data_for_plot, color='red', label='Generated', kde=True, stat='density', alpha=0.5, ax=ax, bins=50)
        elif not generated_data_for_plot.empty:
            sns.histplot(generated_data_for_plot, color='red', label='Generated', stat='density', alpha=0.5, ax=ax, bins=1)
        
        ax.set_title(f'Distribution of {feature}')
        ax.legend()
    plt.tight_layout(rect=[0, 0.03, 1, 0.96])
    plt.savefig(f'comparison_plots/{category}_histograms_RVAE.png')
    plt.close(fig_hist)
    print(f"  - Histograms saved to comparison_plots/{category}_histograms_RVAE.png")


# Entry point for the script
def run_comparison():
    """
    Orchestrates the comparison process: loading models and data, performing numerical
    and visual comparisons for each RVAE model category.
    """
    os.makedirs('comparison_plots', exist_ok=True) # Create directory for plots

    # 1. Load all RVAE models
    global rvae_models_data # Make this global so tune_temperatures can access it
    rvae_models_data = load_all_rvaes_for_comparison()
    if not rvae_models_data:
        print("No RVAE models loaded. Exiting comparison.")
        return

    # 2. Tune temperatures for all models
    print("\n--- Tuning Temperatures for VAE Models ---")
    best_Ts = tune_temperatures(rvae_models_data, seq_len=SEQUENCE_LENGTH)
    print("\nTemperature tuning complete. Best temperatures per category:")
    print(best_Ts)

    # Process each category/model
    for category_name, model_info in rvae_models_data.items():
        print(f"\n===== Comparing for Category: '{category_name}' =====")
        
        model = model_info['model']
        scaler = model_info['scaler']
        features = model_info['features'] # These are the features used for VAE training
        original_min_max = model_info['original_min_max'] # Use this for accurate comparison with original scale
        
        # Get the best temperature for this category
        current_best_T = best_Ts.get(category_name, 1.0) # Default to 1.0 if not found

        # 3. Load original data for this category
        try:
            # The sheet name in the Excel file is derived by removing 'category_' prefix.
            sheet_name_in_excel = category_name.replace("category_", "") 
            
            # Load the original data directly from the Excel sheet
            original_df_full = pd.read_excel(EXCEL_SUBSET_PATH, sheet_name=sheet_name_in_excel)
            original_df = original_df_full[features].copy() # Select only the features the VAE was trained on

            # Convert to numeric and fill any initial non-numeric values with 0
            for col in original_df.columns:
                original_df[col] = pd.to_numeric(original_df[col], errors='coerce').fillna(0)
            
            # Ensure non-negativity for values that cannot be negative, and clamp to original min/max if available
            for feature in features: # Iterate through features used by VAE
                if feature in original_df.columns:
                    # Apply clamping to original min/max (if available)
                    if feature in original_min_max:
                        min_val = original_min_max[feature]['min']
                        max_val = original_min_max[feature]['max']
                        original_df[feature] = np.clip(original_df[feature], min_val, max_val)

                    # Round specific features to integers if they are inherently discrete
                    if feature in ['AllocatedProcessors']: # Add other integer features if any
                        original_df[feature] = original_df[feature].round().astype(int)
                    
                    # Ensure non-negativity
                    if feature in ['RunTime', 'AverageCPUTimeUsed', 'UsedMemory', 'AllocatedProcessors']:
                        original_df[feature] = original_df[feature].apply(lambda x: max(0.0, x))
            
            # After all transformations, ensure all values are finite by explicitly replacing any leftover NaNs/Infs
            for col in original_df.columns:
                if original_df[col].dtype == 'float64' or original_df[col].dtype == 'float32':
                    original_df[col] = np.nan_to_num(original_df[col], nan=0.0, posinf=np.finfo(np.float64).max, neginf=np.finfo(np.float64).min)
            
            print(f"Loaded original data for '{category_name}': {len(original_df)} samples.")
            
        except Exception as e:
            print(f"Could not load original data for category '{category_name}' from '{EXCEL_SUBSET_PATH}' (sheet: '{sheet_name_in_excel}'): {e}. Skipping.")
            continue
        
        if original_df.empty:
            print(f"Original data for category '{category_name}' is empty. Skipping comparison.")
            continue

        # 4. Generate synthetic data (number of sequences, not jobs) using the best temperature
        # 4. Generate synthetic data (targetting equal number of jobs) using the best temperature
        target_num_generated_jobs = len(original_df)

        if target_num_generated_jobs == 0:
            print(f"Original data for category '{category_name}' is empty. Skipping generation.")
            continue

        # Calculate how many sequences are needed to reach or exceed the target number of jobs
        # Use ceil to ensure we generate at least the target number of jobs,
        # even if it means generating slightly more than original_df length due to SEQUENCE_LENGTH
        num_sequences_to_generate = int(np.ceil(target_num_generated_jobs / SEQUENCE_LENGTH))

        if num_sequences_to_generate == 0: # This case should ideally not happen if target_num_generated_jobs > 0
            num_sequences_to_generate = 1 # Ensure at least one sequence is generated if target is small

        generated_df = generate_synthetic_data(model_info, num_sequences_to_generate, SEQUENCE_LENGTH, temperature=current_best_T)

        # Crucially, trim the generated data to exactly match the original number of samples
        if len(generated_df) > target_num_generated_jobs:
            generated_df = generated_df.head(target_num_generated_jobs)
        elif len(generated_df) < target_num_generated_jobs:
            print(f"Warning: Generated fewer jobs ({len(generated_df)}) than target ({target_num_generated_jobs}) for '{category_name}'. This might happen if num_sequences_to_generate was very small.")

        print(f"Generated {len(generated_df)} synthetic jobs (targeting {target_num_generated_jobs} original jobs) for '{category_name}' with temperature {current_best_T:.2f}.")
        if generated_df.empty:
            print(f"Generated data for category '{category_name}' is empty. Skipping comparison.")
            continue
        
        # 5. Perform Numerical Comparison
        numerical_comparison(original_df, generated_df, category_name, features)

        # 6. Perform Visual Comparison
        visual_comparison(original_df, generated_df, category_name, features)
        
    print("\nComparison complete. Check 'comparison_plots' directory for visualizations.")

if __name__ == '__main__':
    run_comparison()