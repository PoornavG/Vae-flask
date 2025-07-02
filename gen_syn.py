# generate_synthetic_data.py
import os
import numpy as np
import pandas as pd
import torch
import glob
import logging
import random # For random category selection if needed

# --- CONFIGURATION IMPORTS ---
# Assuming config.py is in the same directory or accessible via PYTHONPATH
try:
    from config import (
        WEIGHTS_DIR, DEFAULT_LATENT_DIM, HIDDEN_DIMS, # HIDDEN_DIMS might not be used by RVAE directly
        DEFAULT_SIM_JOBS, # Use this as the default number of jobs to generate per category
        BURST_LEVEL_MAP, POISSON_RATES, P_TRANSITION, # For MMPP inter-arrival times
        EXCEL_SUBSET_PATH # Path to the real categorized data for category names
    )
except ImportError:
    print("Warning: config.py not found or incomplete. Using default hardcoded configurations.")
    # --- Fallback hardcoded configurations ---
    WEIGHTS_DIR = 'vae_models'
    DEFAULT_LATENT_DIM = 64
    HIDDEN_DIMS = [256, 128, 64] # Mockup, might not be used by RVAE
    DEFAULT_SIM_JOBS = 500 # Default jobs to generate per category for comparison
    BURST_LEVEL_MAP = {'Low': 0, 'Mid': 1, 'High': 2}
    POISSON_RATES = {'Low': 0.05, 'Mid': 0.5, 'High': 2.0}
    P_TRANSITION = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]
    EXCEL_SUBSET_PATH = 'swf_utils/subsets/categorized_subsets.xlsx' # Mockup path

# --- VAE MODEL AND PREPROCESSING IMPORTS ---
# Import from your new vae_training3.py
try:
    from vae_training3 import VAE, load_and_preprocess , set_seed # We need load_and_preprocess for its original_min_max and features logic
except ImportError:
    print("Error: vae_training3.py not found. Please ensure it's in the same directory or PYTHONPATH.")
    exit() # Exit if VAE module is not available

# --- LOGGING CONFIG ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('SyntheticDataGenerator')

# --- OUTPUT DIRECTORY FOR SYNTHETIC DATA ---
# This should be the directory where compare_data.py expects synthetic files
SYNTHETIC_OUTPUT_DIR = 'synthetic_data_for_comparison'
os.makedirs(SYNTHETIC_OUTPUT_DIR, exist_ok=True)

# --- MMPP GENERATOR (Copied from simser10.py) ---
def generate_mmpp_interarrival_time(current_burst_level_str, num_jobs_to_generate):
    """
    Generates a sequence of inter-arrival times using an MMPP model.
    Uses BURST_LEVEL_MAP to convert string label to 0-indexed state.
    """
    interarrivals = []
    current_state_idx = BURST_LEVEL_MAP.get(current_burst_level_str, 0) # Default to 0 (Low)
    
    for _ in range(num_jobs_to_generate):
        current_rate = POISSON_RATES[list(BURST_LEVEL_MAP.keys())[list(BURST_LEVEL_MAP.values()).index(current_state_idx)]]
        if current_rate > 0:
            interarrival_time = np.random.exponential(scale=1/current_rate)
        else:
            interarrival_time = 0.0
        
        interarrivals.append(max(0.0, interarrival_time))
        
        transition_probs = P_TRANSITION[current_state_idx]
        current_state_idx = np.random.choice(len(transition_probs), p=transition_probs)

    return interarrivals

# --- VAE RELATED FUNCTIONS (Adapted from simser10.py) ---
def load_all_vaes_for_generation():
    """
    Loads all trained RVAE models from the weights directory into a dictionary,
    including scaler, features, and original_min_max for post-processing.
    """
    logger.info("Loading RVAE models for generation...")
    models = {}
    
    for ckpt_path in glob.glob(os.path.join(WEIGHTS_DIR, "*_vae_ckpt.pth")):
        category = os.path.basename(ckpt_path).replace("_vae_ckpt.pth", "")
        try:
            # Ensure map_location="cpu" if not using GPU for generation
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            state = ckpt["model_state"]
            scaler = ckpt["scaler"]
            feats = ckpt["features"]
            latent_d = ckpt["latent_dim"]
            original_min_max = ckpt.get("original_min_max", {})

            # Initialize VAE with the correct parameters for RVAE (from vae_training3.py)
            # Assuming default num_rnn_layers=1, rnn_hidden_size=128 as in vae_training3.py
            model = VAE(input_dim=len(feats), latent_dim=latent_d)
            model.load_state_dict(state)
            model.eval() # Set to evaluation mode

            models[category] = {
                "model": model,
                "scaler": scaler,
                "latent_dim": latent_d,
                "features": feats,
                "original_min_max": original_min_max
            }
            logger.info(f"Loaded RVAE for category '{category}' (input_dim={len(feats)}, latent_dim={latent_d}).")
        except Exception as e:
            logging.error(f"Failed to load RVAE for category '{category}' from {ckpt_path}: {e}")
            
    return models

def generate_synthetic_jobs_for_category(model_data, num_jobs_to_generate, sequence_length=10):
    """
    Generates a specified number of synthetic jobs for a given category
    using the loaded RVAE model.
    """
    model = model_data['model']
    scaler = model_data['scaler']
    latent_d = model_data['latent_dim']
    features = model_data['features']
    original_min_max = model_data['original_min_max']
    category = model_data['category'] # Add category to model_data for logging

    generated_jobs_df = pd.DataFrame(columns=features)
    
    # Check which features were log-transformed during training
    skewed_features = ['RunTime', 'WaitTime', 'AverageCPUTimeUsed']

    # Check if 'UsedMemory' was constant at zero in the original data for this category
    force_zero_memory = False
    if 'UsedMemory' in original_min_max:
        if original_min_max['UsedMemory']['min'] == 0 and original_min_max['UsedMemory']['max'] == 0:
            force_zero_memory = True

    # Determine how many sequences are needed
    # Each sequence contains 'sequence_length' jobs
    num_sequences_needed = (num_jobs_to_generate + sequence_length - 1) // sequence_length
    
    all_generated_sequences = []

    with torch.no_grad():
        for _ in range(num_sequences_needed):
            z = torch.randn(1, latent_d).to(model.fc_mu.weight.device) # Generate one latent vector per sequence
            # Decode to get a sequence of jobs
            # model.decode returns (batch_size, sequence_length, input_dim)
            generated_sequence_scaled = model.decode(z, sequence_length=sequence_length).cpu().numpy()
            
            # Reshape for inverse_transform: (sequence_length, input_dim)
            generated_sequence_scaled = generated_sequence_scaled.reshape(-1, len(features))
            
            # Inverse transform from scaled space to (potentially log-transformed) original scale
            real_vals = scaler.inverse_transform(generated_sequence_scaled)
            
            # Apply inverse log-transformation for specific features
            for i, feature_name in enumerate(features):
                if feature_name in skewed_features:
                    real_vals[:, i] = np.expm1(real_vals[:, i]) # exp(x) - 1

            # Post-processing to enforce original data ranges (clamping)
            for i, feature_name in enumerate(features):
                if feature_name in original_min_max:
                    orig_min = original_min_max[feature_name]['min']
                    orig_max = original_min_max[feature_name]['max']
                    real_vals[:, i] = np.clip(real_vals[:, i], orig_min, orig_max)
            
            # Convert to DataFrame and append
            sequence_df = pd.DataFrame(real_vals, columns=features)
            
            # Apply specific logic for 'pes' and 'ram' as in simser10.py
            if 'AllocatedProcessors' in sequence_df.columns:
                sequence_df['AllocatedProcessors'] = sequence_df['AllocatedProcessors'].apply(lambda x: max(1, int(round(x))))
            if 'UsedMemory' in sequence_df.columns:
                if force_zero_memory:
                    sequence_df['UsedMemory'] = 0.0
                else:
                    sequence_df['UsedMemory'] = sequence_df['UsedMemory'].apply(lambda x: max(0.0, x))
            
            # Ensure 'RunTime' and 'AverageCPUTimeUsed' are positive
            if 'RunTime' in sequence_df.columns:
                sequence_df['RunTime'] = sequence_df['RunTime'].apply(lambda x: max(0.0, x))
            if 'AverageCPUTimeUsed' in sequence_df.columns:
                sequence_df['AverageCPUTimeUsed'] = sequence_df['AverageCPUTimeUsed'].apply(lambda x: max(0.0, x))

            all_generated_sequences.append(sequence_df)

    # Concatenate all generated sequences
    if all_generated_sequences:
        generated_jobs_df = pd.concat(all_generated_sequences, ignore_index=True)
        # Trim to the exact number of jobs requested
        generated_jobs_df = generated_jobs_df.head(num_jobs_to_generate)
    else:
        logger.warning(f"No sequences generated for category {category}.")
        
    logger.info(f"Generated {len(generated_jobs_df)} synthetic jobs for category '{category}'.")
    return generated_jobs_df

def main():
    set_seed(42) # Ensure reproducibility for generation
    
    # 1. Load all trained RVAE models
    all_vae_models = load_all_vaes_for_generation()
    if not all_vae_models:
        logger.error("No VAE models loaded. Cannot generate synthetic data. Please ensure you have trained models using train_all_vae2.py.")
        return

    # 2. Get the list of categories from the real data Excel file
    # This ensures we generate for categories that actually exist and have trained models
    try:
        excel_sheets = pd.ExcelFile(EXCEL_SUBSET_PATH).sheet_names
        categories_to_generate = [cat for cat in excel_sheets if cat in all_vae_models]
        if not categories_to_generate:
            logger.warning(f"No matching categories between '{EXCEL_SUBSET_PATH}' and trained VAE models in '{WEIGHTS_DIR}'.")
            # Fallback to generating for all loaded models if no categories from Excel match
            categories_to_generate = list(all_vae_models.keys())
            if not categories_to_generate:
                logger.error("No categories to generate synthetic data for. Exiting.")
                return
            logger.info(f"Generating for all loaded VAE model categories: {categories_to_generate}")

    except FileNotFoundError:
        logger.error(f"Real categorized subsets Excel file not found at '{EXCEL_SUBSET_PATH}'. Cannot determine categories to generate. Generating for all loaded models.")
        categories_to_generate = list(all_vae_models.keys())
        if not categories_to_generate:
            logger.error("No categories to generate synthetic data for. Exiting.")
            return

    # 3. Generate synthetic data for each category
    for category_name in categories_to_generate:
        logger.info(f"Starting synthetic data generation for category: '{category_name}'")
        model_info = all_vae_models[category_name]
        
        # Add category name to model_info for easier access within the generation function
        model_info['category'] = category_name 

        # Define the sequence length used during training for this model
        # You might need to retrieve this from the saved checkpoint if it varies per model
        # For now, assuming a default or consistent SEQUENCE_LENGTH (e.g., 10)
        # If vae_training3.py's load_and_preprocess was called with a specific sequence_length,
        # that same length should be used here.
        # For simplicity, let's assume a fixed SEQUENCE_LENGTH=10 for now.
        # In a more robust system, SEQUENCE_LENGTH should be saved in the checkpoint.
        current_sequence_length = 10 # IMPORTANT: Match this to what you used in vae_training3.py's load_and_preprocess

        synthetic_df = generate_synthetic_jobs_for_category(
            model_data=model_info,
            num_jobs_to_generate=DEFAULT_SIM_JOBS, # Use configured default for comparison
            sequence_length=current_sequence_length
        )
        
        # Save the generated data to an Excel file
        output_file_name = f"{category_name}_synthetic_{DEFAULT_SIM_JOBS}.xlsx"
        output_path = os.path.join(SYNTHETIC_OUTPUT_DIR, output_file_name)
        synthetic_df.to_excel(output_path, index=False)
        logger.info(f"Saved synthetic data for '{category_name}' to '{output_path}'")

    logger.info("Synthetic data generation complete for all categories.")

if __name__ == "__main__":
    main()