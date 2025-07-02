# config.py
# This file stores all key project configurations.

import os

# --- PATHS ---
DATA_DIR = "generated_data" # Directory for generated data
WEIGHTS_DIR = 'vae_models' # Directory for VAE model weights
SUBSETS_DIR = "subsets" # Directory for data subsets
LOG_DIR = "runs" # Directory for training logs
SWF_PATH = "/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf" # Path to the SWF dataset
EXCEL_SUBSET_PATH = os.path.join(SUBSETS_DIR, "categorized_subsets.xlsx") # Path to categorized subsets in Excel

# --- VAE HYPERPARAMETERS ---
DEFAULT_LATENT_DIM = 64 # Default dimensionality of the VAE's latent space
ANOMALY_LATENT_DIM = 16 # Latent dimension specifically for anomaly detection models
HIDDEN_DIMS = [256, 128, 64] # Dimensions of the hidden layers in the VAE's encoder/decoder MLPs
BATCH_SIZE = 64 # Number of samples per training batch
EPOCHS = 100 # Total number of training epochs
PATIENCE = 10 # Number of epochs to wait for improvement before early stopping
LEARNING_RATE = 1e-3 # Learning rate for the optimizer
KL_ANNEAL_EPOCHS = 30 # Number of epochs over which to anneal the KL divergence loss weight
GRAD_CLIP_NORM = 5.0 # Maximum norm for gradient clipping
MIN_TRAINING_SIZE = 200 # Minimum number of samples required to train a VAE for a category
SEED = 42 # Random seed for reproducibility
SEQUENCE_LENGTH = 10 # Length of sequences for Recurrent VAEs (RVAEs)
ANOMALY_PCT=1
# --- SIMULATION CONFIG ---
FLASK_API_URL = "http://127.0.0.1:5000/simulate" # URL for the simulation API
DEFAULT_SIM_JOBS = 1000 # Default number of jobs to simulate
DEFAULT_GRANULARITY = 'hour' # Default time granularity for simulation
DEFAULT_BURST_THRESH_SECONDS = 60 # Default threshold in seconds to define burst periods
DEFAULT_NUM_BINS = 3 # Default number of bins for categorization
POISSON_RATES = {'Low': 0.05, 'Mid': 0.5, 'High': 2.0} # Poisson rates for different burst levels in MMPP
P_TRANSITION = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]] # Transition probabilities between burst levels in MMPP
BURST_LEVEL_MAP = { # Mapping of string labels to 0-indexed states for MMPP
    'Low': 0,
    'Mid': 1,
    'High': 2
}

# --- GENERAL CONFIG ---
# (Note: SEED is duplicated, can keep one or ensure consistency if used differently)
# SEED = 42
LOGGING_LEVEL = 'INFO' # Logging verbosity level