import os
from pyexpat import features
import uuid
import threading
import numpy as np
import random
from queue import Queue
from datetime import datetime, timedelta, time # Added timedelta, time
import logging
from collections import Counter, defaultdict
import glob
import json 
import sys # Added sys for path modification

from flask import Flask, request, jsonify, abort
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import torch.serialization
from sklearn.preprocessing import PowerTransformer # Ensure PowerTransformer is importable for loading models
from compare_Rvae import tune_temperatures
# Add the project's root directory to the Python path
# This ensures that `config` and `swf_utils` can be imported regardless of the current working directory
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    parent_dir = os.path.abspath(os.path.join(script_dir, os.pardir))
    if parent_dir not in sys.path:
        sys.path.insert(0, parent_dir)
except NameError:
    pass

# Ensure StandardScaler and PowerTransformer are added to safe globals for torch.load
torch.serialization.add_safe_globals([StandardScaler, np._core.multiarray._reconstruct, np.ndarray]) # Added numpy globals
torch.serialization.add_safe_globals([PowerTransformer]) # Added PowerTransformer
# Add numpy._core.multiarray._reconstruct and np.ndarray for numpy objects in pickles
torch.serialization.add_safe_globals([np.dtype])
try:
    torch.serialization.add_safe_globals([np._core.multiarray._reconstruct, np.ndarray])
except AttributeError:
    # Older numpy versions might not have _reconstruct in _core.multiarray
    # Fallback for broader compatibility, though less precise
    logging.warning("Could not add np._core.multiarray._reconstruct to safe globals. Ensure numpy is updated if issues arise.")


# Import all configurations from the new config.py file
try:
    from config import (
        DEFAULT_GRANULARITY, WEIGHTS_DIR, HIDDEN_DIMS,
        POISSON_RATES, P_TRANSITION, BURST_LEVEL_MAP,
        MIN_TRAINING_SIZE, DEFAULT_LATENT_DIM, SWF_PATH,
        EXCEL_SUBSET_PATH, SEQUENCE_LENGTH # Ensure SEQUENCE_LENGTH is imported
    )
except ImportError:
    # --- MOCKUP CONFIG for standalone running if config.py is missing ---
    logging.warning("config.py not found or incomplete. Using mockup configurations.")
    WEIGHTS_DIR = 'vae_models'
    HIDDEN_DIMS = [128, 64]
    POISSON_RATES = {'Low': 0.05, 'Mid': 0.5, 'High': 2.0}
    P_TRANSITION = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]
    BURST_LEVEL_MAP = {'Low': 0, 'Mid': 1, 'High': 2}
    MIN_TRAINING_SIZE = 200
    DEFAULT_GRANULARITY = 'hour'
    DEFAULT_LATENT_DIM = 64
    SWF_PATH = 'path/to/your/SDSC-SP2-1998-4.2-cln.swf' # Placeholder, user must update
    EXCEL_SUBSET_PATH = 'subsets/categorized_subsets.xlsx'
    SEQUENCE_LENGTH = 10
    

# Import the new and updated functions from swf_categorizer3
# We import its main function to call it if needed
try:
    from swf_utils.swf_categorizer3 import (
        compute_user_burst_metrics, # Still used by load_user_profiles_and_burst_metrics
        main as swf_categorizer_main # Import the main function
    )
except ImportError:
    logging.warning("swf_utils/swf_categorizer3.py not found. Categorization setup will be skipped.")
    swf_categorizer_main = None

# Import the VAE model and helpers from the updated training module
# We import its main function to call it if needed
try:
    from vae_training3 import VAE
    from train_all_vae2 import train_all_vaes as train_all_vaes # Import the main function
except ImportError:
    logging.warning("vae_training3.py or train_all_vae2.py not found. VAE training setup will be skipped.")
    VAE = None
    train_all_vaes = None


# ─── LOGGING SETUP ────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('JobSimulator')

# ─── GLOBAL STATE ─────────────────────────────────────────────────────────
app = Flask(__name__)
vae_models = {}
user_profiles = {}
category_user_counts = {}
job_queue = Queue() # For future background job processing if needed
time_category_distribution = {}
granularity_minutes = 10 # Default from swf_categorizer3.py
temperature_map = {}
# Paths for generated data files (consistent with swf_categorizer3.py)
USER_PROFILES_FILE = "user_profiles.csv"
TIME_CATEGORY_DIST_FILE = "time_category_distribution.json"

# Lock for user profile access (if modified concurrently)
profile_lock = threading.Lock()


# ─── MMPP GENERATOR ───────────────────────────────────────────────────────
# (Keep this as is, it's a utility for generating inter-arrival times)
def generate_mmpp_interarrival_time(num_jobs_to_generate, current_burst_level):
    """
    Simulates inter-arrival times using a Markov Modulated Poisson Process (MMPP).
    """
    inter_arrival_times = []
    # Map burst level string to its index in P_TRANSITION
    current_state_idx = BURST_LEVEL_MAP.get(current_burst_level, 0) # Default to 'Low' (index 0)

    for _ in range(num_jobs_to_generate):
        # Get the Poisson rate for the current state
        rate = POISSON_RATES.get(list(BURST_LEVEL_MAP.keys())[current_state_idx])

        # Generate inter-arrival time from exponential distribution (Poisson process)
        # Handle cases where rate might be zero or negative to prevent errors
        if rate <= 0:
            inter_arrival_time = 0.0 # Or some sensible default if no arrivals expected
        else:
            inter_arrival_time = np.random.exponential(1 / rate)
        
        inter_arrival_times.append(inter_arrival_time)

        # Transition to the next state based on transition probabilities
        next_state_idx = np.random.choice(
            len(P_TRANSITION), 
            p=P_TRANSITION[current_state_idx]
        )
        current_state_idx = next_state_idx
    
    return inter_arrival_times


# ─── VAE MODEL LOADING ────────────────────────────────────────────────────
# (Keep this as is, it's responsible for loading the actual models into memory)
def load_all_vaes():
    loaded_models = {}
    for ckpt_path in glob.glob(os.path.join(WEIGHTS_DIR, '*_vae_ckpt.pth')):
        cat = os.path.basename(ckpt_path).replace('_vae_ckpt.pth','')
        if cat.startswith("category_"):
            cat = cat[len("category_"):]
        try:
            ckpt = torch.load(ckpt_path, map_location='cpu', weights_only=False)
            # sequence length & feature list must be saved in ckpt
            seq_len = ckpt.get('sequence_length', 1)
            features = ckpt['features']
            latent = ckpt.get('latent_dim', DEFAULT_LATENT_DIM)
            scaler = ckpt['scaler']
            pt = ckpt.get('power_transformer', None)
            orig_mm = ckpt.get('original_min_max', {})

            model = VAE(input_dim=seq_len*len(features), latent_dim=latent)
            state_key = 'model_state_dict' if 'model_state_dict' in ckpt else 'model_state'
            model.load_state_dict(ckpt[state_key])
            model.eval()

            loaded_models[cat] = {
                'model': model,
                'scaler': scaler,
                'power_transformer': pt,
                'features': features,
                'sequence_length': seq_len,
                'latent_dim': latent,
                'original_min_max': orig_mm
            }
            logger.info(f"Loaded VAE '{cat}' (seq_len={seq_len}, feats={len(features)})")
        except Exception as e:
            logger.error(f"Error loading VAE '{cat}': {e}")
    return loaded_models

# ─── USER PROFILE HANDLING ────────────────────────────────────────────────
# (This still loads the file, but the _interactive_setup ensures the file exists)
def load_user_profiles_and_burst_metrics(file_path):
    """
    Loads user profiles from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        # Convert DataFrame to a dictionary for quick lookup: {user_id: burst_level_string}
        # Assuming the CSV has 'user_id' and 'classified_burst_level' columns
        return df.set_index('UserID')['BurstLevel'].to_dict()
    except FileNotFoundError:
        logger.warning(f"User profiles file '{file_path}' not found. User burst levels will be randomized.")
        return {}
    except Exception as e:
        logger.error(f"Error loading user profiles from '{file_path}': {e}")
        return {}

def classify_user_burst_level(user_id):
    """
    Retrieves a user's burst level or assigns a random one if unknown.
    """
    with profile_lock:
        return user_profiles.get(user_id, random.choice(list(BURST_LEVEL_MAP.keys())))

# ─── CATEGORY SELECTION FOR TIME SLOT ─────────────────────────────────────
# (This still uses the loaded distribution, but _interactive_setup ensures it's available)
def get_category_for_time_slot(day, current_time_str):
    """
    Selects a job category for a given 10‑minute interval and day,
    based on historical distribution (or a wrapped distribution if only flat counts are present).
    """
    # 1. Parse time string
    try:
        current_dt = datetime.strptime(current_time_str, '%H:%M')
    except ValueError:
        logger.error(f"Invalid time format: {current_time_str}. Expected HH:MM.")
        return None

    # 2. Round down to nearest granularity bucket
    minute_bucket = (current_dt.minute // granularity_minutes) * granularity_minutes
    interval_start = current_dt.replace(minute=minute_bucket, second=0, microsecond=0)
    interval_str = interval_start.strftime('%H:%M')
    day_key = day.capitalize()

    # 3. Look up raw data for this interval
    raw = time_category_distribution.get(day_key, {}).get(interval_str)
    if raw is None:
        logger.warning(f"No historical data for {day_key} {interval_str}. Choosing random category.")
        return random.choice(list(vae_models.keys())) if vae_models else None

    # 4. Build a {category: count} map
    if isinstance(raw, (int, float)):
        # Flat count → uniform weighting across all categories
        total = int(raw)
        cats = list(vae_models.keys())
        dist_for_interval = {c: 1 for c in cats}
        logger.info(f"Wrapped flat count={total} into uniform category_counts for {day_key} {interval_str}.")
    elif isinstance(raw, dict):
        # Either nested under "category_counts" or already a direct map
        dist_for_interval = raw.get("category_counts", raw)
        if not dist_for_interval:
            logger.warning(f"No category_counts for {day_key} {interval_str}. Choosing random category.")
            return random.choice(list(vae_models.keys())) if vae_models else None
    else:
        logger.warning(f"Unexpected data type for {day_key} {interval_str}. Choosing random category.")
        return random.choice(list(vae_models.keys())) if vae_models else None

    # 5. Sample one category according to the distribution
    categories, counts = zip(*dist_for_interval.items())
    total_count = sum(counts)
    if total_count > 0:
        probabilities = [count / total_count for count in counts]
        return np.random.choice(categories, p=probabilities)
    else:
        logger.warning(f"Zero total_count at {day_key} {interval_str}. Choosing random category.")
        return random.choice(list(vae_models.keys())) if vae_models else None


# ─── CORE JOB SAMPLING LOGIC ──────────────────────────────────────────────
# (This is the most critical function and has been updated for sequence VAEs)
def sample_valid_jobs_for_user(model_info, need, burst_level, category):    # Extract metadata
    m = model_info['model']
    scaler = model_info['scaler']
    pt = model_info['power_transformer']
    feats = model_info['features']
    seq_len = model_info['sequence_length']
    latent = model_info['latent_dim']
    orig_mm = model_info['original_min_max']

    T = float(temperature_map.get(category, 1.0))
    # how many sequences we need
    num_seq = max(1, int(np.ceil(need/seq_len)))
    jobs = []
    ia_times = generate_mmpp_interarrival_time(need, burst_level)
    ia_iter = iter(ia_times)

    for _ in range(num_seq):
        z = torch.randn(1, latent) * T  # Sample from a normal distribution
        with torch.no_grad():
            out = m.decode(z, seq_len).cpu().numpy().reshape(seq_len, len(feats))
        inv = scaler.inverse_transform(out)
        if pt is not None:
            temp_df = pd.DataFrame(inv, columns=feats)
            pt_cols = list(pt.feature_names_in_)
            common = [c for c in pt_cols if c in feats]
            if common:
                # Create DataFrame in the original PT order
                pt_input = pd.DataFrame(temp_df[common].values, columns=pt_cols)
                pt_inv = pt.inverse_transform(pt_input)
                temp_df[common] = pd.DataFrame(pt_inv, columns=common)
            inv = temp_df.values
        # clipping
        for i,f in enumerate(feats):
            mm = orig_mm.get(f)
            if mm:
                inv[:,i] = np.clip(inv[:,i], mm['min'], mm['max'])
        # build job dicts
        for row in inv:
            job = {feats[i]: float(row[i]) for i in range(len(feats))}
            job['interarrival'] = next(ia_iter, 0.0)
            job['category'] = category
            if 'AllocatedProcessors' in job:
                # round half-to-even by default; cast to int
                job['AllocatedProcessors'] = int(round(job['AllocatedProcessors']))
            jobs.append(job)
    jobs = jobs[:need]
    # ← NEW: assign each job a real UserID sampled from history for this category
    if category in category_user_counts:
        users, probs = category_user_counts[category]
        logger.debug(f"Sampling user for category {category} from {len(users)} candidates")
        for job in jobs:
            job['user_id'] = np.random.choice(users, p=probs)
    else:
        logger.warning(f"No usercounts for category {category}, leaving user_id blank")        
    return jobs


# ─── FLASK ENDPOINTS ──────────────────────────────────────────────────────
@app.route('/simulate', methods=['POST'])
def simulate_jobs_endpoint():
    """
    Endpoint to simulate a specific number of jobs for a given category.
    """
    data = request.get_json()
    category = data.get('category')
    job_count = data.get('job_count')
    user_id = data.get('user_id') # Optional user ID

    if not category or not job_count:
        abort(400, description="Missing 'category' or 'job_count' in request.")

    logger.info(f"Received request for {job_count} jobs in category '{category}' for user '{user_id}'.")

    model_data = vae_models.get(category)
    if not model_data:
        abort(404, description=f"No VAE model found for category '{category}'.")
    
    # Classify the user's burst level
    burst_level = classify_user_burst_level(user_id) if user_id else random.choice(list(BURST_LEVEL_MAP.keys()))

    valid_jobs = sample_valid_jobs_for_user(
        model_info=model_data,
        need=job_count,
        burst_level=burst_level,
        category=category
    )

    return jsonify(valid_jobs)

@app.route('/simulate_by_time_range', methods=['POST'])
def simulate_jobs_by_time_range_endpoint():
    """
    Endpoint to simulate a total number of jobs distributed over a time range
    and day based on historical distribution.
    """
    data = request.get_json()
    day = data.get('day')
    start_time_str = data.get('start_time')  # e.g., "09:00"
    end_time_str = data.get('end_time')      # e.g., "17:00"
    total_job_count = data.get('total_job_count')
    user_id = data.get('user_id')            # Optional user ID

    if not all([day, start_time_str, end_time_str, total_job_count is not None]):
        abort(400, description="Missing 'day', 'start_time', 'end_time', or 'total_job_count' in request.")

    logger.info(f"Received request for {total_job_count} jobs on {day} from {start_time_str} to {end_time_str} for user '{user_id}'.")

    # Parse times
    try:
        start_dt = datetime.strptime(start_time_str, '%H:%M')
        end_dt   = datetime.strptime(end_time_str,   '%H:%M')
    except ValueError:
        abort(400, description="Invalid time format. Use HH:MM.")

    if start_dt >= end_dt:
        abort(400, description="start_time must be before end_time.")

    # Classify the user's burst level once for the entire range
    burst_level = classify_user_burst_level(user_id) if user_id else random.choice(list(BURST_LEVEL_MAP.keys()))

    # 1️⃣ Compute total historical jobs in range
    total_hist_jobs_in_range = 0
    temp_dt = start_dt
    day_key = day.capitalize()
    while temp_dt < end_dt:
        interval_str = temp_dt.strftime('%H:%M')
        raw = time_category_distribution.get(day_key, {}).get(interval_str)

        if raw is None:
            hist_count = 0
        elif isinstance(raw, dict):
            counts_map = raw.get("category_counts", raw)
            hist_count = sum(counts_map.values())
        else:
            hist_count = int(raw)

        total_hist_jobs_in_range += hist_count
        temp_dt += timedelta(minutes=granularity_minutes)

    # 2️⃣ Distribute jobs across intervals
    generated_jobs = []
    remaining = total_job_count
    current_dt = start_dt

    # Pre-calc number of intervals for fallback
    total_intervals = int((end_dt - start_dt).total_seconds() / 60 / granularity_minutes) or 1

    while current_dt < end_dt and remaining > 0:
        interval_str = current_dt.strftime('%H:%M')

        # Determine jobs for this interval
        if total_hist_jobs_in_range > 0:
            raw = time_category_distribution.get(day_key, {}).get(interval_str)
            if raw is None:
                hist_interval = 0
            elif isinstance(raw, dict):
                counts_map = raw.get("category_counts", raw)
                hist_interval = sum(counts_map.values())
            else:
                hist_interval = int(raw)

            share = (hist_interval / total_hist_jobs_in_range) * total_job_count
            jobs_this = min(int(round(share)), remaining)
        else:
            # Even distribution fallback
            jobs_this = remaining // total_intervals
            # Last interval takes any remainder
            if current_dt + timedelta(minutes=granularity_minutes) >= end_dt:
                jobs_this = remaining

        jobs_this = max(0, jobs_this)

        # If there are jobs to generate, split across all categories in this interval
        if jobs_this > 0:
            raw = time_category_distribution.get(day_key, {}).get(interval_str)
            if isinstance(raw, dict) and raw:
                counts_map = raw.get("category_counts", raw)
            else:
                counts_map = {c: 1 for c in vae_models.keys()}

            total_hist = sum(counts_map.values())
            if total_hist > 0:
                for cat, hist_count in counts_map.items():
                    if remaining <= 0:
                        break
                    n_cat = int(round(jobs_this * hist_count / total_hist))
                    if n_cat <= 0:
                        continue

                    model_data = vae_models.get(cat)
                    if not model_data:
                        logger.warning(f"No model for category '{cat}' at {day_key} {interval_str}. Skipping.")
                        continue

                    logger.info(f"Generating {n_cat} jobs for {day_key} {interval_str} in category '{cat}'")
                    seq = sample_valid_jobs_for_user(
                        model_info=model_data,
                        need=n_cat,
                        burst_level=burst_level,
                        category=cat
                    )
                    generated_jobs.extend(seq)
                    remaining -= len(seq)

        current_dt += timedelta(minutes=granularity_minutes)

    # Final adjustment
    if len(generated_jobs) > total_job_count:
        generated_jobs = generated_jobs[:total_job_count]
    elif len(generated_jobs) < total_job_count:
        logger.warning(f"Generated only {len(generated_jobs)} of requested {total_job_count} jobs.")

    return jsonify(generated_jobs)



# ─── INTERACTIVE SETUP FUNCTION ──────────────────────────────────────────
def _interactive_setup():
    """
    Guides the user through setting up necessary data files and training models.
    """
    global user_profiles, time_category_distribution, granularity_minutes, vae_models
    
    logger.info("\n--- Job Simulator Setup ---")
    logger.info("This process ensures all necessary data and models are prepared.")

    # 1. Categorization and User Profiles Setup
    needs_categorization = False
    if not os.path.exists(USER_PROFILES_FILE) or not os.path.exists(TIME_CATEGORY_DIST_FILE):
        logger.warning(f"Data files ('{USER_PROFILES_FILE}', '{TIME_CATEGORY_DIST_FILE}') not found.")
        needs_categorization = True
    
    if needs_categorization:
        if swf_categorizer_main:
            response = input("Do you want to run the data categorization and create user profiles/time distributions? (y/n): ").lower()
            if response == 'y':
                if not os.path.exists(SWF_PATH) or not os.path.isfile(SWF_PATH):
                    logger.error(f"SWF log file not found at configured path: {SWF_PATH}")
                    logger.error("Please update SWF_PATH in config.py or provide a valid path.")
                    # Abort or fallback to limited functionality
                    sys.exit("SWF log file missing. Cannot run categorization.")
                
                logger.info("Running SWF data categorization...")
                try:
                    swf_categorizer_main() # Call the main function of swf_categorizer3.py
                    logger.info("SWF data categorization complete.")
                except Exception as e:
                    logger.error(f"Error during SWF data categorization: {e}")
                    sys.exit("Categorization failed. Exiting.")
            else:
                logger.warning("Skipping data categorization. Simulator functionality may be limited without historical data.")
        else:
            logger.error("`swf_categorizer3.py`'s main function not imported. Cannot run categorization.")
    else:
        response = input("Data files already exist. Do you want to re-run data categorization? (y/n): ").lower()
        if response == 'y':
            if swf_categorizer_main:
                logger.info("Re-running SWF data categorization...")
                try:
                    swf_categorizer_main()
                    logger.info("SWF data categorization complete.")
                except Exception as e:
                    logger.error(f"Error during SWF data categorization: {e}")
                    sys.exit("Categorization failed. Exiting.")
            else:
                 logger.error("`swf_categorizer3.py`'s main function not imported. Cannot re-run categorization.")
        else:
            logger.info("Using existing data files.")

    # Load the generated/existing data files
    with profile_lock:
        user_profiles = load_user_profiles_and_burst_metrics(USER_PROFILES_FILE)
    
    if os.path.exists(TIME_CATEGORY_DIST_FILE):
        try:
            with open(TIME_CATEGORY_DIST_FILE, 'r') as f:
                dist_data = json.load(f)
            time_category_distribution = dist_data.get("category_distribution", {})
            granularity_minutes = dist_data.get("granularity_minutes", 10)
            # ← NEW: load the real-user counts per category
            raw_counts = dist_data.get("category_user_counts", {})
            # build sampling arrays
            for cat, u2c in raw_counts.items():
                users = list(u2c.keys())
                counts = np.array(list(u2c.values()), dtype=float)
                probs = counts / counts.sum() if counts.sum()>0 else np.ones_like(counts)/len(counts)
                category_user_counts[cat] = (users, probs)
            logger.info(f"Loaded user‐count distributions for {len(category_user_counts)} categories: "
            f"{list(category_user_counts.keys())}")
            logger.info(f"Loaded time-based category distribution (granularity: {granularity_minutes} min).")
        except Exception as e:
            logger.error(f"Failed to load time-based category distribution: {e}. Time-based simulation will fall back to random categories.")
            time_category_distribution = {}
    else:
        logger.warning(f"Time-based category distribution file '{TIME_CATEGORY_DIST_FILE}' not found. Time-based simulation will fall back to random categories.")


    # 2. VAE Model Training Setup
    needs_vae_training = False
    if not os.path.exists(WEIGHTS_DIR) or not glob.glob(os.path.join(WEIGHTS_DIR, '*.pth')):
        logger.warning(f"No VAE models found in '{WEIGHTS_DIR}'.")
        needs_vae_training = True
    
    if needs_vae_training:
        if train_all_vaes:
            response = input("Do you want to train the VAE models? This can take a while. (y/n): ").lower()
            if response == 'y':
                logger.info("Starting VAE model training...")
                try:
                    train_all_vaes() # Call the main function of train_all_vae2.py
                    logger.info("VAE model training complete.")
                except Exception as e:
                    logger.error(f"Error during VAE model training: {e}")
                    sys.exit("VAE training failed. Exiting.")
            else:
                logger.warning("Skipping VAE model training. Simulator cannot generate jobs.")
                sys.exit("Cannot proceed without VAE models. Exiting.") # Cannot function without models
        else:
            logger.error("`train_all_vae2.py`'s main function not imported. Cannot train VAE models.")
            sys.exit("Cannot proceed without VAE models. Exiting.")
    else:
        response = input("VAE models already exist. Do you want to re-train VAE models? (y/n): ").lower()
        if response == 'y':
            if train_all_vaes:
                logger.info("Re-training VAE models...")
                try:
                    train_all_vaes()
                    logger.info("VAE model training complete.")
                except Exception as e:
                    logger.error(f"Error during VAE model training: {e}")
                    sys.exit("VAE training failed. Exiting.")
            else:
                logger.error("`train_all_vae2.py`'s main function not imported. Cannot re-train VAE models.")
                sys.exit("Cannot proceed without VAE models. Exiting.")
        else:
            logger.info("Using existing VAE models.")

    # Attempt to load VAE models regardless (they might have been trained or already existed)
    global vae_models
    vae_models = load_all_vaes()
    if not vae_models:
        logger.error("No VAE models available after setup. Simulator will not be able to generate jobs.")
        sys.exit("No VAE models loaded. Exiting.")

    logger.info("Tuning temperatures for each VAE model (this may take a moment)...")
    temperature_map = tune_temperatures(vae_models)
    logger.info(f"Tuned temperatures: {temperature_map}")


    logger.info("Job simulator setup complete. Starting Flask server...")


# ─── SERVER INITIALIZATION ────────────────────────────────────────────────
def run_server():
    """
    Initializes and runs the Flask server.
    """
    # Run the interactive setup first
    _interactive_setup()

    logger.info("Starting Flask server...")
    # Use threaded=True for concurrent requests. Set host to '0.0.0.0' to be accessible externally.
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    run_server()