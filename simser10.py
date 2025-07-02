import os
import uuid
import threading
import numpy as np
import random
from queue import Queue
from datetime import datetime, timedelta, time # Added timedelta
import logging
from collections import Counter, defaultdict
import glob
import json # Added json import

from flask import Flask, request, jsonify, abort
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
import torch.serialization

torch.serialization.add_safe_globals([StandardScaler])
# Import all configurations from the new config.py file
try:
    from config import (
        DEFAULT_GRANULARITY, WEIGHTS_DIR, HIDDEN_DIMS,
        POISSON_RATES, P_TRANSITION, BURST_LEVEL_MAP,
        MIN_TRAINING_SIZE, DEFAULT_LATENT_DIM # Also import DEFAULT_LATENT_DIM if used
    )
except ImportError:
    # --- MOCKUP CONFIG ---
    WEIGHTS_DIR = 'vae_models'
    HIDDEN_DIMS = [128, 64]
    POISSON_RATES = {'Low': 0.05, 'Mid': 0.5, 'High': 2.0}
    P_TRANSITION = [[0.9, 0.1, 0.0], [0.1, 0.8, 0.1], [0.0, 0.1, 0.9]]
    BURST_LEVEL_MAP = {'Low': 0, 'Mid': 1, 'High': 2}
    MIN_TRAINING_SIZE = 200

# Import the new and updated functions from swf_categorizer3
from swf_utils.swf_categorizer3 import (
    parse_sdsc_sp2_log,
    detect_and_remove_anomalies,
    compute_bin_edges,
    compute_burst_activity_edges,
    label_and_categorize_jobs,
    compute_user_burst_metrics,
    classify_job,
)

# Import the VAE model and helpers from the updated training module
try:
    from vae_training2 import VAE
    # Import preprocessing function from the new module
    from data_preprocessing import load_and_preprocess as vae_load_and_preprocess
    print(f"DEBUG: VAE class imported from: {VAE.__module__}.{VAE.__name__}") # Add this line
except ImportError:
    # ─── MOCKUPS FOR MISSING MODULES (for standalone running) ───────────────
    logging.warning("Using mockup classes for VAE and preprocessing. Please ensure vae_training2.py and data_preprocessing.py are in the python path.")
    class VAE(torch.nn.Module):
        def __init__(self, input_dim, hidden_dims, latent_dim):
            super().__init__()
            self.decoder = torch.nn.Linear(latent_dim, input_dim)
            self.center_bias = torch.zeros(input_dim) # Ensure center_bias is always defined
            self.post_norm = torch.nn.Identity()
        def decode(self, z): return self.decoder(z)
        # Add a placeholder for .features to prevent AttributeError when using model.features
        @property
        def features(self):
            return ['RunTime', 'AllocatedProcessors', 'AverageCPUTimeUsed', 'UsedMemory']
        @features.setter
        def features(self, value):
            pass # Do nothing, just allow it to be set

    def vae_load_and_preprocess(*args, **kwargs):
        mock_features = ['SubmitTime', 'WaitTime', 'RunTime', 'AllocatedProcessors', 'AverageCPUTimeUsed']
        mock_data = torch.randn(100, len(mock_features))
        mock_scaler = StandardScaler()
        mock_scaler.fit(mock_data.numpy()) # Fit for a minimal mock functionality
        mock_original_min_max = {f: {'min': -1.0, 'max': 1.0} for f in mock_features} # Dummy min/max
        return mock_data, mock_scaler, mock_features, mock_original_min_max
# ──────────────────────────────────────────────────────────────────

# ─── LOGGING CONFIG ───────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('JobSim')
logging.getLogger('werkzeug').setLevel(logging.WARNING) # Suppress Flask server logs

# ─── FLASK APP ────────────────────────────────────────────────────────────
app = Flask(__name__)
# Global cache for VAE models and user profiles
vae_models = {}
user_profiles = {}
job_queue = Queue()
# Lock for thread-safe access to the user profiles
profile_lock = threading.Lock()

# Global variable for time-based category distribution
time_category_distribution = {}
granularity_minutes = 10 # Default, will be updated from JSON

# Path to the pre-computed distribution file
TIME_CATEGORY_DIST_FILE = "time_category_distribution.json"

# ─── MMPP GENERATOR (NEW) ─────────────────────────────────────────────
# Define the states of the Markov chain (0: Low, 1: Mid, 2: High)
# and the transition probability matrix
# These are loaded from config.py now.

def generate_mmpp_interarrival_time(current_burst_level_str, num_jobs_to_generate):
    """
    Generates a sequence of inter-arrival times using an MMPP model.
    Uses BURST_LEVEL_MAP to convert string label to 0-indexed state.
    """
    interarrivals = []
    
    # Get the starting state index from the classified burst level string
    # E.g., 'Low' -> 0, 'Mid' -> 1, 'High' -> 2
    current_state_idx = BURST_LEVEL_MAP.get(current_burst_level_str, 0) # Default to 0 (Low) if not found
    
    for _ in range(num_jobs_to_generate):
        # 1. Get the Poisson rate for the current state (using the 0-indexed state)
        # Ensure POISSON_RATES is defined globally or imported from config.py
        current_rate = POISSON_RATES[list(BURST_LEVEL_MAP.keys())[list(BURST_LEVEL_MAP.values()).index(current_state_idx)]]
        
        # 2. Generate an inter-arrival time from a Poisson process (exponential distribution)
        if current_rate > 0:
            interarrival_time = np.random.exponential(scale=1/current_rate)
        else:
            interarrival_time = 0.0 # No arrivals if rate is 0
        
        interarrivals.append(max(0.0, interarrival_time)) # Ensure non-negative
        
        # 3. Transition to the next state based on the probability matrix
        # Ensure P_TRANSITION is defined globally or imported from config.py
        transition_probs = P_TRANSITION[current_state_idx]
        current_state_idx = np.random.choice(len(transition_probs), p=transition_probs)

    return interarrivals

# ─── VAE RELATED FUNCTIONS ────────────────────────────────────────────
def load_all_vaes():
    """
    Loads all trained VAE models from the weights directory into a dictionary.
    """
    logger.info("Loading VAE models and scalers...")
    models = {}
    
    for ckpt_path in glob.glob(os.path.join(WEIGHTS_DIR, "*_vae_ckpt.pth")):
        category = os.path.basename(ckpt_path).replace("_vae_ckpt.pth", "")
        try:
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            
            state = ckpt["model_state"]
            scaler = ckpt["scaler"]
            feats = ckpt["features"]
            latent_d = ckpt["latent_dim"] # Directly get latent_dim from checkpoint
            
            # --- NEW: Load original_min_max from checkpoint ---
            original_min_max = ckpt.get("original_min_max", {}) # Get with default empty dict

            model = VAE(input_dim=len(feats), hidden_dims=HIDDEN_DIMS, latent_dim=latent_d)
            model.load_state_dict(state)
            model.eval()

            models[category] = {
                "model": model,
                "scaler": scaler,
                "latent_dim": latent_d,
                "features": feats,
                "original_min_max": original_min_max # --- NEW: Store it ---
            }
            logger.info(f"Loaded VAE for category '{category}' with input_dim={len(feats)}, latent_dim={latent_d}.")
        except Exception as e:
            logging.error(f"Failed to load VAE for category '{category}': {e}")
            
    return models

def sample_valid_jobs_for_user(model, scaler, latent_d, need, burst_level, category):
    """
    Sample valid jobs from a VAE model using the user's burst level for inter-arrival times.
    """
    jobs = []
    attempts = need * 5 # Allow more attempts for sampling valid jobs
    
    # --- NEW: Use MMPP to generate all inter-arrival times for this batch ---
    interarrival_sequence = generate_mmpp_interarrival_time(burst_level, need)
    interarrival_iterator = iter(interarrival_sequence)
    
    # Retrieve original min/max for this category
    model_info = vae_models.get(category)
    original_min_max = model_info.get('original_min_max', {}) if model_info else {}

    # Check if 'UsedMemory' was constant at zero in the original data for this category
    force_zero_memory = False
    if 'UsedMemory' in original_min_max:
        if original_min_max['UsedMemory']['min'] == 0 and original_min_max['UsedMemory']['max'] == 0:
            force_zero_memory = True

    while len(jobs) < need and attempts > 0:
        batch_size = min(need - len(jobs), 64)
        z = torch.randn(batch_size, latent_d)
        
        with torch.no_grad():
            outp = model.decode(z)
            outp = outp + model.center_bias # Add back the mean of the original data (learned as center_bias)
            real_vals_scaled = model.post_norm(outp).numpy()
            
        real_vals = scaler.inverse_transform(real_vals_scaled)
        
        # --- NEW: Post-processing to enforce original data ranges (clamping) ---
        for i, feature_name in enumerate(model.features): # Iterate through features in the order they were trained
            if feature_name in original_min_max:
                orig_min = original_min_max[feature_name]['min']
                orig_max = original_min_max[feature_name]['max']
                
                # Apply clamping
                real_vals[:, i] = np.clip(real_vals[:, i], orig_min, orig_max)

        for vals in real_vals:
            try:
                # IMPORTANT: Ensure these features match the order in FEATURES list in train_all_vae2.py
                # This order must be consistent with the model.features list
                submit_time_idx      = model.features.index('SubmitTime')
                wait_time_idx        = model.features.index('WaitTime')
                rt_idx               = model.features.index('RunTime')
                pes_idx              = model.features.index('AllocatedProcessors')
                cpu_time_idx         = model.features.index('AverageCPUTimeUsed')
                ram_idx              = model.features.index('UsedMemory') # UsedMemory is kept!

                # Ensure non-negativity and reasonable defaults
                submit_time = max(float(vals[submit_time_idx]), 0.0)
                wait_time = max(float(vals[wait_time_idx]), 0.0)
                length = max(float(vals[rt_idx]), 0.0)
                pes = max(int(round(vals[pes_idx])), 1) # Must be at least 1
                cpu_time = max(float(vals[cpu_time_idx]), 0.0)
                
                # --- CONDITIONAL ZEROING OF UsedMemory ---
                ram = float(vals[ram_idx])
                if force_zero_memory: # If original data had UsedMemory all zeros
                    ram = 0.0
                else: # Otherwise, apply non-negativity to generated RAM
                    ram = max(ram, 0.0) 

            except ValueError as e:
                logger.error(f"Feature index not found in model's features list for category {category}: {e}. Falling back to default.")
                submit_time = 0.0
                wait_time = 0.0
                length = 10.0
                pes = 1
                cpu_time = 10.0
                ram = 0.0 # Ensure ram is 0 if feature missing or constant

            try:
                interarrival_time = next(interarrival_iterator)
            except StopIteration:
                interarrival_time = 0.0
            
            jobs.append({
                "submit_time": submit_time,
                "wait_time": wait_time,
                "length": length if length > 0.0 else cpu_time, # Ensure length is positive
                "pes": pes,
                "cpu_time": cpu_time if cpu_time > 0.0 else length, # Ensure cpu_time is positive
                "ram": ram,
                "interarrival": interarrival_time,
                "category": category,
            })
            
            if len(jobs) == need:
                break
        attempts -= batch_size

    # Fallback to fill remaining jobs if attempts run out
    while len(jobs) < need:
        jobs.append({
            "submit_time": 0.0, "wait_time": 0.0,
            "length": 0.0, "pes": 1, "cpu_time": 0.0, "ram": 0.0, "interarrival": 0.0, "category": category
        })
    
    logger.info(f"[GEN] Generated {len(jobs)}/{need} jobs for category {category}")
    return jobs

# --- USER PROFILE HANDLING ---
def load_user_profiles_and_burst_metrics(file_path):
    """
    Loads user profiles from a CSV file.
    """
    try:
        df = pd.read_csv(file_path)
        # Convert DataFrame to a dictionary for faster lookup
        profiles = df.set_index('UserID').to_dict('index')
        return profiles
    except FileNotFoundError:
        logger.error(f"User profiles file not found at {file_path}. Generating jobs without profile-based burst levels.")
        return {}

def classify_user_burst_level(user_id):
    """
    Classifies a user's burst level based on their historical metrics.
    Returns a string like 'Low', 'Mid', 'High' consistent with BURST_LEVEL_MAP keys.
    """
    with profile_lock:
        profile = user_profiles.get(user_id)
    
    if profile:
        if 'burst_level' in profile:
            # Assume profile['burst_level'] is already 'Low', 'Mid', 'High'
            if profile['burst_level'] in BURST_LEVEL_MAP:
                return profile['burst_level']
        
        # Fallback to a random selection from the keys of BURST_LEVEL_MAP
        return random.choice(list(BURST_LEVEL_MAP.keys()))
    
    return random.choice(list(BURST_LEVEL_MAP.keys())) # If no profile, choose randomly from available levels

def get_category_for_time_slot(day: str, interval_str: str) -> str:
    """
    Selects a job category for a given day and 10-minute interval
    based on historical distribution.
    """
    global time_category_distribution
    day_dist = time_category_distribution.get("category_distribution", {}).get(day)
    
    if not day_dist:
        logger.warning(f"No historical data for day '{day}'. Choosing a random category.")
        return random.choice(list(vae_models.keys())) # Fallback to any loaded VAE category

    interval_data = day_dist.get(interval_str)

    if not interval_data:
        logger.warning(f"No historical data for interval '{interval_str}' on '{day}'. Choosing a random category from day's overall distribution.")
        # Fallback: aggregate all categories for the day and choose
        all_categories_for_day = Counter()
        for _, cats in day_dist.items():
            all_categories_for_day.update(cats)
        
        if not all_categories_for_day:
            return random.choice(list(vae_models.keys())) # Final fallback if day also has no data

        categories, counts = zip(*all_categories_for_day.items())
        chosen_category = random.choices(categories, weights=counts, k=1)[0]
        return chosen_category
    
    categories, counts = zip(*interval_data.items())
    chosen_category = random.choices(categories, weights=counts, k=1)[0]
    return chosen_category

# ─── FLASK ROUTES ─────────────────────────────────────────────────────────
@app.route('/simulate', methods=['POST'])
def simulate_jobs_endpoint():
    """
    (Original endpoint - remains for backward compatibility, though not used by new feature)
    Generates a batch of synthetic jobs for a specified category and user.
    """
    data = request.get_json()
    category = data.get('category')
    job_count = data.get('job_count', 100)
    user_id = data.get('user_id', None)

    if not category:
        abort(400, description="Missing 'category' parameter.")

    logger.info(f"[REQ] Received request for {job_count} jobs in category '{category}' for user '{user_id}'.")

    model_data = vae_models.get(category)
    if not model_data:
        abort(404, description=f"No VAE model found for category '{category}'.")
    
    model = model_data['model']
    scaler = model_data['scaler']
    latent_d = model_data['latent_dim']
    model.features = model_data['features'] # Attach features to the model for sampling

    # Classify the user's burst level
    burst_level = classify_user_burst_level(user_id) if user_id else random.choice(['Low', 'Mid', 'High'])

    valid_jobs = sample_valid_jobs_for_user(
        model=model,
        scaler=scaler,
        latent_d=latent_d,
        need=job_count,
        burst_level=burst_level,
        category=category
    )

    return jsonify(valid_jobs)


@app.route('/simulate_by_time_range', methods=['POST'])
def simulate_jobs_by_time_range_endpoint():
    """
    Generates a batch of synthetic jobs for a specified day and time range,
    distributing jobs proportionally based on historical patterns.
    """
    # FIX: Declare global variables at the very beginning of the function
    global time_category_distribution, granularity_minutes 

    data = request.get_json()
    day = data.get('day') # e.g., "Monday"
    start_time_str = data.get('start_time') # e.g., "09:00"
    end_time_str = data.get('end_time') # e.g., "11:00"
    total_job_count = data.get('total_job_count', 100)
    user_id = data.get('user_id', None)

    if not all([day, start_time_str, end_time_str]):
        abort(400, description="Missing 'day', 'start_time', or 'end_time' parameters.")

    logger.info(f"[REQ] Received request for {total_job_count} jobs for '{day}' from {start_time_str} to {end_time_str} for user '{user_id}'.")

    all_generated_jobs = []
    
    # Parse start and end times
    try:
        start_hour, start_minute = map(int, start_time_str.split(':'))
        end_hour, end_minute = map(int, end_time_str.split(':'))
        
        current_time = datetime(1, 1, 1, start_hour, (start_minute // granularity_minutes) * granularity_minutes)
        end_time = datetime(1, 1, 1, end_hour, (end_minute // granularity_minutes) * granularity_minutes)

    except ValueError:
        abort(400, description="Invalid time format. Use HH:MM (e.g., 09:00).")

    # Get the total historical job count for the specified day and time range
    historical_interval_totals = time_category_distribution.get("interval_total_jobs", {})
    
    total_historical_jobs_in_range = 0
    intervals_in_range = []

    temp_time = current_time
    while temp_time <= end_time:
        interval_str = temp_time.strftime(f'%H:%M')
        intervals_in_range.append(interval_str)
        
        # Add historical count for this interval
        interval_count = historical_interval_totals.get(day, {}).get(interval_str, 0)
        total_historical_jobs_in_range += interval_count
        
        temp_time += timedelta(minutes=granularity_minutes)

    if total_historical_jobs_in_range == 0:
        logger.warning(f"No historical job data found for '{day}' {start_time_str}-{end_time_str}. Distributing jobs evenly.")
        # Fallback: distribute jobs evenly if no historical data for the range
        num_intervals = len(intervals_in_range)
        jobs_per_interval_base = total_job_count // num_intervals if num_intervals > 0 else 0
        jobs_remainder = total_job_count % num_intervals
    else:
        jobs_per_interval_base = 0 # Will be calculated proportionally
        jobs_remainder = 0

    # Iterate through each 10-minute interval in the requested range
    current_time = datetime(1, 1, 1, start_hour, (start_minute // granularity_minutes) * granularity_minutes)
    
    for _ in range(len(intervals_in_range)):
        interval_str = current_time.strftime(f'%H:%M')
        
        jobs_for_this_interval = 0
        if total_historical_jobs_in_range > 0:
            # Proportional distribution
            historical_count_for_interval = historical_interval_totals.get(day, {}).get(interval_str, 0)
            jobs_for_this_interval = int(round(total_job_count * (historical_count_for_interval / total_historical_jobs_in_range)))
        else:
            # Even distribution fallback
            jobs_for_this_interval = jobs_per_interval_base
            if jobs_remainder > 0:
                jobs_for_this_interval += 1
                jobs_remainder -= 1

        if jobs_for_this_interval > 0:
            # Randomly select a category based on the historical distribution for this interval
            selected_category = get_category_for_time_slot(day, interval_str)
            
            model_data = vae_models.get(selected_category)
            if not model_data:
                logger.error(f"No VAE model found for selected category '{selected_category}' for interval {interval_str} on {day}. Skipping generation for this slot.")
                current_time += timedelta(minutes=granularity_minutes)
                continue # Skip this interval if model is missing

            model = model_data['model']
            scaler = model_data['scaler']
            latent_d = model_data['latent_dim']
            model.features = model_data['features'] # Attach features to the model for sampling

            burst_level = classify_user_burst_level(user_id) if user_id else random.choice(['Low', 'Mid', 'High'])

            jobs_generated_in_slot = sample_valid_jobs_for_user(
                model=model,
                scaler=scaler,
                latent_d=latent_d,
                need=jobs_for_this_interval,
                burst_level=burst_level,
                category=selected_category
            )
            all_generated_jobs.extend(jobs_generated_in_slot)
            logger.info(f"[GEN_RANGE] Generated {len(jobs_generated_in_slot)} jobs for {day} {interval_str} (Category: {selected_category})")
        
        current_time += timedelta(minutes=granularity_minutes)
        if current_time > end_time and interval_str == end_time.strftime(f'%H:%M'): # Stop if we processed the last interval
            break

    # If the exact total job count wasn't met due to rounding, add/remove the difference
    if len(all_generated_jobs) < total_job_count:
        needed = total_job_count - len(all_generated_jobs)
        logger.info(f"Padding {needed} jobs to meet total_job_count.")
        # Sample from the last selected category or a random one
        last_category = selected_category if 'selected_category' in locals() else random.choice(list(vae_models.keys()))
        model_data = vae_models.get(last_category)
        if model_data:
            model = model_data['model']
            scaler = model_data['scaler']
            latent_d = model_data['latent_dim']
            model.features = model_data['features']
            burst_level = classify_user_burst_level(user_id) if user_id else random.choice(['Low', 'Mid', 'High'])
            all_generated_jobs.extend(sample_valid_jobs_for_user(model, scaler, latent_d, needed, burst_level, last_category))
        else: # Fallback if even last_category model is missing
            for _ in range(needed):
                all_generated_jobs.append({"length": 0.0, "pes": 1, "cpu_time": 0.0, "ram": 0.0, "interarrival": 0.0, "category": "fallback"})
    elif len(all_generated_jobs) > total_job_count:
        logger.info(f"Trimming {len(all_generated_jobs) - total_job_count} jobs to meet total_job_count.")
        all_generated_jobs = all_generated_jobs[:total_job_count]

    return jsonify(all_generated_jobs)


def run_server():
    """
    Initializes and runs the Flask server.
    """
    global user_profiles, vae_models, time_category_distribution, granularity_minutes
    
    # Load user profiles from a user_profiles.csv file created by swf_categorizer3.py
    with profile_lock:
        user_profiles = load_user_profiles_and_burst_metrics('user_profiles.csv')

    # Load all VAE models ONLY ONCE at startup
    vae_models = load_all_vaes()
    if not vae_models:
        logger.error("No VAE models were loaded. Job generation will fail.")
        # Consider a stronger exit or mock behavior if no models are available

    # Load the time-based category distribution
    if os.path.exists(TIME_CATEGORY_DIST_FILE):
        try:
            with open(TIME_CATEGORY_DIST_FILE, 'r') as f:
                dist_data = json.load(f)
            time_category_distribution = dist_data.get("category_distribution", {})
            granularity_minutes = dist_data.get("granularity_minutes", 10) # Update granularity
            logger.info(f"Loaded time-based category distribution (granularity: {granularity_minutes} min).")
        except Exception as e:
            logger.error(f"Failed to load time-based category distribution from {TIME_CATEGORY_DIST_FILE}: {e}")
            time_category_distribution = {}
    else:
        logger.warning(f"Time-based category distribution file '{TIME_CATEGORY_DIST_FILE}' not found. Time-based simulation will fall back to random categories.")


    logger.info("Starting Flask server...")
    # Use threaded=True for concurrent requests.
    app.run(debug=False, host='0.0.0.0', port=5000, threaded=True)

if __name__ == '__main__':
    run_server()