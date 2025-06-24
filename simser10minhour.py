import os
import uuid
import threading
import numpy as np
from queue import Queue
from datetime import datetime, time, timezone
import logging
from collections import Counter, defaultdict

from flask import Flask, request, jsonify, abort
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler

# Import the new and updated functions
from swf_utils.swf_categorizer2 import (
    parse_sdsc_sp2_log,
    detect_and_remove_anomalies,
    compute_bin_edges,
    compute_burst_activity_edges,
    label_and_categorize_jobs, 
    compute_user_burst_metrics,
    calculate_interval_burstiness,
    determine_burstiness_thresholds,
    classify_job
)

# Make sure your vae_training.py and train_all_vaes.py are accessible
try:
    from vae_training import VAE, set_seed, HIDDEN_DIMS
    from train_all_vae import train_all_vaes
except ImportError:
    # ─── MOCKUPS FOR MISSING MODULES (for standalone running) ───────────────
    class VAE(torch.nn.Module):
        def __init__(self, input_dim, hidden_dims, latent_dim):
            super().__init__()
            self.encoder = torch.nn.Linear(input_dim, latent_dim)
            self.decoder = torch.nn.Linear(latent_dim, input_dim)
            self.center_bias = torch.nn.Parameter(torch.zeros(input_dim))
        def post_norm(self, x): return x
    def set_seed(s): pass
    def train_all_vaes(): print("Mock training function called.")
    HIDDEN_DIMS = [128, 64]
    # ─────────────────────────────────────────────────────────────────────────

# ─── CONFIG ────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
SWF_PATH = "/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf"
ANOMALY_PCT = 1.0
BURST_THRESHOLD_SECONDS = 60 # Max seconds between jobs to be in the same burst
SUBSETS_DIR = os.path.join(BASE_DIR, "subsets")
WEIGHTS_DIR = os.path.join(BASE_DIR, "vae_models")
DEFAULT_GRAN = 600 # 10 minutes in seconds

os.makedirs(SUBSETS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─── INIT FLASK APP ────────────────────────────────────────────────────────
app = Flask(__name__)
set_seed(42)

job_queue = Queue()
job_status = {}  # job_id -> {'status': ..., 'result': ...}

# Global variables for user analysis
user_profiles = {}  # UserID -> profile data
user_burstiness_edges = []

# ─── 1) PARSE SWF & COMPUTE GLOBAL BIN EDGES ──────────────────────────────
logging.info("[INIT] Parsing SWF and computing global bin edges...")
df_all = parse_sdsc_sp2_log(SWF_PATH)
_, df_clean = detect_and_remove_anomalies(df_all, ANOMALY_PCT / 100.0)

# Compute VAEs' runtime/cpu edges
RT_EDGES, CPU_EDGES = compute_bin_edges(df_clean)
logging.info(f"[INIT] Global Runtime edges: {RT_EDGES}")
logging.info(f"[INIT] Global CPU-util edges: {CPU_EDGES}")

# Label all jobs in the clean dataset with their category
df_clean = label_and_categorize_jobs(df_clean, RT_EDGES, CPU_EDGES)

# Add weekday and time of day information
df_clean['Weekday'] = df_clean['SubmitDateTime'].dt.weekday
df_clean['SubmitTimeOfDay'] = df_clean['SubmitDateTime'].dt.strftime('%H:%M:%S')

# Compute user-activity (burstiness) edges
BURST_ACTIVITY_EDGES = compute_burst_activity_edges(df_clean, BURST_THRESHOLD_SECONDS)
logging.info(f"[INIT] Global Burst Activity edges: {BURST_ACTIVITY_EDGES}")

# Calculate burstiness metrics for 10-minute and 1-hour intervals
logging.info("[INIT] Calculating burstiness metrics for intervals...")
df_10min_metrics = calculate_interval_burstiness(df_clean, 10,BURST_THRESHOLD_SECONDS)
df_1hour_metrics = calculate_interval_burstiness(df_clean, 60,BURST_THRESHOLD_SECONDS)

# Determine burstiness thresholds for intervals
thresholds_10min = determine_burstiness_thresholds(df_10min_metrics)
thresholds_1hour = determine_burstiness_thresholds(df_1hour_metrics)

logging.info(f"[INIT] 10-minute burstiness thresholds: {thresholds_10min}")
logging.info(f"[INIT] 1-hour burstiness thresholds: {thresholds_1hour}")

# ─── 2) USER PROFILE ANALYSIS ──────────────────────────────────────────────
def analyze_user_profiles():
    """Analyze user behavior patterns and burstiness levels."""
    global user_profiles, user_burstiness_edges
    
    logging.info("[INIT] Analyzing user profiles...")
    
    # Compute user burstiness metrics
    user_max_bursts = compute_user_burst_metrics(df_clean, BURST_THRESHOLD_SECONDS)
    user_burstiness_edges = BURST_ACTIVITY_EDGES
    
    # Create user profiles
    for user_id in df_clean['UserID'].unique():
        user_data = df_clean[df_clean['UserID'] == user_id].copy()
        
        # Classify user burstiness level
        max_burst = user_max_bursts.get(user_id, 1)
        burst_level = classify_burstiness(max_burst, user_burstiness_edges)
        
        # Analyze job categories by time intervals (hour-based for simplicity)
        user_data['Hour'] = user_data['SubmitDateTime'].dt.hour
        category_by_hour = {}
        interarrival_by_hour = {}
        
        for hour in range(24):
            hour_jobs = user_data[user_data['Hour'] == hour]
            if not hour_jobs.empty:
                # Most common categories in this hour
                cat_counts = hour_jobs['Category'].value_counts()
                category_by_hour[hour] = cat_counts.to_dict()
                
                # Average interarrival time in this hour
                avg_interarrival = hour_jobs['Interarrival'].mean()
                interarrival_by_hour[hour] = avg_interarrival
        
        # Store user profile
        user_profiles[user_id] = {
            'burst_level': burst_level,
            'max_burst_size': max_burst,
            'total_jobs': len(user_data),
            'category_by_hour': category_by_hour,
            'interarrival_by_hour': interarrival_by_hour,
            'active_weekdays': user_data['Weekday'].unique().tolist(),
            'primary_categories': user_data['Category'].value_counts().head(3).to_dict()
        }
    
    logging.info(f"[INIT] Analyzed {len(user_profiles)} user profiles")
    
    # Log burstiness distribution
    burst_dist = Counter([profile['burst_level'] for profile in user_profiles.values()])
    logging.info(f"[INIT] Burstiness distribution: {dict(burst_dist)}")

def convert_numpy_types(obj):
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    return obj


def classify_burstiness(burst_size, edges):
    """Classify burstiness level based on edges."""
    if burst_size <= edges[1]:
        return 'Low'
    elif burst_size <= edges[2]:
        return 'Mid'
    else:
        return 'High'

# ─── 3) EXPORT WEEKDAY DATA FOR VAE TRAINING ───────────────────────────────
def export_weekday_data():
    """Export clean data split by weekday for VAE training."""
    for weekday in range(7):
        weekday_df = df_clean[df_clean['Weekday'] == weekday].copy()
        if not weekday_df.empty:
            output_path = os.path.join(SUBSETS_DIR, f"weekday_{weekday}.xlsx")
            weekday_df.to_excel(output_path, index=False)
            logging.info(f"[EXPORT] Exported {len(weekday_df)} jobs for weekday {weekday}")

# ─── 4) PROMPT TO RETRAIN VAEs ──────────────────────────────────────────────
def export_and_train_all():
    export_weekday_data()
    train_all_vaes()

def prompt_and_train():
    if os.getenv("RETRAIN_VAES", "false").lower() == "true":
        logging.info("[TRAIN] Exporting subsets & retraining VAEs…")
        export_and_train_all()
        logging.info("[TRAIN] Retraining complete.")
    else:
        logging.info("[TRAIN] Skipping retraining; using existing checkpoints.")

prompt_and_train()
analyze_user_profiles()

# ─── 5) ENHANCED JOB SAMPLING WITH USER CONTEXT ────────────────────────────
def sample_valid_jobs_for_user(model, scaler, latent_d, need, base_interarrival, burst_level):
    """Sample valid jobs from VAE model with user-specific adjustments."""
    jobs = []
    attempts = need * 3
    
    # Adjust interarrival based on burst level
    if burst_level == 'High':
        interarrival_factor = 0.5  # Shorter intervals for bursty users
    elif burst_level == 'Mid':
        interarrival_factor = 0.75
    else:
        interarrival_factor = 1.0
    
    adjusted_interarrival = base_interarrival * interarrival_factor
    
    while len(jobs) < need and attempts > 0:
        batch_size = min(need - len(jobs), 64)
        z = torch.randn(batch_size, latent_d)
        with torch.no_grad():
            outp = model.decoder(z) + model.center_bias
            outp = model.post_norm(outp).numpy()
        real_vals = scaler.inverse_transform(outp)
        
        for vals in real_vals:
            length = max(float(vals[1]), 0.0)
            cpu_time = max(float(vals[3]), 0.0)
            if length == 0.0 and cpu_time == 0.0:
                continue
            pes = max(int(round(vals[2])), 1)
            ram = max(float(vals[4]), 0.0)
            
            # Add some randomness to interarrival for burst patterns
            if burst_level == 'High':
                # For high burst users, add more variation
                ia_variation = np.random.exponential(adjusted_interarrival)
            else:
                ia_variation = max(adjusted_interarrival * np.random.uniform(0.5, 1.5), 0.0)
            
            jobs.append({
                "length": length if length > 0.0 else cpu_time,
                "pes": pes,
                "cpu_time": cpu_time if cpu_time > 0.0 else length,
                "ram": ram,
                "interarrival": ia_variation
            })
            if len(jobs) == need:
                break
        attempts -= batch_size

    # Pad with zero-length jobs if needed
    while len(jobs) < need:
        jobs.append({
            "length": 0.0,
            "pes": 1,
            "cpu_time": 0.0,
            "ram": 0.0,
            "interarrival": max(adjusted_interarrival, 0.0)
        })
    return jobs

# ─── 6) USER-BASED JOB GENERATION ──────────────────────────────────────────
def get_active_users_in_interval(start_dt, end_dt, weekday):
    """Get users who were historically active in the given time interval."""
    start_time = start_dt.time()
    end_time = end_dt.time()
    
    # Find users active in this time window on this weekday
    df_interval = df_clean[
        (df_clean['Weekday'] == weekday) &
        (df_clean['SubmitDateTime'].dt.time >= start_time) &
        (df_clean['SubmitDateTime'].dt.time < end_time)
    ]
    
    return df_interval['UserID'].unique()

def calculate_user_jobs_in_interval(user_id, start_dt, end_dt, weekday):
    """Calculate how many jobs a user should generate based on historical patterns and burstiness."""
    if user_id not in user_profiles:
        return 0, []
    
    profile = user_profiles[user_id]
    hour = start_dt.hour
    
    # Get historical job count for this user in similar time intervals
    historical_data = df_clean[
        (df_clean['UserID'] == user_id) &
        (df_clean['Weekday'] == weekday) &
        (df_clean['SubmitDateTime'].dt.time >= start_dt.time()) &
        (df_clean['SubmitDateTime'].dt.time < end_dt.time())
    ]
    
    base_count = len(historical_data)
    if base_count == 0:
        # Check if user has any activity in this hour across all days
        hour_activity = df_clean[
            (df_clean['UserID'] == user_id) &
            (df_clean['SubmitDateTime'].dt.hour == hour)
        ]
        base_count = max(1, len(hour_activity) // 30)  # Rough estimate
    
    # Calculate burstiness for this interval
    interval_10min_data = df_clean[
        (df_clean['UserID'] == user_id) &
        (df_clean['SubmitDateTime'] >= start_dt) &
        (df_clean['SubmitDateTime'] < end_dt)
    ]
    
    interval_1hour_start = start_dt.replace(minute=0, second=0)
    interval_1hour_end = interval_1hour_start + pd.Timedelta(hours=1)
    interval_1hour_data = df_clean[
        (df_clean['UserID'] == user_id) &
        (df_clean['SubmitDateTime'] >= interval_1hour_start) &
        (df_clean['SubmitDateTime'] < interval_1hour_end)
    ]
    
    # Classify burstiness level
    if len(interval_10min_data) >= 5:
        # Calculate burstiness metrics for 10-minute interval
        job_count = len(interval_10min_data)
        interval_duration = (end_dt - start_dt).total_seconds() / 60  # in minutes
        arrival_rate = job_count / interval_duration if interval_duration > 0 else 0
        
        # Classify based on 10-minute thresholds
        if (job_count >= thresholds_10min['job_count']['medium'] and
            arrival_rate >= thresholds_10min['arrival_rate']['medium']):
            burst_level = 'High'
        elif (job_count >= thresholds_10min['job_count']['low'] and
              arrival_rate >= thresholds_10min['arrival_rate']['low']):
            burst_level = 'Mid'
        else:
            burst_level = 'Low'
    elif len(interval_1hour_data) >= 5:
        # Calculate burstiness metrics for 1-hour interval
        job_count = len(interval_1hour_data)
        interval_duration = 60  # 1 hour in minutes
        arrival_rate = job_count / interval_duration
        
        # Classify based on 1-hour thresholds
        if (job_count >= thresholds_1hour['job_count']['medium'] and
            arrival_rate >= thresholds_1hour['arrival_rate']['medium']):
            burst_level = 'High'
        elif (job_count >= thresholds_1hour['job_count']['low'] and
              arrival_rate >= thresholds_1hour['arrival_rate']['low']):
            burst_level = 'Mid'
        else:
            burst_level = 'Low'
    else:
        # Fall back to precomputed user-level burstiness
        burst_level = profile['burst_level']
    
    # Adjust based on burstiness level
    burst_multiplier = {'Low': 0.8, 'Mid': 1.0, 'High': 1.5}
    adjusted_count = int(base_count * burst_multiplier[burst_level])
    adjusted_count = max(1, adjusted_count)  # At least 1 job
    
    # Get user's preferred categories for this hour
    preferred_categories = []
    if hour in profile['category_by_hour']:
        # Sort categories by frequency for this hour
        hour_cats = profile['category_by_hour'][hour]
        preferred_categories = sorted(hour_cats.keys(), key=lambda x: hour_cats[x], reverse=True)
    else:
        # Fall back to user's overall primary categories
        preferred_categories = list(profile['primary_categories'].keys())
    
    return adjusted_count, preferred_categories, burst_level

# ─── 7) WORKER THREAD ──────────────────────────────────────────────────────
def worker():
    while True:
        jid, start_ts, end_ts, gran = job_queue.get()
        logging.info(f"[WORKER] Picked job {jid}")
        job_status[jid]["status"] = "running"
        try:
            result = generate_cloudlets_with_users(start_ts, end_ts, gran)
            job_status[jid].update({"status": "done", "result": result})
            total = sum(len(iv["jobs"]) for iv in result)
            logging.info(f"[WORKER] Job {jid} done, generated {total} jobs total")
        except Exception as e:
            logging.error(f"[WORKER] Job {jid} error: {e}", exc_info=True)
            job_status[jid].update({"status": "error", "result": {"error": str(e)}})
        finally:
            job_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

# ─── 8) REST ENDPOINTS ──────────────────────────────────────────────────────
@app.route("/simulate", methods=["POST"])
def submit():
    data = request.get_json(force=True)
    
    # Support both old API (day, start_time, end_time) and new API (start_time, end_time as timestamps)
    if "day" in data and "start_time" in data and "end_time" in data:
        # Old API format
        if not all(k in data for k in ("day", "start_time", "end_time")):
            abort(400, "Missing 'day', 'start_time', or 'end_time'")
        try:
            day = int(data["day"])
            if day < 0 or day > 6:
                raise ValueError
        except ValueError:
            abort(400, "'day' must be an integer 0–6")
        try:
            sh, sm = map(int, data["start_time"].split(":"))
            eh, em = map(int, data["end_time"].split(":"))
            stime = time(sh, sm)
            etime = time(eh, em)
        except Exception:
            abort(400, "'start_time'/'end_time' must be 'HH:MM'")
        if etime <= stime:
            abort(400, "'end_time' must be after 'start_time'")

        # compute UTC timestamps for the given weekday & times
        first_date = df_all["SubmitDateTime"].iloc[0].normalize()
        offset = (day - first_date.weekday()) % 7
        base_date = first_date + pd.Timedelta(days=offset)
        start_dt = datetime.combine(base_date, stime)
        end_dt = datetime.combine(base_date, etime)
        start_ts = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
        end_ts = int(end_dt.replace(tzinfo=timezone.utc).timestamp())
        gran = data.get("granularity", DEFAULT_GRAN)
        
    else:
        # New API format with timestamps
        start_ts_str = data.get("start_time")
        end_ts_str = data.get("end_time")
        gran_str = data.get("granularity", DEFAULT_GRAN)

        if not all([start_ts_str, end_ts_str]):
            abort(400, description="Missing start_time or end_time")
        
        try:
            start_ts = int(start_ts_str)
            end_ts = int(end_ts_str)
            gran = int(gran_str)
        except (ValueError, TypeError) as e:
            abort(400, description=f"Invalid type for timestamp or granularity: {e}. Ensure they are numbers.")

        if start_ts >= end_ts:
            abort(400, description="start_time must be less than end_time")

    job_id = str(uuid.uuid4())
    job_status[job_id] = {"status": "pending"}
    job_queue.put((job_id, start_ts, end_ts, gran))

    logging.info(f"[API] Submitted job {job_id} for interval {start_ts}-{end_ts} with granularity {gran}")
    return jsonify({"job_id": job_id}), 202

@app.route("/simulate/<job_id>", methods=["GET"])
def status(job_id):
    if job_id not in job_status:
        abort(404, description="Job not found")
    info = job_status[job_id]
    logging.info(f"[API] Status check for job {job_id}: {info['status']}")
    return jsonify(convert_numpy_types({
        "job_id": job_id,
        "status": info["status"],
        "result": info.get("result")
    }))

# ─── 9) ENHANCED CLOUDLET GENERATION WITH USER ANALYSIS ────────────────────
# ─── 9) ENHANCED CLOUDLET GENERATION WITH USER ANALYSIS ────────────────────
def generate_cloudlets_with_users(start_ts, end_ts, gran):
    """Generate cloudlets considering user behavior patterns and burstiness."""
    start_dt = datetime.fromtimestamp(start_ts, tz=None)  # Remove timezone info
    end_dt = datetime.fromtimestamp(end_ts, tz=None)      # Remove timezone info
    weekday = start_dt.weekday()
    logging.info(f"[GEN] Generating cloudlets for weekday {weekday} from {start_dt.time()} to {end_dt.time()}")
    
    intervals = pd.interval_range(
        start=start_dt, 
        end=end_dt, 
        freq=pd.Timedelta(seconds=gran), 
        closed="left"
    )
    output = []

    for iv in intervals:
        logging.info(f"[GEN]  Interval {iv.left.time()} - {iv.right.time()}")
        
        # Get active users in this interval
        active_users = get_active_users_in_interval(iv.left, iv.right, weekday)
        logging.info(f"[GEN]   Found {len(active_users)} historically active users")
        
        if len(active_users) == 0:
            logging.info("[GEN]   No active users in this interval. Generating 0 jobs.")
            output.append({
                "interval_start": int(iv.left.timestamp()),
                "interval_end": int(iv.right.timestamp()),
                "user_activity": {},
                "category_mix": {},
                "jobs": []
            })
            continue

        user_activity = {}
        all_jobs = []
        category_counter = Counter()
        
        # Generate jobs for each active user
        for user_id in active_users:
            job_count, preferred_categories, burst_level = calculate_user_jobs_in_interval(
                user_id, iv.left, iv.right, weekday
            )
            
            if job_count == 0 or not preferred_categories:
                continue
                
            user_profile = user_profiles.get(user_id, {})
            
            user_activity[int(user_id)] = {
                'job_count': job_count,
                'burst_level': burst_level,
                'preferred_categories': preferred_categories[:3]  # Top 3
            }
            
            # Generate jobs for this user based on their preferred categories
            user_jobs = []
            jobs_per_category = max(1, job_count // len(preferred_categories))
            remaining_jobs = job_count
            
            for i, category in enumerate(preferred_categories):
                if remaining_jobs <= 0:
                    break
                    
                # For the last category, assign all remaining jobs
                if i == len(preferred_categories) - 1:
                    cat_job_count = remaining_jobs
                else:
                    cat_job_count = min(jobs_per_category, remaining_jobs)
                
                remaining_jobs -= cat_job_count
                category_counter[category] += cat_job_count
                
                # Generate jobs using the appropriate VAE
                cat_jobs = generate_jobs_for_category(
                    category, cat_job_count, burst_level, user_profile
                )
                user_jobs.extend(cat_jobs)
            
            # Add user context to jobs
            for job in user_jobs:
                job['user_id'] = user_id
                job['user_burst_level'] = burst_level
            
            all_jobs.extend(user_jobs)
        
        # Calculate category mix
        total_jobs = sum(category_counter.values())
        category_mix = {cat: count/total_jobs for cat, count in category_counter.items()} if total_jobs > 0 else {}
        
        output.append({
            "interval_start": int(iv.left.timestamp()),
            "interval_end": int(iv.right.timestamp()),
            "user_activity": user_activity,
            "category_mix": category_mix,
            "jobs": all_jobs
        })
        
        logging.info(f"[GEN]   Generated {len(all_jobs)} jobs from {len(active_users)} users")

    return output

def generate_jobs_for_category(category, job_count, burst_level, user_profile):
    """Generate jobs for a specific category using the appropriate VAE."""
    ckpt_file = f"{category.lower()}_data_vae.pt"
    ckpt_path = os.path.join(WEIGHTS_DIR, ckpt_file)
    
    if not os.path.exists(ckpt_path):
        logging.warning(f"[GEN] Missing checkpoint for {category} at {ckpt_file}")
        # Generate mock jobs as fallback
        return [{
            "length": 10.0, 
            "pes": 1, 
            "cpu_time": 10.0, 
            "ram": 100.0, 
            "interarrival": 5.0
        } for _ in range(job_count)]

    try:
        # Load full checkpoint (model + scaler)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        feats, scaler, state = ckpt["features"], ckpt["scaler"], ckpt["model_state"]
        latent_d = state["fc_mu.weight"].shape[0]

        model = VAE(input_dim=len(feats), hidden_dims=HIDDEN_DIMS, latent_dim=latent_d)
        model.load_state_dict(state)
        model.eval()

        # Get base interarrival from user profile
        base_interarrival = 30.0  # Default
        if user_profile and 'interarrival_by_hour' in user_profile:
            current_hour = datetime.now().hour
            if current_hour in user_profile['interarrival_by_hour']:
                base_interarrival = user_profile['interarrival_by_hour'][current_hour]

        valid_jobs = sample_valid_jobs_for_user(
            model, scaler, latent_d, job_count, base_interarrival, burst_level
        )
        
        logging.info(f"[GEN]   Generated {len(valid_jobs)}/{job_count} jobs for category {category}")
        return valid_jobs
        
    except Exception as e:
        logging.error(f"[GEN] Error loading/using VAE for {category}: {e}")
        # Generate mock jobs as fallback
        return [{
            "length": 10.0, 
            "pes": 1, 
            "cpu_time": 10.0, 
            "ram": 100.0, 
            "interarrival": 5.0
        } for _ in range(job_count)]

if __name__ == "__main__":
    # Note: For production, use a proper WSGI server like Gunicorn or uWSGI
    app.run(host="0.0.0.0", port=5000, debug=False)