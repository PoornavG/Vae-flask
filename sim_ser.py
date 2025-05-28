import os
import uuid
import threading
import numpy as np
from queue import Queue
from datetime import datetime, time, timezone

from flask import Flask, request, jsonify, abort
import pandas as pd
import torch
from collections import Counter

from swf_utils.swf_categorizer import (
    parse_sdsc_sp2_log,
    detect_and_remove_anomalies,
    compute_bin_edges,
    classify_job,
    split_by_weekday
)
from vae_training import (
    VAE, set_seed, HIDDEN_DIMS
)
from train_all_vae import (train_all_vaes)

# ─── CONFIG ────────────────────────────────────────────────────────────────
BASE_DIR    = os.path.abspath(os.path.dirname(__file__))
SWF_PATH    = "/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf"
ANOMALY_PCT = 1.0
SUBSETS_DIR = os.path.join(BASE_DIR, "subsets")
WEIGHTS_DIR = os.path.join(BASE_DIR, "vae_models")
DEFAULT_GRAN = 600  # 10 minutes in seconds

os.makedirs(SUBSETS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─── INIT FLASK APP ────────────────────────────────────────────────────────
app = Flask(__name__)
set_seed(42)

job_queue  = Queue()
job_status = {}  # job_id -> {'status': ..., 'result': ...}

# ─── 1) PARSE SWF & COMPUTE BIN EDGES ───────────────────────────────────────
print("[INIT] Parsing SWF and computing bin edges...")
df_all = parse_sdsc_sp2_log(SWF_PATH)
split_by_weekday(df_all)
_, df_clean = detect_and_remove_anomalies(df_all, ANOMALY_PCT / 100.0)
RT_EDGES, CPU_EDGES = compute_bin_edges(df_clean)
print(f"[INIT] Runtime edges: {RT_EDGES}")
print(f"[INIT] CPU-util edges: {CPU_EDGES}")

# ─── 2) PROMPT TO RETRAIN VAEs ──────────────────────────────────────────────
def export_and_train_all():
    train_all_vaes()

def prompt_and_train():
    resp = input("Retrain VAEs now? (y/n): ").strip().lower()
    if resp.startswith("y"):
        print("[TRAIN] Exporting subsets & retraining VAEs…")
        export_and_train_all()
        print("[TRAIN] Retraining complete.")
    else:
        print("[TRAIN] Skipping retraining; using existing checkpoints.")

prompt_and_train()

# ─── 3) JOB SAMPLING ────────────────────────────────────────────────────────
def sample_valid_jobs(model, scaler, latent_d, need, max_attempts=3):
    """
    Sample `need` jobs from the VAE decoder, extracting interarrival directly
    from the decoded output instead of using a fixed mean.
    """
    jobs = []
    attempts = need * max_attempts

    while len(jobs) < need and attempts > 0:
        batch_size = min(need - len(jobs), 64)
        z = torch.randn(batch_size, latent_d)
        with torch.no_grad():
            outp = model.decoder(z) + model.center_bias
            outp = model.post_norm(outp).numpy()
        real_vals = scaler.inverse_transform(outp)

        for vals in real_vals:
            # Extract interarrival directly
            interarrival = max(float(vals[0]), 0.0)
            length       = max(float(vals[1]), 0.0)
            pes          = max(int(round(vals[2])), 1)
            cpu_time     = max(float(vals[3]), 0.0)
            ram          = max(float(vals[4]), 0.0)

            # Skip degenerate samples
            if length == 0.0 and cpu_time == 0.0 and interarrival == 0.0:
                continue

            jobs.append({
                "length":       length if length > 0.0 else cpu_time,
                "pes":          pes,
                "cpu_time":     cpu_time if cpu_time > 0.0 else length,
                "ram":          ram,
                "interarrival": interarrival
            })

            if len(jobs) == need:
                break

        attempts -= batch_size

    # pad with zero-length jobs if needed
    while len(jobs) < need:
        jobs.append({
            "length":       0.0,
            "pes":          1,
            "cpu_time":     0.0,
            "ram":          0.0,
            "interarrival": 0.0
        })

    return jobs

# ─── 4) WORKER THREAD ───────────────────────────────────────────────────────
def worker():
    while True:
        jid, start_ts, end_ts, gran = job_queue.get()
        print(f"[WORKER] Picked job {jid}")
        job_status[jid]["status"] = "running"
        try:
            result = generate_cloudlets(start_ts, end_ts, gran)
            job_status[jid].update({"status": "done", "result": result})
            total = sum(len(iv["jobs"]) for iv in result)
            print(f"[WORKER] Job {jid} done, generated {total} jobs total")
        except Exception as e:
            print(f"[WORKER] Job {jid} error: {e}")
            job_status[jid].update({"status": "error", "result": {"error": str(e)}})
        finally:
            job_queue.task_done()

threading.Thread(target=worker, daemon=True).start()

# ─── 5) REST ENDPOINTS ──────────────────────────────────────────────────────
@app.route("/simulate", methods=["POST"])
def submit():
    data = request.get_json(force=True)
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
    offset     = (day - first_date.weekday()) % 7
    base_date  = first_date + pd.Timedelta(days=offset)
    start_dt   = datetime.combine(base_date, stime)
    end_dt     = datetime.combine(base_date, etime)
    start_ts   = int(start_dt.replace(tzinfo=timezone.utc).timestamp())
    end_ts     = int(end_dt.replace(tzinfo=timezone.utc).timestamp())

    jid = str(uuid.uuid4())
    job_status[jid] = {"status": "pending", "result": None}
    job_queue.put((jid, start_ts, end_ts, DEFAULT_GRAN))
    print(f"[API] Queued job {jid}: day={day}, {data['start_time']}–{data['end_time']}")
    return jsonify({"job_id": jid}), 202

@app.route("/simulate/<job_id>", methods=["GET"])
def status(job_id):
    info = job_status.get(job_id)
    if info is None:
        abort(404, "Unknown job_id")
    print(f"[API] Status check for job {job_id}: {info['status']}")
    return jsonify({
        "job_id": job_id,
        "status": info["status"],
        "result": info["result"]
    })

# ─── 6) CLOUDLET GENERATION ─────────────────────────────────────────────────
def generate_cloudlets(start_ts, end_ts, gran):
    start_dt = datetime.fromtimestamp(start_ts, tz=timezone.utc)
    end_dt   = datetime.fromtimestamp(end_ts, tz=timezone.utc)
    weekday  = start_dt.weekday()
    print(f"[GEN] Generating cloudlets for weekday {weekday} from {start_dt.time()} to {end_dt.time()}")

    path = os.path.join(SUBSETS_DIR, f"weekday_{weekday}_data.xlsx")
    if not os.path.exists(path):
        raise FileNotFoundError(f"No data for weekday {weekday}")

    df = pd.read_excel(path)
    df = df[df["Category"].notna()]  # only labeled jobs
    df["TimeOfDay"] = pd.to_datetime(df["SubmitTimeOfDay"], format="%H:%M:%S").dt.time

    intervals = pd.interval_range(start=start_dt, end=end_dt, freq="10min", closed="left")
    output = []
    for iv in intervals:
        left_t, right_t = iv.left.time(), iv.right.time()
        bucket = df[(df["TimeOfDay"] >= left_t) & (df["TimeOfDay"] < right_t)]
        real_count = len(bucket)
        to_gen = real_count

        source_df = bucket if real_count > 0 else df
        cats      = [classify_job(r, RT_EDGES, CPU_EDGES) for _, r in source_df.iterrows()]
        counts    = Counter(cats)
        total     = sum(counts.values())
        mix       = {c: counts[c] / total for c in counts}

        alloc = {c: int(round(mix[c] * to_gen)) for c in mix}
        jobs  = []
        for cat, n_cat in alloc.items():
            if n_cat < 1:
                continue
            ckpt_file = f"{cat.lower()}_data_vae.pt"
            ckpt_path = os.path.join(WEIGHTS_DIR, ckpt_file)
            if not os.path.exists(ckpt_path):
                print(f"[GEN]     Missing checkpoint for {cat}")
                continue

            # load full checkpoint (model + scaler)
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            feats, scaler, state = ckpt["features"], ckpt["scaler"], ckpt["model_state"]
            latent_d = state["fc_mu.weight"].shape[0]

            model = VAE(input_dim=len(feats), hidden_dims=HIDDEN_DIMS, latent_dim=latent_d)
            model.load_state_dict(state)
            model.eval()

            valid_jobs = sample_valid_jobs(model, scaler, latent_d, n_cat)
            jobs.extend(valid_jobs)
            print(f"[GEN]     Category {cat}: generated {len(valid_jobs)}/{n_cat} jobs")

        output.append({
            "interval_start": int(iv.left.timestamp()),
            "interval_end":   int(iv.right.timestamp()),
            "category_mix":   mix,
            "jobs":           jobs
        })

    return output

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
