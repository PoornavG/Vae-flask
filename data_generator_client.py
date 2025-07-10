import asyncio
import httpx
import pandas as pd
import logging
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
FLASK_API_URL = "http://127.0.0.1:5000/simulate_by_time_range"
OUTPUT_DIR = "generated_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = "synthetic_jobs_time_range.xlsx"  # Changed to .xlsx

GENERATION_PLAN = [
    {'day': 'Monday', 'start_time': '09:00', 'end_time': '11:00', 'total_job_count': 10000},
    {'day': 'Tuesday', 'start_time': '13:00', 'end_time': '15:00', 'total_job_count': 5000},
    {'day': 'Wednesday', 'start_time': '10:00', 'end_time': '10:30', 'total_job_count': 2500},
]

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "generator_time_range.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TimeRangeDataGenerator')

# --- MAIN ASYNC GENERATOR ---
async def fetch_jobs_by_time_range(client, day, start_time, end_time, total_job_count):
    try:
        logger.info(f"Sending request for {total_job_count} jobs on {day} from {start_time} to {end_time}...")
        payload = {
            'day': day,
            'start_time': start_time,
            'end_time': end_time,
            'total_job_count': total_job_count
        }
        response = await client.post(
            FLASK_API_URL, 
            json=payload, 
            timeout=300.0
        )
        response.raise_for_status()
        data = response.json()
        logger.info(f"Successfully received {len(data)} jobs for {day} {start_time}-{end_time}.")
        return data
    except httpx.RequestError as exc:
        logger.error(f"Error fetching jobs for {day} {start_time}-{end_time}: {exc}")
        return []

async def generate_dataset_in_parallel_by_time_range():
    all_generated_jobs = []

    async with httpx.AsyncClient() as client:
        tasks = [
            fetch_jobs_by_time_range(client, entry['day'], entry['start_time'], entry['end_time'], entry['total_job_count'])
            for entry in GENERATION_PLAN
        ]
        results = await asyncio.gather(*tasks)

    for jobs_list in results:
        all_generated_jobs.extend(jobs_list)

    if not all_generated_jobs:
        logger.warning("No jobs were generated. Exiting.")
        return

    df_jobs = pd.DataFrame(all_generated_jobs)
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)

    try:
        if os.path.exists(output_path):
            # Read existing Excel file and append new data
            existing_df = pd.read_excel(output_path)
            combined_df = pd.concat([existing_df, df_jobs], ignore_index=True)
            combined_df.to_excel(output_path, index=False)
            logger.info(f"Appended {len(df_jobs)} jobs to existing Excel file at {output_path}.")
        else:
            df_jobs.to_excel(output_path, index=False)
            logger.info(f"Saved {len(df_jobs)} jobs to new Excel file: {output_path}.")
    except Exception as e:
        logger.error(f"Failed to save data to Excel file: {e}")
        fallback_path = output_path.replace(".xlsx", ".csv")
        df_jobs.to_csv(fallback_path, index=False)
        logger.info(f"Saved data to CSV instead at: {fallback_path}")

if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting parallel data generation for time ranges...")

    try:
        asyncio.run(generate_dataset_in_parallel_by_time_range())
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user.")

    end_time = time.time()
    logger.info(f"Generation finished in {end_time - start_time:.2f} seconds.")
    logger.info(f"Dataset is available in the '{OUTPUT_DIR}' directory.")
