import asyncio
import httpx
import pandas as pd
import logging
import os
import time
from datetime import datetime, timedelta

# --- CONFIGURATION ---
# Updated URL to point to the new time-range simulation endpoint
FLASK_API_URL = "http://127.0.0.1:5000/simulate_by_time_range" 
OUTPUT_DIR = "generated_data"
os.makedirs(OUTPUT_DIR, exist_ok=True)
OUTPUT_FILENAME = "synthetic_jobs_time_range.parquet" # Changed output filename for clarity

# Define the generation plan for time ranges.
# Each entry now specifies day, start_time, end_time, and total_job_count for the range.
# The 'day' should be a full day name (e.g., "Monday", "Tuesday").
# The 'start_time' and 'end_time' should be in "HH:MM" format (24-hour).
GENERATION_PLAN = [
    # Example: 10,000 jobs on Monday from 9 AM to 11 AM
    {'day': 'Monday', 'start_time': '09:00', 'end_time': '11:00', 'total_job_count': 10000},
    # Example: 5,000 jobs on Tuesday from 13:00 (1 PM) to 15:00 (3 PM)
    {'day': 'Tuesday', 'start_time': '13:00', 'end_time': '15:00', 'total_job_count': 5000},
    # Example: 2,500 jobs on Wednesday from 10:00 AM to 10:30 AM
    {'day': 'Wednesday', 'start_time': '10:00', 'end_time': '10:30', 'total_job_count': 2500},
    # Add more entries as needed
]

# --- LOGGING SETUP ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(OUTPUT_DIR, "generator_time_range.log")), # Updated log filename
        logging.StreamHandler()
    ]
)
logger = logging.getLogger('TimeRangeDataGenerator')

# --- MAIN ASYNC GENERATOR ---
async def fetch_jobs_by_time_range(client, day, start_time, end_time, total_job_count):
    """
    Sends a single asynchronous request to the Flask API to generate jobs for a time range.
    """
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
            timeout=300.0  # Increased timeout for potentially larger job counts
        )
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        
        data = response.json()
        logger.info(f"Successfully received {len(data)} jobs for {day} {start_time}-{end_time}.")
        return data
        
    except httpx.RequestError as exc:
        logger.error(f"An error occurred while fetching jobs for {day} {start_time}-{end_time}: {exc}")
        return []

async def generate_dataset_in_parallel_by_time_range():
    """
    Coordinates the parallel generation of the dataset based on time ranges.
    """
    all_generated_jobs = []
    
    async with httpx.AsyncClient() as client:
        # Create a list of tasks, one for each time range in the plan
        tasks = [
            fetch_jobs_by_time_range(client, entry['day'], entry['start_time'], entry['end_time'], entry['total_job_count'])
            for entry in GENERATION_PLAN
        ]
        
        # Run all tasks concurrently
        results = await asyncio.gather(*tasks)
    
    # Collect all the jobs from the results
    for jobs_list in results:
        all_generated_jobs.extend(jobs_list)

    if not all_generated_jobs:
        logger.warning("No jobs were generated. Exiting.")
        return

    # Convert the list of jobs into a pandas DataFrame
    df_jobs = pd.DataFrame(all_generated_jobs)
    
    # --- SAVE TO PARQUET ---
    output_path = os.path.join(OUTPUT_DIR, OUTPUT_FILENAME)
    try:
        if os.path.exists(output_path):
            # If the file exists, read it and append the new data
            existing_df = pd.read_parquet(output_path)
            combined_df = pd.concat([existing_df, df_jobs], ignore_index=True)
            combined_df.to_parquet(output_path, engine='pyarrow', index=False)
            logger.info(f"Appended {len(df_jobs)} new jobs to {output_path}.")
        else:
            # If the file doesn't exist, create it
            df_jobs.to_parquet(output_path, engine='pyarrow', index=False)
            logger.info(f"Saved {len(df_jobs)} jobs to a new file: {output_path}.")
            
    except Exception as e:
        logger.error(f"Failed to save data to Parquet file: {e}")
        # As a fallback, save to CSV if Parquet fails
        df_jobs.to_csv(output_path.replace('.parquet', '.csv'), index=False)
        logger.info("Saved data to CSV instead.")


if __name__ == "__main__":
    start_time = time.time()
    logger.info("Starting parallel data generation for time ranges...")
    
    # Run the asynchronous main function
    try:
        asyncio.run(generate_dataset_in_parallel_by_time_range())
    except KeyboardInterrupt:
        logger.warning("Generation interrupted by user.")
        
    end_time = time.time()
    logger.info(f"Generation finished in {end_time - start_time:.2f} seconds.")
    logger.info(f"Dataset is available in the '{OUTPUT_DIR}' directory.")