import pandas as pd
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('TimeDistGenerator')

# --- CONFIGURATION ---
SUBSETS_DIR = "subsets"
CATEGORIZED_SUBSETS_EXCEL = os.path.join(SUBSETS_DIR, "categorized_subsets.xlsx")
OUTPUT_JSON_FILE = "time_category_distribution.json"
GRANULARITY_MINUTES = 10 # Desired granularity: 10 minutes

def generate_time_category_distribution(excel_path):
    """
    Analyzes the categorized jobs Excel file to build a distribution
    of job categories per day of the week and 10-minute interval.
    """
    if not os.path.exists(excel_path):
        logger.error(f"Error: Categorized subsets Excel file not found at {excel_path}")
        logger.info("Please ensure swf_categorizer3.py has been run to generate this file.")
        return None

    try:
        df = pd.read_excel(excel_path)
    except Exception as e:
        logger.error(f"Error reading Excel file {excel_path}: {e}")
        return None

    logger.info(f"Loaded {len(df)} jobs from {excel_path}")

    # Ensure SubmitTime is in datetime format
    df['SubmitTime'] = pd.to_datetime(df['SubmitTime'])

    # Initialize nested defaultdict for the distribution
    # Structure: {DayOfWeek: {HH:MM_interval: {Category: Count}}}
    time_category_counts = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
    
    # Also keep track of total jobs per interval for proportional sampling
    interval_total_jobs = defaultdict(lambda: defaultdict(int))

    for index, row in df.iterrows():
        submit_time = row['SubmitTime']
        category = row['Category']
        
        day_of_week = submit_time.strftime('%A') # e.g., 'Monday'
        
        # Calculate 10-minute interval string (e.g., "09:00", "09:10")
        hour = submit_time.hour
        minute = (submit_time.minute // GRANULARITY_MINUTES) * GRANULARITY_MINUTES
        interval_str = f"{hour:02d}:{minute:02d}"

        time_category_counts[day_of_week][interval_str][category] += 1
        interval_total_jobs[day_of_week][interval_str] += 1

    # Convert defaultdicts to regular dicts for JSON serialization
    final_distribution = {
        day: {
            interval: dict(categories) for interval, categories in intervals.items()
        } for day, intervals in time_category_counts.items()
    }
    
    final_interval_totals = {
        day: dict(intervals) for day, intervals in interval_total_jobs.items()
    }

    # Combine into a single JSON object for saving
    output_data = {
        "category_distribution": final_distribution,
        "interval_total_jobs": final_interval_totals,
        "granularity_minutes": GRANULARITY_MINUTES
    }

    with open(OUTPUT_JSON_FILE, 'w') as f:
        json.dump(output_data, f, indent=4)
    
    logger.info(f"Time-based category distribution saved to {OUTPUT_JSON_FILE}")
    return output_data

if __name__ == "__main__":
    logger.info("Starting generation of time-based category distribution...")
    # Ensure the subsets directory exists for the excel file
    if not os.path.exists(SUBSETS_DIR):
        os.makedirs(SUBSETS_DIR)
        logger.warning(f"'{SUBSETS_DIR}' directory not found. Please run swf_categorizer3.py first to create '{CATEGORIZED_SUBSETS_EXCEL}'.")
    
    distribution_data = generate_time_category_distribution(CATEGORIZED_SUBSETS_EXCEL)
    if distribution_data:
        logger.info("Distribution generation complete.")
    else:
        logger.error("Distribution generation failed.")