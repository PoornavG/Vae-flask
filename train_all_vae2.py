import os
import glob
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional
# --- FIXED: Import the correct functions from swf_categorizer3.py ---
from swf_utils.swf_categorizer3 import (
    parse_sdsc_sp2_log,
    detect_and_remove_anomalies,
    compute_bin_edges,
    compute_burst_activity_edges,
    label_and_categorize_jobs,
)
from vae_training3 import (
    VAE,
    set_seed,
    load_and_preprocess,
    make_dataloaders,
    train_vae,
)
from config import (
    DEFAULT_LATENT_DIM, ANOMALY_LATENT_DIM, HIDDEN_DIMS,
    BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE,
    WEIGHTS_DIR, KL_ANNEAL_EPOCHS, GRAD_CLIP_NORM,
    MIN_TRAINING_SIZE # Also include MIN_TRAINING_SIZE
)

# ───────── CONFIG ─────────
SWF_PATH    = "/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf"
ANOMALY_PCT = 1.0
SUBSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "subsets")
MIN_SIZE    = 200

os.makedirs(SUBSETS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
# ──────────────────────────

def export_subsets_to_excel():
    """
    Parses the SWF log, removes anomalies, categorizes jobs, and exports
    subsets to an Excel file for VAE training.
    
    This function now uses the updated logic from your swf_categorizer3.py.
    """
    # --- CHANGE START ---
    # Removed "SubmitTime" and "WaitTime" from the FEATURES list
    FEATURES = ["RunTime", "AllocatedProcessors", "AverageCPUTimeUsed", "UsedMemory"] 
    # --- CHANGE END ---
    
    print("Parsing SWF log...")
    df = parse_sdsc_sp2_log(SWF_PATH)
    
    print("Detecting and removing anomalies...")
    anoms, clean_data = detect_and_remove_anomalies(df)
    
    if clean_data.empty:
        print("No clean data available after anomaly removal. Cannot create subsets.")
        return []

    # --- FIXED: Compute the bin edges first ---
    print("Computing Jenks breaks for resource bins...")
    rt_edges, cpu_edges = compute_bin_edges(clean_data)
    
    print("Computing Jenks breaks for burst activity...")
    burst_count_edges = compute_burst_activity_edges(clean_data)

    # --- FIXED: Use the correct categorization function with the new parameters ---
    print("Labeling and categorizing jobs...")
    categorized_df = label_and_categorize_jobs(
        clean_data,
        rt_edges=rt_edges,
        cpu_edges=cpu_edges,
        burst_count_edges=burst_count_edges
    )

    category_counts = categorized_df['Category'].value_counts()
    valid_categories = category_counts[category_counts >= 100].index  # e.g., keep only those with 100+ jobs
    filtered_df = categorized_df[categorized_df['Category'].isin(valid_categories)]

    # Export categorized subsets to an Excel file
    excel_path = os.path.join(SUBSETS_DIR, 'categorized_subsets.xlsx')
    with pd.ExcelWriter(excel_path) as writer:
        for category, subset_df in filtered_df.groupby('Category'):
            if not subset_df.empty:
                # Save each category as a separate sheet in the Excel file
                subset_df.to_excel(writer, sheet_name=category, index=False)
                print(f"  - Saved {len(subset_df)} jobs for category '{category}'")
    
    print(f"\nCategorized subsets exported to '{excel_path}'")
    return FEATURES, excel_path # Return the updated FEATURES list

def train_all_vaes():
    set_seed(42)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # --- FIXED: Call the updated export function to get features and file path ---
    FEATURES, excel_file = export_subsets_to_excel() # FEATURES here will now include the missing ones
    if not FEATURES:
        return # Exit if data processing failed

    if not os.path.exists(excel_file):
        print(f"Error: {excel_file} not found. Please check data processing.")
        return

    # Process each sheet in the Excel file as a subset for training
    excel_sheets = pd.ExcelFile(excel_file).sheet_names
    for sheet_name in excel_sheets:
        subset = f'category_{sheet_name}'
        path = excel_file
        
        try:
            # --- FIXED: load_and_preprocess needs the sheet_name argument ---
            X, scaler, pt, feats, original_min_max = load_and_preprocess(path, FEATURES, sheet_name=sheet_name)
        except Exception as e:
            print(f"Failed to preprocess sheet '{sheet_name}': {e}")
            continue

        if X.size(0) <= MIN_SIZE:
            print(f"\n▶ Subset: {subset} has only {X.size(0)} samples—skipping VAE training.")
            continue

        print(f"\n▶ Subset: {subset}")
        tr_loader, vl_loader = make_dataloaders(X, BATCH_SIZE)

        latent = DEFAULT_LATENT_DIM
        model = VAE(input_dim=len(feats), latent_dim=latent).to(device) 
        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE)
        writer = SummaryWriter(log_dir=f"runs/{subset}")

        trained = train_vae(
            model, tr_loader, vl_loader,
            optimizer, scheduler,
            epochs=EPOCHS,
            patience=PATIENCE,
            writer=writer,features=feats,
        )
        writer.close()

        # Save the final checkpoint
        ckpt = {
            'model_state': trained.state_dict(),
            'scaler': scaler,
            'power_transformer': pt,
            'features': FEATURES, # This FEATURES will now be the complete list
            'latent_dim': latent,
            'original_min_max': original_min_max # Store original min/max for synthesis
        }
        ckpt_path = os.path.join(WEIGHTS_DIR, f'{subset}_vae_ckpt.pth')
        torch.save(ckpt, ckpt_path)
        print(f"Saved checkpoint to {ckpt_path}")

# ─── MAIN EXECUTION BLOCK ──────────────────────────────────────────
if __name__ == '__main__':
    train_all_vaes()