import os
import glob
import torch
import pandas as pd

from torch.utils.tensorboard import SummaryWriter

from swf_utils.swf_categorizer import (
    parse_sdsc_sp2_log,
    detect_and_remove_anomalies,
    label_categories
)
from vae_training import (
    VAE,
    set_seed,
    load_and_preprocess,
    make_dataloaders,
    train_vae,
    DEFAULT_LATENT_DIM,
    ANOMALY_LATENT_DIM,
    HIDDEN_DIMS,
    BATCH_SIZE,
    EPOCHS,
    PATIENCE,
    LEARNING_RATE,
    WEIGHTS_DIR
)

# ───────── CONFIG ─────────
SWF_PATH    = "/home/poornav/cloudsim-simulator/SDSC-SP2-1998-4.2-cln.swf"
ANOMALY_PCT = 1.0
SUBSETS_DIR = "subsets"
MIN_SIZE    = 200

os.makedirs(SUBSETS_DIR, exist_ok=True)
os.makedirs(WEIGHTS_DIR, exist_ok=True)
# ──────────────────────────

def export_subsets_to_excel():
    df = parse_sdsc_sp2_log(SWF_PATH)
    anoms, clean = detect_and_remove_anomalies(df, ANOMALY_PCT/100.0)
    labeled = label_categories(clean)

    # Export anomaly subset
    anoms.to_excel(f"{SUBSETS_DIR}/anomalies_data.xlsx", index=False)
    print(f"Exported subset: anomalies_data.xlsx ({len(anoms)} rows)")

    # Export one file per category (9 total)
    for cat, subdf in labeled.groupby("Category"):
        fname = f"{cat.lower()}_data.xlsx"
        subdf.to_excel(os.path.join(SUBSETS_DIR, fname), index=False)
        print(f"Exported subset: {fname} ({len(subdf)} rows)")


def train_all_vaes():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", device)

    FEATURES = [
        'Interarrival','RunTime','AllocatedProcessors',
        'AverageCPUTimeUsed','UsedMemory','RequestedTime',
        'RequestedMemory','CPUUtil'
    ]

    pattern = os.path.join(SUBSETS_DIR, "*_data.xlsx")
    for path in sorted(glob.glob(pattern)):
        fname = os.path.basename(path)

        # Skip weekday and anomaly files
        if fname.startswith("weekday_") or "anomal" in fname:
            print(f"Skipping: {fname}")
            continue

        # Check column existence before processing
        df_check = pd.read_excel(path, nrows=1)
        if not all(col in df_check.columns for col in FEATURES):
            print(f"Skipping {fname}: missing required columns.")
            continue

        subset = os.path.splitext(fname)[0]
        try:
            X, scaler, feats = load_and_preprocess(path, FEATURES)
        except Exception as e:
            print(f"Failed to preprocess {fname}: {e}")
            continue

        if X.size(0) <= MIN_SIZE:
            print(f"\n▶ Subset: {subset} has only {X.size(0)} samples—skipping VAE training.")
            continue

        print(f"\n▶ Subset: {subset}")
        tr_loader, vl_loader = make_dataloaders(X, BATCH_SIZE)

        latent = DEFAULT_LATENT_DIM
        model = VAE(input_dim=len(feats), hidden_dims=HIDDEN_DIMS, latent_dim=latent).to(device)

        optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-5)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=PATIENCE)
        writer    = SummaryWriter(log_dir=f"runs/{subset}")

        trained = train_vae(
            model, tr_loader, vl_loader,
            optimizer, scheduler,
            epochs=EPOCHS,
            patience=PATIENCE,
            writer=writer,
            beta=1.0
        )
        writer.close()

        ckpt = {
            'model_state': trained.state_dict(),
            'scaler': scaler,
            'features': feats
        }
        out_path = os.path.join(WEIGHTS_DIR, f"{subset}_vae.pt")
        torch.save(ckpt, out_path)
        print(f"Saved → {subset}_vae.pt")

if __name__ == "__main__":
    set_seed(42)
    export_subsets_to_excel()
    train_all_vaes()
