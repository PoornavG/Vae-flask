import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter

# ─── HYPERPARAMETERS ───────────────────────────────────────────────
DEFAULT_LATENT_DIM   = 32
ANOMALY_LATENT_DIM   = 16
HIDDEN_DIMS          = [128, 64]
BATCH_SIZE           = 64
EPOCHS               = 100
PATIENCE             = 10
LEARNING_RATE        = 1e-3
WEIGHTS_DIR          = 'vae_models'
KL_ANNEAL_EPOCHS     = 20   # how many epochs to ramp β from 0 → 1
GRAD_CLIP_NORM       = 5.0  # max norm for gradient clipping

os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─── HELPERS ───────────────────────────────────────────────────────
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_preprocess(path, features, drop_thresh=0.5):
    df_full = pd.read_excel(path, sheet_name=0)
    df = df_full[features]
    to_drop = [c for c in df.columns if df[c].isna().mean() > drop_thresh]
    df = df.drop(columns=to_drop).fillna(0.0)
    scaler = StandardScaler().fit(df.values)
    X = scaler.transform(df.values).astype('float32')
    return torch.from_numpy(X), scaler, [c for c in features if c not in to_drop]

def make_dataloaders(X, bs, val_frac=0.2):
    n_val = int(len(X) * val_frac)
    tr, vl = random_split(X, [len(X) - n_val, n_val])
    return (DataLoader(tr, batch_size=bs, shuffle=True,  num_workers=2),
            DataLoader(vl, batch_size=bs, shuffle=False, num_workers=2))

# ─── VAE MODEL ─────────────────────────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        # Encoder
        layers, d = [], input_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True)]
            d = h
        self.encoder = nn.Sequential(*layers)
        self.fc_mu     = nn.Linear(d, latent_dim)
        self.fc_logvar = nn.Linear(d, latent_dim)

        # Decoder
        layers, d = [], latent_dim
        for h in reversed(hidden_dims):
            layers += [nn.Linear(d, h), nn.ReLU(inplace=True)]
            d = h
        layers += [nn.Linear(d, input_dim)]
        self.decoder = nn.Sequential(*layers)

        # Learned centering bias + post-decode norm (as before)
        self.center_bias = nn.Parameter(torch.zeros(input_dim))
        self.post_norm   = nn.BatchNorm1d(input_dim, affine=False)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        h  = self.encoder(x)
        mu = self.fc_mu(h)
        lv = self.fc_logvar(h)
        z  = self.reparameterize(mu, lv)
        out = self.decoder(z) + self.center_bias
        out = self.post_norm(out)
        return out, mu, lv

# ─── LOSS ─────────────────────────────────────────────────────────
def vae_loss(recon, x, mu, logvar, beta=1.0):
    recons = nn.functional.mse_loss(recon, x, reduction='sum')
    kld    = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    return recons + beta * kld

# ─── TRAIN FUNCTION ────────────────────────────────────────────────
def train_vae(model,
              train_loader,
              val_loader,
              optimizer,
              scheduler,
              epochs=EPOCHS,
              patience=PATIENCE,
              writer: SummaryWriter = None,
              weights_dir: str = WEIGHTS_DIR,
              beta: float = None):
    """
    Train VAE model. If beta is None: use linear KL-annealing over KL_ANNEAL_EPOCHS.
    Otherwise use fixed beta for all epochs (backwards-compatible).
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    best_val_loss = float('inf')
    wait = 0
    best_path = os.path.join(weights_dir, 'vae_best.pt')

    for epoch in range(1, epochs+1):
        # determine beta for this epoch
        cur_beta = beta if beta is not None else min(1.0, epoch / KL_ANNEAL_EPOCHS)

        # --- TRAIN ---
        model.train()
        train_loss = 0.0
        for xb in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, xb, mu, logvar, cur_beta)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        # --- VAL ---
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for xb in val_loader:
                xb = xb.to(device)
                recon, mu, logvar = model(xb)
                val_loss += vae_loss(recon, xb, mu, logvar, cur_beta).item()

        avg_train = train_loss / len(train_loader.dataset)
        avg_val   = val_loss   / len(val_loader.dataset)

        # TensorBoard logs
        if writer:
            writer.add_scalar('Loss/Train', avg_train, epoch)
            writer.add_scalar('Loss/Val',   avg_val,   epoch)
            writer.add_scalar('beta',       cur_beta,  epoch)
            for name, tensor in [('mu', mu), ('logvar', logvar)]:
                writer.add_histogram(name, tensor.cpu(), epoch)

        print(f"Epoch {epoch:03d} β={cur_beta:.3f}: train={avg_train:.4f}, val={avg_val:.4f}")

        scheduler.step(avg_val)

        # early stopping + checkpoint
        if avg_val < best_val_loss:
            best_val_loss = avg_val
            wait = 0
            torch.save(model.state_dict(), best_path)
        else:
            wait += 1
            if wait >= patience:
                print(f"Early stopping at epoch {epoch}")
                break

    # load best weights
    model.load_state_dict(torch.load(best_path, map_location=device))
    return model

# ─── USAGE ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Module loaded. Import and call `train_vae(...)` from your driver.")
