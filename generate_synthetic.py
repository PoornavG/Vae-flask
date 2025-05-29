# generate_synthetic.py

import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

# ─── VAE DEFINITION (must match training) ────────────────────────────────
class VAE(nn.Module):
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        # Encoder (not used here but must match saved architecture)
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
        self.decoder    = nn.Sequential(*layers)
        # Centering bias + normalization
        self.center_bias = nn.Parameter(torch.zeros(input_dim))
        self.post_norm   = nn.BatchNorm1d(input_dim, affine=False)

    def reparameterize(self, mu, logvar):
        std = (0.5 * logvar).exp()
        return mu + std * torch.randn_like(std)

    def forward(self, x):
        # Not used for generation
        h  = self.encoder(x)
        mu = self.fc_mu(h)
        lv = self.fc_logvar(h)
        z  = self.reparameterize(mu, lv)
        out = self.decoder(z) + self.center_bias
        return self.post_norm(out), mu, lv

# ─── SYNTHETIC SAMPLE GENERATOR ─────────────────────────────────────────
def generate_synthetic(n_samples=500, noise_scale=0.01, hidden_dims=[128,64]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    # Allow StandardScaler unpickling
    import sklearn.preprocessing._data as _sd
    torch.serialization.add_safe_globals([_sd.StandardScaler])

    for ckpt_path in glob.glob(os.path.join("vae_models", "*_vae.pt")):
        subset = os.path.basename(ckpt_path).replace("_vae.pt", "")
        print(f"\n▶ Generating {n_samples} samples for: {subset}")

        # Load full checkpoint (weights + scaler + features)
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        features    = ckpt['features']
        scaler      = ckpt['scaler']
        state_dict  = ckpt['model_state']
        latent_dim  = state_dict['fc_mu.weight'].shape[0]

        # Build and load model
        model = VAE(input_dim=len(features),
                    hidden_dims=hidden_dims,
                    latent_dim=latent_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        # Sample latents and decode
        with torch.no_grad():
            z   = torch.randn(n_samples, latent_dim, device=device)
            out = model.decoder(z) + model.center_bias
            out = model.post_norm(out).cpu().numpy()

        # Add small Gaussian noise in feature space
        out += np.random.randn(*out.shape) * noise_scale

        # Inverse‐scale to original units
        gen_data = scaler.inverse_transform(out)
        df_gen   = pd.DataFrame(gen_data, columns=features)

        # Compute column‐wise stats
        stats = pd.DataFrame({
            'count':  df_gen.count(),
            'mean':   df_gen.mean(),
            'median': df_gen.median(),
            'std':    df_gen.std(ddof=1),
            'min':    df_gen.min(),
            'max':    df_gen.max()
        })

        # Write Excel
        out_file = f"{subset}_synthetic_500.xlsx"
        with pd.ExcelWriter(out_file, engine='xlsxwriter') as writer:
            df_gen.to_excel(writer, sheet_name='Data', index=False)
            stats.to_excel(writer, sheet_name='Stats')
        print(f"  ✅ Saved synthetic dataset: {out_file}")

if __name__ == "__main__":
    generate_synthetic()
