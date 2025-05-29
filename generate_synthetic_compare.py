import os
import glob
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import pdist, squareform
import matplotlib.pyplot as plt

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

    def forward(self, x):
        h  = self.encoder(x)
        mu = self.fc_mu(h)
        lv = self.fc_logvar(h)
        z  = mu + (0.5*lv).exp() * torch.randn_like(mu)
        out = self.decoder(z) + self.center_bias
        return self.post_norm(out), mu, lv

# ─── SYNTHETIC SAMPLE GENERATOR & COMPARISON ─────────────────────────────
def compare_synthetic(n_samples=500, runs=10, noise_scale=0.01, hidden_dims=[128,64]):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on device:", device)

    # allow StandardScaler in checkpoints
    import sklearn.preprocessing._data as _sd
    torch.serialization.add_safe_globals([_sd.StandardScaler])

    # ensure output directory exists
    out_dir = "vae_models"
    os.makedirs(out_dir, exist_ok=True)

    for ckpt_path in glob.glob(os.path.join(out_dir, "*_vae.pt")):
        subset = os.path.basename(ckpt_path).replace("_vae.pt", "")
        print(f"\n▶ Processing subset: {subset}")

        # Load checkpoint
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
        features   = ckpt['features']
        scaler     = ckpt['scaler']
        state_dict = ckpt['model_state']
        latent_dim = state_dict['fc_mu.weight'].shape[0]

        # Build and load model
        model = VAE(input_dim=len(features),
                    hidden_dims=hidden_dims,
                    latent_dim=latent_dim).to(device)
        model.load_state_dict(state_dict)
        model.eval()

        mean_vectors = []  # store column means for each run

        # generate and save synthetic datasets
        for run in range(1, runs+1):
            print(f"  • Run {run}/{runs}…", end=" ")

            # Sample & decode
            with torch.no_grad():
                z   = torch.randn(n_samples, latent_dim, device=device)
                out = model.decoder(z) + model.center_bias
                out = model.post_norm(out).cpu().numpy()

            # add noise & inverse-scale
            out += np.random.randn(*out.shape) * noise_scale
            gen_data = scaler.inverse_transform(out)
            df_gen   = pd.DataFrame(gen_data, columns=features)

            # save Excel
            excel_path = os.path.join(out_dir, f"{subset}_synthetic_{run}.xlsx")
            with pd.ExcelWriter(excel_path, engine='xlsxwriter') as writer:
                df_gen.to_excel(writer, sheet_name='Data', index=False)
            print("saved.")

            # record means
            mean_vectors.append(df_gen.mean().values)

        # compute pairwise distance matrix
        mean_mat = np.vstack(mean_vectors)           # shape (runs, n_features)
        dists    = pdist(mean_mat, metric='euclidean')
        dist_mat = squareform(dists)

        # --- Numeric summaries --- #
        # Pairwise-distance stats
        dist_stats = {
            'mean':   np.mean(dists),
            'median': np.median(dists),
            'min':    np.min(dists),
            'max':    np.max(dists),
            'std':    np.std(dists, ddof=1)
        }
        df_dist = pd.DataFrame.from_dict(dist_stats, orient='index', columns=['value'])
        df_dist.to_csv(os.path.join(out_dir, f"{subset}_distance_summary.csv"))

        # Per-feature stats
        feat_means = mean_mat.mean(axis=0)
        feat_stds  = mean_mat.std(axis=0, ddof=1)
        feat_ranges= mean_mat.max(axis=0) - mean_mat.min(axis=0)
        df_feat = pd.DataFrame({
            'feature': features,
            'mean_of_means': feat_means,
            'std_of_means':  feat_stds,
            'range_of_means':feat_ranges
        })
        df_feat.to_csv(os.path.join(out_dir, f"{subset}_feature_summary.csv"), index=False)

        # --- Plots --- #
        # 1. Heatmap (existing)
        fig, ax = plt.subplots()
        im = ax.imshow(dist_mat, aspect='auto', cmap='viridis')
        ax.set_title(f"Euclidean distances ({subset})")
        ax.set_xlabel("run")
        ax.set_ylabel("run")
        fig.colorbar(im, ax=ax, label='distance')
        fig_path = os.path.join(out_dir, f"{subset}_heatmap.png")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

        # 2. Bar chart of distance stats
        fig, ax = plt.subplots()
        df_dist.plot.bar(ax=ax, legend=False)
        ax.set_title(f"Distance summary ({subset})")
        ax.set_ylabel("value")
        ax.set_xticklabels(df_dist.index, rotation=45)
        fig_path = os.path.join(out_dir, f"{subset}_distance_bar.png")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

        # 3. Error-bar plot for per-feature means
        fig, ax = plt.subplots()
        ax.errorbar(
            x=np.arange(len(features)),
            y=feat_means,
            yerr=feat_stds,
            fmt='o',
            capsize=5
        )
        ax.set_xticks(np.arange(len(features)))
        ax.set_xticklabels(features, rotation=45, ha='right')
        ax.set_title(f"Feature means ±1 std ({subset})")
        ax.set_ylabel("value")
        fig_path = os.path.join(out_dir, f"{subset}_feature_errorbar.png")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

        # 4. Line plot of feature means across runs (existing)
        fig, ax = plt.subplots()
        for idx, feat in enumerate(features):
            ax.plot(range(1, runs+1), mean_mat[:, idx], label=feat)
        ax.set_title(f"Feature means over runs ({subset})")
        ax.set_xlabel("run")
        ax.set_ylabel("mean value")
        ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize='small')
        fig.tight_layout()
        fig_path = os.path.join(out_dir, f"{subset}_feature_means.png")
        fig.savefig(fig_path, bbox_inches='tight')
        plt.close(fig)

        print(f"  ✓ Summaries & plots saved for subset `{subset}`.")

if __name__ == "__main__":
    compare_synthetic()
