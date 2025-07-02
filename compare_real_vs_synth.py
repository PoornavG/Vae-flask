import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from scipy.stats import ks_2samp, wasserstein_distance

# ─── CONFIG ─────────────────────────────────────────────────────────
N_SAMPLES   = 1000  # max points per dataset
BASE_DIR    = os.path.dirname(__file__)
REAL_DIR    = os.path.join(BASE_DIR, 'subsets')
SYN_PATTERN = os.path.join(BASE_DIR, 'synthetic_data_for_comparison', '*_synthetic_500.xlsx')
OUT_DIR     = os.path.join(BASE_DIR, 'comparison_reports')
os.makedirs(OUT_DIR, exist_ok=True)

# ─── UTILS ────────────────────────────────────────────────────────────
def sample_data(df, cols, n=N_SAMPLES, seed=0):
    return df[cols].sample(n=min(n, len(df)), random_state=seed)

# ─── UNIVARIATE COMPARISON ───────────────────────────────────────────
def summary_statistics(real, synth):
    """Compute means, stds, and Wasserstein & KS distances for each feature."""
    stats = []
    for col in real.columns:
        r = real[col].dropna()
        s = synth[col].dropna()
        stats.append({
            'feature': col,
            'real_mean': r.mean(),
            'synth_mean': s.mean(),
            'real_std': r.std(),
            'synth_std': s.std(),
            'ks_stat': ks_2samp(r, s).statistic,
            'wasserstein': wasserstein_distance(r, s)
        })
    return pd.DataFrame(stats)

# ─── PLOTS ───────────────────────────────────────────────────────────
def plot_pca(real, synth, ax):
    combined = pd.concat([real, synth])
    scaler   = StandardScaler().fit(combined)
    Zr = PCA(n_components=2).fit_transform(scaler.transform(real))
    Zs = PCA(n_components=2).fit_transform(scaler.transform(synth))
    ax.scatter(Zr[:,0], Zr[:,1], s=6, alpha=0.5, label='Real')
    ax.scatter(Zs[:,0], Zs[:,1], s=6, alpha=0.5, label='Synthetic')
    ax.set_xlabel('PC1')
    ax.set_ylabel('PC2')
    ax.legend()
    ax.set_title('PCA')

def plot_tsne(real, synth, ax):
    combined = pd.concat([real, synth])
    scaler   = StandardScaler().fit(combined)
    Zr = TSNE(n_components=2, random_state=0).fit_transform(scaler.transform(real))
    Zs = TSNE(n_components=2, random_state=0).fit_transform(scaler.transform(synth))
    ax.scatter(Zr[:,0], Zr[:,1], s=6, alpha=0.5, label='Real')
    ax.scatter(Zs[:,0], Zs[:,1], s=6, alpha=0.5, label='Synthetic')
    ax.set_xlabel('t-SNE1')
    ax.set_ylabel('t-SNE2')
    ax.legend()
    ax.set_title('t-SNE')

def plot_correlations(real, synth, out_path):
    corr_r = real.corr()
    corr_s = synth.corr()
    diff   = corr_r - corr_s

    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    for ax, mat, title in zip(axes, [corr_r, corr_s, diff], ['Real Corr', 'Synth Corr', 'Difference']):
        im = ax.imshow(mat, vmin=-1, vmax=1)
        ax.set_title(title)
        ax.set_xticks(range(len(mat)))
        ax.set_xticklabels(mat.columns, rotation=90)
        ax.set_yticks(range(len(mat)))
        ax.set_yticklabels(mat.index)
        fig.colorbar(im, ax=ax)
    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

def plot_univariate_distributions(real, synth, out_path, bins=30):
    cols = real.columns.tolist()
    n = len(cols)
    cols_per_row = 3
    n_rows = (n + cols_per_row - 1) // cols_per_row

    fig, axes = plt.subplots(n_rows, cols_per_row, figsize=(5 * cols_per_row, 4 * n_rows))
    axes = axes.flatten()

    for i, col in enumerate(cols):
        axes[i].hist(real[col].dropna(), bins=bins, density=True, alpha=0.5, label='Real')
        axes[i].hist(synth[col].dropna(), bins=bins, density=True, alpha=0.5, label='Synthetic')
        axes[i].set_title(col)
        axes[i].legend()

    for ax in axes[n:]:
        ax.axis('off')

    plt.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

# ─── MAIN ─────────────────────────────────────────────────────────────
def main():
    # 1) Discover files
    real_paths = sorted(glob.glob(os.path.join(REAL_DIR, '*.xlsx')))
    synth_paths = sorted(glob.glob(SYN_PATTERN))

    real_bases = {os.path.splitext(os.path.basename(p))[0]: p for p in real_paths}
    synth_bases = {os.path.splitext(os.path.basename(p))[0].replace('_synthetic_500',''): p
                   for p in synth_paths}

    paired = set(real_bases) & set(synth_bases)
    if not paired:
        print("No matching pairs found.")
        return

    for base in paired:
        real_df  = pd.read_excel(real_bases[base])
        synth_df = pd.read_excel(synth_bases[base])

        # numeric common columns
        numeric = real_df.select_dtypes(include=np.number).columns.tolist()
        common  = [c for c in numeric if c in synth_df.columns]
        if not common:
            print(f"{base}: no numeric columns in common.")
            continue

        # sampling
        real_s  = sample_data(real_df, common)
        synth_s = sample_data(synth_df, common)

        # create report folder
        rpt_dir = os.path.join(OUT_DIR, base)
        os.makedirs(rpt_dir, exist_ok=True)

        # univariate stats
        stats_df = summary_statistics(real_s, synth_s)
        stats_df.to_csv(os.path.join(rpt_dir, 'summary_statistics.csv'), index=False)

        # plots: PCA + t-SNE
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        plot_pca(real_s, synth_s, axes[0])
        plot_tsne(real_s, synth_s, axes[1])
        fig.suptitle(f"{base}: Multivariate Embeddings")
        fig.tight_layout(rect=[0, 0, 1, 0.95])
        fig.savefig(os.path.join(rpt_dir, 'multivariate_comparison.png'), dpi=150)
        plt.close(fig)

        # correlations
        plot_correlations(real_s, synth_s, os.path.join(rpt_dir, 'correlation_comparison.png'))

        # univariate distributions
        plot_univariate_distributions(real_s, synth_s,
                                      os.path.join(rpt_dir, 'univariate_distributions.png'))

        print(f"Report generated for {base} in {rpt_dir}")

if __name__ == "__main__":
    main()
