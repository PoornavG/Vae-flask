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

from config import (
    DEFAULT_LATENT_DIM, ANOMALY_LATENT_DIM, HIDDEN_DIMS,
    BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE,
    WEIGHTS_DIR, KL_ANNEAL_EPOCHS, GRAD_CLIP_NORM
)
os.makedirs(WEIGHTS_DIR, exist_ok=True)

# ─── HELPERS ───────────────────────────────────────────────────────
def set_seed(seed=0):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def load_and_preprocess(path, features, sheet_name=0, drop_thresh=0.5):
    """
    Load data, handle missing values, and preprocess for VAE training.
    
    Args:
        path (str): Path to the Excel file.
        features (list): List of feature columns to use.
        sheet_name (str or int): Name or index of the Excel sheet to load.
        drop_thresh (float): Threshold for dropping columns with too many NaNs.
    """
    df_full = pd.read_excel(path, sheet_name=sheet_name)
    
    # Handle missing values by dropping columns with more than `drop_thresh` NaNs
    initial_cols = df_full.shape[1]
    df_full.dropna(thresh=len(df_full) * (1 - drop_thresh), axis=1, inplace=True)
    if initial_cols != df_full.shape[1]:
        print(f"  - Dropped {initial_cols - df_full.shape[1]} columns due to missing values.")
    
    # Ensure all required features are present and handle remaining NaNs
    missing_features = [f for f in features if f not in df_full.columns]
    if missing_features:
        print(f"  - Warning: Missing features in data: {missing_features}. Using available features.")
        features = [f for f in features if f in df_full.columns]
        
    df_clean = df_full[features].copy()
    
    # --- FIXED: Explicitly convert columns to numeric type to prevent DType errors ---
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    # Fill remaining NaNs with 0
    df_clean.fillna(0, inplace=True)
    
    # Standardize the data
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Convert to PyTorch tensor
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    
    print(f"  - Loaded {X_tensor.size(0)} samples with {X_tensor.size(1)} features.")
    
    return X_tensor, scaler, features


def make_dataloaders(X, batch_size):
    """
    Splits data into training and validation sets and creates DataLoaders.
    """
    total_size = len(X)
    train_size = int(0.8 * total_size)
    val_size = total_size - train_size
    
    train_data, val_data = random_split(X, [train_size, val_size])
    
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, pin_memory=True)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, pin_memory=True)
    
    print(f"  - Data split: {train_size} train, {val_size} validation.")
    
    return train_loader, val_loader

def vae_loss(recon_x, x, mu, logvar, beta=1.0):
    """
    Calculates the VAE loss, which is a sum of Reconstruction Loss (BCE) and
    KL Divergence Loss.
    """
    # Reconstruction Loss: Mean Squared Error (MSE) is better for continuous data
    recon_loss = nn.functional.mse_loss(recon_x, x, reduction='sum')
    
    # KL Divergence Loss
    kl_div = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    return recon_loss + beta * kl_div

class VAE(nn.Module):
    """
    Variational Autoencoder (VAE) model.
    """
    def __init__(self, input_dim, hidden_dims, latent_dim):
        super().__init__()
        
        # Encoder layers
        encoder_layers = []
        in_dim = input_dim
        for h_dim in hidden_dims:
            encoder_layers.append(nn.Linear(in_dim, h_dim))
            encoder_layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.encoder_mlp = nn.Sequential(*encoder_layers)
        self.fc_mu = nn.Linear(in_dim, latent_dim)
        self.fc_logvar = nn.Linear(in_dim, latent_dim)
        
        # Decoder layers
        decoder_layers = []
        in_dim = latent_dim
        for h_dim in reversed(hidden_dims):
            decoder_layers.append(nn.Linear(in_dim, h_dim))
            decoder_layers.append(nn.ReLU())
            in_dim = h_dim
            
        self.decoder_mlp = nn.Sequential(*decoder_layers)
        self.final_layer = nn.Linear(in_dim, input_dim)
        self.register_buffer('center_bias', torch.zeros(input_dim))
        self.post_norm = nn.Identity()
        
    def encode(self, x):
        """Encodes the input into latent space (mu and logvar)."""
        h = self.encoder_mlp(x)
        return self.fc_mu(h), self.fc_logvar(h)

    def reparameterize(self, mu, logvar):
        """Reparameterization trick to sample from N(mu, exp(logvar))."""
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps

    def decode(self, z):
        """Decodes a latent vector back to the input space."""
        h = self.decoder_mlp(z)
        return self.final_layer(h)

    def forward(self, x):
        """Forward pass through the VAE."""
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        recon = self.decode(z)
        return recon, mu, logvar
    
    def set_center_bias(self, training_data_mean):
        """Set the center bias to the mean of training data"""
        if isinstance(training_data_mean, np.ndarray):
            training_data_mean = torch.tensor(training_data_mean, dtype=torch.float32)
        self.center_bias = training_data_mean

def train_vae(model, train_loader, val_loader, optimizer, scheduler, epochs, patience, writer=None, beta=1.0):
    """
    Trains a VAE model with early stopping and learning rate scheduling.
    """
    device = next(model.parameters()).device
    best_val_loss = float('inf')
    wait = 0
    best_path = 'best_model_temp.pth'
    
    model.train()
    
    for epoch in range(epochs):
        train_loss = 0.0
        
        # Linear annealing of beta
        cur_beta = min(beta * (epoch / KL_ANNEAL_EPOCHS), beta)
        
        for xb in train_loader:
            xb = xb.to(device)
            optimizer.zero_grad()
            recon, mu, logvar = model(xb)
            loss = vae_loss(recon, xb, mu, logvar, cur_beta)
            
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            train_loss += loss.item()

        # Validation
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

    # Load the best model's state
    model.load_state_dict(torch.load(best_path))
    os.remove(best_path)
    print("Training finished.")
    return model