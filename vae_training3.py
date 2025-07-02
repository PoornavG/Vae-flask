import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.tensorboard import SummaryWriter
from sklearn.preprocessing import PowerTransformer
import torch.nn.functional as F
# Assuming config.py is accessible in the Python path
try:
    from config import (
        DEFAULT_LATENT_DIM, ANOMALY_LATENT_DIM,
        HIDDEN_DIMS,
        BATCH_SIZE, EPOCHS, PATIENCE, LEARNING_RATE,
        WEIGHTS_DIR, KL_ANNEAL_EPOCHS, GRAD_CLIP_NORM,
        MIN_TRAINING_SIZE, SEED
    )
except ImportError:
    print("Warning: config.py not found or incomplete. Using default hardcoded configurations.")
    # --- Fallback hardcoded configurations for standalone testing ---
    DEFAULT_LATENT_DIM = 64
    ANOMALY_LATENT_DIM = 16
    HIDDEN_DIMS = [256, 128, 64]
    BATCH_SIZE = 64
    EPOCHS = 300
    PATIENCE = 30
    LEARNING_RATE = 1e-3
    WEIGHTS_DIR = 'vae_weights'
    KL_ANNEAL_EPOCHS = 50
    GRAD_CLIP_NORM = 1.0
    MIN_TRAINING_SIZE = 100
    SEED = 42

# Ensure reproducibility
def set_seed(seed_value):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

set_seed(SEED)

class JobSequenceDataset(Dataset):
    def __init__(self, sequences):
        self.sequences = sequences

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

def make_dataloaders(X_sequences_tensor, batch_size, val_split=0.2):
    dataset = JobSequenceDataset(X_sequences_tensor)
    
    val_size = int(len(dataset) * val_split)
    train_size = len(dataset) - val_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader

# Define the VAE model (Recurrent Variational Autoencoder - RVAE)
class VAE(nn.Module):
    def __init__(self, input_dim, latent_dim):
        super(VAE, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        
        self.num_rnn_layers = 1 # Fixed number of RNN layers
        self.rnn_hidden_size = 128 # Fixed RNN hidden size

        # Encoder
        # The encoder maps the input sequence to a hidden state
        # The LSTM processes the sequence step-by-step
        self.encoder_rnn = nn.LSTM(
            input_dim, self.rnn_hidden_size, self.num_rnn_layers,
            batch_first=True
        )
        
        # Linear layers to map the final RNN hidden state to mu and log_var
        self.fc_mu = nn.Linear(self.rnn_hidden_size, latent_dim)
        self.fc_log_var = nn.Linear(self.rnn_hidden_size, latent_dim)

        # Decoder
        # The decoder takes a latent vector z and generates a sequence
        # We'll use a simple linear layer to project z to the initial hidden state for the decoder LSTM
        self.decoder_hidden_init = nn.Linear(latent_dim, self.rnn_hidden_size)
        self.decoder_rnn = nn.LSTM(
            input_dim, self.rnn_hidden_size, self.num_rnn_layers,
            batch_first=True # Input to decoder is previous output, so this needs to match
        )
        
        # Output layer to map RNN output to feature dimension
        self.fc_decode_output = nn.Linear(self.rnn_hidden_size, input_dim)

        self.to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))


    def encode(self, x):
        # x shape: (batch_size, sequence_length, input_dim)
        _, (hidden_state, cell_state) = self.encoder_rnn(x)
        # Use the hidden state from the last layer for mu and log_var
        # hidden_state shape: (num_layers, batch_size, hidden_size)
        # We take the last layer's hidden state: hidden_state[-1, :, :]
        mu = self.fc_mu(hidden_state[-1, :, :])
        log_var = self.fc_log_var(hidden_state[-1, :, :])
        return mu, log_var

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z, sequence_length):
        # z shape: (batch_size, latent_dim)
        
        # Project latent vector to initial hidden state for decoder RNN
        # This will be the initial hidden state for all layers if num_rnn_layers > 1
        initial_hidden = self.decoder_hidden_init(z) # shape: (batch_size, rnn_hidden_size)
        
        # Repeat initial_hidden for num_rnn_layers
        initial_hidden_state = initial_hidden.unsqueeze(0).repeat(self.num_rnn_layers, 1, 1)
        initial_cell_state = torch.zeros_like(initial_hidden_state) # Initial cell state is typically zeros

        # Create a placeholder for the initial input to the decoder
        # For simplicity, start with a zero vector for the first input
        decoder_input = torch.zeros(z.size(0), 1, self.input_dim).to(z.device) # shape: (batch_size, 1, input_dim)
        
        outputs = []
        hidden = (initial_hidden_state, initial_cell_state)

        for _ in range(sequence_length):
            # Pass through decoder RNN
            # decoder_input shape: (batch_size, 1, input_dim)
            # hidden state: (num_layers, batch_size, hidden_size)
            decoder_output, hidden = self.decoder_rnn(decoder_input, hidden)
            
            # Map RNN output to feature dimension
            # decoder_output shape: (batch_size, 1, rnn_hidden_size)
            decoded_step = self.fc_decode_output(decoder_output.squeeze(1)) # (batch_size, input_dim)
            
            outputs.append(decoded_step.unsqueeze(1)) # Add sequence dimension back: (batch_size, 1, input_dim)
            
            # For next step, use the current decoded output as input (teacher forcing-like, or just auto-regressive)
            # In a true auto-regressive setup, you'd feed the *actual* output.
            # For training, we usually use the actual target input (teacher forcing).
            # For generation, we feed the previous output. Here, we feed the decoded output.
            decoder_input = decoded_step.unsqueeze(1) # Use the generated output as input for the next step

        return torch.cat(outputs, dim=1) # Concatenate along the sequence_length dimension

    def forward(self, x):
        mu, log_var = self.encode(x)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decode(z, sequence_length=x.size(1))
        return x_reconstructed, mu, log_var

# VAE Loss Function
def vae_loss(reconstructed_x, x, mu, logvar, kl_weight):
    # Reconstruction loss (e.g., MSE)
    recon_loss = F.mse_loss(reconstructed_x, x, reduction='sum') # Using sum as in the diff
    
    # KL divergence
    kl_loss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    
    # Total loss with beta-weighting
    loss = recon_loss + kl_weight * kl_loss
    return loss

KL_ANNEAL_EPOCHS = 30 # This will be effectively replaced by warmup_epochs
GRAD_CLIP_NORM = 1.0
# Training function
def train_vae(model, train_loader, val_loader, optimizer, scheduler, epochs, patience, writer):
    device = next(model.parameters()).device
    best_val_loss = float('inf')
    epochs_no_improve = 0

    print(f"Starting training on {device}...")

    # hyperparameters from the diff
    target_beta = 0.1      # how much weight to give KL relative to recon
    warmup_epochs = 30     # linearly ramp beta from 0->target_beta over these epochs

    for epoch in range(epochs):
        # linear warm-up schedule for beta (KL weight)
        if epoch < warmup_epochs: # Use epoch < warmup_epochs as epoch starts from 0
            beta = target_beta * ((epoch + 1) / warmup_epochs) # epoch + 1 to avoid 0 beta in first epoch
        else:
            beta = target_beta
        
        # Log the current beta value
        writer.add_scalar('Hyperparameters/beta', beta, epoch)

        model.train()
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            
            reconstructed_data, mu, log_var = model(data)
            
            # Use the calculated beta as kl_weight
            loss = vae_loss(reconstructed_data, data, mu, log_var, kl_weight=beta)
            loss.backward()
            
            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
            
            optimizer.step()
            train_loss += loss.item()
        
        avg_train_loss = train_loss / len(train_loader.dataset) # This assumes vae_loss is sum-reduced per batch
        writer.add_scalar('Loss/train', avg_train_loss, epoch)

        # Validation
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                reconstructed_data, mu, log_var = model(data)
                # For validation, typically you'd use the full target_beta or 1.0 for KL weight
                # The diff uses full weight for validation (implied by no beta calculation for validation)
                # Let's stick to target_beta for consistency with the training phase's eventual beta.
                loss = vae_loss(reconstructed_data, data, mu, log_var, kl_weight=target_beta) 
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader.dataset) # This assumes vae_loss is sum-reduced per batch
        writer.add_scalar('Loss/val', avg_val_loss, epoch)
        
        scheduler.step(avg_val_loss)

        print(f"Epoch {epoch+1}/{epochs}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}, Beta: {beta:.4f}")

        # Early stopping
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
            # Save the best model state (optional, as the original comment says final state is saved)
            # torch.save(model.state_dict(), 'best_vae_model.pth') 
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f"Early stopping triggered after {patience} epochs with no improvement.")
                break
                
    return model

# --- UPDATED load_and_preprocess function ---
def load_and_preprocess(path, features, sheet_name=0, drop_thresh=0.5, sequence_length=10):
    """
    Load data from Excel, handle missing values, apply log-transformation, standardize,
    and convert flat data into fixed-length sequences for RVAE training.
    
    Args:
        path (str): Path to the Excel file.
        features (list): List of feature columns to use.
        sheet_name (str or int): Name or index of the Excel sheet to load.
        drop_thresh (float): Threshold for dropping columns with too many NaNs.
        sequence_length (int): The fixed length of each job sequence for the RVAE.
                                Adjust this based on how you define a "trace" or "session".
    """
    df_full = pd.read_excel(path, sheet_name=sheet_name)
    
    # Store columns before dropping
    columns_before_initial_drop = set(df_full.columns)
    
    # Handle missing values by dropping columns with more than `drop_thresh` NaNs from the full dataframe
    df_full.dropna(thresh=len(df_full) * (1 - drop_thresh), axis=1, inplace=True)
    
    # Identify dropped columns
    columns_after_initial_drop = set(df_full.columns)
    dropped_full_df_columns = list(columns_before_initial_drop - columns_after_initial_drop)

    if dropped_full_df_columns: # Check if the list is not empty
        print(f"  - Dropped columns from the full Excel sheet due to missing values: {', '.join(dropped_full_df_columns)}")
    else:
        print("  - No columns dropped from the full Excel sheet due to missing values.")
    
    # Ensure all required features are present and handle remaining NaNs
    # This 'features' list is the one passed as an argument to this function,
    # e.g., ["RunTime", "AllocatedProcessors", "AverageCPUTimeUsed", "UsedMemory"]
    
    missing_features_from_target = [f for f in features if f not in df_full.columns]
    if missing_features_from_target:
        print(f"  - Warning: User-requested features not found in processed data (likely dropped or not in original sheet): {', '.join(missing_features_from_target)}. Using available features.")
        # Update the 'features' list to only include those that are actually present in df_full after initial dropping
        features = [f for f in features if f in df_full.columns] 
        
    df_clean = df_full[features].copy() # Now, df_clean only contains the features intended for VAE training that survived
    
    # Explicitly convert columns of df_clean to numeric type to prevent DType errors
    for col in df_clean.columns:
        df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce')
        
    # Fill any remaining NaNs within the selected features with 0 (a common strategy for job data)
    df_clean.fillna(0, inplace=True)
    
    # Store original min/max for clamping in simulation (applied to un-transformed scale)
    original_min_max = {}
    for feature in features: # Use the potentially updated 'features' list here
        original_min_max[feature] = {
            'min': df_clean[feature].min(),
            'max': df_clean[feature].max()
        }
    
    # Apply log-transformation to highly-skewed features
    # === NEW: Power-transform highly-skewed features ===
    skewed_features = ['RunTime', 'AverageCPUTimeUsed']
    # Only keep those actually present
    skewed_present = [f for f in skewed_features if f in df_clean.columns]
    if skewed_present:
        pt = PowerTransformer(method='yeo-johnson', standardize=False)
        # reshape to 2d array, fit on raw values
        df_clean[skewed_present] = pt.fit_transform(df_clean[skewed_present])
    else:
        pt = None
            
    # Standardize the data (mean=0, std=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(df_clean)
    
    # Reshape into sequences for RVAE using a sliding window
    sequences = []
    num_jobs = X_scaled.shape[0]
    num_features = X_scaled.shape[1] # This will be the actual number of features used for training

    for i in range(0, num_jobs - sequence_length + 1):
        sequences.append(X_scaled[i : i + sequence_length, :])
    
    if not sequences:
        print(f"  - Not enough samples ({num_jobs}) to form sequences of length {sequence_length}. Returning empty data.")
        return torch.empty(0), StandardScaler(), [], {}

    X_sequences_tensor = torch.tensor(np.array(sequences), dtype=torch.float32)
    
    print(f"  - Loaded {num_jobs} samples, formed {X_sequences_tensor.size(0)} sequences of length {sequence_length} with {len(features)} features (actual features used for training).")
    
    return X_sequences_tensor, scaler, pt, features, original_min_max # This 'features' list contains the names of features used for training