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
