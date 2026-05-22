import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from collections import Counter

# --- Configuration ---
DATA_PATH = 'data/raw/XSS_dataset.csv'
PROCESSED_DATA_DIR = 'data/processed/'
NUM_CLIENTS = 5
MAX_LEN = 100
VOCAB_SIZE = 256  # Character-level tokenization (ASCII)
TEST_SIZE = 0.2
RANDOM_STATE = 42

# --- Custom Dataset Class ---
class XSSDataset(Dataset):
    def __init__(self, payloads, labels):
        self.payloads = payloads
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.payloads[idx], self.labels[idx]

# --- Preprocessing and Tokenization Function ---
def char_tokenize(payload):
    """
    Character-level tokenization and padding.
    Uses ASCII values (0-255) as tokens.
    """
    tokens = [ord(c) for c in payload.lower() if ord(c) < VOCAB_SIZE]
    
    # Padding/Truncating
    if len(tokens) < MAX_LEN:
        # Pad with 0s
        tokens = tokens + [0] * (MAX_LEN - len(tokens))
    else:
        # Truncate
        tokens = tokens[:MAX_LEN]
        
    return torch.tensor(tokens, dtype=torch.long)

# --- Non-IID Data Distribution Function (Label Skew) ---
def non_iid_split(X_train, y_train, num_clients):
    """
    Splits the training data into a specified number of clients with a non-IID
    distribution based on label skew.
    Clients 0, 1, 2 will have a higher proportion of XSS (label 1).
    Clients 3, 4 will have a higher proportion of Normal (label 0).
    """
    print("Applying Non-IID split...")
    
    # Separate by class
    X_malicious = X_train[y_train == 1]
    y_malicious = y_train[y_train == 1]
    X_normal = X_train[y_train == 0]
    y_normal = y_train[y_train == 0]
    
    # Calculate base split size for each class
    mal_per_client = len(X_malicious) // num_clients
    norm_per_client = len(X_normal) // num_clients
    
    client_data = []
    
    # Define skew ratios (e.g., 80% of data from one class, 20% from the other)
    # Malicious-heavy clients (0, 1, 2)
    mal_heavy_clients = [0, 1, 2]
    mal_heavy_ratio = 0.8
    
    # Normal-heavy clients (3, 4)
    norm_heavy_clients = [3, 4]
    norm_heavy_ratio = 0.8
    
    # Track indices used
    mal_indices = np.arange(len(X_malicious))
    norm_indices = np.arange(len(X_normal))
    np.random.shuffle(mal_indices)
    np.random.shuffle(norm_indices)
    
    mal_start, norm_start = 0, 0
    
    for i in range(num_clients):
        if i in mal_heavy_clients:
            # Malicious-heavy client
            mal_count = int(mal_per_client * mal_heavy_ratio)
            norm_count = mal_per_client - mal_count # Use remaining budget for normal
            
            # Ensure we don't exceed available data
            mal_count = min(mal_count, len(X_malicious) - mal_start)
            norm_count = min(norm_count, len(X_normal) - norm_start)
            
            # Select indices
            mal_idx = mal_indices[mal_start : mal_start + mal_count]
            norm_idx = norm_indices[norm_start : norm_start + norm_count]
            
            mal_start += mal_count
            norm_start += norm_count
            
        elif i in norm_heavy_clients:
            # Normal-heavy client
            norm_count = int(norm_per_client * norm_heavy_ratio)
            mal_count = norm_per_client - norm_count # Use remaining budget for malicious
            
            # Ensure we don't exceed available data
            norm_count = min(norm_count, len(X_normal) - norm_start)
            mal_count = min(mal_count, len(X_malicious) - mal_start)
            
            # Select indices
            mal_idx = mal_indices[mal_start : mal_start + mal_count]
            norm_idx = norm_indices[norm_start : norm_start + norm_count]
            
            mal_start += mal_count
            norm_start += norm_count
        
        # Combine and shuffle
        X_client = np.concatenate([X_malicious.iloc[mal_idx], X_normal.iloc[norm_idx]])
        y_client = np.concatenate([y_malicious.iloc[mal_idx], y_normal.iloc[norm_idx]])
        
        # Shuffle the client's data internally
        p = np.random.permutation(len(X_client))
        X_client, y_client = X_client[p], y_client[p]
        
        client_data.append((X_client, y_client))
        
        print(f"Client {i}: Samples={len(X_client)}, Malicious={len(mal_idx)}, Normal={len(norm_idx)}, Malicious Ratio={len(mal_idx)/len(X_client):.2f}")

    # Handle remaining data (if any) - not strictly necessary for this simple split, but good practice
    # For simplicity, we will discard remaining data in this basic implementation.
    # A more robust implementation would distribute the remainder.
    
    return client_data

# --- Main Processing Function ---
def process_data():
    # 1. Load Data
    try:
        df = pd.read_csv(DATA_PATH, encoding='latin1')
    except FileNotFoundError:
        print(f"Error: Data file not found at {DATA_PATH}")
        return
    
    # Check for required columns
    if 'Sentence' not in df.columns or 'Label' not in df.columns:
        print("Error: CSV must contain 'Sentence' and 'Label' columns.")
        print(f"Found columns: {df.columns.tolist()}")
        return

    # Drop rows with NaN values in the required columns
    df.dropna(subset=['Sentence', 'Label'], inplace=True)
    
    # Convert Label to integer (0 or 1)
    df['Label'] = df['Label'].astype(int)
    
    print(f"Total samples loaded: {len(df)}")
    print(f"Label distribution: {Counter(df['Label'])}")

    # 2. Split into centralized test set and federated training set
    X_train_fed, X_test_central, y_train_fed, y_test_central = train_test_split(
        df['Sentence'], df['Label'], test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=df['Label']
    )
    
    print(f"\nFederated Training Set size: {len(X_train_fed)}")
    print(f"Centralized Test Set size: {len(X_test_central)}")

    # 3. Non-IID Split for clients
    client_splits = non_iid_split(X_train_fed, y_train_fed, NUM_CLIENTS)

    # 4. Tokenize and Save Data
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(PROCESSED_DATA_DIR, exist_ok=True)
    
    # Tokenize and save centralized test set
    X_test_tokens = torch.stack([char_tokenize(p) for p in X_test_central])
    y_test_tensor = torch.tensor(y_test_central.values, dtype=torch.long)
    torch.save(X_test_tokens, os.path.join(PROCESSED_DATA_DIR, 'X_test_central.pt'))
    torch.save(y_test_tensor, os.path.join(PROCESSED_DATA_DIR, 'y_test_central.pt'))
    print(f"\nSaved Centralized Test Set to {PROCESSED_DATA_DIR}")
    
    # Tokenize and save client data
    for i, (X_client, y_client) in enumerate(client_splits):
        X_client_tokens = torch.stack([char_tokenize(p) for p in X_client])
        y_client_tensor = torch.tensor(y_client, dtype=torch.long)
        
        torch.save(X_client_tokens, os.path.join(PROCESSED_DATA_DIR, f'client_{i}_X.pt'))
        torch.save(y_client_tensor, os.path.join(PROCESSED_DATA_DIR, f'client_{i}_y.pt'))
        print(f"Saved Client {i} data (Samples: {len(X_client)})")

if __name__ == '__main__':
    process_data()
