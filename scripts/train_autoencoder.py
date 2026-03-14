import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import sqlite3
import pickle
import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

from models.lstm_autoencoder import LSTMAutoencoder, create_sliding_windows, train_autoencoder


def load_normal_data(db_path='network_metrics.db'):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM perfsonar_metrics WHERE is_anomaly = 0"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def preprocess_data(df, seq_len=30, stride=1, test_size=0.2, random_state=42):
    # Extract features
    feature_cols = ['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits']
    data = df[feature_cols].values

    # Fit scaler on training data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create sliding windows
    windows = create_sliding_windows(data_scaled, seq_len=seq_len, stride=stride)
    print(f"Created {len(windows)} sliding windows (seq_len={seq_len}, stride={stride})")

    # Train/validation split
    train_windows, val_windows = train_test_split(
        windows, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Convert to PyTorch tensors
    train_tensor = torch.FloatTensor(train_windows)
    val_tensor = torch.FloatTensor(val_windows)

    # Create data loaders
    train_dataset = TensorDataset(train_tensor)
    val_dataset = TensorDataset(val_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

    print(f"Train samples: {len(train_windows)}, Validation samples: {len(val_windows)}")

    return train_loader, val_loader, scaler


def compute_threshold(model, train_loader, percentile=95, device='cpu'):
    model.eval()
    reconstruction_errors = []

    with torch.no_grad():
        for batch in train_loader:
            batch = batch[0].to(device)
            reconstructed = model(batch)
            mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
            reconstruction_errors.extend(mse.cpu().numpy())

    threshold = np.percentile(reconstruction_errors, percentile)
    print(f"Anomaly detection threshold (95th percentile): {threshold:.6f}")
    return threshold


def main():
    # Configuration
    DB_PATH = 'network_metrics.db'
    SEQ_LEN = 30
    STRIDE = 1
    EPOCHS = 50
    LR = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("LSTM Autoencoder Training")
    print("="*60)

    # Load data
    print("\n[1/5] Loading normal network traffic data")
    df = load_normal_data(DB_PATH)
    print(f"Loaded {len(df)} normal samples")

    # Preprocess
    print("\n[2/5] Preprocessing data")
    train_loader, val_loader, scaler = preprocess_data(
        df, seq_len=SEQ_LEN, stride=STRIDE
    )

    # Initialize model
    print("\n[3/5] Initializing LSTM Autoencoder")
    model = LSTMAutoencoder(
        n_features=4,
        hidden_dim=64,
        latent_dim=32,
        seq_len=SEQ_LEN
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n[4/5] Training model")
    history = train_autoencoder(
        model, train_loader, val_loader,
        epochs=EPOCHS, lr=LR, device=DEVICE
    )

    # Compute threshold
    print("\n[5/5] Computing anomaly detection threshold")
    threshold = compute_threshold(model, train_loader, percentile=95, device=DEVICE)

    # Save artifacts
    print("\nSaving model artifacts")
    Path('saved_models').mkdir(exist_ok=True)

    torch.save(model.state_dict(), 'saved_models/autoencoder.pth')
    print("[OK] Model saved: saved_models/autoencoder.pth")

    with open('saved_models/scaler_autoencoder.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[OK] Scaler saved: saved_models/scaler_autoencoder.pkl")

    with open('saved_models/autoencoder_threshold.pkl', 'wb') as f:
        pickle.dump(threshold, f)
    print("[OK] Threshold saved: saved_models/autoencoder_threshold.pkl")

    with open('saved_models/autoencoder_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("[OK] Training history saved: saved_models/autoencoder_history.pkl")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.6f}")
    print(f"Anomaly Threshold: {threshold:.6f}")


if __name__ == '__main__':
    main()
