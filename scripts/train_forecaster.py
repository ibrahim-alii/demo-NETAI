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
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.lstm_forecaster import LSTMForecaster, train_forecaster
from models.lstm_autoencoder import create_sliding_windows


def load_all_data(db_path='network_metrics.db'):
    conn = sqlite3.connect(db_path)
    query = "SELECT * FROM perfsonar_metrics"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df


def create_forecast_dataset(data, seq_len=30, stride=1):
    X, y = [], []
    n_samples = data.shape[0]

    for i in range(0, n_samples - seq_len, stride):
        X.append(data[i:i + seq_len])
        y.append(data[i + seq_len])

    return np.array(X), np.array(y)


def preprocess_data(df, seq_len=30, stride=1, test_size=0.2, random_state=42):
    # Extract features
    feature_cols = ['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits']
    data = df[feature_cols].values

    # Fit scaler on training data
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    # Create forecast dataset
    X, y = create_forecast_dataset(data_scaled, seq_len=seq_len, stride=stride)
    print(f"Created {len(X)} samples (seq_len={seq_len}, stride={stride})")

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, shuffle=True
    )

    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.FloatTensor(y_train)
    X_test_tensor = torch.FloatTensor(X_test)
    y_test_tensor = torch.FloatTensor(y_test)

    # Create data loaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    print(f"Train samples: {len(X_train)}, Test samples: {len(X_test)}")

    return train_loader, test_loader, scaler, (X_test_tensor, y_test_tensor)


def evaluate_forecaster(model, X_test, y_test, scaler, device='cpu'):
    model.eval()
    model = model.to(device)

    with torch.no_grad():
        X_test = X_test.to(device)
        predictions = model(X_test).cpu().numpy()
        y_test_np = y_test.cpu().numpy()

    # Inverse transform to original scale
    predictions_orig = scaler.inverse_transform(predictions)
    y_test_orig = scaler.inverse_transform(y_test_np)

    # Compute metrics
    rmse = np.sqrt(mean_squared_error(y_test_orig, predictions_orig))
    mae = mean_absolute_error(y_test_orig, predictions_orig)

    # Per-feature metrics
    feature_names = ['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits']
    feature_metrics = {}

    for i, name in enumerate(feature_names):
        feature_rmse = np.sqrt(mean_squared_error(y_test_orig[:, i], predictions_orig[:, i]))
        feature_mae = mean_absolute_error(y_test_orig[:, i], predictions_orig[:, i])
        feature_metrics[name] = {'rmse': feature_rmse, 'mae': feature_mae}

    return {
        'overall_rmse': rmse,
        'overall_mae': mae,
        'feature_metrics': feature_metrics,
        'predictions': predictions_orig,
        'actuals': y_test_orig
    }


def main():
    # Configuration
    DB_PATH = 'network_metrics.db'
    SEQ_LEN = 30
    STRIDE = 1
    EPOCHS = 50
    LR = 0.001
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    print("="*60)
    print("LSTM Forecaster Training")
    print("="*60)

    # Load data
    print("\n[1/5] Loading network traffic data")
    df = load_all_data(DB_PATH)
    print(f"Loaded {len(df)} samples (including anomalies)")

    # Preprocess
    print("\n[2/5] Preprocessing data")
    train_loader, test_loader, scaler, (X_test, y_test) = preprocess_data(
        df, seq_len=SEQ_LEN, stride=STRIDE
    )

    # Initialize model
    print("\n[3/5] Initializing LSTM Forecaster")
    model = LSTMForecaster(
        n_features=4,
        hidden_dim=50,
        num_layers=2
    )
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")

    # Train
    print("\n[4/5] Training model")
    history = train_forecaster(
        model, train_loader, test_loader,
        epochs=EPOCHS, lr=LR, device=DEVICE
    )

    # Evaluate
    print("\n[5/5] Evaluating on test set")
    metrics = evaluate_forecaster(model, X_test, y_test, scaler, device=DEVICE)

    print(f"\nTest Set Performance:")
    print(f"  Overall RMSE: {metrics['overall_rmse']:.2f}")
    print(f"  Overall MAE: {metrics['overall_mae']:.2f}")
    print(f"\nPer-Feature Metrics:")
    for name, vals in metrics['feature_metrics'].items():
        print(f"  {name}:")
        print(f"    RMSE: {vals['rmse']:.4f}")
        print(f"    MAE: {vals['mae']:.4f}")

    # Save artifacts
    print("\nSaving model artifacts")
    Path('saved_models').mkdir(exist_ok=True)

    torch.save(model.state_dict(), 'saved_models/forecaster.pth')
    print("[OK] Model saved: saved_models/forecaster.pth")

    with open('saved_models/scaler_forecaster.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    print("[OK] Scaler saved: saved_models/scaler_forecaster.pkl")

    with open('saved_models/forecaster_history.pkl', 'wb') as f:
        pickle.dump(history, f)
    print("[OK] Training history saved: saved_models/forecaster_history.pkl")

    with open('saved_models/forecaster_predictions.pkl', 'wb') as f:
        pickle.dump({
            'predictions': metrics['predictions'],
            'actuals': metrics['actuals'],
            'metrics': {
                'overall_rmse': metrics['overall_rmse'],
                'overall_mae': metrics['overall_mae'],
                'feature_metrics': metrics['feature_metrics']
            }
        }, f)
    print("Test predictions saved: saved_models/forecaster_predictions.pkl")

    print("\n" + "="*60)
    print("Training Complete!")
    print("="*60)
    print(f"Final Train Loss: {history['train_loss'][-1]:.6f}")
    print(f"Final Val Loss: {history['val_loss'][-1]:.6f}")
    print(f"Test RMSE: {metrics['overall_rmse']:.2f}")
    print(f"Test MAE: {metrics['overall_mae']:.2f}")


if __name__ == '__main__':
    main()
