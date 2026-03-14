import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import sqlite3
import pickle
import torch
from sklearn.ensemble import IsolationForest
from sklearn.metrics import precision_score, recall_score, f1_score
import matplotlib.pyplot as plt
from matplotlib.dates import DateFormatter
import matplotlib.dates as mdates

from models.lstm_autoencoder import LSTMAutoencoder, create_sliding_windows
from models.lstm_forecaster import LSTMForecaster


def load_all_data(db_path='network_metrics.db'):
    conn = sqlite3.connect(db_path)
    df = pd.read_sql_query("SELECT * FROM perfsonar_metrics", conn)
    conn.close()
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def evaluate_lstm_autoencoder(df, device='cpu'):
    print("\n[1/3] Evaluating LSTM Autoencoder")

    # Load model and artifacts
    model = LSTMAutoencoder(n_features=4, hidden_dim=64, latent_dim=32, seq_len=30)
    model.load_state_dict(torch.load('saved_models/autoencoder.pth'))
    model = model.to(device)
    model.eval()

    with open('saved_models/scaler_autoencoder.pkl', 'rb') as f:
        scaler = pickle.load(f)
    with open('saved_models/autoencoder_threshold.pkl', 'rb') as f:
        threshold = pickle.load(f)

    # Preprocess data
    feature_cols = ['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits']
    data = df[feature_cols].values
    data_scaled = scaler.transform(data)

    # Create windows
    windows = create_sliding_windows(data_scaled, seq_len=30, stride=1)
    X_tensor = torch.FloatTensor(windows).to(device)

    # Compute reconstruction errors
    reconstruction_errors = []
    with torch.no_grad():
        for i in range(0, len(X_tensor), 32):
            batch = X_tensor[i:i+32]
            reconstructed = model(batch)
            mse = torch.mean((batch - reconstructed) ** 2, dim=(1, 2))
            reconstruction_errors.extend(mse.cpu().numpy())

    reconstruction_errors = np.array(reconstruction_errors)

    # Align with original data (first 29 samples don't have windows)
    errors_aligned = np.full(len(df), np.nan)
    errors_aligned[29:29+len(reconstruction_errors)] = reconstruction_errors

    # Predict anomalies
    predictions_aligned = np.zeros(len(df))
    predictions_aligned[29:29+len(reconstruction_errors)] = (reconstruction_errors > threshold).astype(int)

    # Evaluate (only on samples with predictions)
    valid_mask = ~np.isnan(errors_aligned)
    y_true = df.loc[valid_mask, 'is_anomaly'].values
    y_pred = predictions_aligned[valid_mask]

    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'errors': errors_aligned,
        'predictions': predictions_aligned,
        'threshold': threshold
    }


def evaluate_isolation_forest(df):
    print("\n[2/3] Evaluating Isolation Forest")

    # Feature engineering
    df = df.copy()
    df['hour'] = df['timestamp'].dt.hour
    df['day_of_week'] = df['timestamp'].dt.dayofweek

    # Rolling statistics
    for col in ['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits']:
        df[f'{col}_rolling_mean'] = df[col].rolling(6, min_periods=1).mean()
        df[f'{col}_rolling_std'] = df[col].rolling(6, min_periods=1).std()

    # Select features
    feature_cols = (
        ['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits'] +
        ['hour', 'day_of_week'] +
        [f'{col}_rolling_mean' for col in ['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits']] +
        [f'{col}_rolling_std' for col in ['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits']]
    )

    X = df[feature_cols].fillna(0).values

    # Train Isolation Forest
    iso_forest = IsolationForest(contamination=0.05, random_state=42)
    predictions_iso = iso_forest.fit_predict(X)
    predictions_iso = (predictions_iso == -1).astype(int)  # -1 = anomaly -> 1

    # Evaluate
    y_true = df['is_anomaly'].values
    precision = precision_score(y_true, predictions_iso, zero_division=0)
    recall = recall_score(y_true, predictions_iso, zero_division=0)
    f1 = f1_score(y_true, predictions_iso, zero_division=0)

    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1 Score: {f1:.4f}")

    return {
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'predictions': predictions_iso
    }


def load_forecaster_results():
    print("\n[3/3] Loading LSTM Forecaster results...")
    with open('saved_models/forecaster_predictions.pkl', 'rb') as f:
        results = pickle.load(f)
    print(f"  Test RMSE: {results['metrics']['overall_rmse']:.2f}")
    print(f"  Test MAE: {results['metrics']['overall_mae']:.2f}")
    return results


def plot_raw_timeseries(df):
    print("\nGenerating Raw Time-Series Dashboard")

    fig, axes = plt.subplots(2, 2, figsize=(14, 8))
    fig.suptitle('Raw Network Metrics Time-Series', fontsize=14, fontweight='bold')

    metrics = [
        ('throughput_bps', 'Throughput (bps)', axes[0, 0]),
        ('latency_ms', 'Latency (ms)', axes[0, 1]),
        ('packet_loss_rate', 'Packet Loss Rate', axes[1, 0]),
        ('retransmits', 'Retransmits', axes[1, 1])
    ]

    anomaly_df = df[df['is_anomaly'] == 1]

    for col, title, ax in metrics:
        ax.plot(df['timestamp'], df[col], linewidth=0.8, alpha=0.7, label='Normal')
        ax.scatter(anomaly_df['timestamp'], anomaly_df[col], color='red', s=30,
                   alpha=0.8, label='Anomaly', zorder=5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xlabel('Time', fontsize=9)
        ax.set_ylabel(title, fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    plt.tight_layout()
    Path('figures').mkdir(exist_ok=True)
    plt.savefig('figures/raw_timeseries.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/raw_timeseries.png")


def plot_reconstruction_error(df, lstm_results):
    print("\nGenerating Reconstruction Error Plot")

    fig, ax = plt.subplots(figsize=(14, 5))

    errors = lstm_results['errors']
    threshold = lstm_results['threshold']

    # Plot reconstruction error
    valid_mask = ~np.isnan(errors)
    ax.plot(df.loc[valid_mask, 'timestamp'], errors[valid_mask],
            linewidth=1, alpha=0.7, label='Reconstruction Error', color='blue')

    # Threshold line
    ax.axhline(threshold, color='orange', linestyle='--', linewidth=2,
               label=f'Threshold (95th percentile = {threshold:.4f})')

    # True anomalies
    anomaly_mask = (df['is_anomaly'] == 1) & valid_mask
    ax.scatter(df.loc[anomaly_mask, 'timestamp'], errors[anomaly_mask],
               color='red', s=40, alpha=0.8, label='True Anomalies', zorder=5)

    # Predicted anomalies (above threshold)
    pred_anomaly_mask = (errors > threshold) & valid_mask
    ax.scatter(df.loc[pred_anomaly_mask, 'timestamp'], errors[pred_anomaly_mask],
               color='purple', s=20, alpha=0.5, marker='x', label='Predicted Anomalies', zorder=4)

    ax.set_title('LSTM Autoencoder: Reconstruction Error Over Time', fontsize=13, fontweight='bold')
    ax.set_xlabel('Time', fontsize=11)
    ax.set_ylabel('Reconstruction Error (MSE)', fontsize=11)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/reconstruction_error.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/reconstruction_error.png")


def plot_forecast_predictions(forecaster_results):
    print("\nGenerating Forecast Predictions")

    predictions = forecaster_results['predictions']
    actuals = forecaster_results['actuals']

    # Focus on throughput (most visually interesting)
    throughput_pred = predictions[:, 0]
    throughput_actual = actuals[:, 0]

    # Take first 200 samples for visibility
    n_samples = min(200, len(throughput_pred))

    fig, ax = plt.subplots(figsize=(14, 5))

    time_index = np.arange(n_samples)
    ax.plot(time_index, throughput_actual[:n_samples], linewidth=2,
            label='Actual Throughput', alpha=0.8, color='blue')
    ax.plot(time_index, throughput_pred[:n_samples], linewidth=2,
            linestyle='--', label='Predicted Throughput', alpha=0.8, color='orange')

    ax.set_title('LSTM Forecaster: Throughput Predictions on Test Set', fontsize=13, fontweight='bold')
    ax.set_xlabel('Test Sample Index', fontsize=11)
    ax.set_ylabel('Throughput (bps)', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/forecast_predictions.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/forecast_predictions.png")


def plot_model_comparison(lstm_metrics, iso_metrics):
    print("\nGenerating Model Comparison")

    metrics = ['Precision', 'Recall', 'F1 Score']
    lstm_values = [lstm_metrics['precision'], lstm_metrics['recall'], lstm_metrics['f1']]
    iso_values = [iso_metrics['precision'], iso_metrics['recall'], iso_metrics['f1']]

    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 6))

    bars1 = ax.bar(x - width/2, lstm_values, width, label='LSTM Autoencoder',
                   color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, iso_values, width, label='Isolation Forest',
                   color='coral', alpha=0.8)

    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)

    ax.set_title('Anomaly Detection Model Comparison', fontsize=14, fontweight='bold')
    ax.set_ylabel('Score', fontsize=11)
    ax.set_xticks(x)
    ax.set_xticklabels(metrics, fontsize=11)
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.1)
    ax.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.savefig('figures/model_comparison.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/model_comparison.png")


def plot_training_loss():
    print("\nGenerating Training Loss Curves")

    with open('saved_models/autoencoder_history.pkl', 'rb') as f:
        ae_history = pickle.load(f)
    with open('saved_models/forecaster_history.pkl', 'rb') as f:
        fc_history = pickle.load(f)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

    # Autoencoder
    epochs_ae = range(1, len(ae_history['train_loss']) + 1)
    ax1.plot(epochs_ae, ae_history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    ax1.plot(epochs_ae, ae_history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
    ax1.set_title('LSTM Autoencoder Training', fontsize=12, fontweight='bold')
    ax1.set_xlabel('Epoch', fontsize=10)
    ax1.set_ylabel('Loss (MSE)', fontsize=10)
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Forecaster
    epochs_fc = range(1, len(fc_history['train_loss']) + 1)
    ax2.plot(epochs_fc, fc_history['train_loss'], label='Train Loss', linewidth=2, color='blue')
    ax2.plot(epochs_fc, fc_history['val_loss'], label='Validation Loss', linewidth=2, color='orange')
    ax2.set_title('LSTM Forecaster Training', fontsize=12, fontweight='bold')
    ax2.set_xlabel('Epoch', fontsize=10)
    ax2.set_ylabel('Loss (MSE)', fontsize=10)
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/training_loss.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/training_loss.png")


def plot_forecast_error_distribution(df, forecaster_results):
    print("\nGenerating Forecast Error Distribution")

    # We'll compute errors on the full dataset for visualization
    # Load model and make predictions
    with open('saved_models/scaler_forecaster.pkl', 'rb') as f:
        scaler = pickle.load(f)

    # Use saved test predictions
    predictions = forecaster_results['predictions']
    actuals = forecaster_results['actuals']

    # Compute absolute errors across all features
    errors = np.abs(predictions - actuals).mean(axis=1)  # Average error across features

    # Since we don't have labels for test set, we'll use a simpler visualization
    # showing distribution of prediction errors

    fig, ax = plt.subplots(figsize=(10, 6))

    ax.hist(errors, bins=50, alpha=0.7, color='steelblue', edgecolor='black')

    ax.set_title('LSTM Forecaster: Prediction Error Distribution (Test Set)',
                 fontsize=13, fontweight='bold')
    ax.set_xlabel('Mean Absolute Error (Normalized)', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')

    # Add statistics
    mean_error = np.mean(errors)
    median_error = np.median(errors)
    ax.axvline(mean_error, color='red', linestyle='--', linewidth=2,
               label=f'Mean: {mean_error:.4f}')
    ax.axvline(median_error, color='orange', linestyle='--', linewidth=2,
               label=f'Median: {median_error:.4f}')
    ax.legend(fontsize=10)

    plt.tight_layout()
    plt.savefig('figures/forecast_error_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("  Saved: figures/forecast_error_distribution.png")


def main():
    print("="*60)
    print("Model Evaluation and Visualization")
    print("="*60)

    # Load data
    print("\nLoading data")
    df = load_all_data()
    print(f"Loaded {len(df)} samples")

    # Evaluate models
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    lstm_results = evaluate_lstm_autoencoder(df, device=device)
    iso_results = evaluate_isolation_forest(df)
    forecaster_results = load_forecaster_results()

    # Generate all visualizations
    print("\n" + "="*60)
    print("Generating Visualizations")
    print("="*60)

    plot_raw_timeseries(df)
    plot_reconstruction_error(df, lstm_results)
    plot_forecast_predictions(forecaster_results)
    plot_model_comparison(lstm_results, iso_results)
    plot_training_loss()
    plot_forecast_error_distribution(df, forecaster_results)

    # Summary
    print("\n" + "="*60)
    print("Evaluation Complete!")
    print("="*60)
    print("\nAnomaly Detection Results:")
    print(f"  LSTM Autoencoder - Precision: {lstm_results['precision']:.4f}, "
          f"Recall: {lstm_results['recall']:.4f}, F1: {lstm_results['f1']:.4f}")
    print(f"  Isolation Forest - Precision: {iso_results['precision']:.4f}, "
          f"Recall: {iso_results['recall']:.4f}, F1: {iso_results['f1']:.4f}")
    print(f"\nForecasting Results:")
    print(f"  LSTM Forecaster - RMSE: {forecaster_results['metrics']['overall_rmse']:.2f}, "
          f"MAE: {forecaster_results['metrics']['overall_mae']:.2f}")
    print(f"\nAll visualizations saved to figures/")


if __name__ == '__main__':
    main()
