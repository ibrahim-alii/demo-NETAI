import numpy as np
import pandas as pd
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path


def generate_synthetic_data(n_samples=1000, anomaly_rate=0.05, seed=42):
    np.random.seed(seed)

    # Generate timestamps at 4-hour intervals
    start_time = datetime(2025, 1, 1, 0, 0, 0)
    timestamps = [start_time + timedelta(hours=4*i) for i in range(n_samples)]

    # Time indices for sinusoidal patterns (period = 6 steps = 24 hours)
    t = np.arange(n_samples)

    # Generate throughput (bps) - baseline 8 Gbps with daily pattern
    throughput_baseline = 8e9
    throughput_pattern = 0.2 * throughput_baseline * np.sin(2 * np.pi * t / 6)
    throughput_noise = np.random.normal(0, 0.05 * throughput_baseline, n_samples)
    throughput_bps = throughput_baseline + throughput_pattern + throughput_noise

    # Generate latency (ms) - baseline 12ms with daily pattern
    latency_baseline = 12.0
    latency_pattern = 2.0 * np.sin(2 * np.pi * t / 6)
    latency_noise = np.random.normal(0, 0.5, n_samples)
    latency_ms = latency_baseline + latency_pattern + latency_noise
    latency_ms = np.maximum(latency_ms, 0.1)  # Ensure positive

    # Generate packet loss rate - exponential distribution
    packet_loss_rate = np.random.exponential(0.001, n_samples)
    packet_loss_rate = np.clip(packet_loss_rate, 0, 0.1)

    # Generate retransmits - Poisson distribution
    retransmits = np.random.poisson(3, n_samples)

    # Create DataFrame
    df = pd.DataFrame({
        'timestamp': timestamps,
        'throughput_bps': throughput_bps,
        'latency_ms': latency_ms,
        'packet_loss_rate': packet_loss_rate,
        'retransmits': retransmits,
        'is_anomaly': 0
    })

    # Inject anomalies
    n_anomalies = int(n_samples * anomaly_rate)
    anomaly_indices = np.random.choice(n_samples, n_anomalies, replace=False)

    for idx in anomaly_indices:
        # Throughput drops 50-90%
        df.loc[idx, 'throughput_bps'] *= np.random.uniform(0.1, 0.5)

        # Latency spikes 2-5x
        df.loc[idx, 'latency_ms'] *= np.random.uniform(2, 5)

        # Packet loss elevated to 5-30%
        df.loc[idx, 'packet_loss_rate'] = np.random.uniform(0.05, 0.30)

        # Retransmit spikes
        df.loc[idx, 'retransmits'] = np.random.poisson(50)

        # Mark as anomaly
        df.loc[idx, 'is_anomaly'] = 1

    return df


def save_to_sqlite(df, db_path='network_metrics.db'):
    conn = sqlite3.connect(db_path)
    df.to_sql('perfsonar_metrics', conn, if_exists='replace', index=False)
    conn.close()
    print(f"[OK] SQLite database created: {db_path}")


def save_to_csv(df, csv_path='data/synthetic_perfsonar.csv'):
    Path(csv_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(csv_path, index=False)
    print(f"[OK] CSV backup created: {csv_path}")


if __name__ == '__main__':
    print("Generating synthetic perfSONAR network metrics")

    # Generate data
    df = generate_synthetic_data(n_samples=1000, anomaly_rate=0.05, seed=42)

    # Display summary
    print(f"\nDataset Summary:")
    print(f"  Total samples: {len(df)}")
    print(f"  Anomalies: {df['is_anomaly'].sum()} ({df['is_anomaly'].mean()*100:.1f}%)")
    print(f"  Time range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    print(f"\nMetric Statistics:")
    print(df[['throughput_bps', 'latency_ms', 'packet_loss_rate', 'retransmits']].describe())

    # Save to SQLite and CSV
    print("\nSaving data")
    save_to_sqlite(df, 'network_metrics.db')
    save_to_csv(df, 'data/synthetic_perfsonar.csv')

    print("\nData generation complete")
