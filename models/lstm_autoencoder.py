import torch
import torch.nn as nn
import numpy as np


class LSTMAutoencoder(nn.Module):
    def __init__(self, n_features=4, hidden_dim=64, latent_dim=32, seq_len=30):
        super(LSTMAutoencoder, self).__init__()
        self.n_features = n_features
        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.seq_len = seq_len

        # Encoder: 2-layer LSTM (input -> 64 -> 32)
        self.encoder_lstm1 = nn.LSTM(
            input_size=n_features,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.encoder_lstm2 = nn.LSTM(
            input_size=hidden_dim,
            hidden_size=latent_dim,
            batch_first=True
        )

        # Decoder: 2-layer LSTM (32 -> 64 -> input)
        self.decoder_lstm1 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=latent_dim,
            batch_first=True
        )
        self.decoder_lstm2 = nn.LSTM(
            input_size=latent_dim,
            hidden_size=hidden_dim,
            batch_first=True
        )
        self.decoder_output = nn.Linear(hidden_dim, n_features)

    def forward(self, x):
        batch_size = x.size(0)

        # Encoder
        encoded, (hidden1, cell1) = self.encoder_lstm1(x)
        encoded, (hidden2, cell2) = self.encoder_lstm2(encoded)

        # Use final hidden state as latent representation
        # Repeat it for decoder input
        latent = hidden2[-1]  # [batch_size, latent_dim]
        latent = latent.unsqueeze(1).repeat(1, self.seq_len, 1)  # [batch_size, seq_len, latent_dim]

        # Decoder
        decoded, _ = self.decoder_lstm1(latent)
        decoded, _ = self.decoder_lstm2(decoded)
        reconstructed = self.decoder_output(decoded)  # [batch_size, seq_len, n_features]

        return reconstructed


def create_sliding_windows(data, seq_len=30, stride=1):
    n_samples, n_features = data.shape
    windows = []

    for i in range(0, n_samples - seq_len + 1, stride):
        window = data[i:i + seq_len]
        windows.append(window)

    return np.array(windows)


def train_autoencoder(model, train_loader, val_loader, epochs=50, lr=0.001, device='cpu'):
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    history = {
        'train_loss': [],
        'val_loss': []
    }

    print(f"Training LSTM Autoencoder on {device}...")
    print(f"Epochs: {epochs}, Learning Rate: {lr}")

    for epoch in range(epochs):
        # Training phase
        model.train()
        train_losses = []
        for batch in train_loader:
            batch = batch[0].to(device)  # Extract tensor from tuple
            optimizer.zero_grad()

            # Forward pass
            reconstructed = model(batch)
            loss = criterion(reconstructed, batch)

            # Backward pass
            loss.backward()
            optimizer.step()

            train_losses.append(loss.item())

        # Validation phase
        model.eval()
        val_losses = []
        with torch.no_grad():
            for batch in val_loader:
                batch = batch[0].to(device)  # Extract tensor from tuple
                reconstructed = model(batch)
                loss = criterion(reconstructed, batch)
                val_losses.append(loss.item())

        # Record losses
        avg_train_loss = np.mean(train_losses)
        avg_val_loss = np.mean(val_losses)
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)

        # Print progress every 10 epochs
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}] - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")

    print("Training complete!")
    return history
