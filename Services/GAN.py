import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, TensorDataset
from DataPreprocessor import DataPreprocessor

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # Load and preprocess daily data
# df_day = pd.read_csv("data/VM-CPU-data-day.csv", index_col=0, parse_dates=True)
# preprocessed_day = DataPreprocessor(df_day).get_data_frame()
# scaler_day = MinMaxScaler()
# scaled_day = scaler_day.fit_transform(preprocessed_day)
# print(f"Scaled daily data shape: {scaled_day.shape}")
# # Load and preprocess monthly data
# df_month = pd.read_csv("data/VM-CPU-data-month.csv", index_col=0, parse_dates=True)
# preprocessed_month = DataPreprocessor(df_month).get_data_frame()
# scaler_month = MinMaxScaler()
# scaled_month = scaler_month.fit_transform(preprocessed_month)

# Function to create sequences
def create_sequences(data, seq_len):
    sequences = []
    for i in range(len(data) - seq_len):
        sequences.append(data[i:i + seq_len])
    return np.array(sequences)

# # Create sequences for daily and monthly data
# # Set seq_len for daily and monthly datasets
# seq_len_day = 4  # 1 hour for daily data
# seq_len_month = 60  # 1 day for monthly data
# sequences_day = create_sequences(scaled_day, seq_len_day)
# sequences_month = create_sequences(scaled_month, seq_len_month)
# print(f"Number of sequences (daily): {len(sequences_day)}")
# print(f"Number of sequences (monthly): {len(sequences_month)}")

# # Prepare DataLoaders
# tensor_data_day = torch.tensor(sequences_day, dtype=torch.float32)
# dataset_day = TensorDataset(tensor_data_day)
# loader_day = DataLoader(dataset_day, batch_size=64, shuffle=True)

# tensor_data_month = torch.tensor(sequences_month, dtype=torch.float32)
# dataset_month = TensorDataset(tensor_data_month)
# loader_month = DataLoader(dataset_month, batch_size=64, shuffle=True)

# Define Generator and Discriminator
class Generator(nn.Module):
    def __init__(self, latent_dim, hidden_dim, seq_len, feature_dim):
        super().__init__()
        self.lstm = nn.LSTM(latent_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, feature_dim)

    def forward(self, z):
        out, _ = self.lstm(z)
        return self.linear(out)

class Discriminator(nn.Module):
    def __init__(self, hidden_dim, seq_len, feature_dim):
        super().__init__()
        self.lstm = nn.LSTM(feature_dim, hidden_dim, batch_first=True)
        self.linear = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        out, _ = self.lstm(x)
        return self.sigmoid(self.linear(out[:, -1, :]))

# Hyperparameters
latent_dim = 100
hidden_dim = 64
feature_dim = 1

# Train GAN for a given DataLoader
def train_gan(loader, scaler, output_file, seq_len, num_epochs=200):
    G = Generator(latent_dim, hidden_dim, seq_len, feature_dim).to(device)
    D = Discriminator(hidden_dim, seq_len, feature_dim).to(device)

    criterion = nn.BCELoss()
    optimizer_G = optim.Adam(G.parameters(), lr=0.0002)
    optimizer_D = optim.Adam(D.parameters(), lr=0.0002)

    for epoch in range(num_epochs):
        for real_seq, in loader:
            real_seq = real_seq.to(device)
            batch_size = real_seq.size(0)

            # Labels
            real_labels = torch.ones((batch_size, 1)).to(device)
            fake_labels = torch.zeros((batch_size, 1)).to(device)

            # Train Discriminator
            z = torch.randn((batch_size, seq_len, latent_dim)).to(device)
            fake_seq = G(z).detach()

            D_loss = (
                criterion(D(real_seq), real_labels) +
                criterion(D(fake_seq), fake_labels)
            )
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            # Train Generator
            z = torch.randn((batch_size, seq_len, latent_dim)).to(device)
            fake_seq = G(z)
            G_loss = criterion(D(fake_seq), real_labels)

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: D_loss={D_loss.item():.4f}, G_loss={G_loss.item():.4f}")

    # Generate synthetic data for a whole year
    total_intervals = 365 * 48  # 365 days * 48 intervals per day (30-minute intervals)
    generated = []

    G.eval()
    with torch.no_grad():
        for _ in range(total_intervals // seq_len + 1):
            z = torch.randn((1, seq_len, latent_dim)).to(device)
            fake_seq = G(z).cpu().numpy().squeeze()
            generated.extend(fake_seq.tolist())

    generated = np.array(generated[:total_intervals]).reshape(-1, 1)

    # Denormalize
    generated_data = scaler.inverse_transform(generated).flatten()

    # Save to CSV
    start_date = pd.Timestamp("2025-01-01 00:00:00")
    date_range = pd.date_range(start=start_date, periods=total_intervals, freq='30T')

    df_synthetic = pd.DataFrame({
        "datetime": date_range,
        "cpu": generated_data
    })

    df_synthetic.to_csv(output_file, index=False)
    print(f"Saved: {output_file}")


# Load and preprocess the 5-month dataset
df_5_months = pd.read_csv("data/cpu_usage_full_period.csv", index_col=0, parse_dates=True)
preprocessed_5_months = DataPreprocessor(df_5_months).get_data_frame()
scaler_5_months = MinMaxScaler()
scaled_5_months = scaler_5_months.fit_transform(preprocessed_5_months)

# Create sequences for the 5-month dataset
seq_len_30min = 48  # Sequence length for 1 day (30-minute intervals)
sequences_5_months = create_sequences(scaled_5_months, seq_len_30min)

# Prepare DataLoader
tensor_data_5_months = torch.tensor(sequences_5_months, dtype=torch.float32)
dataset_5_months = TensorDataset(tensor_data_5_months)
loader_5_months = DataLoader(dataset_5_months, batch_size=64, shuffle=True)

# Train GAN for the 5-month dataset to generate 1 year of data
train_gan(loader_5_months, scaler_5_months, "data/synthetic_cpu_year_2025.csv", seq_len_30min)

# 200 Epochs, D_loss and G_loss 