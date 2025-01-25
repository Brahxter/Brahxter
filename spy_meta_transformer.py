import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SPYDataset(Dataset):
    def __init__(self, csv_file, sequence_length=78):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length

        # Enhanced normalization using percentage changes
        self.data['returns'] = self.data['close'].pct_change()
        self.returns_std = self.data['returns'].std()

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx:idx + self.sequence_length]

        # Normalize using relative changes
        prices = sequence[['open', 'high', 'low', 'close']].values
        base_price = prices[0, -1]  # First close price
        prices = (prices - base_price) / base_price

        volumes = sequence[['Volume']].values
        volume_mean = volumes.mean()
        volumes = (volumes - volume_mean) / volume_mean

        return {
            'prices': torch.FloatTensor(prices),
            'volumes': torch.FloatTensor(volumes),
            'target': torch.FloatTensor([sequence['close'].iloc[-1] / base_price - 1])
        }


class MetaLearningTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()

        self.input_embedding = nn.Linear(5, d_model)
        self.positional_encoding = self._create_positional_encoding(
            78, d_model)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        # Refined prediction head
        self.prediction_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()  # Constrain output to [-1, 1] range
        )

        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x):
        x = torch.cat([x['prices'], x['volumes']], dim=-1)
        x = self.input_embedding(x)
        x = x + self.positional_encoding[:x.size(1)].unsqueeze(0)

        encoded = self.transformer_encoder(x)
        attention_scores = torch.softmax(
            self.attention_weights(encoded), dim=1)
        attended = torch.sum(attention_scores * encoded, dim=1)

        return self.prediction_layers(attended)

    def _create_positional_encoding(self, max_len, d_model):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
