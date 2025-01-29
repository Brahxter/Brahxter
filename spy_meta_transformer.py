import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SPYDataset(Dataset):
    def __init__(self, csv_file, sequence_length=78):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length

        # Convert timestamp to datetime
        self.data['TimeStamp'] = pd.to_datetime(self.data['TimeStamp'])

        # Group data by trading day
        self.data['trading_day'] = self.data['TimeStamp'].dt.date
        self.trading_days = self.data['trading_day'].unique()

    def __len__(self):
        return len(self.trading_days)

    def __getitem__(self, idx):
        trading_day = self.trading_days[idx]
        day_data = self.data[self.data['trading_day'] == trading_day]

        # Get first 30 minutes of trading (6 5-minute candles)
        opening_range = day_data.iloc[:6]

        if len(opening_range) < 6:
            return None

        # Normalize prices relative to opening price
        base_price = opening_range['open'].iloc[0]
        prices = opening_range[['open', 'high', 'low', 'close']].values
        prices = (prices - base_price) / base_price

        # Normalize volume with standard deviation
        volumes = opening_range[['Volume']].values
        volume_std = volumes.std()
        if volume_std > 0:
            volumes = (volumes - volumes.mean()) / volume_std
        else:
            volumes = volumes - volumes.mean()

        # Calculate daily return for target
        daily_close = day_data['close'].iloc[-1]
        target = (daily_close / base_price) - 1

        return {
            'prices': torch.FloatTensor(prices),
            'volumes': torch.FloatTensor(volumes),
            'target': torch.FloatTensor([target])
        }


class MetaLearningTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()

        # 4 price features + 1 volume feature
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

        self.prediction_layers = nn.Sequential(
            nn.Linear(d_model, d_model),
            nn.LayerNorm(d_model),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.ReLU(),
            nn.Linear(d_model // 2, 1),
            nn.Tanh()
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
