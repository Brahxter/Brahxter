import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SPYDataset(Dataset):
    def __init__(self, csv_file, sequence_length=78):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length

        # Normalize the data
        self.price_scaler = self._fit_scaler(
            self.data[['open', 'high', 'low', 'close']])
        self.volume_scaler = self._fit_scaler(self.data[['Volume']])

    def _fit_scaler(self, data):
        return {'mean': data.mean(), 'std': data.std()}

    def __len__(self):
        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):
        sequence = self.data.iloc[idx:idx + self.sequence_length]

        # Normalize the sequence
        prices = (sequence[['open', 'high', 'low', 'close']] -
                  self.price_scaler['mean']) / self.price_scaler['std']
        volumes = (sequence[['Volume']] -
                   self.volume_scaler['mean']) / self.volume_scaler['std']

        return {
            'prices': torch.FloatTensor(prices.values),
            'volumes': torch.FloatTensor(volumes.values),
            'target': torch.FloatTensor([sequence['close'].iloc[-1]])
        }


class MetaLearningTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()

        self.input_embedding = nn.Linear(5, d_model)
        self.positional_encoding = self._create_positional_encoding(
            78, d_model)

        # Enhanced transformer layers
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,  # Increased capacity
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        # Deeper prediction head
        self.prediction_layers = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, d_model // 4),
            nn.ReLU(),
            nn.Linear(d_model // 4, 1)
        )

        # Attention pooling
        self.attention_weights = nn.Linear(d_model, 1)

    def forward(self, x):
        # Combine price and volume features
        x = torch.cat([x['prices'], x['volumes']], dim=-1)

        # Input embedding with positional encoding
        x = self.input_embedding(x)
        x = x + self.positional_encoding[:x.size(1)].unsqueeze(0)

        # Transformer encoding
        encoded = self.transformer_encoder(x)

        # Attention pooling
        attention_scores = torch.softmax(
            self.attention_weights(encoded), dim=1)
        attended = torch.sum(attention_scores * encoded, dim=1)

        # Final prediction
        return self.prediction_layers(attended)

    def _create_positional_encoding(self, max_len, d_model):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
    # Example usage
    # dataset = SPYDataset('spy_data.csv')
    # model = MetaLearningTransformer()
    # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # criterion = nn.MSELoss()
    # for epoch in range(10):
    #     for data in dataset:
    #         optimizer.zero_grad()
    #         output = model(data)
    #         loss = criterion(output, data['target'])
    #         loss.backward()
    #         optimizer.step()
    #         print(f"Epoch {epoch}, Loss: {loss.item()}")
