import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class MultiScaleFeatureProcessor(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.price_convs = nn.ModuleList([
            nn.Conv1d(4, d_model, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.volume_convs = nn.ModuleList([
            nn.Conv1d(1, d_model, kernel_size=k, padding=k//2)
            for k in [3, 5, 7]
        ])
        self.feature_fusion = nn.Linear(d_model * 6, d_model)

    def forward(self, price_data, volume_data):
        price_features = [conv(price_data.transpose(1, 2))
                          for conv in self.price_convs]
        volume_features = [conv(volume_data.transpose(1, 2))
                           for conv in self.volume_convs]
        combined = torch.cat(price_features + volume_features, dim=1)
        return self.feature_fusion(combined.transpose(1, 2))


class AdaptiveAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.temperature = nn.Parameter(
            torch.sqrt(torch.FloatTensor([d_model])))
        self.qkv_transform = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.scale = d_model ** -0.5

    def forward(self, x):
        qkv = self.qkv_transform(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: t.contiguous(), qkv)
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        return self.out_proj(attn @ v)


class SPYDataset(Dataset):
    def __init__(self, csv_file, sequence_length=78):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        self.data['TimeStamp'] = pd.to_datetime(self.data['TimeStamp'])
        self.data['trading_day'] = self.data['TimeStamp'].dt.date
        self.trading_days = self.data['trading_day'].unique()

    def __len__(self):
        return len(self.trading_days)

    def __getitem__(self, idx):
        trading_day = self.trading_days[idx]
        day_data = self.data[self.data['trading_day'] == trading_day]

        opening_range = day_data.iloc[:6]
        if len(opening_range) < 6:
            return None

        base_price = opening_range['open'].iloc[0]
        prices = opening_range[['open', 'high', 'low', 'close']].values
        price_std = np.std(prices) + 1e-6
        prices = np.clip((prices - base_price) / price_std, -0.1, 0.1)

        volumes = opening_range[['Volume']].values
        volume_std = np.std(volumes) + 1e-6
        volumes = np.clip((volumes - np.mean(volumes)) / volume_std, -5, 5)

        daily_close = day_data['close'].iloc[-1]
        target = np.clip((daily_close / base_price) - 1, -0.1, 0.1)

        return {
            'prices': torch.FloatTensor(prices),
            'volumes': torch.FloatTensor(volumes),
            'target': torch.FloatTensor([target])
        }


class MetaLearningTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_layers=num_layers,
            dropout=dropout
        )

    def adapt(self, support_set):
        # Quick adaptation to new patterns
        adapted_params = self.get_quick_weights(support_set)
        return adapted_params

    def get_quick_weights(self, data):
        # MAML-style quick adaptation
        grads = self.compute_gradients(data)
        quick_weights = self.update_weights(grads)
        return quick_weights

    def compute_gradients(self, data):
        loss = self.calculate_loss(data)
        return torch.autograd.grad(loss, self.parameters())

    def update_weights(self, gradients):
        quick_weights = {}
        for (name, param), grad in zip(self.named_parameters(), gradients):
            quick_weights[name] = param - self.learning_rate * grad
        return quick_weights

    def forward(self, x):
        return self.transformer(x)

    def _create_positional_encoding(self, max_len, d_model):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
