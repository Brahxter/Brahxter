import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


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


class TemporalFusionBlock(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.short_conv = nn.Conv1d(d_model, d_model, kernel_size=3, padding=1)
        self.mid_conv = nn.Conv1d(d_model, d_model, kernel_size=5, padding=2)
        self.long_conv = nn.Conv1d(d_model, d_model, kernel_size=7, padding=3)
        self.fusion = nn.Linear(d_model * 3, d_model)
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x):
        x_t = x.transpose(1, 2)
        short = self.short_conv(x_t)
        mid = self.mid_conv(x_t)
        long = self.long_conv(x_t)
        fused = torch.cat([short, mid, long], dim=1)
        return self.norm(self.fusion(fused.transpose(1, 2)))


class SPYDataset(Dataset):
    def __init__(self, csv_file, sequence_length=78):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length

        self.data['TimeStamp'] = pd.to_datetime(self.data['TimeStamp'])
        self.data['trading_day'] = self.data['TimeStamp'].dt.date

        # Enhanced market microstructure features
        self.data['price_velocity'] = self.data['close'].diff() / \
            self.data['Volume']
        self.data['volume_intensity'] = self.data.groupby('trading_day')['Volume'].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-6))
        self.data['price_acceleration'] = self.data['price_velocity'].diff()
        self.data['volume_momentum'] = self.data['volume_intensity'].rolling(
            3).mean()

        self.trading_days = self.data['trading_day'].unique()

    def __len__(self):
        return len(self.trading_days)

    def __getitem__(self, idx):
        trading_day = self.trading_days[idx]
        day_data = self.data[self.data['trading_day'] == trading_day]

        opening_range = day_data.iloc[:6]
        if len(opening_range) < 6:
            return None

        # Dynamic price normalization with volatility adjustment
        base_price = opening_range['open'].iloc[0]
        prices = opening_range[['open', 'high', 'low', 'close']].values
        price_std = np.std(prices) + 1e-6
        prices = np.clip((prices - base_price) / price_std, -0.1, 0.1)

        # Enhanced volume features with momentum
        volumes = np.column_stack([
            opening_range['Volume'].values,
            opening_range['volume_intensity'].values,
            opening_range['price_velocity'].values,
            opening_range['price_acceleration'].values,
            opening_range['volume_momentum'].values
        ])
        volumes = np.nan_to_num(volumes)
        volumes = np.clip(volumes, -1e3, 1e3)
        volume_std = np.std(volumes, axis=0) + 1e-6
        volumes = np.clip(
            (volumes - np.mean(volumes, axis=0)) / volume_std, -5, 5)

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

        # Enhanced embeddings with larger capacity
        self.price_embedding = nn.Linear(4, d_model)
        self.volume_embedding = nn.Linear(5, d_model)

        # Multi-scale temporal fusion
        self.temporal_fusion = TemporalFusionBlock(d_model)

        # Adaptive attention mechanism
        self.adaptive_attention = AdaptiveAttention(d_model)

        # Dynamic feature selection
        self.feature_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2),
            nn.LayerNorm(d_model * 2),
            nn.GELU(),
            nn.Linear(d_model * 2, d_model * 2),
            nn.Sigmoid()
        )

        self.positional_encoding = self._create_positional_encoding(
            78, d_model)

        # Enhanced transformer with increased capacity
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model * 2,
            nhead=nhead,
            dim_feedforward=2048,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers)

        # Advanced prediction head with residual connections
        self.prediction_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.LayerNorm(d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, d_model // 2),
            nn.LayerNorm(d_model // 2),
            nn.GELU(),
            nn.Linear(d_model // 2, d_model // 4),
            nn.LayerNorm(d_model // 4),
            nn.GELU(),
            nn.Linear(d_model // 4, 1),
            nn.Tanh()
        )

        self.attention_weights = nn.Linear(d_model * 2, 1)

    def forward(self, x):
        # Process features with temporal awareness
        price_features = self.price_embedding(x['prices'])
        volume_features = self.volume_embedding(x['volumes'])

        # Apply temporal fusion
        price_temporal = self.temporal_fusion(price_features)
        volume_temporal = self.temporal_fusion(volume_features)

        # Adaptive attention
        price_context = self.adaptive_attention(price_temporal)
        volume_context = self.adaptive_attention(volume_temporal)

        # Combine features with dynamic gating
        combined = torch.cat([price_context, volume_context], dim=-1)
        gates = self.feature_gate(combined)
        x = combined * gates

        # Add positional encoding with temporal awareness
        x = x + \
            self.positional_encoding[:x.size(1)].unsqueeze(0).repeat(1, 1, 2)

        # Multi-scale processing
        encoded = self.transformer_encoder(x)

        # Hierarchical attention
        attention_scores = torch.softmax(
            self.attention_weights(encoded), dim=1)
        context = torch.sum(attention_scores * encoded, dim=1)

        return self.prediction_head(context)

    def _create_positional_encoding(self, max_len, d_model):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(
            0, d_model, 2).float() * (-np.log(10000.0) / d_model))

        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
