import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset

class SPYDataset(Dataset):
    def __init__(self, csv_file, sequence_length=78):
        self.data = pd.read_csv(csv_file)
        self.sequence_length = sequence_length
        
        # Enhanced feature engineering
        self.data['TimeStamp'] = pd.to_datetime(self.data['TimeStamp'])
        self.data['trading_day'] = self.data['TimeStamp'].dt.date
        self.data['time_of_day'] = self.data['TimeStamp'].dt.time
        
        # Volume profile features
        self.data['volume_ma'] = self.data['Volume'].rolling(5).mean()
        self.data['volume_ratio'] = self.data['Volume'] / self.data['volume_ma']
        self.data['cumulative_volume'] = self.data.groupby('trading_day')['Volume'].cumsum()
        
        # Price action features
        self.data['price_momentum'] = self.data['close'].pct_change(3)
        self.data['high_low_range'] = (self.data['high'] - self.data['low']) / self.data['open']
        self.data['opening_gap'] = self.data.groupby('trading_day')['open'].transform(lambda x: x.iloc[0] / x.shift(1).iloc[-1] - 1)
        
        self.trading_days = self.data['trading_day'].unique()

    def __len__(self):
        return len(self.trading_days)

    def __getitem__(self, idx):
        trading_day = self.trading_days[idx]
        day_data = self.data[self.data['trading_day'] == trading_day]
        
        # Get opening range data (first 30 minutes)
        opening_range = day_data.iloc[:6]
        
        if len(opening_range) < 6:
            return None
        
        # Enhanced feature normalization
        base_price = opening_range['open'].iloc[0]
        prices = opening_range[['open', 'high', 'low', 'close']].values
        prices = (prices - base_price) / base_price
        
        # Volume features
        volumes = np.column_stack([
            opening_range['Volume'].values,
            opening_range['volume_ratio'].values,
            opening_range['cumulative_volume'].values
        ])
        volume_std = np.std(volumes, axis=0)
        volume_std[volume_std == 0] = 1
        volumes = (volumes - np.mean(volumes, axis=0)) / volume_std
        
        # Technical features
        tech_features = np.column_stack([
            opening_range['price_momentum'].values,
            opening_range['high_low_range'].values,
            opening_range['opening_gap'].values
        ])
        tech_features = np.nan_to_num(tech_features, nan=0)
        
        # Calculate target with momentum consideration
        daily_close = day_data['close'].iloc[-1]
        target = (daily_close / base_price) - 1
        
        return {
            'prices': torch.FloatTensor(prices),
            'volumes': torch.FloatTensor(volumes),
            'tech_features': torch.FloatTensor(tech_features),
            'target': torch.FloatTensor([target])
        }

class MetaLearningTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        
        # Enhanced input embedding
        self.price_embedding = nn.Linear(4, d_model // 2)
        self.volume_embedding = nn.Linear(3, d_model // 4)
        self.tech_embedding = nn.Linear(3, d_model // 4)
        
        self.feature_combiner = nn.Linear(d_model, d_model)
        self.positional_encoding = self._create_positional_encoding(78, d_model)
        
        # Multi-head attention with enhanced features
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=1024,
            dropout=dropout,
            batch_first=True
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Enhanced prediction head
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
        # Process each feature stream
        price_embed = self.price_embedding(x['prices'])
        volume_embed = self.volume_embedding(x['volumes'])
        tech_embed = self.tech_embedding(x['tech_features'])
        
        # Combine features
        combined = torch.cat([price_embed, volume_embed, tech_embed], dim=-1)
        x = self.feature_combiner(combined)
        
        # Add positional encoding
        x = x + self.positional_encoding[:x.size(1)].unsqueeze(0)
        
        # Enhanced attention mechanism
        encoded = self.transformer_encoder(x)
        attention_scores = torch.softmax(self.attention_weights(encoded), dim=1)
        attended = torch.sum(attention_scores * encoded, dim=1)
        
        return self.prediction_layers(attended)

    def _create_positional_encoding(self, max_len, d_model):
        pos_encoding = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-np.log(10000.0) / d_model))
        
        pos_encoding[:, 0::2] = torch.sin(position * div_term)
        pos_encoding[:, 1::2] = torch.cos(position * div_term)
        return pos_encoding
