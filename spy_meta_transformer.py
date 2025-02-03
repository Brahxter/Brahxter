import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SPYDataset(Dataset):
    def __init__(self, csv_file):
        self.data = pd.read_csv(csv_file)

        # Rename 'TimeStamp' to 'date' for consistency
        self.data.rename(columns={'TimeStamp': 'date'}, inplace=True)

        # Convert 'date' column to datetime and extract trading day
        self.data['trading_day'] = pd.to_datetime(self.data['date']).dt.date
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
            'prices': torch.FloatTensor(prices),   # expected shape: (6, 4)
            'volumes': torch.FloatTensor(volumes),   # expected shape: (6, 1)
            'target': torch.FloatTensor([target])
        }


class MetaLearningTransformer(nn.Module):
    def __init__(self, d_model=256, nhead=8, num_layers=4, dropout=0.2):
        super().__init__()
        self.d_model = d_model
        # Add an input projection layer to map the 4 features to d_model.
        self.input_projection = nn.Linear(4, d_model)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_layers,
            num_decoder_layers=num_layers,
            dropout=dropout,
            batch_first=True
        )
        # Output layer to map transformer output to a single prediction value
        self.output_layer = nn.Linear(d_model, 1)

        # Dummy learning rate for adaptation routines
        self.learning_rate = 0.001

    def forward(self, x):
        # x is expected to be of shape (batch, seq, 4)
        # Apply the linear projection so that features become d_model.
        x = self.input_projection(x)  # Now shape: (batch, seq, d_model)
        # Create a target sequence as a clone of x.
        tgt = x.clone()

        # If x has no batch dim, add one.
        if len(x.shape) == 2:
            x = x.unsqueeze(0)
        if len(tgt.shape) == 2:
            tgt = tgt.unsqueeze(0)

        # Transformer expects inputs as (seq_len, batch, d_model)
        x = x.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        # Shape: (seq_len, batch, d_model)
        output = self.transformer(src=x, tgt=tgt)

        # Get the last output in the sequence and project to a single prediction
        output = output[-1]  # Shape: (batch, d_model)
        output = self.output_layer(output)  # Shape: (batch, 1)
        return output

    # --- Meta-learning Adaptation Mechanisms ---

    def adapt(self, support_set):
        adapted_params = self.get_quick_weights(support_set)
        return adapted_params

    def get_quick_weights(self, data):
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

    def calculate_loss(self, data):
        # Dummy loss; replace with proper meta-learning loss calculation.
        return torch.tensor(0.0, requires_grad=True)
