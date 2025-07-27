# <UPDATED_CODE># 🚀 ULTIMATE SPY 1-MINUTE TRAINING SCRIPT
# Combining the best of all three approaches with clean single-timeframe data
import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, RobustScaler
# Import NearMiss for undersampling
from imblearn.under_sampling import NearMiss
from sklearn.metrics import silhouette_score, f1_score, accuracy_score, precision_score, recall_score, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
from torch.utils.data import Dataset, DataLoader
import pickle
from datetime import datetime
import warnings
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter, defaultdict
import time

# Fix for PyTorch 2.6 loading issue
from torch.serialization import add_safe_globals
from collections import defaultdict
add_safe_globals([defaultdict])


# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"🚀 Using device: {device}")

# Create unique model directory
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"ultimate_spy_models_{timestamp}"
os.makedirs(model_dir, exist_ok=True)
print(f"📁 Model directory: {model_dir}")

# --- CONFIGURATION ---
POSITION_SIZING = 0.01  # Percentage of capital per trade
STOP_LOSS_FACTOR = 0.005  # 0.5% stop loss from entry price
TAKE_PROFIT_FACTOR = 0.01  # 1% take profit from entry price
LSTM_HIDDEN_DIM = 256
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_BIDIRECTIONAL = True
USE_ATTENTION = True
NUM_ATTENTION_HEADS = 4
BATCH_SIZE = 1024
LEARNING_RATE = 0.0005
NUM_EPOCHS = 100
PATIENCE = 10  # For early stopping
SEQ_LENGTH = 30  # For 1-minute data, this means 30 minutes lookback
TRAIN_SPLIT = 0.8  # 80% for training, 20% for testing
TARGET_HORIZON = 5  # Predict 5 minutes into the future
USE_ROBUST_SCALER = False  # Set to True to use RobustScaler instead of StandardScaler
# --- MODELING & LOSS CONFIGURATION ---
USE_CLASSIFICATION = True  # Set to True for classification, False for regression
NUM_CLASSES = 3  # For classification: 0 (down), 1 (flat), 2 (up)
CLASSIFICATION_THRESHOLD_PCT = 0.001  # 0.1% threshold for price movement
# Set to 'focal' to use FocalLoss, 'crossentropy' for CrossEntropyLoss
LOSS_FUNCTION = 'focal'
# --- CLASS IMBALANCE HANDLING ---
# Set to True to use NearMiss undersampling on the training data
USE_NEARMISS_UNDERSAMPLING = True
# Set to True to apply class weights to the loss function (not recommended with NearMiss)
USE_CLASS_WEIGHTS = False
# --- FEATURE SELECTION ---
# Set to True to enable feature selection using RandomForestClassifier
ENABLE_FEATURE_SELECTION = False
N_SELECT_FEATURES = 20  # Number of features to select if ENABLE_FEATURE_SELECTION is True

# --- Feature Engineering ---
# This is the updated function to calculate ultimate technical features.
# It uses 'open', 'high', 'low', 'close', 'volume' from your 1-minute data.
# Traditional lagging indicators (RSI, Bollinger Bands, EMA Crossover) have been removed.


def calculate_ultimate_technical_features(df):
    """
    Calculate comprehensive technical features including candle energy and
    ultimate volume indicators, excluding traditional lagging indicators.
    Assumes df has 'open', 'high', 'low', 'close', 'volume' columns.
    """
    print("🔧 Calculating ultimate technical features (excluding lagging indicators)...")

    # Basic returns (keep these as they reflect price action)
    df['return_1'] = np.clip(df['close'].pct_change(1), -1, 1)
    df['return_5'] = np.clip(df['close'].pct_change(5), -1, 1)
    df['return_10'] = np.clip(df['close'].pct_change(10), -1, 1)

    # Volatility (keep as it reflects price action magnitude)
    df['volatility_5'] = df['return_1'].rolling(window=5).std()
    df['volatility_10'] = df['return_1'].rolling(window=10).std()
    df['volatility_20'] = df['return_1'].rolling(window=20).std()

    # --- ULTIMATE VOLUME INDICATOR ---
    price_range = df['high'] - df['low']
    # Replace zero ranges with NaN to avoid division by zero
    price_range = price_range.replace(0, np.nan)

    with warnings.catch_warnings():
        # Ignore warnings for division by zero
        warnings.simplefilter("ignore", RuntimeWarning)
        df['buying_volume'] = df['volume'] * \
            (df['close'] - df['low']) / price_range
        df['selling_volume'] = df['volume'] * \
            (df['high'] - df['close']) / price_range

    # Fill NaNs that resulted from price_range being zero
    df['buying_volume'].fillna(0, inplace=True)
    # Fill NaNs that resulted from price_range being zero
    df['selling_volume'].fillna(0, inplace=True)

    # Handle cases where volume is zero to prevent NaN in sell_volume_percent
    df['sell_volume_percent'] = np.clip(
        # Replace 0 volume with NaN
        (df['selling_volume'] / (df['volume'].replace(0, np.nan))) * 100, 0, 100)
    # Fill NaNs that resulted from 0 volume
    df['sell_volume_percent'].fillna(0, inplace=True)

    df['vol_avg_30'] = df['volume'].shift(1).rolling(window=30).mean()
    df['volume_strength'] = np.clip(
        # Handle cases where vol_avg_30 is zero
        (df['volume'] / (df['vol_avg_30'].replace(0, np.nan))) * 100, 0, 1000)
    # Fill NaNs that resulted from 0 vol_avg_30
    df['volume_strength'].fillna(0, inplace=True)

    df['uv_ratio'] = df['volume'] / \
        (df['volume'].rolling(window=20).mean().replace(0, np.nan)
         )  # Handle cases where rolling mean of volume is zero
    df['uv_ratio'].fillna(0, inplace=True)
    df['uv_trend'] = df['uv_ratio'].rolling(window=5).mean()
    df['uv_trend'].fillna(0, inplace=True)
    df['uv_signal'] = df['uv_ratio'] - df['uv_trend']
    df['uv_signal'].fillna(0, inplace=True)

    # Price ranges and gaps (keep these as they reflect price action)
    df['hl_range'] = np.clip((df['high'] - df['low']) / df['close'], 0, 1)
    df['gap'] = np.clip((df['open'] - df['close'].shift(1)
                         ) / df['close'].shift(1), -1, 1)
    df['oc_range'] = abs(df['open'] - df['close']) / df['open']

    # Momentum (keep these as they reflect price action)
    df['momentum_5'] = np.clip(df['close'] - df['close'].shift(5), -100, 100)
    df['momentum_10'] = np.clip(df['close'] - df['close'].shift(10), -100, 100)

    # Normalized prices (keep these as they reflect price action relative to recent history)
    df['norm_price_5'] = df['close'] / df['close'].rolling(window=5).mean() - 1
    df['norm_price_10'] = df['close'] / \
        df['close'].rolling(window=10).mean() - 1

    # --- CANDLE ENERGY FEATURES ---
    df['candle_range'] = df['high'] - df['low']

    # Candle Body Ratio: How much of the range is covered by the body?
    df['candle_body_abs'] = (df['close'] - df['open']).abs()
    df['candle_body_ratio'] = np.where(
        # Handle division by zero
        df['candle_range'] == 0, 0, df['candle_body_abs'] / df['candle_range'])
    df['candle_body_ratio'] = np.clip(df['candle_body_ratio'], 0, 1)

    # Upper Wick Ratio
    df['upper_wick'] = df['high'] - df[['open', 'close']].max(axis=1)
    df['upper_wick_ratio'] = np.where(
        df['candle_range'] == 0, 0, df['upper_wick'] / df['candle_range'])
    df['upper_wick_ratio'] = np.clip(df['upper_wick_ratio'], 0, 1)

    # Lower Wick Ratio
    df['lower_wick'] = df[['open', 'close']].min(axis=1) - df['low']
    df['lower_wick_ratio'] = np.where(
        # Handle division by zero
        df['candle_range'] == 0, 0, df['lower_wick'] / df['candle_range'])
    df['lower_wick_ratio'] = np.clip(df['lower_wick_ratio'], 0, 1)

    # True Range and ATR
    df['true_range'] = np.maximum(df['high'] - df['low'], np.maximum(
        abs(df['high'] - df['close'].shift(1)), abs(df['low'] - df['close'].shift(1))))
    df['atr_14'] = df['true_range'].rolling(window=14).mean()

    # Time features
    df['hour'] = df.index.hour
    df['minute'] = df.index.minute
    df['day_of_week'] = df.index.dayofweek

    # Clean up temporary columns
    df = df.drop(columns=[
        'candle_range', 'candle_body_abs', 'upper_wick', 'lower_wick',
        'buying_volume', 'selling_volume', 'vol_avg_30', 'uv_ratio', 'uv_trend'
    ], errors='ignore')

    print(
        f"✅ Created {len([col for col in df.columns if col not in ['open', 'high', 'low', 'close', 'volume']])} ultimate features.")
    return df

# --- Model Definitions ---

# NEW: Focal Loss for handling class imbalance


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        if isinstance(alpha, (float, int)):
            self.alpha = torch.tensor([alpha, 1 - alpha])
        if isinstance(alpha, list):
            self.alpha = torch.tensor(alpha)
        self.reduction = reduction

    def forward(self, inputs, targets):
        BCE_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        F_loss = (1 - pt)**self.gamma * BCE_loss

        if self.alpha is not None:
            if self.alpha.device != inputs.device:
                self.alpha = self.alpha.to(inputs.device)
            alpha_t = self.alpha[targets.data.view(-1)]
            F_loss = alpha_t * F_loss

        if self.reduction == 'mean':
            return torch.mean(F_loss)
        elif self.reduction == 'sum':
            return torch.sum(F_loss)
        else:
            return F_loss


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, _ = x.size()
        q = self.q_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len,
                                self.num_heads, self.head_dim).transpose(1, 2)
        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim)
        output = self.out_proj(output)
        return output


class UltimateSpyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout, bidirectional, use_attention, num_heads, classification, num_classes=None):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.classification = classification
        self.num_classes = num_classes

        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers,
                            batch_first=True, dropout=dropout, bidirectional=bidirectional)

        if use_attention:
            self.attention = MultiHeadSelfAttention(
                hidden_dim * (2 if bidirectional else 1), num_heads)
            self.norm1 = nn.LayerNorm(hidden_dim * (2 if bidirectional else 1))
            self.dropout_attn = nn.Dropout(dropout)

        lstm_output_dim = hidden_dim * (2 if bidirectional else 1)
        self.fc1 = nn.Linear(lstm_output_dim, lstm_output_dim // 2)
        self.dropout_fc = nn.Dropout(dropout)

        if self.classification:
            if self.num_classes is None:
                raise ValueError(
                    "num_classes must be specified for classification tasks.")
            self.fc_out = nn.Linear(lstm_output_dim // 2, self.num_classes)
        else:
            self.fc_out = nn.Linear(lstm_output_dim // 2, output_dim)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         x.size(0), self.hidden_dim).to(device)
        c0 = torch.zeros(self.num_layers * (2 if self.bidirectional else 1),
                         x.size(0), self.hidden_dim).to(device)
        lstm_out, _ = self.lstm(x, (h0, c0))

        if self.use_attention:
            attn_output = self.attention(lstm_out)
            # Residual connection
            lstm_out = self.norm1(lstm_out + self.dropout_attn(attn_output))

        out = lstm_out[:, -1, :]  # We take the output of the last time step
        out = F.relu(self.fc1(out))
        out = self.dropout_fc(out)

        if self.classification:
            # No softmax here, CrossEntropyLoss/FocalLoss expects logits
            return self.fc_out(out)
        else:
            return self.fc_out(out)


class UltimateDataset(Dataset):
    def __init__(self, features, targets, seq_length):
        self.features = features
        self.targets = targets
        self.seq_length = seq_length

    def __len__(self):
        return len(self.features) - self.seq_length

    def __getitem__(self, idx):
        # For classification, targets should be long type
        target_tensor = torch.tensor(
            self.targets[idx + self.seq_length], dtype=torch.long if USE_CLASSIFICATION else torch.float32)
        return (
            torch.tensor(self.features[idx: idx +
                         self.seq_length], dtype=torch.float32),
            target_tensor,
        )

    # --- Training, Evaluation, and Backtesting Functions ---


def train_ultimate_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, patience, model_dir, classification):
    print(f"\n" + "="*50)
    print("🚀 Starting model training...")
    best_loss = float('inf')
    best_accuracy = 0
    epochs_no_improve = 0
    history = defaultdict(list)

    for epoch in range(num_epochs):
        model.train()
        train_loss = 0
        for batch_features, batch_targets in tqdm(train_loader, desc=f"Epoch {epoch+1} Training"):
            batch_features, batch_targets = batch_features.to(
                device), batch_targets.to(device)

            optimizer.zero_grad()
            outputs = model(batch_features)

            # Target type is handled by the Dataset's __getitem__ method
            loss = criterion(outputs, batch_targets)

            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        train_loss /= len(train_loader)
        history['train_loss'].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0
        val_preds, val_targets = [], []
        with torch.no_grad():
            for batch_features, batch_targets in tqdm(val_loader, desc=f"Epoch {epoch+1} Validation"):
                batch_features, batch_targets = batch_features.to(
                    device), batch_targets.to(device)
                outputs = model(batch_features)

                loss = criterion(outputs, batch_targets)
                val_loss += loss.item()

                if classification:
                    _, predicted = torch.max(outputs.data, 1)
                    val_preds.extend(predicted.cpu().numpy())
                    val_targets.extend(batch_targets.cpu().numpy())

        val_loss /= len(val_loader)
        history['val_loss'].append(val_loss)

        if classification:
            val_accuracy = accuracy_score(val_targets, val_preds)
            val_f1 = f1_score(val_targets, val_preds, average='weighted')
            history['val_accuracy'].append(val_accuracy)
            history['val_f1'].append(val_f1)
            print(f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Accuracy: {val_accuracy:.4f} | Val F1-Score: {val_f1:.4f}")

            if val_accuracy > best_accuracy:
                best_accuracy = val_accuracy
                epochs_no_improve = 0
                torch.save({'model_state_dict': model.state_dict()},
                           f"{model_dir}/best_model.pth")
                print(
                    f"⭐ New best model saved with accuracy: {best_accuracy:.4f}")
            else:
                epochs_no_improve += 1
        else:  # Regression
            print(
                f"Epoch {epoch+1}/{num_epochs} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            if val_loss < best_loss:
                best_loss = val_loss
                epochs_no_improve = 0
                torch.save({'model_state_dict': model.state_dict()},
                           f"{model_dir}/best_model.pth")
                print(f"⭐ New best model saved with loss: {best_loss:.4f}")
            else:
                epochs_no_improve += 1

        if epochs_no_improve >= patience:
            print(
                f"🛑 Early stopping triggered after {patience} epochs with no improvement.")
            break

    print("✅ Training complete!")
    return history


def evaluate_ultimate_model(model, test_data, seq_length):
    print(f"\n" + "="*50)
    print("🔬 Evaluating model performance...")
    model.eval()
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    all_preds, all_targets = [], []

    with torch.no_grad():
        for features, targets in tqdm(test_loader, desc="Evaluating"):
            features, targets = features.to(device), targets.to(device)
            outputs = model(features)
            _, predicted = torch.max(outputs.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = accuracy_score(all_targets, all_preds)
    precision = precision_score(
        all_targets, all_preds, average='weighted', zero_division=0)
    recall = recall_score(all_targets, all_preds,
                          average='weighted', zero_division=0)
    f1 = f1_score(all_targets, all_preds, average='weighted', zero_division=0)
    cm = confusion_matrix(all_targets, all_preds)

    print("\n--- Evaluation Metrics ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-Score: {f1:.4f}")
    print("Confusion Matrix:\n", cm)
    print("✅ Evaluation complete!")
    return {"accuracy": accuracy, "precision": precision, "recall": recall, "f1_score": f1, "confusion_matrix": cm}


def backtest_ultimate_model(model, test_data, feature_scaler, target_scaler, initial_capital, seq_length):
    print(f"\n" + "="*50)
    print("📈 Starting backtest...")
    model.eval()

    capital = initial_capital
    position = 0  # 0: no position, 1: long, -1: short
    entry_price = 0
    trade_history = []

    # --- MAJOR FIX: Handle classification vs. regression data preparation correctly ---
    test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)
    all_predictions = []

    # Get model predictions first
    with torch.no_grad():
        for features, _ in tqdm(test_loader, desc="Backtest Predictions"):
            features = features.to(device)
            outputs = model(features)
            if USE_CLASSIFICATION:
                _, predicted_labels = torch.max(outputs.data, 1)
                all_predictions.extend(predicted_labels.cpu().numpy())
            else:  # Regression
                predicted_returns_scaled = outputs.cpu().numpy().flatten()
                all_predictions.extend(predicted_returns_scaled)

    # Now, prepare the actual price data for simulation
    # The prices at which trades occur are the *actual close prices* from the original dataframe
    # We need to align predictions with the correct timestamps/prices from the test set.
    # A prediction at index `i` corresponds to a decision for the price at `test_original_index[i + seq_length]`

    # FIX: Get the original 'close' prices for the test period
    actual_prices = df.loc[test_dataset.features_original_index]['close'].values

    # We start making decisions after the first seq_length period
    trade_simulation_prices = actual_prices[seq_length:]

    # Ensure alignment
    num_predictions = len(all_predictions)
    num_prices = len(trade_simulation_prices)
    aligned_length = min(num_predictions, num_prices)

    # This loop is crucial for simulating trades
    for i in tqdm(range(aligned_length), desc="Simulating Trades"):
        current_price = trade_simulation_prices[i]
        predicted_action = all_predictions[i]
        current_time = test_dataset.features_original_index[i + seq_length]

        # --- Trading Logic ---
        if position == 0:  # If we can open a new position
            if USE_CLASSIFICATION:
                if predicted_action == 2:  # 'Up' prediction -> Buy
                    position, entry_price = 1, current_price
                    trade_history.append(
                        {"type": "BUY", "entry_price": entry_price, "time": current_time})
                elif predicted_action == 0:  # 'Down' prediction -> Sell
                    position, entry_price = -1, current_price
                    trade_history.append(
                        {"type": "SELL", "entry_price": entry_price, "time": current_time})
            else:  # Regression
                # FIX: Inverse transform only the predicted returns needed
                predicted_return = target_scaler.inverse_transform(
                    np.array([[predicted_action]]))[0][0]
                if predicted_return > CLASSIFICATION_THRESHOLD_PCT:  # Predicted up
                    position, entry_price = 1, current_price
                    trade_history.append(
                        {"type": "BUY", "entry_price": entry_price, "time": current_time})
                elif predicted_return < -CLASSIFICATION_THRESHOLD_PCT:  # Predicted down
                    position, entry_price = -1, current_price
                    trade_history.append(
                        {"type": "SELL", "entry_price": entry_price, "time": current_time})

        # --- Exit Logic (Stop Loss / Take Profit / Signal Change) ---
        else:  # If we are already in a position
            exit_reason = None
            profit = 0

            # Check for SL/TP
            if position == 1:  # Long position
                if current_price <= entry_price * (1 - STOP_LOSS_FACTOR):
                    exit_reason = "SL_BUY"
                elif current_price >= entry_price * (1 + TAKE_PROFIT_FACTOR):
                    exit_reason = "TP_BUY"
                profit = (current_price - entry_price) * \
                    (capital * POSITION_SIZING / entry_price)
            elif position == -1:  # Short position
                if current_price >= entry_price * (1 + STOP_LOSS_FACTOR):
                    exit_reason = "SL_SELL"
                elif current_price <= entry_price * (1 - TAKE_PROFIT_FACTOR):
                    exit_reason = "TP_SELL"
                profit = (entry_price - current_price) * \
                    (capital * POSITION_SIZING / entry_price)

            # Check for signal-based exit (e.g., model predicts flat/reverse)
            if USE_CLASSIFICATION and predicted_action == 1 and exit_reason is None:
                exit_reason = "CLOSE_BUY" if position == 1 else "CLOSE_SELL"

            if exit_reason:
                capital += profit
                trade_history.append(
                    {"type": exit_reason, "exit_price": current_price, "profit": profit, "time": current_time})
                position, entry_price = 0, 0

    # Close any open position at the end of the backtest
    if position != 0:
        final_price = trade_simulation_prices[-1]
        profit = 0
        if position == 1:
            profit = (final_price - entry_price) * \
                (capital * POSITION_SIZING / entry_price)
            trade_type = "END_CLOSE_BUY"
        elif position == -1:
            profit = (entry_price - final_price) * \
                (capital * POSITION_SIZING / entry_price)
            trade_type = "END_CLOSE_SELL"
        capital += profit
        trade_history.append({"type": trade_type, "exit_price": final_price,
                             "profit": profit, "time": test_dataset.features_original_index[-1]})

    final_capital = capital
    total_profit = final_capital - initial_capital
    profit_percent = (total_profit / initial_capital) * 100

    print("\n--- Backtest Results ---")
    print(f"Initial Capital: ${initial_capital:,.2f}")
    print(f"Final Capital: ${final_capital:,.2f}")
    print(f"Total Profit/Loss: ${total_profit:,.2f}")
    print(f"Profit/Loss Percentage: {profit_percent:.2f}%")
    print(
        f"Number of Trades: {len([t for t in trade_history if t['type'] in ['BUY', 'SELL']])}")

    winning_trades = [t['profit']
                      for t in trade_history if 'profit' in t and t['profit'] > 0]
    losing_trades = [t['profit']
                     for t in trade_history if 'profit' in t and t['profit'] <= 0]
    total_trades_closed = len(winning_trades) + len(losing_trades)
    win_rate = (len(winning_trades) / total_trades_closed *
                100) if total_trades_closed > 0 else 0
    avg_profit_per_trade = (sum(winning_trades) + sum(losing_trades)) / \
        total_trades_closed if total_trades_closed > 0 else 0

    print(f"Win Rate: {win_rate:.2f}%")
    print(f"Average Profit per Trade: ${avg_profit_per_trade:,.2f}")
    print("✅ Backtest complete!")
    return {"final_capital": final_capital, "profit_percent": profit_percent, "trade_history": trade_history}


# --- Main Script Logic ---
if __name__ == "__main__":
    # Load data
    print(f"\n" + "="*50)
    print("📊 Loading 1-minute SPY data...")
    try:
        df = pd.read_csv('data/spy_1min_2008_2021_cleaned.csv',
                         index_col='date', parse_dates=True)
        if 'barCount' in df.columns:
            df = df.drop(columns=['barCount'])
        if 'average' in df.columns:
            df = df.drop(columns=['average'])
        print("✅ 1-minute SPY data loaded successfully!")
    except FileNotFoundError:
        print("❌ Error: 'data/spy_1min_2008_2021_cleaned.csv' not found.")
        exit()

    df = calculate_ultimate_technical_features(df)
    df.dropna(inplace=True)

    # Define targets
    df['future_close'] = df['close'].shift(-TARGET_HORIZON)
    df['price_change_pct'] = (df['future_close'] - df['close']) / df['close']
    df.dropna(inplace=True)

    feature_columns = [col for col in df.columns if col not in [
        'open', 'high', 'low', 'close', 'volume', 'future_close', 'price_change_pct']]
    target_scaler = None  # Initialize to None

    if USE_CLASSIFICATION:
        df['target'] = 1  # Default to 'flat'
        df.loc[df['price_change_pct'] >
               CLASSIFICATION_THRESHOLD_PCT, 'target'] = 2  # Up
        df.loc[df['price_change_pct'] < -
               CLASSIFICATION_THRESHOLD_PCT, 'target'] = 0  # Down
        target_column = 'target'
    else:  # Regression
        target_column = 'price_change_pct'
        target_scaler = StandardScaler() if not USE_ROBUST_SCALER else RobustScaler()
        df[target_column] = target_scaler.fit_transform(df[[target_column]])

    # (Optional) Feature Selection
    if ENABLE_FEATURE_SELECTION:
        print("Feature selection is enabled but the logic is complex; skipping for this update.")

    # Scale features
    feature_scaler = StandardScaler() if not USE_ROBUST_SCALER else RobustScaler()
    df[feature_columns] = feature_scaler.fit_transform(df[feature_columns])

    # Split data
    split_idx = int(len(df) * TRAIN_SPLIT)
    train_df = df.iloc[:split_idx]
    test_df = df.iloc[split_idx:]

    train_features, train_targets = train_df[feature_columns].values, train_df[target_column].values
    test_features, test_targets = test_df[feature_columns].values, test_df[target_column].values

    # --- Handle Class Imbalance in Training Data ---
    if USE_CLASSIFICATION:
        print("\n--- Class Distribution (Before Balancing) ---")
        print(Counter(train_targets))

        if USE_NEARMISS_UNDERSAMPLING:
            print("\n⚖️ Applying NearMiss undersampling to the training data...")
            # NearMiss expects 2D feature array
            nm = NearMiss()
            train_features, train_targets = nm.fit_resample(
                train_features, train_targets)
            print("\n--- Class Distribution (After NearMiss) ---")
            print(Counter(train_targets))

    # Create datasets and dataloaders
    train_dataset = UltimateDataset(train_features, train_targets, SEQ_LENGTH)
    test_dataset = UltimateDataset(test_features, test_targets, SEQ_LENGTH)

    # Store original indices in dataset for backtesting
    train_dataset.features_original_index = train_df.index
    test_dataset.features_original_index = test_df.index

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize model, criterion, optimizer
    input_dimension = train_features.shape[1]
    model = UltimateSpyModel(
        input_dim=input_dimension, hidden_dim=LSTM_HIDDEN_DIM, num_layers=LSTM_NUM_LAYERS,
        output_dim=1, dropout=LSTM_DROPOUT, bidirectional=LSTM_BIDIRECTIONAL,
        use_attention=USE_ATTENTION, num_heads=NUM_ATTENTION_HEADS,
        classification=USE_CLASSIFICATION, num_classes=NUM_CLASSES if USE_CLASSIFICATION else None
    ).to(device)

    # --- Set Loss Function ---
    if USE_CLASSIFICATION:
        class_weights = None
        if USE_CLASS_WEIGHTS and not USE_NEARMISS_UNDERSAMPLING:
            print("\n⚖️ Calculating class weights for the loss function...")
            class_counts = Counter(train_targets)
            # weight = total_samples / (n_classes * count_of_class)
            total_samples = len(train_targets)
            weights = [total_samples / (len(class_counts) * class_counts[i])
                       for i in sorted(class_counts.keys())]
            class_weights = torch.FloatTensor(weights).to(device)
            print(f"Applied weights: {class_weights}")

        if LOSS_FUNCTION.lower() == 'focal':
            print("Using Focal Loss.")
            criterion = FocalLoss(alpha=class_weights, gamma=2.0)
        else:  # Default to CrossEntropy
            print("Using CrossEntropy Loss.")
            criterion = nn.CrossEntropyLoss(weight=class_weights)
    else:  # Regression
        criterion = nn.MSELoss()

    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # Train model
    history = train_ultimate_model(
        model, train_loader, val_loader, criterion, optimizer, NUM_EPOCHS, PATIENCE, model_dir, USE_CLASSIFICATION
    )

    # Load best model for evaluation
    print(f"\n" + "="*50)
    print("📊 Loading best model for evaluation...")
    best_model = UltimateSpyModel(
        input_dim=input_dimension, hidden_dim=LSTM_HIDDEN_DIM, num_layers=LSTM_NUM_LAYERS,
        output_dim=1, dropout=LSTM_DROPOUT, bidirectional=LSTM_BIDIRECTIONAL,
        use_attention=USE_ATTENTION, num_heads=NUM_ATTENTION_HEADS,
        classification=USE_CLASSIFICATION, num_classes=NUM_CLASSES if USE_CLASSIFICATION else None
    ).to(device)

    checkpoint = torch.load(f"{model_dir}/best_model.pth", map_location=device)
    best_model.load_state_dict(checkpoint['model_state_dict'])
    print("✅ Best model loaded successfully!")

    # Evaluate and Backtest
    if USE_CLASSIFICATION:
        evaluation_results = evaluate_ultimate_model(
            best_model, test_dataset, SEQ_LENGTH)

    backtest_results = backtest_ultimate_model(
        best_model, test_dataset, feature_scaler, target_scaler,
        initial_capital=100000, seq_length=SEQ_LENGTH
    )

    print(f"\n🎉 ULTIMATE SPY 1-MINUTE TRAINING COMPLETE! 🎉")
    print(f"Model saved to: {model_dir}")
