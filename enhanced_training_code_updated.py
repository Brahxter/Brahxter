import torch
import torch.nn as nn
import torch.nn.functional as F
import pandas as pd
import numpy as np
from sklearn.preprocessing import RobustScaler
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import Dataset, DataLoader
import pickle
from datetime import datetime
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import defaultdict
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


def create_sequences(data, seq_length):
    """
    Create sequences for time series prediction
    Args:
        data: numpy array of shape (n_samples, n_features)
        seq_length: length of each sequence
    Returns:
        numpy array of shape (n_sequences, seq_length, n_features)
    """
    sequences = []
    n_samples = len(data)

    for i in range(n_samples - seq_length + 1):
        sequence = data[i:(i + seq_length)]
        sequences.append(sequence)

    return np.array(sequences)


# Set device and create model directory
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
model_dir = f"spy_trading_models_{timestamp}"
os.makedirs(model_dir, exist_ok=True)

# Configuration
POSITION_SIZING_CONFIG = {
    'base_size': 0.2,
    'confidence_multiplier': 1.5,
    'volatility_adjuster': 0.7,
    'max_position': 0.5,
}

RISK_MANAGEMENT_CONFIG = {
    'base_stop_loss': -0.01,
    'dynamic_stop_loss_multiplier': 0.5,
    'take_profit_target': 0.02,
}

# Training Parameters
SEQ_LENGTH = 20
NUM_EPOCHS = 50
BATCH_SIZE = 64
LEARNING_RATE = 0.001


def detect_market_regime(returns, n_regimes=4):
    """Detect market regimes using K-means clustering on volatility, trend, and skewness"""
    print("🎯 Detecting market regimes...")

    # Calculate regime features
    volatility = returns.rolling(window=20).std().fillna(0)
    trend = returns.rolling(window=20).mean().fillna(0)
    skewness = returns.rolling(window=20).skew().fillna(0)

    features = np.column_stack([volatility, trend, skewness])
    features = np.nan_to_num(features, nan=0.0)

    # Fit K-means
    kmeans = KMeans(n_clusters=n_regimes, random_state=42, n_init=10)
    regimes = kmeans.fit_predict(features)

    # Analyze regime characteristics
    regime_stats = {}
    for i in range(n_regimes):
        mask = regimes == i
        regime_stats[i] = {
            'count': np.sum(mask),
            'avg_volatility': np.mean(volatility[mask]),
            'avg_trend': np.mean(trend[mask]),
            'avg_return': np.mean(returns[mask]),
            'skew': np.mean(skewness[mask])
        }

    return regimes, regime_stats


def calculate_ultimate_technical_features(df):
    """Calculate comprehensive technical features including volume profile and market microstructure"""
    print("🔧 Calculating ultimate technical features...")

    # Returns and Volatility
    df['return_1'] = np.clip(df['close'].pct_change(1), -1, 1)
    df['return_5'] = np.clip(df['close'].pct_change(5), -1, 1)
    df['volatility_5'] = df['return_1'].rolling(window=5).std()
    df['volatility_20'] = df['return_1'].rolling(window=20).std()

    # Volume Profile
    price_range = df['high'] - df['low']
    epsilon = 1e-9
    price_range_safe = np.where(price_range == 0, epsilon, price_range)

    df['buying_volume'] = df['volume'] * \
        (df['close'] - df['low']) / price_range_safe
    df['selling_volume'] = df['volume'] * \
        (df['high'] - df['close']) / price_range_safe
    df['sell_volume_percent'] = np.clip(
        (df['selling_volume'] / df['volume']) * 100, 0, 100)

    # Market Microstructure
    df['vol_avg_30'] = df['volume'].rolling(window=30).mean()
    df['volume_strength'] = np.clip(
        (df['volume'] / df['vol_avg_30']) * 100, 0, 1000)

    # Candle Energy Features
    df['candle_range'] = df['high'] - df['low']
    df['candle_body'] = abs(df['close'] - df['open'])
    df['candle_body_ratio'] = np.clip(
        df['candle_body'] / df['candle_range'], 0, 1)

    # Price Action
    df['upper_wick'] = (df['high'] - df[['open', 'close']
                                        ].max(axis=1)) / df['candle_range']
    df['lower_wick'] = (df[['open', 'close']].min(
        axis=1) - df['low']) / df['candle_range']

    # Technical Indicators
    df['rsi_14'] = calculate_rsi(df['close'], 14)
    df['bb_position'] = calculate_bollinger_position(df['close'])

    return df


def calculate_rsi(prices, period=14):
    """Calculate RSI indicator"""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def calculate_bollinger_position(prices, period=20):
    """Calculate position within Bollinger Bands"""
    ma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    upper = ma + 2 * std
    lower = ma - 2 * std
    return (prices - lower) / (upper - lower)


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.embed_dim = embed_dim

        self.wq = nn.Linear(embed_dim, embed_dim)
        self.wk = nn.Linear(embed_dim, embed_dim)
        self.wv = nn.Linear(embed_dim, embed_dim)
        self.fc_out = nn.Linear(embed_dim, embed_dim)

    def forward(self, x):
        batch_size, seq_len, hidden_dim = x.size()

        q = self.wq(x).view(batch_size, seq_len, self.num_heads,
                            self.head_dim).transpose(1, 2)
        k = self.wk(x).view(batch_size, seq_len, self.num_heads,
                            self.head_dim).transpose(1, 2)
        v = self.wv(x).view(batch_size, seq_len, self.num_heads,
                            self.head_dim).transpose(1, 2)

        scores = torch.matmul(q, k.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        context = torch.matmul(attention_weights, v)

        context = context.transpose(1, 2).contiguous().view(
            batch_size, seq_len, self.embed_dim)
        output = self.fc_out(context)
        return output, attention_weights


class ResidualLSTMBlock(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, bidirectional):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim,
                            batch_first=True, bidirectional=bidirectional)
        self.dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.fc_res = nn.Linear(
            input_dim, self.output_dim) if input_dim != self.output_dim else None

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        residual = self.fc_res(x) if self.fc_res else x
        return self.dropout(lstm_out + residual)


class UltimateSpyModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim, dropout,
                 bidirectional=True, use_attention=True, num_heads=4,
                 classification=True, num_classes=3):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.use_attention = use_attention
        self.classification = classification

        # Stacked Residual LSTM Blocks
        self.lstm_blocks = nn.ModuleList([
            ResidualLSTMBlock(
                input_dim if i == 0 else hidden_dim * 2,
                hidden_dim, dropout, bidirectional
            ) for i in range(num_layers)
        ])

        if use_attention:
            self.attention = MultiHeadAttention(hidden_dim * 2, num_heads)
            self.layer_norm = nn.LayerNorm(hidden_dim * 2)

        # Prediction heads
        final_dim = hidden_dim * 2
        self.regression_head = nn.Sequential(
            nn.Linear(final_dim, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, output_dim)
        )

        if classification:
            self.classification_head = nn.Sequential(
                nn.Linear(final_dim, 128),
                nn.BatchNorm1d(128),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(128, num_classes)
            )

    def forward(self, x):
        # Process through LSTM blocks
        for block in self.lstm_blocks:
            x = block(x)

        if self.use_attention:
            attn_out, _ = self.attention(x)
            x = self.layer_norm(x + attn_out)

        # Get final timestep features
        features = x[:, -1, :]

        # Generate predictions
        reg_out = self.regression_head(
            features).squeeze()  # Remove extra dimension

        if self.classification:
            cls_out = self.classification_head(features)
            return reg_out, cls_out
        return reg_out


class StockDataset(Dataset):
    def __init__(self, X, y_reg, y_cls=None):
        self.X = torch.FloatTensor(X)
        self.y_reg = torch.FloatTensor(y_reg).squeeze()  # Ensure 1D tensor
        self.y_cls = torch.LongTensor(y_cls) if y_cls is not None else None

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        if self.y_cls is not None:
            return self.X[idx], self.y_reg[idx], self.y_cls[idx]
        return self.X[idx], self.y_reg[idx]


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2):
        super().__init__()
        self.alpha = torch.tensor(alpha) if alpha is not None else None
        if self.alpha is not None:
            self.alpha = self.alpha.to(device)
        self.gamma = gamma

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)

        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * (1 - pt)**self.gamma * ce_loss
        else:
            focal_loss = (1 - pt)**self.gamma * ce_loss

        return focal_loss.mean()


def train_model(model, train_loader, val_loader, num_epochs, learning_rate, classification=True, alpha_weights=None):
    print("🚀 Starting model training...")

    # Initialize optimizer and schedulers
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5)

    # Loss functions
    regression_criterion = nn.MSELoss()
    classification_criterion = FocalLoss(
        alpha=alpha_weights) if classification else None

    # Tracking metrics
    metrics = {
        'train_losses': [],
        'val_losses': [],
        'train_accuracies': [],
        'val_accuracies': [],
        'f1_scores': [],
        'precision': [],
        'recall': [],
        'confusion_matrices': []
    }

    best_val_loss = float('inf')
    best_val_f1 = 0
    patience_counter = 0

    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0
        train_correct = 0
        train_total = 0

        for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
            optimizer.zero_grad()

            if classification:
                X_batch, y_reg_batch, y_cls_batch = [
                    b.to(device) for b in batch]
                reg_out, cls_out = model(X_batch)
                reg_loss = regression_criterion(reg_out, y_reg_batch)
                cls_loss = classification_criterion(cls_out, y_cls_batch)
                loss = reg_loss + cls_loss

                # Calculate training accuracy
                _, predicted = torch.max(cls_out.data, 1)
                train_total += y_cls_batch.size(0)
                train_correct += (predicted == y_cls_batch).sum().item()
            else:
                X_batch, y_reg_batch = [b.to(device) for b in batch]
                reg_out = model(X_batch)
                loss = regression_criterion(reg_out.squeeze(), y_reg_batch)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()

        avg_train_loss = train_loss / len(train_loader)
        train_accuracy = 100 * train_correct / train_total if classification else None

        # Validation phase
        model.eval()
        val_loss = 0
        val_correct = 0
        val_total = 0
        predictions = []
        true_values = []

        with torch.no_grad():
            for batch in val_loader:
                if classification:
                    X_batch, y_reg_batch, y_cls_batch = [
                        b.to(device) for b in batch]
                    reg_out, cls_out = model(X_batch)
                    reg_loss = regression_criterion(
                        reg_out.squeeze(), y_reg_batch)
                    cls_loss = classification_criterion(cls_out, y_cls_batch)
                    loss = reg_loss + cls_loss

                    _, predicted = torch.max(cls_out.data, 1)
                    val_total += y_cls_batch.size(0)
                    val_correct += (predicted == y_cls_batch).sum().item()
                    predictions.extend(predicted.cpu().numpy())
                    true_values.extend(y_cls_batch.cpu().numpy())
                else:
                    X_batch, y_reg_batch = [b.to(device) for b in batch]
                    reg_out = model(X_batch)
                    loss = regression_criterion(reg_out.squeeze(), y_reg_batch)

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = 100 * val_correct / val_total if classification else None

        # Calculate F1 score for classification
        if classification:
            f1 = f1_score(true_values, predictions, average='weighted')
            metrics['f1_scores'].append(f1)

            # New metrics
            precision = precision_score(
                true_values, predictions, average='weighted')
            recall = recall_score(true_values, predictions, average='weighted')
            conf_matrix = confusion_matrix(true_values, predictions)

            # Store in metrics dictionary
            metrics['precision'].append(precision)
            metrics['recall'].append(recall)
            metrics['confusion_matrices'].append(conf_matrix)

            # Enhanced printing
            print(f"Precision: {precision:.4f}, Recall: {recall:.4f}")
            print("\nConfusion Matrix:")
            print(conf_matrix)

        # Update learning rate
        lr_scheduler.step(avg_val_loss)

        # Store metrics
        metrics['train_losses'].append(avg_train_loss)
        metrics['val_losses'].append(avg_val_loss)
        if classification:
            metrics['train_accuracies'].append(train_accuracy)
            metrics['val_accuracies'].append(val_accuracy)

        # Early stopping and model saving
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_val_loss,
                'metrics': metrics
            }, f"{model_dir}/best_model.pth")
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= 10:
                print("🛑 Early stopping triggered!")
                break

        # Print epoch results
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        print(
            f"Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
        if classification:
            print(
                f"Train Acc: {train_accuracy:.2f}%, Val Acc: {val_accuracy:.2f}%")
            print(f"F1 Score: {f1:.4f}")
        print(f"Learning Rate: {optimizer.param_groups[0]['lr']:.6f}")

    # Plot training metrics
    plot_training_metrics(metrics, classification)

    return model, metrics


def plot_training_metrics(metrics, classification):
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(metrics['train_losses'], label='Train Loss')
    plt.plot(metrics['val_losses'], label='Val Loss')
    plt.title('Loss Over Time')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()

    if classification:
        plt.subplot(1, 2, 2)
        plt.plot(metrics['train_accuracies'], label='Train Acc')
        plt.plot(metrics['val_accuracies'], label='Val Acc')
        plt.plot(metrics['f1_scores'], label='F1 Score')
        plt.title('Accuracy & F1 Score')
        plt.xlabel('Epoch')
        plt.ylabel('Percentage')
        plt.legend()

    plt.tight_layout()
    plt.savefig(f"{model_dir}/training_metrics.png")
    plt.close()


def evaluate_model(model, test_loader):
    model.eval()
    all_preds = []
    all_targets = []
    test_loss = 0

    with torch.no_grad():
        for batch in test_loader:
            X_batch, y_reg_batch, y_cls_batch = [b.to(device) for b in batch]
            reg_out, cls_out = model(X_batch)

            _, predicted = torch.max(cls_out.data, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y_cls_batch.cpu().numpy())

    # Calculate metrics
    results = {
        'accuracy': accuracy_score(all_targets, all_preds),
        'f1_score': f1_score(all_targets, all_preds, average='weighted'),
        'precision': precision_score(all_targets, all_preds, average='weighted'),
        'recall': recall_score(all_targets, all_preds, average='weighted'),
        'confusion_matrix': confusion_matrix(all_targets, all_preds)
    }

    return results


def backtest_model(model, test_data, feature_scaler, target_scaler, initial_capital=100000):
    print("💰 Starting backtest simulation...")

    model.eval()
    portfolio_value = [initial_capital]
    current_position = 0  # -1: Short, 0: Neutral, 1: Long
    position_size = 0
    trades = []
    current_capital = initial_capital
    num_shares = 0

    with torch.no_grad():
        for i in range(len(test_data['X'])):
            X = torch.FloatTensor(test_data['X'][i:i+1]).to(device)
            current_price = test_data['prices'][i]
            timestamp = test_data['timestamps'][i]

            # Get model predictions
            if model.classification:
                reg_out, cls_pred = model(X)
                cls_probs = F.softmax(cls_pred, dim=1)
                predicted_class = torch.argmax(cls_pred, dim=1).item()
                confidence = cls_probs[0][predicted_class].item()
                predicted_return = target_scaler.inverse_transform(
                    reg_out.cpu().numpy().reshape(-1, 1))[0][0]
            else:
                reg_out = model(X)
                predicted_return = target_scaler.inverse_transform(
                    reg_out.cpu().numpy().reshape(-1, 1))[0][0]
                confidence = abs(predicted_return)

            # Dynamic position sizing based on confidence
            base_position_size = POSITION_SIZING_CONFIG['base_size'] * \
                current_capital
            position_size = min(
                base_position_size *
                (1 + confidence *
                 POSITION_SIZING_CONFIG['confidence_multiplier']),
                current_capital * POSITION_SIZING_CONFIG['max_position']
            )

            # Trading logic
            if current_position == 0:  # No position
                if predicted_class == 2:  # Buy signal
                    num_shares = position_size / current_price
                    current_capital -= (current_price * num_shares)
                    current_position = 1
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'BUY',
                        'price': current_price,
                        'shares': num_shares,
                        'confidence': confidence
                    })
                elif predicted_class == 0:  # Short signal
                    num_shares = position_size / current_price
                    current_capital += (current_price * num_shares)
                    current_position = -1
                    trades.append({
                        'timestamp': timestamp,
                        'type': 'SHORT',
                        'price': current_price,
                        'shares': num_shares,
                        'confidence': confidence
                    })

            # Position management
            elif current_position != 0:
                current_trade = trades[-1]
                entry_price = current_trade['price']

                if current_position == 1:  # Long position
                    profit_loss_pct = (
                        current_price - entry_price) / entry_price
                else:  # Short position
                    profit_loss_pct = (
                        entry_price - current_price) / entry_price

                # Dynamic stop loss
                stop_loss = RISK_MANAGEMENT_CONFIG['base_stop_loss'] * (
                    1 + abs(predicted_return) *
                    RISK_MANAGEMENT_CONFIG['dynamic_stop_loss_multiplier']
                )

                # Exit conditions
                exit_signal = (
                    profit_loss_pct <= stop_loss or  # Stop loss
                    # Take profit
                    profit_loss_pct >= RISK_MANAGEMENT_CONFIG['take_profit_target'] or
                    # Signal reversal for long
                    (current_position == 1 and predicted_class == 0) or
                    # Signal reversal for short
                    (current_position == -1 and predicted_class == 2)
                )

                if exit_signal:
                    if current_position == 1:
                        current_capital += (current_price * num_shares)
                        trade_type = "SELL"
                    else:
                        current_capital -= (current_price * num_shares)
                        trade_type = "COVER"

                    profit_loss = (current_price - entry_price) * \
                        num_shares * current_position
                    trades[-1].update({
                        'exit_timestamp': timestamp,
                        'exit_price': current_price,
                        'exit_type': trade_type,
                        'profit_loss': profit_loss,
                        'return_pct': profit_loss_pct
                    })

                    num_shares = 0
                    current_position = 0

            # Update portfolio value
            portfolio_value.append(
                current_capital + (num_shares * current_price * current_position))

    # Calculate performance metrics
    total_trades = len([t for t in trades if 'exit_price' in t])
    winning_trades = len(
        [t for t in trades if 'profit_loss' in t and t['profit_loss'] > 0])
    total_profit = sum(t['profit_loss'] for t in trades if 'profit_loss' in t)
    max_drawdown = calculate_max_drawdown(portfolio_value)

    results = {
        'portfolio_value': portfolio_value,
        'trades': trades,
        'total_return': (portfolio_value[-1] - initial_capital) / initial_capital,
        'win_rate': winning_trades / total_trades if total_trades > 0 else 0,
        'total_trades': total_trades,
        'total_profit': total_profit,
        'max_drawdown': max_drawdown
    }

    print(f"✅ Backtest Results:")
    print(f"Total Return: {results['total_return']:.2%}")
    print(f"Win Rate: {results['win_rate']:.2%}")
    print(f"Total Trades: {results['total_trades']}")
    print(f"Max Drawdown: {results['max_drawdown']:.2%}")

    return results


def calculate_max_drawdown(portfolio_values):
    peak = portfolio_values[0]
    max_drawdown = 0

    for value in portfolio_values:
        if value > peak:
            peak = value
        drawdown = (peak - value) / peak
        max_drawdown = max(max_drawdown, drawdown)

    return max_drawdown


if __name__ == "__main__":
    print("🚀 Starting SPY Trading System")

    try:
        # Load and prepare data
        df = pd.read_csv("data/spy_5min_data_2021_2024.csv")
        print(f"✅ Loaded {len(df)} rows of data")

        # Standardize column names
        df.columns = df.columns.str.lower()
        required_columns = ['timestamp', 'open',
                            'high', 'low', 'close', 'volume']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"Missing required column: {col}")

        # Convert timestamp and set index
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        df.set_index('timestamp', inplace=True)
        print(f"📅 Date range: {df.index.min()} to {df.index.max()}")

        # Calculate features
        df = calculate_ultimate_technical_features(df)
        print(f"✨ Created {len(df.columns)} features")

        # Clean data
        df = df.ffill().bfill()  # Updated to use newer methods

        # Detect market regimes
        regimes, regime_stats = detect_market_regime(df['return_1'])
        df['market_regime'] = regimes
        print("\n🎯 Market Regime Statistics:")
        for regime, stats in regime_stats.items():
            print(
                f"Regime {regime}: {stats['count']} samples, Avg Return: {stats['avg_return']:.4f}")

        # Prepare features and targets
        feature_columns = [col for col in df.columns if col not in
                           ['open', 'high', 'low', 'close', 'volume', 'return_1']]
        X = df[feature_columns].values
        y = df['return_1'].values

        # Create classification targets
        returns_for_classification = df['return_1'].copy()
        y_class = pd.qcut(returns_for_classification, q=3,
                          labels=[0, 1, 2]).astype(int)

        print("\n📊 Class distribution:")
        class_distribution = pd.Series(y_class).value_counts(normalize=True)
        for cls, pct in class_distribution.items():
            print(f"Class {cls}: {pct:.2%}")

        # Split data
        train_size = int(len(df) * 0.7)
        val_size = int(len(df) * 0.15)

        splits = {
            'train': slice(None, train_size),
            'val': slice(train_size, train_size + val_size),
            'test': slice(train_size + val_size, None)
        }

        # Create sequences for each split
        X_dict = {k: X[v] for k, v in splits.items()}
        X_dict = {k: create_sequences(v, SEQ_LENGTH)
                  for k, v in X_dict.items()}

        y_dict = {k: y[v] for k, v in splits.items()}
        y_dict = {k: y_dict[k][SEQ_LENGTH-1:]
                  for k in splits.keys()}  # Align with sequences

        y_class_dict = {k: y_class[v] for k, v in splits.items()}
        y_class_dict = {k: y_class_dict[k][SEQ_LENGTH-1:]
                        for k in splits.keys()}  # Align with sequences

        print("\n📈 Sequence shapes:")
        for split, data in X_dict.items():
            print(f"{split}: {data.shape}")

        # Scale data
        feature_scaler = RobustScaler()
        target_scaler = RobustScaler()

        # Scale each timestep in sequences
        X_scaled = {}
        for split in splits.keys():
            # Reshape to 2D for scaling
            shape = X_dict[split].shape
            X_reshaped = X_dict[split].reshape(-1, shape[-1])

            if split == 'train':
                X_scaled_reshaped = feature_scaler.fit_transform(X_reshaped)
            else:
                X_scaled_reshaped = feature_scaler.transform(X_reshaped)

            # Reshape back to sequences
            X_scaled[split] = X_scaled_reshaped.reshape(shape)

        y_scaled = {
            'train': target_scaler.fit_transform(y_dict['train'].reshape(-1, 1)),
            'val': target_scaler.transform(y_dict['val'].reshape(-1, 1)),
            'test': target_scaler.transform(y_dict['test'].reshape(-1, 1))
        }

        # Create datasets and dataloaders
        datasets = {
            'train': StockDataset(X_scaled['train'], y_scaled['train'], y_class_dict['train']),
            'val': StockDataset(X_scaled['val'], y_scaled['val'], y_class_dict['val']),
            'test': StockDataset(X_scaled['test'], y_scaled['test'], y_class_dict['test'])
        }

        dataloaders = {
            'train': DataLoader(datasets['train'], batch_size=BATCH_SIZE, shuffle=True),
            'val': DataLoader(datasets['val'], batch_size=BATCH_SIZE),
            'test': DataLoader(datasets['test'], batch_size=BATCH_SIZE)
        }

        # Initialize model
        model = UltimateSpyModel(
            input_dim=len(feature_columns),
            hidden_dim=256,
            num_layers=2,
            output_dim=1,
            dropout=0.3,
            bidirectional=True,
            use_attention=True,
            num_heads=4,
            classification=True
        ).to(device)

        print(f"\n🧠 Model architecture:")
        print(f"Input features: {len(feature_columns)}")
        print(f"Parameters: {sum(p.numel() for p in model.parameters())}")

        # Train model - now capturing both model and metrics
        trained_model, training_metrics = train_model(
            model,
            dataloaders['train'],
            dataloaders['val'],
            num_epochs=NUM_EPOCHS,
            learning_rate=LEARNING_RATE,
            classification=True
        )

        print("\n📊 Evaluating model performance...")
        evaluation_results = evaluate_model(trained_model, dataloaders['test'])

        print("\n🎯 Test Set Performance:")
        print(f"Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"F1 Score: {evaluation_results['f1_score']:.4f}")
        print(f"Precision: {evaluation_results['precision']:.4f}")
        print(f"Recall: {evaluation_results['recall']:.4f}")
        print("\nConfusion Matrix:")
        print(evaluation_results['confusion_matrix'])

        print("\n💰 Running backtest simulation...")
        test_data = {
            'X': X_scaled['test'],
            'prices': df['close'].values[splits['test']][SEQ_LENGTH-1:],
            'timestamps': df.index[splits['test']][SEQ_LENGTH-1:]
        }

        backtest_results = backtest_model(
            trained_model,
            test_data,
            feature_scaler,
            target_scaler
        )

        # Save comprehensive results
        results = {
            'training_metrics': training_metrics,
            'evaluation_results': evaluation_results,
            'backtest_results': backtest_results,
            'model_config': {
                'feature_columns': feature_columns,
                'seq_length': SEQ_LENGTH,
                'hidden_dim': 256,
                'num_layers': 2
            },
            'feature_importance': pd.Series(
                trained_model.regression_head[0].weight.data.cpu(
                ).numpy().squeeze(),
                index=feature_columns
            ).sort_values(ascending=False)
        }

        # Save results and model
        results_path = f"{model_dir}/results.pkl"
        with open(results_path, "wb") as f:
            pickle.dump(results, f)

        print("\n✨ Training and evaluation complete!")
        print(f"📁 Results saved to: {results_path}")
        print("\n📊 Key Performance Metrics:")
        print(
            f"Final Training Accuracy: {training_metrics['train_accuracies'][-1]:.4f}")
        print(
            f"Final Validation Accuracy: {training_metrics['val_accuracies'][-1]:.4f}")
        print(f"Test Set Accuracy: {evaluation_results['accuracy']:.4f}")
        print(f"Backtest Return: {backtest_results['total_return']:.2%}")

    except Exception as e:
        print(f"\n❌ Error during execution: {str(e)}")
        import traceback
        print(traceback.format_exc())
