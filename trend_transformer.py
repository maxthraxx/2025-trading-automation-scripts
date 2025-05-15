# ================================================================
# WARNING: THIS CODE IS FUNDAMENTALLY BROKEN – DO NOT USE
# This implementation is deeply flawed and unreliable.
# It was originally based on the open-source GPT-2 model,
# but contains critical design and logic issues that render it unsafe
# or unsuitable for any real-world application.
# KEEP THIS FILE FOR REFERENCE ONLY – NOT FOR USE.
# ================================================================


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
from torch import nn
from torch.utils.data import DataLoader, TensorDataset, random_split

# --- Config ---
SEQUENCE_LENGTH = 64
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
HORIZON = 1


# --- Feature Engineering ---
def create_features(df, trend_lookback=5):
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    rng.replace(0, np.nan, inplace=True)

    body_size = (c - o) / (rng + 1e-9)
    close_position = (c - l) / (rng + 1e-9)
    trend_momentum = (c - c.shift(trend_lookback)) / (rng + 1e-9)
    breakout_high = (h - h.shift(1)) / (rng + 1e-9)
    breakout_low_inv = -(l - l.shift(1)) / (rng + 1e-9)

    upper_wick = (h - np.maximum(c, o)) / (rng + 1e-9)
    lower_wick = (np.minimum(c, o) - l) / (rng + 1e-9)
    wick_polarity = np.tanh(lower_wick - upper_wick)
    wick_ratio = (lower_wick - upper_wick) / (lower_wick + upper_wick + 1e-9)
    wick_ratio = wick_ratio.clip(-10, 10)
    range_percent = rng / (o + 1e-9)
    body_vs_range = body_size * range_percent

    features = pd.DataFrame(
        {
            "close": c,
            "body_size": body_size,
            "close_position": close_position,
            "trend_momentum": trend_momentum,
            "breakout_high": breakout_high,
            "breakout_low_inv": breakout_low_inv,
            "upper_wick": upper_wick,
            "lower_wick": lower_wick,
            "wick_polarity": wick_polarity,
            "wick_ratio": wick_ratio,
            "range_percent": range_percent,
            "body_vs_range": body_vs_range,
        },
        index=df.index,
    )

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    features.dropna(inplace=True)
    return features


def create_labels(df, horizon=1):
    df["future_return"] = df["close"].shift(-horizon) / df["close"] - 1
    df["label"] = (df["future_return"] > 0).astype(int)
    return df.dropna().reset_index(drop=True)


def to_sequences(X, y, sequence_length=64):
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i : i + sequence_length])
        y_seq.append(y[i + sequence_length])
    return np.array(X_seq), np.array(y_seq)


# --- Transformer Model ---
class TransformerClassifier(nn.Module):
    def __init__(self, input_dim, seq_len, n_heads=2, n_layers=1, dim_feedforward=64, dropout=0.3):
        super().__init__()
        self.embedding = nn.Linear(input_dim, 32)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=32,
            nhead=n_heads,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * seq_len, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 2),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformer(x)
        return self.classifier(x)


# --- Visualization ---
def plot_predictions(y_true, y_pred):
    plt.figure(figsize=(12, 5))
    plt.plot(y_true, label="True", alpha=0.7)
    plt.plot(y_pred, label="Predicted", alpha=0.7)
    plt.title("Up/Down Regime Prediction")
    plt.xlabel("Sample")
    plt.ylabel("Direction (0=Down, 1=Up)")
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_confusion(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Down", "Up"])
    disp.plot(cmap="Blues")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()


# --- Load and Prepare Data ---
df = pd.read_csv("historical_price_data.csv")  # Replace with your file path
df.columns = [col.strip().lower() for col in df.columns]

# Safety check
required_cols = {"open", "high", "low", "close"}
if not required_cols.issubset(df.columns):
    raise ValueError(f"Missing required OHLC columns: {required_cols - set(df.columns)}")

# Dynamically set SEQUENCE_LENGTH and HORIZON based on your timeframe
# Example: if each row is 15 minutes, then 96 = 1 day
candle_interval = 15  # minutes
lookback_days = 1
prediction_days = 0.25 # 6 hours

# Calculate how many rows = 1 day
candles_per_day = int(24 * 60 / candle_interval)

SEQUENCE_LENGTH = candles_per_day * lookback_days     # how many candles to look back
HORIZON = candles_per_day * prediction_days           # how far ahead to predict

print(f"Lookback window: {SEQUENCE_LENGTH} candles (~{lookback_days} day)")
print(f"Prediction horizon: {HORIZON} candles (~{prediction_days} day)")

# Feature + Label
features = create_features(df, trend_lookback=SEQUENCE_LENGTH)
features = create_labels(features, horizon=HORIZON)

# Only keep model input columns
feature_cols = [col for col in features.columns if col not in ["close", "future_return", "label"]]
X, y = to_sequences(features[feature_cols].values, features["label"].values, sequence_length=SEQUENCE_LENGTH)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
dataset = TensorDataset(torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.long))
train_len = int(0.8 * len(dataset))
train_set, test_set = random_split(dataset, [train_len, len(dataset) - train_len])
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE)

# --- Model Setup ---
model = TransformerClassifier(input_dim=X.shape[2], seq_len=SEQUENCE_LENGTH).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

# --- Training ---
model.train()
for epoch in range(EPOCHS):
    epoch_loss, correct, total = 0, 0, 0
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        outputs = model(X_batch)
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item()
        _, predicted = torch.max(outputs, 1)
        correct += (predicted == y_batch).sum().item()
        total += y_batch.size(0)
    print(f"Epoch {epoch + 1}/{EPOCHS} - Loss: {epoch_loss:.4f} - Accuracy: {100 * correct / total:.2f}%")

# --- Evaluation ---
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for X_batch, y_batch in test_loader:
        X_batch = X_batch.to(device)
        outputs = model(X_batch)
        _, predicted = torch.max(outputs, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(y_batch.numpy())

# --- Visual Diagnostics ---
plot_predictions(all_labels, all_preds)
plot_confusion(all_labels, all_preds)

# --- Predict Next Regime with Confidence ---
with torch.no_grad():
    latest_seq = torch.tensor(X[-1:], dtype=torch.float32).to(device)
    logits = model(latest_seq)
    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
    next_regime = np.argmax(probs)
    confidence_up = probs[1] * 100
    confidence_down = probs[0] * 100

    print(f"Predicted Next Regime: {'UP' if next_regime == 1 else 'DOWN'}")
    print(f"Confidence: {confidence_up:.2f}% UP | {confidence_down:.2f}% DOWN")
