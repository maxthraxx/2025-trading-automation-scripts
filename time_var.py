"""Copyright (C) 2025 James Sawyer
All rights reserved.

This script and the associated files are private
and confidential property. Unauthorized copying of
this file, via any medium, and the divulgence of any
contained information without express written consent
is strictly prohibited.

This script is intended for personal use only and should
not be distributed or used in any commercial or public
setting unless otherwise authorized by the copyright holder.
By using this script, you agree to abide by these terms.

DISCLAIMER: This script is provided 'as is' without warranty
of any kind, either express or implied, including, but not
limited to, the implied warranties of merchantability,
fitness for a particular purpose, or non-infringement. In no
event shall the authors or copyright holders be liable for
any claim, damages, or other liability, whether in an action
of contract, tort or otherwise, arising from, out of, or in
connection with the script or the use or other dealings in
the script.
"""


import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.fft import fft
from sklearn.ensemble import RandomForestRegressor
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# Constants
TREND_LOOKBACK = 3
TRAIN_SPLIT_RATIO = 0.8
RANDOM_STATE = 42

# Watermark settings
WATERMARK_TEXT = "Time-Variant Features - James Sawyer 2025"
WATERMARK_ALPHA = 0.25
WATERMARK_FONTSIZE = 16
WATERMARK_COLOR = "gray"
WATERMARK_ROTATION = 30

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger()


def load_and_prepare_data(filepath="backtest_prices.csv"):
    """Load and prepare the initial dataset."""
    df = pd.read_csv(filepath)
    df["snapshotTime"] = pd.to_datetime(df["snapshotTime"], format="%Y:%m:%d-%H:%M:%S")
    df = df.sort_values("snapshotTime").reset_index(drop=True)
    logger.info(f"Loaded {len(df)} rows. Head:\n{df.head()}")
    return df


def normalize_within_window(df, window_size, time_col="snapshotTime"):
    """Create time-invariant features within specified time windows."""
    window_label = f"window_{window_size}min"
    df[window_label] = df[time_col].astype(np.int64) // (window_size * 60 * 1_000_000_000)
    grouped = df.groupby(window_label)

    for col in ["open", "close", "high", "low"]:
        df[f"{col}_min_{window_size}"] = grouped[col].transform("min")
        df[f"{col}_max_{window_size}"] = grouped[col].transform("max")
        df[f"{col}_norm_{window_size}"] = (df[col] - df[f"{col}_min_{window_size}"]) / (df[f"{col}_max_{window_size}"] - df[f"{col}_min_{window_size}"] + 1e-12)

    df[f"range_{window_size}"] = df[f"high_max_{window_size}"] - df[f"low_min_{window_size}"]
    df[f"pos_in_{window_size}"] = grouped.cumcount() / grouped.cumcount().groupby(
        df[window_label],
    ).transform("max").replace(0, 1)
    return df


def create_window_features(df, windows=[60, 480]):
    """Create time-invariant features for multiple time windows."""
    logger.info("Creating time-invariant features for windows...")

    # Apply normalization for all windows
    for window_size in windows:
        df = normalize_within_window(df, window_size)

    # Create derived features for all windows
    for win in windows:
        window_label = f"window_{win}min"
        df[f"dist_to_high_{win}"] = (df[f"high_max_{win}"] - df["close"]) / (df[f"range_{win}"] + 1e-12)
        df[f"dist_to_low_{win}"] = (df["close"] - df[f"low_min_{win}"]) / (df[f"range_{win}"] + 1e-12)
        df[f"is_new_high_{win}"] = (df["close"] >= df[f"high_max_{win}"]).astype(int)
        df[f"is_new_low_{win}"] = (df["close"] <= df[f"low_min_{win}"]).astype(int)
        df[f"return_{win}"] = df["close"] / df.groupby(window_label)["open"].transform("first") - 1

    return df


def create_candlestick_features(df):
    """Create candlestick and price action features."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l
    rng.replace(0, np.nan, inplace=True)

    # Price action/candlestick features
    df["body_size"] = (c - o) / (rng + 1e-9)
    df["close_position"] = (c - l) / (rng + 1e-9)
    df["trend_momentum"] = (c - c.shift(TREND_LOOKBACK)) / (rng + 1e-9)
    df["breakout_high"] = (h - h.shift(1)) / (rng + 1e-9)
    df["breakout_low_inv"] = -(l - l.shift(1)) / (rng + 1e-9)
    df["upper_wick"] = (h - np.maximum(c, o)) / (rng + 1e-9)
    df["lower_wick"] = (np.minimum(c, o) - l) / (rng + 1e-9)
    df["wick_polarity"] = np.tanh(df["lower_wick"] - df["upper_wick"])
    df["wick_ratio"] = (df["lower_wick"] - df["upper_wick"]) / (df["lower_wick"] + df["upper_wick"] + 1e-9)
    df["wick_ratio"] = df["wick_ratio"].clip(-10, 10)
    df["range_percent"] = rng / (o + 1e-9)
    df["body_vs_range"] = df["body_size"] * df["range_percent"]

    return df


def create_technical_indicators(df):
    """Create technical analysis indicators."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]
    rng = h - l

    # Rolling features
    df["roll_range_60"] = rng.rolling(6).mean()
    df["roll_vol_60"] = c.rolling(6).std()
    df["roll_body_60"] = df["body_size"].rolling(6).mean()
    df["roll_momentum_60"] = c.rolling(6).apply(
        lambda x: x.iloc[-1] - x.iloc[0] if len(x) == 6 else np.nan,
        raw=False,
    )

    # Z-scores
    df["z_close_20"] = (c - c.rolling(20).mean()) / (c.rolling(20).std() + 1e-9)
    df["z_vol_20"] = (df["roll_vol_60"] - df["roll_vol_60"].rolling(20).mean()) / (df["roll_vol_60"].rolling(20).std() + 1e-9)

    # ATR
    tr = pd.concat([h - l, abs(h - c.shift(1)), abs(l - c.shift(1))], axis=1).max(axis=1)
    df["atr_14"] = tr.rolling(14).mean()
    df["rel_atr"] = df["atr_14"] / (o + 1e-9)

    # RSI
    delta = c.diff()
    up, down = delta.clip(lower=0), -delta.clip(upper=0)
    roll_up = up.rolling(14).mean()
    roll_down = down.rolling(14).mean()
    rs = roll_up / (roll_down + 1e-9)
    df["rsi_14"] = 100 - (100 / (1 + rs))

    # Rolling min/max position
    df["roll_min_20"] = c.rolling(20).min()
    df["roll_max_20"] = c.rolling(20).max()
    df["pos_in_roll"] = (c - df["roll_min_20"]) / (df["roll_max_20"] - df["roll_min_20"] + 1e-9)

    return df


def create_time_features(df):
    """Create time-based cyclical features."""
    df["minute_of_hour"] = df["snapshotTime"].dt.minute
    df["hour_of_day"] = df["snapshotTime"].dt.hour
    df["slot_in_8h"] = ((df["hour_of_day"] % 8) * 60 + df["minute_of_hour"]) // 10
    df["sin_hour"] = np.sin(2 * np.pi * df["hour_of_day"] / 24)
    df["cos_hour"] = np.cos(2 * np.pi * df["hour_of_day"] / 24)
    df["sin_min"] = np.sin(2 * np.pi * df["minute_of_hour"] / 60)
    df["cos_min"] = np.cos(2 * np.pi * df["minute_of_hour"] / 60)
    return df


def create_advanced_features(df):
    """Create advanced quantitative features for better prediction."""
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    # Kaufman Efficiency Ratio (10-period)
    def kaufman_efficiency_ratio(price, period=10):
        direction = abs(price - price.shift(period))
        volatility = abs(price.diff()).rolling(period).sum()
        return direction / (volatility + 1e-9)

    df["kaufman_er_10"] = kaufman_efficiency_ratio(c, 10)
    df["kaufman_er_20"] = kaufman_efficiency_ratio(c, 20)

    # Rolling Hurst Exponent (simplified approximation)
    def rolling_hurst(price, period=20):
        def hurst_exponent(ts):
            if len(ts) < 4:
                return 0.5
            lags = range(2, min(len(ts) // 2, 10))
            tau = [np.sqrt(np.std(np.subtract(ts[lag:], ts[:-lag]))) for lag in lags]
            poly = np.polyfit(np.log(lags), np.log(tau), 1)
            return poly[0] * 2.0

        return price.rolling(period).apply(hurst_exponent, raw=False)

    df["hurst_20"] = rolling_hurst(c, 20)

    # Session breakout detection (assuming 24h sessions)
    def session_breakouts(df, session_hours=24):
        session_id = df["snapshotTime"].dt.floor(f"{session_hours}h")
        session_groups = df.groupby(session_id)

        df["session_high"] = session_groups["high"].transform("max")
        df["session_low"] = session_groups["low"].transform("min")
        df["session_open"] = session_groups["open"].transform("first")

        df["breakout_above_session"] = (df["close"] > df["session_high"].shift(1)).astype(int)
        df["breakout_below_session"] = (df["close"] < df["session_low"].shift(1)).astype(int)
        df["session_range"] = df["session_high"] - df["session_low"]
        df["pos_in_session_range"] = (df["close"] - df["session_low"]) / (df["session_range"] + 1e-9)

        return df

    df = session_breakouts(df)

    # Candlestick pattern detectors
    def detect_candlestick_patterns(df):
        o, h, l, c = df["open"], df["high"], df["low"], df["close"]
        body = abs(c - o)
        upper_shadow = h - np.maximum(c, o)
        lower_shadow = np.minimum(c, o) - l
        range_val = h - l

        # Doji (small body relative to range)
        df["doji"] = (body < 0.1 * range_val).astype(int)

        # Hammer/Hanging Man (long lower shadow, small upper shadow)
        df["hammer"] = ((lower_shadow > 2 * body) & (upper_shadow < 0.5 * body)).astype(int)

        # Shooting Star (long upper shadow, small lower shadow)
        df["shooting_star"] = ((upper_shadow > 2 * body) & (lower_shadow < 0.5 * body)).astype(int)

        # Engulfing patterns
        prev_body = body.shift(1)
        df["bullish_engulfing"] = ((c > o) & (c.shift(1) < o.shift(1)) & (body > prev_body) & (o < c.shift(1)) & (c > o.shift(1))).astype(int)

        df["bearish_engulfing"] = ((c < o) & (c.shift(1) > o.shift(1)) & (body > prev_body) & (o > c.shift(1)) & (c < o.shift(1))).astype(int)

        return df

    df = detect_candlestick_patterns(df)

    # Statistical regime change indicators
    def regime_change_indicators(df):
        c = df["close"]
        returns = c.pct_change()

        # Rolling Sharpe ratio (20-period)
        rolling_returns = returns.rolling(20)
        df["rolling_sharpe"] = rolling_returns.mean() / (rolling_returns.std() + 1e-9) * np.sqrt(252)

        # Rolling Sortino ratio (downside deviation)
        downside_returns = returns.where(returns < 0, 0)
        downside_std = downside_returns.rolling(20).std()
        df["rolling_sortino"] = rolling_returns.mean() / (downside_std + 1e-9) * np.sqrt(252)

        # Rolling skewness and kurtosis
        df["rolling_skew"] = returns.rolling(20).skew()
        df["rolling_kurtosis"] = returns.rolling(20).kurt()

        # Regime change detection via rolling correlation breakdown
        returns_lag = returns.shift(1)
        df["regime_instability"] = 1 - abs(returns.rolling(10).corr(returns_lag))

        return df

    df = regime_change_indicators(df)

    # Wavelet and FFT energy features
    def spectral_features(df, window=50):
        c = df["close"]

        def wavelet_energy(series):
            if len(series) < 8:
                return 0
            # Simple wavelet approximation using differences
            diffs = np.diff(series.values)
            return np.sum(diffs**2) / len(diffs)

        def fft_dominant_freq(series):
            if len(series) < 8:
                return 0
            fft_vals = np.abs(fft(series.values - series.mean()))
            freqs = np.fft.fftfreq(len(series))
            dominant_idx = np.argmax(fft_vals[1 : len(fft_vals) // 2]) + 1
            return abs(freqs[dominant_idx])

        def fft_energy(series):
            if len(series) < 8:
                return 0
            fft_vals = np.abs(fft(series.values - series.mean()))
            return np.sum(fft_vals**2) / len(fft_vals)

        df["wavelet_energy"] = c.rolling(window).apply(wavelet_energy, raw=False)
        df["fft_dominant_freq"] = c.rolling(window).apply(fft_dominant_freq, raw=False)
        df["fft_energy"] = c.rolling(window).apply(fft_energy, raw=False)

        return df

    df = spectral_features(df)

    return df


def get_feature_columns():
    """Define the feature columns for modeling."""
    return [
        # Time-invariant features
        "open_norm_60",
        "close_norm_60",
        "high_norm_60",
        "low_norm_60",
        "dist_to_high_60",
        "dist_to_low_60",
        "pos_in_60",
        "open_norm_480",
        "close_norm_480",
        "high_norm_480",
        "low_norm_480",
        "dist_to_high_480",
        "dist_to_low_480",
        "pos_in_480",
        "return_60",
        "return_480",
        "is_new_high_60",
        "is_new_low_60",
        "is_new_high_480",
        "is_new_low_480",
        # Time features
        "minute_of_hour",
        "hour_of_day",
        "slot_in_8h",
        "sin_hour",
        "cos_hour",
        "sin_min",
        "cos_min",
        # Technical features
        "body_size",
        "close_position",
        "trend_momentum",
        "breakout_high",
        "breakout_low_inv",
        "upper_wick",
        "lower_wick",
        "wick_polarity",
        "wick_ratio",
        "range_percent",
        "body_vs_range",
        "roll_range_60",
        "roll_vol_60",
        "roll_body_60",
        "roll_momentum_60",
        "z_close_20",
        "z_vol_20",
        "atr_14",
        "rel_atr",
        "rsi_14",
        "pos_in_roll",
        # Advanced features
        "kaufman_er_10",
        "kaufman_er_20",
        "hurst_20",
        "breakout_above_session",
        "breakout_below_session",
        "pos_in_session_range",
        "doji",
        "hammer",
        "shooting_star",
        "bullish_engulfing",
        "bearish_engulfing",
        "rolling_sharpe",
        "rolling_sortino",
        "rolling_skew",
        "rolling_kurtosis",
        "regime_instability",
        "wavelet_energy",
        "fft_dominant_freq",
        "fft_energy",
    ]


def train_price_forecast_model(df_model, feature_cols):
    """Train RandomForest model for price forecasting."""
    X = df_model[feature_cols]
    y = df_model["close_next"]
    split_idx = int(len(df_model) * TRAIN_SPLIT_RATIO)

    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    model = RandomForestRegressor(n_estimators=200, random_state=RANDOM_STATE)
    model.fit(X_train, y_train)
    df_model["forecast_close"] = model.predict(X)

    logger.info("RandomForest forecasting complete.")
    return df_model, model


def detect_regimes(df_model):
    """Detect market regimes using Gaussian Mixture Model."""
    regime_feats = [
        "return_60",
        "return_480",
        "roll_vol_60",
        "body_vs_range",
        "wick_polarity",
        "z_close_20",
        "rsi_14",
        "pos_in_roll",
    ]

    scaler = StandardScaler()
    regime_X = scaler.fit_transform(df_model[regime_feats].fillna(0))
    gmm = GaussianMixture(n_components=3, covariance_type="full", random_state=RANDOM_STATE)
    df_model["regime"] = gmm.fit_predict(regime_X)

    regime_probs = gmm.predict_proba(regime_X)
    for i in range(regime_probs.shape[1]):
        df_model[f"regime_prob_{i}"] = regime_probs[:, i]

    logger.info(f"Regime assignment done. Counts: {df_model['regime'].value_counts().to_dict()}")
    return df_model


def analyze_regimes(df_model):
    """Analyze what each regime represents for trading interpretation."""
    regime_analysis = {}

    for regime in sorted(df_model["regime"].unique()):
        regime_data = df_model[df_model["regime"] == regime]

        # Calculate regime characteristics
        analysis = {
            "count": len(regime_data),
            "avg_return_60": regime_data["return_60"].mean(),
            "avg_return_480": regime_data["return_480"].mean(),
            "avg_volatility": regime_data["roll_vol_60"].mean(),
            "avg_rsi": regime_data["rsi_14"].mean(),
            "avg_sharpe": regime_data["rolling_sharpe"].mean(),
            "avg_body_vs_range": regime_data["body_vs_range"].mean(),
            "avg_wick_polarity": regime_data["wick_polarity"].mean(),
            "trend_direction": "Bullish" if regime_data["return_60"].mean() > 0 else "Bearish",
            "volatility_level": "High" if regime_data["roll_vol_60"].mean() > df_model["roll_vol_60"].median() else "Low",
        }

        # Determine regime interpretation
        if analysis["avg_return_60"] > 0.001 and analysis["avg_rsi"] < 70:
            interpretation = "BUY SIGNAL - Bullish momentum with room to grow"
        elif analysis["avg_return_60"] < -0.001 and analysis["avg_rsi"] > 30:
            interpretation = "SELL SIGNAL - Bearish momentum with room to fall"
        elif analysis["avg_volatility"] > df_model["roll_vol_60"].quantile(0.7):
            interpretation = "HIGH VOLATILITY - Uncertain/choppy market"
        elif abs(analysis["avg_return_60"]) < 0.0005:
            interpretation = "SIDEWAYS - Consolidation/range-bound"
        else:
            interpretation = "MIXED SIGNALS - Requires caution"

        analysis["trading_signal"] = interpretation
        regime_analysis[f"Regime_{regime}"] = analysis

    return regime_analysis


def create_trading_signals(df_model):
    """Create explicit trading signals based on regime and features."""
    # Map regimes to trading signals based on characteristics
    regime_signals = {regime: analyze_regimes(df_model)[f"Regime_{regime}"]["trading_signal"] for regime in sorted(df_model["regime"].unique())}

    # Create signal column
    df_model["regime_signal"] = df_model["regime"].map(regime_signals)

    # Add confidence score based on regime probability
    df_model["signal_confidence"] = df_model.apply(
        lambda row: row[f"regime_prob_{row['regime']}"],
        axis=1,
    )

    # Create simplified buy/sell/hold signals
    def get_simple_signal(row):
        signal = row["regime_signal"]
        confidence = row["signal_confidence"]

        if "BUY" in signal and confidence > 0.6:
            return "BUY"
        if "SELL" in signal and confidence > 0.6:
            return "SELL"
        return "HOLD"

    df_model["simple_signal"] = df_model.apply(get_simple_signal, axis=1)

    return df_model


def create_dashboard(df_model, watermark_text=WATERMARK_TEXT, watermark_alpha=WATERMARK_ALPHA, 
                    watermark_fontsize=WATERMARK_FONTSIZE, watermark_color=WATERMARK_COLOR,
                    watermark_rotation=WATERMARK_ROTATION):
    """Create comprehensive dashboard plot with improved formatting and watermark."""
    # Set smaller font sizes globally
    plt.rcParams.update({
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
        "legend.fontsize": 7,
    })

    sns.set_style("whitegrid")
    fig, axes = plt.subplots(
        8,
        1,
        figsize=(16, 20),
        sharex=True,
        gridspec_kw={"height_ratios": [2, 1, 1, 1, 1, 1, 1, 1]},
    )

    # Add watermark to the figure
    def add_watermark(fig, text, alpha=0.15, fontsize=12, color="gray", rotation=30):
        """Add a watermark to the entire figure."""
        # Add watermark text across the entire figure
        fig.text(0.5, 0.5, text, fontsize=fontsize, color=color, alpha=alpha,
                ha='center', va='center', rotation=rotation, weight='bold',
                transform=fig.transFigure, zorder=0)
        
        # Add smaller watermarks in corners for additional protection
        fig.text(0.02, 0.98, text.split(' - ')[0] if ' - ' in text else text, 
                fontsize=fontsize-4, color=color, alpha=alpha*0.7,
                ha='left', va='top', rotation=0, weight='normal',
                transform=fig.transFigure, zorder=0)
        
        fig.text(0.98, 0.02, text.split(' - ')[0] if ' - ' in text else text, 
                fontsize=fontsize-4, color=color, alpha=alpha*0.7,
                ha='right', va='bottom', rotation=0, weight='normal',
                transform=fig.transFigure, zorder=0)

    add_watermark(fig, watermark_text, watermark_alpha, watermark_fontsize, 
                 watermark_color, watermark_rotation)

    # 1. Price + Forecast + Regime + Trading Signals
    axes[0].plot(df_model["snapshotTime"], df_model["close"], label="Actual", c="k", lw=1.5)
    axes[0].plot(df_model["snapshotTime"], df_model["forecast_close"], 
                label="Forecast", c="tab:blue", ls="--", lw=1.5, alpha=0.8)

    for r in sorted(df_model["regime"].unique()):
        axes[0].fill_between(df_model["snapshotTime"], df_model["close"].min(), 
                           df_model["close"].max(), where=df_model["regime"] == r,
                           alpha=0.1, color=f"C{r}", label=f"Regime {r}" if r == 0 else None)

    # Add trading signal markers
    buy_signals = df_model[df_model["simple_signal"] == "BUY"]
    sell_signals = df_model[df_model["simple_signal"] == "SELL"]

    if len(buy_signals) > 0:
        axes[0].scatter(buy_signals["snapshotTime"], buy_signals["close"], 
                       marker="^", color="green", s=30, alpha=0.8, 
                       label="BUY Signal", zorder=5)

    if len(sell_signals) > 0:
        axes[0].scatter(sell_signals["snapshotTime"], sell_signals["close"], 
                       marker="v", color="red", s=30, alpha=0.8, 
                       label="SELL Signal", zorder=5)

    # Add text annotations for high-confidence signals
    high_conf_buys = df_model[(df_model["simple_signal"] == "BUY") & (df_model["signal_confidence"] > 0.8)]
    high_conf_sells = df_model[(df_model["simple_signal"] == "SELL") & (df_model["signal_confidence"] > 0.8)]

    # Sample every 10th high-confidence signal to avoid overcrowding
    for i, (idx, row) in enumerate(high_conf_buys.iterrows()):
        if i % 10 == 0:  # Show every 10th signal
            axes[0].annotate(f"BUY\n{row['signal_confidence']:.2f}", 
                           xy=(row["snapshotTime"], row["close"]),
                           xytext=(5, 15), textcoords="offset points",
                           fontsize=6, color="green", weight="bold",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    for i, (idx, row) in enumerate(high_conf_sells.iterrows()):
        if i % 10 == 0:  # Show every 10th signal
            axes[0].annotate(f"SELL\n{row['signal_confidence']:.2f}", 
                           xy=(row["snapshotTime"], row["close"]),
                           xytext=(5, -20), textcoords="offset points",
                           fontsize=6, color="red", weight="bold",
                           bbox=dict(boxstyle="round,pad=0.2", facecolor="white", alpha=0.7))

    axes[0].set_ylabel("Price")
    axes[0].set_title("Price Forecast with Market Regimes & Trading Signals")
    axes[0].legend(loc="upper left", ncol=3, fontsize=6)

    # 2. Trading Signal Confidence Over Time
    axes[1].plot(df_model["snapshotTime"], df_model["signal_confidence"], label="Signal Confidence", color="tab:purple", lw=1)

    # Color-code the confidence line by signal type
    buy_mask = df_model["simple_signal"] == "BUY"
    sell_mask = df_model["simple_signal"] == "SELL"
    hold_mask = df_model["simple_signal"] == "HOLD"

    axes[1].scatter(df_model[buy_mask]["snapshotTime"], df_model[buy_mask]["signal_confidence"], color="green", s=8, alpha=0.6, label="BUY Confidence")
    axes[1].scatter(df_model[sell_mask]["snapshotTime"], df_model[sell_mask]["signal_confidence"], color="red", s=8, alpha=0.6, label="SELL Confidence")
    axes[1].scatter(df_model[hold_mask]["snapshotTime"], df_model[hold_mask]["signal_confidence"], color="gray", s=4, alpha=0.3, label="HOLD Confidence")

    axes[1].axhline(y=0.6, color="orange", linestyle="--", alpha=0.7, linewidth=1, label="Confidence Threshold")
    axes[1].set_ylabel("Confidence")
    axes[1].legend(loc="upper left", ncol=2)
    axes[1].set_title("Trading Signal Confidence & Thresholds")

    # 3. Advanced Indicators: Kaufman ER & Hurst (moved down)
    axes[2].plot(df_model["snapshotTime"], df_model["kaufman_er_10"], label="Kaufman ER(10)", color="tab:orange", lw=1)
    axes[2].plot(df_model["snapshotTime"], df_model["hurst_20"], label="Hurst(20)", color="tab:purple", lw=1)
    axes[2].axhline(y=0.5, color="gray", linestyle=":", alpha=0.5)
    axes[2].set_ylabel("Efficiency/Hurst")
    axes[2].legend(loc="upper left")
    axes[2].set_title("Market Efficiency & Mean Reversion")

    # 4. Session Breakouts & Patterns
    pattern_sum = df_model["doji"] + df_model["hammer"] + df_model["shooting_star"] + df_model["bullish_engulfing"] + df_model["bearish_engulfing"]
    axes[3].plot(df_model["snapshotTime"], df_model["pos_in_session_range"], label="Session Position", color="tab:green", lw=1)
    axes[3].scatter(df_model["snapshotTime"], pattern_sum * 0.1, s=10, alpha=0.6, color="red", label="Patterns")
    axes[3].set_ylabel("Session/Patterns")
    axes[3].legend(loc="upper left")
    axes[3].set_title("Session Breakouts & Candlestick Patterns")

    # 5. Statistical Regime Indicators
    axes[4].plot(df_model["snapshotTime"], df_model["rolling_sharpe"], label="Sharpe(20)", color="tab:cyan", lw=1)
    axes[4].plot(df_model["snapshotTime"], df_model["rolling_sortino"], label="Sortino(20)", color="tab:brown", lw=1)
    axes[4].plot(df_model["snapshotTime"], df_model["regime_instability"], label="Regime Instability", color="tab:red", lw=1)
    axes[4].set_ylabel("Risk Metrics")
    axes[4].legend(loc="upper left")
    axes[4].set_title("Risk-Adjusted Returns & Regime Stability")

    # 6. Spectral Features
    axes[5].plot(df_model["snapshotTime"], df_model["wavelet_energy"], label="Wavelet Energy", color="tab:olive", lw=1)
    axes[5].plot(df_model["snapshotTime"], df_model["fft_energy"], label="FFT Energy", color="tab:pink", lw=1)
    axes[5].set_ylabel("Spectral Energy")
    axes[5].legend(loc="upper left")
    axes[5].set_title("Wavelet & FFT Energy Analysis")

    # 7. Regime Probabilities (compressed)
    for i in range(3):
        axes[6].plot(df_model["snapshotTime"], df_model[f"regime_prob_{i}"], label=f"R{i}", color=f"C{i}", lw=1)
    axes[6].set_ylabel("Regime Prob")
    axes[6].legend(loc="upper left", ncol=3)
    axes[6].set_title("Regime Probabilities")

    # 8. Oscillators with signal zones
    rsi_norm = df_model["rsi_14"] / 100  # Normalize RSI to 0-1
    axes[7].plot(df_model["snapshotTime"], rsi_norm, label="RSI/100", color="tab:cyan", lw=1)
    axes[7].plot(df_model["snapshotTime"], df_model["z_close_20"].clip(-3, 3) / 6 + 0.5, label="Z-Score(norm)", color="tab:brown", lw=1)

    # Add overbought/oversold zones
    axes[7].axhline(y=0.8, color="red", linestyle=":", alpha=0.5, linewidth=0.5, label="Overbought")
    axes[7].axhline(y=0.2, color="green", linestyle=":", alpha=0.5, linewidth=0.5, label="Oversold")
    axes[7].fill_between(df_model["snapshotTime"], 0.8, 1.0, alpha=0.1, color="red")
    axes[7].fill_between(df_model["snapshotTime"], 0.0, 0.2, alpha=0.1, color="green")

    axes[7].set_ylabel("Oscillators")
    axes[7].legend(loc="upper left", ncol=2)
    axes[7].set_title("Normalized Oscillators with Signal Zones")

    plt.xlabel("Time")
    plt.tight_layout(pad=0.5)
    plt.subplots_adjust(hspace=0.3)
    plt.show()
    plt.close()
    logger.info("Enhanced dashboard with trading signals and watermark displayed successfully")


def main():
    """Main execution function."""
    # Load and prepare data
    df = load_and_prepare_data()

    # Feature engineering pipeline
    df = create_window_features(df)
    df = create_candlestick_features(df)
    df = create_technical_indicators(df)
    df = create_time_features(df)
    df = create_advanced_features(df)

    # Add forward-looking target
    df["close_next"] = df["close"].shift(-1)

    # Prepare modeling dataset - be more selective about NaN handling
    feature_cols = get_feature_columns()

    # Check how many NaN values we have per column
    nan_counts = df[feature_cols + ["close_next"]].isnull().sum()
    logger.info(f"NaN counts per feature:\n{nan_counts[nan_counts > 0]}")

    # Instead of dropping all NaN, let's fill them strategically
    df_model = df.copy()

    # Fill NaN values with appropriate strategies
    for col in feature_cols:
        if col in df_model.columns:
            if "norm" in col or "pos_in" in col or "dist_to" in col:
                # For normalized features, fill with 0.5 (middle)
                df_model[col] = df_model[col].fillna(0.5)
            elif any(pattern in col for pattern in ["return", "momentum", "sharpe", "sortino"]):
                # For return-like features, fill with 0
                df_model[col] = df_model[col].fillna(0)
            elif any(pattern in col for pattern in ["rsi", "efficiency", "hurst"]):
                # For bounded indicators, fill with neutral values
                if "rsi" in col:
                    df_model[col] = df_model[col].fillna(50)
                elif "hurst" in col or "efficiency" in col:
                    df_model[col] = df_model[col].fillna(0.5)
                else:
                    df_model[col] = df_model[col].fillna(0)
            elif any(pattern in col for pattern in ["doji", "hammer", "shooting", "engulfing", "breakout", "new_high", "new_low"]):
                # For binary features, fill with 0
                df_model[col] = df_model[col].fillna(0)
            else:
                # For other features, use forward fill then backward fill
                df_model[col] = df_model[col].fillna(method="ffill").fillna(method="bfill").fillna(0)

    # Remove only rows where target is NaN (last row typically)
    df_model = df_model.dropna(subset=["close_next"])

    # Ensure we have enough data for training
    if len(df_model) < 100:
        logger.error(f"Insufficient data after cleaning: {len(df_model)} rows. Need at least 100.")
        return

    logger.info(f"Enhanced feature matrix shape: {df_model[feature_cols].shape}")

    # Display feature sample
    logger.info(
        "\n"
        + tabulate(
            df_model[["snapshotTime", "close"] + feature_cols[:15]].head(10),
            headers="keys",
            tablefmt="pretty",
        )
    )

    # Model training and regime detection
    df_model, model = train_price_forecast_model(df_model, feature_cols)
    df_model = detect_regimes(df_model)

    # Analyze regimes and create trading signals
    regime_analysis = analyze_regimes(df_model)
    logger.info("\n=== REGIME ANALYSIS ===")
    for regime_name, analysis in regime_analysis.items():
        logger.info(f"\n{regime_name}:")
        logger.info(f"  Count: {analysis['count']} observations")
        logger.info(f"  Avg 60min Return: {analysis['avg_return_60']:.4f}")
        logger.info(f"  Avg Volatility: {analysis['avg_volatility']:.6f}")
        logger.info(f"  Avg RSI: {analysis['avg_rsi']:.1f}")
        logger.info(f"  Trend: {analysis['trend_direction']}")
        logger.info(f"  Volatility: {analysis['volatility_level']}")
        logger.info(f"  TRADING SIGNAL: {analysis['trading_signal']}")

    df_model = create_trading_signals(df_model)

    # Show trading signal distribution
    signal_counts = df_model["simple_signal"].value_counts()
    logger.info("\n=== TRADING SIGNALS DISTRIBUTION ===")
    logger.info(f"BUY signals: {signal_counts.get('BUY', 0)}")
    logger.info(f"SELL signals: {signal_counts.get('SELL', 0)}")
    logger.info(f"HOLD signals: {signal_counts.get('HOLD', 0)}")

    # Visualization and output with customizable watermark
    create_dashboard(df_model, 
                    watermark_text=WATERMARK_TEXT,
                    watermark_alpha=WATERMARK_ALPHA,
                    watermark_fontsize=WATERMARK_FONTSIZE,
                    watermark_color=WATERMARK_COLOR,
                    watermark_rotation=WATERMARK_ROTATION)

    # Save results
    df_model.to_csv("full_model_output.csv", index=False)
    logger.info("Saved enhanced DataFrame with all features, forecasts, and regimes")

    # Feature importance analysis
    feature_importance = pd.DataFrame({
        "feature": feature_cols,
        "importance": model.feature_importances_,
    }).sort_values("importance", ascending=False)

    logger.info(f"\nTop 10 most important features:\n{feature_importance.head(10)}")


if __name__ == "__main__":
    main()
