# -*- coding: utf-8 -*-
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

# -*- coding: utf-8 -*-
# pylint: disable=C0116, W0621, W1203, C0103, C0301, W1201, W0511, E0401, E1101, E0606
# C0116: Missing function or method docstring
# W0621: Redefining name %r from outer scope (line %s)
# W1203: Use % formatting in logging functions and pass the % parameters as arguments
# C0103: Constant name "%s" doesn't conform to UPPER_CASE naming style
# C0301: Line too long (%s/%s)
# W1201: Specify string format arguments as logging function parameters
# W0511: TODOs
# E1101: Module 'holidays' has no 'US' member (no-member) ... it does, so ignore this
# E0606: possibly-used-before-assignment, ignore this
# UP018: native-literals (UP018)
"""Comprehensive HMM-based short-term trading signal script with unified plotting of results.
Features: RSI (Wilder), MACD histogram, Bollinger Z-score, ATR, momentum,
candlestick body & wicks, Vortex Indicator. Regimes inferred via HMM with temporal smoothing.
"""

import logging
import sys

import colorlog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
from tabulate import tabulate

# Setup color logging
handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red", "CRITICAL": "bold_red"},
    ),
)
logger = colorlog.getLogger("hmm_trader")
logger.addHandler(handler)
logger.setLevel(logging.DEBUG)


# --- Indicator computations ---
def compute_rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    up = delta.clip(lower=0)
    down = -delta.clip(upper=0)
    avg_up = up.ewm(alpha=1 / period, min_periods=period).mean()
    avg_down = down.ewm(alpha=1 / period, min_periods=period).mean()
    rs = avg_up / (avg_down + 1e-6)
    return (100 - (100 / (1 + rs))).rename("rsi").fillna(50)


def compute_macd_hist(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> pd.Series:
    ema_fast = series.ewm(span=fast, adjust=False).mean()
    ema_slow = series.ewm(span=slow, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=signal, adjust=False).mean()
    return (macd - macd_signal).rename("macd_hist")


def compute_bollinger_z(series: pd.Series, period: int = 20) -> pd.Series:
    ma = series.rolling(period).mean()
    std = series.rolling(period).std()
    return ((series - ma) / (std + 1e-6)).rename("boll_z")


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    high, low, close = df["high"], df["low"], df["close"]
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.ewm(alpha=1 / period, adjust=False).mean().rename("atr")


def compute_momentum(series: pd.Series, period: int = 5) -> pd.Series:
    return series.diff(period).fillna(0).rename("momentum")


def compute_price_action(df: pd.DataFrame) -> pd.DataFrame:
    o, c, h, l = df["open"], df["close"], df["high"], df["low"]
    body = (c - o).rename("body")
    rng = (h - l).replace(0, np.nan)
    upper = (h - np.maximum(o, c)) / rng
    lower = (np.minimum(o, c) - l) / rng
    return pd.concat([body / rng, upper.rename("upper_wick"), lower.rename("lower_wick")], axis=1).fillna(0)


def compute_vortex(df: pd.DataFrame, length: int = 14) -> pd.DataFrame:
    high, low, close = df["high"], df["low"], df["close"]
    prev_low, prev_high = low.shift(1), high.shift(1)
    tr = pd.concat(
        [
            high - low,
            (high - close.shift(1)).abs(),
            (low - close.shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)
    vm_plus = (high - prev_low).abs()
    vm_minus = (low - prev_high).abs()
    sum_tr = tr.rolling(length).sum()
    sum_vm_plus = vm_plus.rolling(length).sum()
    sum_vm_minus = vm_minus.rolling(length).sum()
    vi_plus = (sum_vm_plus / sum_tr).rename("vi_plus")
    vi_minus = (sum_vm_minus / sum_tr).rename("vi_minus")
    return pd.concat([vi_plus, vi_minus, (vi_plus - vi_minus).rename("vi_diff")], axis=1).fillna(0)


# --- Assemble features ---
def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["rsi"] = compute_rsi(df["close"])
    df["macd_hist"] = compute_macd_hist(df["close"])
    df["boll_z"] = compute_bollinger_z(df["close"])
    df["atr"] = compute_atr(df)
    df["momentum"] = compute_momentum(df["close"])
    pa = compute_price_action(df)
    vi = compute_vortex(df)
    feats = pd.concat([df[["rsi", "macd_hist", "boll_z", "atr", "momentum"]], pa, vi], axis=1).dropna()
    feats = feats.drop(columns=["momentum"])
    print(tabulate(feats.tail(10), headers="keys", tablefmt="psql"))
    return feats


# --- Fit HMM ---
def fit_hmm(X: np.ndarray, n_states: int = 3):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X)
    model = GaussianHMM(n_components=n_states, covariance_type="diag", n_iter=200, random_state=42)
    model.fit(Xs)
    logger.info(f"Fitted HMM with {n_states} states")
    return model, scaler


# --- Analysis & Unified Plotting ---
def analyze_and_plot(df: pd.DataFrame, threshold: float = 0.2):
    feats = prepare_features(df)
    X = feats.values
    model, scaler = fit_hmm(X, n_states=3)
    Xs = scaler.transform(X)
    labels = model.predict(Xs)

    # Map states to regimes
    scores = model.means_.sum(axis=1)
    buy_c, sell_c = np.argmax(scores), np.argmin(scores)
    names = ["HOLD"] * model.n_components
    names[buy_c], names[sell_c] = "BUY", "SELL"
    logger.info(f"Cluster labels: {names}")

    # 1) Compute next-bar returns and align to feature index
    returns = df["close"].pct_change().shift(-3)
    returns = returns.reindex(feats.index)

    # 2) Build a small DataFrame to group by state
    state_df = pd.DataFrame(
        {
            "state": labels,
            "returns": returns,
        },
        index=feats.index,
    )

    # 3) Calculate mean return per HMM state
    mean_returns = state_df.groupby("state")["returns"].mean()
    logger.info("Mean next-bar returns by state:\n%s", mean_returns)

    # 4) Relabel BUY and SELL based on realized returns
    buy_c = mean_returns.idxmax()  # state with best avg return
    sell_c = mean_returns.idxmin()  # state with worst avg return
    names = ["HOLD"] * model.n_components
    names[buy_c], names[sell_c] = "BUY", "SELL"
    logger.info(f"Relabeled states by return: BUY→S{buy_c}, SELL→S{sell_c}")

    mean_returns = state_df.groupby("state")["returns"].mean()
    mean_returns_named = mean_returns.rename(index={i: names[i] for i in mean_returns.index})
    logger.info(f"Mean next-bar returns by regime:\n{mean_returns_named}")

    # Smooth posterior: average last N
    window = min(20, len(Xs))
    post = model.predict_proba(Xs)
    avg_post = post[-window:].mean(axis=0)
    buy_p, sell_p = avg_post[buy_c], avg_post[sell_c]
    signal = "BUY" if (buy_p - sell_p) > threshold else "SELL" if (sell_p - buy_p) > threshold else "HOLD"
    logger.info(f"Signal={signal} BUY_P={buy_p:.2f} SELL_P={sell_p:.2f}")

    # Unified figure: 2x2 subplots
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))

    # 1) Probability bar
    ax = axes[0, 0]
    ax.bar(["BUY", "SELL"], [buy_p, sell_p], color=["green", "red"])
    ax.set_title(f"Signal: {signal}")
    ax.set_ylim(0, 1)

    # 2) Close price with regime shading
    ax = axes[0, 1]
    times = feats.index
    ax.plot(times, df.loc[times, "close"], label="Close")
    ymin, ymax = df.loc[times, "close"].min(), df.loc[times, "close"].max()
    ax.fill_between(times, ymin, ymax, where=(labels == buy_c), color="green", alpha=0.1)
    ax.fill_between(times, ymin, ymax, where=(labels == sell_c), color="red", alpha=0.1)
    ax.set_title("Price & Regimes")

    # 3) PCA projection
    if Xs.shape[1] >= 2:
        ax = axes[1, 0]
        proj = PCA(n_components=2).fit_transform(Xs)
        sc = ax.scatter(proj[:, 0], proj[:, 1], c=labels, cmap="tab10", alpha=0.7)
        cent = PCA(n_components=2).fit(Xs).transform(model.means_)
        for i, (x, y) in enumerate(cent):
            ax.text(x, y, f"S{i}:{names[i]}")
        ax.set_title("HMM (PCA)")

    # 4) t-SNE projection
    ax = axes[1, 1]
    n_samples = len(Xs)
    perp = min(30, max(1, n_samples - 1))
    if n_samples > 1:
        proj2 = TSNE(n_components=2, init="pca", learning_rate="auto", perplexity=perp, random_state=42).fit_transform(Xs)
        ax.scatter(proj2[:, 0], proj2[:, 1], c=labels, cmap="tab10", alpha=0.7)
        ax.set_title("HMM (t-SNE)")
    else:
        ax.text(0.5, 0.5, "Not enough samples for TSNE", ha="center")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) < 2:
        logger.error("Usage: python3 hmm_trader.py data.csv")
        sys.exit(1)
    logger.info("Starting HMM-based trading signal analysis.")
    date_format = "%Y:%m:%d-%H:%M:%S"
    df = pd.read_csv(
        sys.argv[1],
        parse_dates=[0],
        index_col=0,
        date_parser=lambda x: pd.to_datetime(x, format=date_format),
    )
    analyze_and_plot(df)
    logger.info("Done.")
