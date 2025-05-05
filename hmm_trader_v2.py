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

"""Final Rolling HMM Forecasting Strategy with Regime Visualization Fix
========================================================================

- Fixes regime detection by assigning BUY/SELL based on sorted means if thresholds fail.
- Adds shaded regime coloring.
- Visualizes HMM clusters with PCA and t-SNE.
- Includes test accuracy and posterior regime probabilities.
"""

import itertools
import logging
import sys
from functools import partial
from multiprocessing import Pool, cpu_count

import colorlog
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

plt.rcParams.update({"font.size": 8})

handler = colorlog.StreamHandler()
handler.setFormatter(
    colorlog.ColoredFormatter(
        "%(log_color)s%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        log_colors={"DEBUG": "cyan", "INFO": "green", "WARNING": "yellow", "ERROR": "red"},
    ),
)
logger = colorlog.getLogger("hmm_rolling")
logger.addHandler(handler)
logger.setLevel(logging.INFO)


def compute_features(df, config):
    df = df.copy()
    o, h, l, c = df["open"], df["high"], df["low"], df["close"]

    macd = c.ewm(span=config["macd_fast"]).mean() - c.ewm(span=config["macd_slow"]).mean()
    signal = macd.ewm(span=config["macd_signal"]).mean()
    df["macd_hist"] = macd - signal

    df["momentum"] = c.diff(config["momentum"])
    df["log_return"] = np.log(c / c.shift(1)).fillna(0)
    df["trend_strength"] = c.pct_change().rolling(config["trend"]).mean() / c.pct_change().rolling(config["trend"]).std()

    prev_low, prev_high = l.shift(1), h.shift(1)
    tr = pd.concat([h - l, (h - c.shift(1)).abs(), (l - c.shift(1)).abs()], axis=1).max(axis=1)
    vm_plus = (h - prev_low).abs()
    vm_minus = (l - prev_high).abs()
    df["vi_diff"] = (vm_plus.rolling(config["vortex"]).sum() - vm_minus.rolling(config["vortex"]).sum()) / tr.rolling(config["vortex"]).sum()

    tp = (h + l + c) / 3
    df["cci"] = (tp - tp.rolling(config["cci"]).mean()) / (0.015 * tp.rolling(config["cci"]).std())
    df["boll_z"] = (c - c.rolling(config["boll"]).mean()) / c.rolling(config["boll"]).std()
    df["atr"] = tr.ewm(alpha=1 / config["atr"]).mean()

    features = ["macd_hist", "momentum", "log_return", "trend_strength", "vi_diff", "cci", "boll_z", "atr"]
    return df[features].dropna()


def evaluate_config(df, config):
    try:
        feats = compute_features(df, config)
        if len(feats) < 30:
            return (config, -np.inf)

        X = StandardScaler().fit_transform(feats)
        model = GaussianHMM(n_components=3, covariance_type="full", random_state=42, n_iter=1000)
        model.fit(X)
        labels = model.predict(X)

        returns = df["close"].pct_change().shift(-4).reindex(feats.index)
        state_df = pd.DataFrame({"state": labels, "returns": returns}, index=feats.index)

        # Assign regimes based on mean return
        state_returns = state_df.groupby("state")["returns"].mean()
        buy = state_returns.idxmax()
        sell = state_returns.idxmin()

        # Create strategy return: +1 for BUY, -1 for SELL, 0 otherwise
        signal = state_df["state"].map({buy: 1, sell: -1}).fillna(0)
        strategy_return = (signal * state_df["returns"]).sum()

        logger.info(f"Tried config: {config} | Strategy Return: {strategy_return:.4f}")
        return (config, strategy_return)

    except Exception as e:
        logger.warning(f"Failed config {config}: {e}")
        return (config, -np.inf)


def brute_force_search(df):
    param_grid = {
        "macd_fast": [8],
        "macd_slow": [26],
        "macd_signal": [9],
        "momentum": [5],
        "trend": [15],
        "vortex": [14],
        "cci": [20],
        "boll": [20],
        "atr": [14],
    }
    # Uncomment the following lines to enable dynamic parameter ranges
    max_lookback = int(len(df) * (1/3))

    param_grid = {
        "macd_fast": range(5, min(12, max_lookback), 2),
        "macd_slow": range(26, min(50, max_lookback), 5),
        "macd_signal": range(8, min(12, max_lookback), 2),
        "momentum": range(3, min(20, max_lookback), 2),
        "trend": range(5, min(20, max_lookback), 5),
        "vortex": range(10, min(18, max_lookback), 3),
        "cci": range(10, min(20, max_lookback), 5),
        "boll": range(10, min(20, max_lookback), 5),
        "atr": range(5, min(35, max_lookback), 5),
    }

    keys, values = zip(*param_grid.items(), strict=False)
    configs = [dict(zip(keys, v, strict=False)) for v in itertools.product(*values)]
    logger.info(f"Evaluating {len(configs)} configurations...")
    with Pool(cpu_count()) as pool:
        results = pool.map(partial(evaluate_config, df), configs)
    best_config = max(results, key=lambda x: x[1])[0]
    logger.info(f"Best config: {best_config}")
    return best_config


def run_expanding_forecast(df, config, train_frac=0.7):
    import warnings

    warnings.filterwarnings("ignore")

    feats = compute_features(df, config)
    returns = df["close"].pct_change().shift(-3).reindex(feats.index)

    train_size = int(len(feats) * train_frac)
    test_index = feats.index[train_size:]

    predicted_states = []
    predicted_returns = []
    predicted_dates = []

    for i in range(train_size, len(feats)):
        X_train = StandardScaler().fit_transform(feats.iloc[:i])
        model = GaussianHMM(n_components=3, covariance_type="full", random_state=42, n_iter=1000)
        model.fit(X_train)
        X_pred = StandardScaler().fit_transform([feats.iloc[i].values])
        state = model.predict(X_pred)[0]
        predicted_states.append(state)
        predicted_returns.append(returns.iloc[i])
        predicted_dates.append(feats.index[i])

    state_series = pd.Series(predicted_states, index=predicted_dates)
    return_series = pd.Series(predicted_returns, index=predicted_dates)

    means = return_series.groupby(state_series).mean()
    sorted_states = means.sort_values()
    regime_map = pd.Series(0, index=means.index)
    regime_map[sorted_states.index[-1]] = 1
    regime_map[sorted_states.index[0]] = -1
    logger.info(f"Regime map: {regime_map.to_dict()}")

    regime_series = state_series.map(regime_map)
    true_labels = return_series.apply(lambda x: 1 if x > 0.002 else -1 if x < -0.002 else 0)
    accuracy = (true_labels == regime_series).mean()
    logger.info(f"Test Accuracy (forecasted direction match): {accuracy:.2%}")

    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 2)

    ax0 = fig.add_subplot(gs[0, 0])
    recent = regime_series
    probs = recent.value_counts(normalize=True).reindex([-1, 1]).fillna(0)
    bars = ax0.bar(["SELL", "BUY"], probs.values, color=["red", "green"])
    for bar, val in zip(bars, probs.values, strict=False):
        ax0.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005, f"{val:.2f}", ha="center")
    ax0.set_title("Posterior Probabilities (Last 20 Bars)")
    ax0.set_ylim(0, 1)
    ax0.set_ylabel("Probability")

    ax1 = fig.add_subplot(gs[0, 1])
    ax1.plot(df["close"], color="black", label="Close Price", linewidth=1)
    ax1.axvline(predicted_dates[0], color="blue", linestyle="--", label="Train/Test Split")
    last_label = None
    start = None
    for date, label in regime_series.items():
        if label != last_label:
            if last_label in [1, -1] and start is not None:
                ax1.axvspan(start, date, color="green" if last_label == 1 else "red", alpha=0.2)
            start = date
            last_label = label
    if last_label in [1, -1] and start is not None:
        ax1.axvspan(start, regime_series.index[-1], color="green" if last_label == 1 else "red", alpha=0.2)
    ax1.set_title("Price with Regime Shading")
    ax1.set_ylabel("Price")
    ax1.set_xlabel("Time")
    ax1.legend(fontsize=8)

    ax2 = fig.add_subplot(gs[1, 0])
    X_test_scaled = StandardScaler().fit_transform(feats.loc[predicted_dates])
    X_pca = PCA(n_components=2).fit_transform(X_test_scaled)
    sc2 = ax2.scatter(X_pca[:, 0], X_pca[:, 1], c=state_series.values, cmap="coolwarm", alpha=0.7)
    fig.colorbar(sc2, ax=ax2, label="State")
    ax2.set_title("HMM Clusters (PCA)")
    ax2.set_xlabel("PC1")
    ax2.set_ylabel("PC2")

    ax3 = fig.add_subplot(gs[1, 1])
    if len(X_test_scaled) > 5:
        try:
            tsne = TSNE(n_components=2, perplexity=min(30, len(X_test_scaled) - 1), random_state=42)
            X_tsne = tsne.fit_transform(X_test_scaled)
            sc3 = ax3.scatter(X_tsne[:, 0], X_tsne[:, 1], c=state_series.values, cmap="coolwarm", alpha=0.7)
            fig.colorbar(sc3, ax=ax3, label="State")
            ax3.set_title("HMM Clusters (t-SNE)")
            ax3.set_xlabel("Dim 1")
            ax3.set_ylabel("Dim 2")
        except Exception:
            ax3.text(0.5, 0.5, "t-SNE failed", ha="center")
    else:
        ax3.text(0.5, 0.5, "t-SNE skipped (too few samples)", ha="center")
        ax3.set_title("HMM Clusters (t-SNE)")

    ax4 = fig.add_subplot(gs[2, 0])
    ax4.barh(["Accuracy"], [accuracy], color="royalblue")
    ax4.set_xlim(0, 1)
    ax4.set_title("Test Signal Accuracy")
    ax4.text(accuracy + 0.01, 0, f"{accuracy:.1%}", va="center")

    ax5 = fig.add_subplot(gs[2, 1])
    ax5.plot(df["close"], label="Close Price")
    ax5.axvline(predicted_dates[0], linestyle="--", color="orange", label=f"Train/Test split @ {predicted_dates[0]}")
    ax5.legend(fontsize=8)
    ax5.set_title("Train/Test Split Visualization")
    ax5.set_xlabel("Time")
    ax5.set_ylabel("Price")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    df = pd.read_csv(sys.argv[1], index_col=0)
    best_config = brute_force_search(df)
    run_expanding_forecast(df, best_config)
