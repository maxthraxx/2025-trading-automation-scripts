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

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# --- OU parameter estimation function ---
def find(x):
    if len(x) < 3:
        return np.nan, np.nan, np.nan
    s_x = np.sum(x[:-1])
    s_y = np.sum(x[1:])
    s_xx = np.sum(x[:-1] ** 2)
    s_yy = np.sum(x[1:] ** 2)
    s_xy = np.sum(x[:-1] * x[1:])
    n = len(x) - 1
    delta = 1
    mu = ((s_y * s_xx) - (s_x * s_xy)) / (n * (s_xx - s_xy) - ((s_x**2) - s_x * s_y))
    theta = -(1 / delta) * np.log((s_xy - mu * s_x - mu * s_y + n * mu**2) / (s_xx - 2 * mu * s_x + n * mu**2))
    alpha = np.exp(-theta * delta)
    sigma_h = np.sqrt((1 / n) * (s_yy - (2 * alpha * s_xy) + ((alpha**2) * s_xx) - (2 * mu * (1 - alpha) * (s_y - alpha * s_x)) + (n * (mu**2) * (1 - alpha) ** 2)))
    sigma = np.sqrt((sigma_h**2) * (2 * theta / (1 - alpha**2)))
    return mu, sigma, theta


# --- Feature: close_pos (price position in trailing window) ---
def compute_price_position(df, window=100):
    if len(df) < window:
        return np.nan
    recent = df.tail(window)
    low = recent["low"].min()
    high = recent["high"].max()
    mid = df["close"].iloc[-1]
    position = (mid - low) / (high - low + 1e-9)
    return np.round(position, 4)


# --- Load data ---
df = pd.read_csv("backtest_prices.csv", parse_dates=["snapshotTime"])
df.set_index("snapshotTime", inplace=True)
df.sort_index(inplace=True)

# --- Calculate close_pos (rolling position) ---
window = 3
df["close_pos"] = [compute_price_position(df.iloc[: i + 1], window=window) for i in range(len(df))]

# --- Rolling OU estimation on close_pos ---
mus, sigmas, thetas = [], [], []
for i in range(len(df)):
    if i < window or np.isnan(df["close_pos"].iloc[i]):
        mus.append(np.nan)
        sigmas.append(np.nan)
        thetas.append(np.nan)
    else:
        feature_slice = df["close_pos"].iloc[i - window + 1 : i + 1].values
        mu, sigma, theta = find(feature_slice)
        mus.append(mu)
        sigmas.append(sigma)
        thetas.append(theta)

df["mu"] = mus
df["sigma"] = sigmas
df["theta"] = thetas

# --- Z-score on close_pos ---
df["Z_Score"] = (df["close_pos"] - df["mu"]) / df["sigma"]

# --- Trading signals (using z-score of close_pos) ---
upper_threshold = 1.5
lower_threshold = -1.5
df["Signal"] = 0
df.loc[df["Z_Score"] > upper_threshold, "Signal"] = -1  # Sell
df.loc[df["Z_Score"] < lower_threshold, "Signal"] = 1  # Buy
df["Trade"] = df["Signal"].diff()

# --- Plot ---
plt.figure(figsize=(14, 7))
plt.plot(df.index, df["close"], label="Close Price", color="blue", lw=1)

buy_signals = df[df["Trade"] == 2]
sell_signals = df[df["Trade"] == -2]
plt.scatter(buy_signals.index, buy_signals["close"], label="Buy Signal", marker="^", color="green", s=70)
plt.scatter(sell_signals.index, sell_signals["close"], label="Sell Signal", marker="v", color="red", s=70)

plt.title("OU Mean Reversion Strategy Using Engineered Feature: close_pos")
plt.xlabel("Date")
plt.ylabel("Price")
plt.legend()
plt.grid(True)
plt.show()
