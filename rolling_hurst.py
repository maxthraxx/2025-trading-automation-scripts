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

import json
import logging

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from hurst import compute_Hc
from sklearn.linear_model import TheilSenRegressor

# ------------------------ Logging Setup ------------------------
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
handler.setFormatter(formatter)
if not logger.hasHandlers():
    logger.addHandler(handler)


class HurstAnalyzer:
    def __init__(self, window: int = 100, kind: str = "random_walk"):
        self.window = window
        self.kind = kind
        self.hurst_series = None
        self.latest_value = None

    def parse_ohlc(self, df: pd.DataFrame) -> pd.Series:
        if "snapshotTime" not in df.columns or "close" not in df.columns:
            logger.error("DataFrame must contain 'snapshotTime' and 'close' columns")
            raise ValueError("Missing required columns in DataFrame")

        df = df.copy()
        df["snapshotTime"] = pd.to_datetime(df["snapshotTime"], format="%Y:%m:%d-%H:%M:%S", errors="coerce")
        df.dropna(subset=["snapshotTime"], inplace=True)
        df.set_index("snapshotTime", inplace=True)

        logger.info(f"Parsed OHLC data: {len(df)} rows retained after cleaning")
        return df["close"]

    def compute_rolling(self, series: pd.Series) -> pd.Series:
        logger.info(f"Computing rolling Hurst exponent with window = {self.window}")
        series_values = series.to_numpy()
        hurst_vals = np.full(len(series_values), np.nan)

        for i in range(self.window, len(series_values)):
            window_data = series_values[i - self.window : i]
            if np.isnan(window_data).any():
                logger.debug(f"Skipping window {i - self.window}:{i} due to NaNs")
                continue
            try:
                h, _, _ = compute_Hc(window_data, kind=self.kind, min_window=5, simplified=True)
                hurst_vals[i] = np.clip(h, 0.0, 1.0)
                logger.debug(f"Hurst at index {i} = {hurst_vals[i]:.4f}")
            except Exception as e:
                logger.warning(f"Error computing Hurst at index {i}: {e}")

        self.hurst_series = pd.Series(hurst_vals, index=series.index)
        logger.info(f"Computed {np.count_nonzero(~np.isnan(hurst_vals))} valid Hurst values")
        return self.hurst_series

    def plot(self, output_file: str = None, headless: bool = False) -> float:
        if self.hurst_series is None:
            logger.error("Hurst series has not been computed.")
            raise ValueError("Run compute_rolling() first")

        valid_hurst = self.hurst_series.dropna()
        if valid_hurst.empty:
            logger.error("Hurst series is empty after dropping NaNs")
            raise ValueError("Hurst series contains no valid values")

        x = np.arange(len(valid_hurst)).reshape(-1, 1)
        y = valid_hurst.values

        # Fit Theil-Sen regression
        model = TheilSenRegressor()
        model.fit(x, y)
        trend_line = model.predict(x)

        if not headless:
            plt.figure(figsize=(14, 6))
            plt.plot(valid_hurst.index, valid_hurst, label="Rolling Hurst", color="blue")
            plt.plot(valid_hurst.index, trend_line, label="Theil-Sen Trend", color="red", linestyle="--")
            plt.axhline(0.5, color="gray", linestyle=":", label="H = 0.5 (Random Walk)")

            # Shaded regions for different regimes
            plt.fill_between(valid_hurst.index, 0, 1, where=(valid_hurst < 0.45), color="green", alpha=0.1, label="Mean-Reverting")
            plt.fill_between(valid_hurst.index, 0, 1, where=((valid_hurst >= 0.45) & (valid_hurst <= 0.55)), color="gray", alpha=0.1, label="Random Walk")
            plt.fill_between(valid_hurst.index, 0, 1, where=(valid_hurst > 0.55), color="orange", alpha=0.1, label="Trending")

            plt.title("Rolling Hurst Exponent with Trend Detection")
            plt.xlabel("Time")
            plt.ylabel("Hurst Exponent")
            plt.legend()
            plt.grid(True)

            if output_file:
                plt.savefig(output_file, bbox_inches="tight")
                logger.info(f"Saved plot to {output_file}")
            else:
                plt.show()

        self.latest_value = valid_hurst.iloc[-1]
        logger.info(f"Latest Hurst Value: {self.latest_value:.4f}")
        return self.latest_value


    @staticmethod
    def interpret(h: float) -> dict:
        if h > 0.65:
            return {
                "code": "MOMENTUM",
                "label": "Strong persistence",
                "strategy": "Trend is strong and likely to continue --->  favor momentum strategies",
            }
        if 0.55 < h <= 0.65:
            return {
                "code": "MOMENTUM",
                "label": "Mild persistence",
                "strategy": "Possible trend, but not strongly established --->  trade trend cautiously",
            }
        if 0.45 < h <= 0.55:
            return {
                "code": "NEUTRAL",
                "label": "Random walk",
                "strategy": "No clear memory or structure --->  avoid directional bias",
            }
        return {
            "code": "MEAN_REV",
            "label": "Mean-reversion (anti-persistence)",
            "strategy": "Price likely to reverse --->  favor fading/extreme-reversal strategies",
        }


# ------------------------ Example Usage ------------------------
if __name__ == "__main__":
    import sys

    csv_path = "backtest_prices.csv" if len(sys.argv) < 2 else sys.argv[1]
    df = pd.read_csv(csv_path)

    analyzer = HurstAnalyzer(window=np.min([20, len(df) // 2]), kind="random_walk")
    close_series = analyzer.parse_ohlc(df)
    hurst_series = analyzer.compute_rolling(close_series)
    latest = analyzer.plot(headless=False)

    interpretation = HurstAnalyzer.interpret(latest)
    print(
        "\n"
        + json.dumps(
            {
                "hurst": round(latest, 4),
                "code": interpretation["code"],
                "interpretation": interpretation["label"],
                "strategy": interpretation["strategy"],
            },
            indent=2,
        ),
    )
