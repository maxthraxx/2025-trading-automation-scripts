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

import io

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from scipy.optimize import minimize


class OptimalPLRSignal:
    """Optimal Path of Least Resistance (PLR) signal generator that works
    with smaller datasets and focuses solely on high-quality signals.
    """

    def __init__(self, lookback_period=3, alpha=0.4, beta=0.4, gamma=0.2, signal_threshold=1.5, optimization_window=5):
        """Initialize the Optimal PLR Signal generator.

        Parameters
        ----------
        lookback_period : int
            The number of periods to consider for calculations
        alpha : float
            Weight for the momentum component
        beta : float
            Weight for the path efficiency component
        gamma : float
            Weight for the candlestick resistance component
        signal_threshold : float
            Threshold multiplier for signal generation (in standard deviations)
        optimization_window : int
            Number of periods to use for parameter optimization

        """
        self.lookback_period = lookback_period
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.signal_threshold = signal_threshold
        self.optimization_window = optimization_window
        self.regime_adjustments = {"trending": {"alpha": 0.5, "beta": 0.4, "gamma": 0.1, "threshold": 1.2}, "reversal": {"alpha": 0.3, "beta": 0.3, "gamma": 0.4, "threshold": 1.8}, "neutral": {"alpha": 0.4, "beta": 0.4, "gamma": 0.2, "threshold": 1.5}}

    def calculate_momentum(self, close_prices):
        """Calculate price momentum as a weighted average of price changes."""
        if len(close_prices) < 2:
            return 0

        # Calculate price changes
        price_changes = np.diff(close_prices)

        # Generate weights (more recent changes have higher weights)
        weights = np.linspace(1, 2, len(price_changes))
        weights = weights / np.sum(weights)  # Normalize weights

        # Calculate weighted momentum
        momentum = np.sum(price_changes * weights)
        return momentum

    def calculate_path_efficiency(self, close_prices):
        """Calculate the path efficiency ratio."""
        if len(close_prices) < 2:
            return 1.0

        # Net movement
        net_movement = abs(close_prices[-1] - close_prices[0])

        # Total movement (sum of absolute period-to-period changes)
        total_movement = np.sum(np.abs(np.diff(close_prices)))

        # Avoid division by zero
        if total_movement == 0:
            return 1.0

        per = net_movement / total_movement
        return per

    def calculate_candlestick_resistance(self, ohlc):
        """Calculate the candlestick resistance factor for a single candle."""
        open_price, high, low, close_price = ohlc

        # Candle range
        candle_range = high - low

        # Avoid division by zero
        if candle_range == 0:
            return 0.0

        # Real body
        real_body = abs(close_price - open_price)

        # Wasted movement proportion
        crf = (candle_range - real_body) / candle_range
        return crf

    def calculate_volatility(self, close_prices):
        """Calculate price volatility as standard deviation of returns."""
        if len(close_prices) < 2:
            return 1.0

        returns = np.diff(close_prices) / close_prices[:-1]
        volatility = np.std(returns)

        # Avoid division by zero or near-zero
        return max(volatility, 0.001)

    def detect_regime_small_dataset(self, prices, window=None):
        """Detect the current market regime using methods suitable for small datasets.
        Uses a combination of momentum and volatility analysis rather than Hurst.

        Parameters
        ----------
        prices : array-like
            Price series to analyze
        window : int, optional
            Window size for analysis, defaults to lookback_period if None

        Returns
        -------
        str
            Market regime ('trending', 'reversal', or 'neutral')

        """
        if window is None:
            window = min(len(prices) - 1, self.lookback_period)

        if len(prices) < window + 1:
            return "neutral"  # Default if not enough data

        # Use multiple timeframe momentum analysis
        short_window = max(2, window // 2)
        med_window = window

        # Calculate short-term and medium-term momentum
        short_mom = prices[-1] - prices[-short_window - 1]
        med_mom = prices[-1] - prices[-med_window - 1]

        # Calculate path efficiency for both windows
        short_pe = self.calculate_path_efficiency(prices[-short_window - 1 :])
        med_pe = self.calculate_path_efficiency(prices[-med_window - 1 :])

        # Determine if momentum is strong
        returns = np.diff(prices[-window - 1 :])
        mean_return = np.mean(returns)
        std_return = np.std(returns)

        # Calculate z-scores of momentum
        short_z = short_mom / (std_return * np.sqrt(short_window)) if std_return > 0 else 0
        med_z = med_mom / (std_return * np.sqrt(med_window)) if std_return > 0 else 0

        # Check if momentum is aligned across timeframes
        aligned = np.sign(short_mom) == np.sign(med_mom)

        # Check efficiency (directness of movement)
        high_efficiency = short_pe > 0.7 and med_pe > 0.6

        # Price pattern analysis (last few candles)
        last_moves = np.diff(prices[-min(4, len(prices) - 1) :]) if len(prices) >= 3 else np.array([0])
        consistent_direction = np.all(last_moves > 0) or np.all(last_moves < 0)

        # Combine factors to determine regime
        if aligned and (abs(med_z) > 1.0) and (high_efficiency or consistent_direction):
            return "trending"
        if not aligned and (abs(short_z) > 1.2) and (short_pe < 0.5):
            return "reversal"
        return "neutral"

    def optimize_parameters(self, df, min_samples=5):
        """Optimize model parameters using historical data.
        Adapted for smaller datasets by using simpler optimization criteria.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with OHLC prices
        min_samples : int
            Minimum number of samples required for optimization

        Returns
        -------
        tuple
            Optimized (alpha, beta, gamma) parameters

        """
        # For very small datasets, use predetermined parameters
        if len(df) < min_samples:
            # Detect current market regime using small dataset method
            if len(df) >= 4:  # Need at least a few points for regime detection
                close_prices = df["close"].values
                current_regime = self.detect_regime_small_dataset(close_prices)

                # Get parameters based on regime
                regime_params = self.regime_adjustments[current_regime]
                return (regime_params["alpha"], regime_params["beta"], regime_params["gamma"])
            # Not enough data even for regime detection
            return (self.alpha, self.beta, self.gamma)

        # Detect current market regime
        close_prices = df["close"].values
        current_regime = self.detect_regime_small_dataset(close_prices)

        # Get initial parameters based on regime
        regime_params = self.regime_adjustments[current_regime]
        initial_params = (regime_params["alpha"], regime_params["beta"], regime_params["gamma"])

        if len(df) < self.optimization_window:
            return initial_params

        # Define simplified objective function for smaller datasets
        def objective_function(params):
            alpha, beta, gamma = params

            # Calculate PLR values using the parameters
            plr_values = []
            future_returns = []

            for i in range(self.lookback_period, len(df)):
                window = slice(i - self.lookback_period, i + 1)
                close_window = df["close"].values[window]

                momentum = self.calculate_momentum(close_window)
                per = self.calculate_path_efficiency(close_window)

                current_ohlc = (df["open"].values[i], df["high"].values[i], df["low"].values[i], df["close"].values[i])
                crf = self.calculate_candlestick_resistance(current_ohlc)

                # Determine signs
                momentum_sign = np.sign(momentum)
                net_direction_sign = np.sign(close_window[-1] - close_window[0])

                # Calculate PLR
                plr = alpha * momentum_sign * abs(momentum) + beta * net_direction_sign * per - gamma * crf

                plr_values.append(plr)

                # Get future return if available
                if i + 1 < len(df):
                    future_return = (df["close"].values[i + 1] - df["close"].values[i]) / df["close"].values[i]
                    future_returns.append(future_return)

            if len(plr_values) < 2 or len(future_returns) < 2:
                return 0

            # Calculate directional accuracy
            plr_signs = np.sign(plr_values[:-1])  # Skip last one as we don't have future return for it
            future_signs = np.sign(future_returns)

            # Cut to the same length
            min_len = min(len(plr_signs), len(future_signs))
            plr_signs = plr_signs[:min_len]
            future_signs = future_signs[:min_len]

            # Calculate accuracy - how often does PLR correctly predict direction
            matches = plr_signs == future_signs
            accuracy = np.mean(matches)

            # Incentivize higher accuracy
            return -accuracy

        # Parameter bounds
        bounds = [(0.1, 0.9), (0.1, 0.9), (0.1, 0.9)]

        # Constraint: sum of parameters = 1
        def param_constraint(params):
            return sum(params) - 1.0

        constraints = {"type": "eq", "fun": param_constraint}

        # Optimize
        try:
            result = minimize(fun=objective_function, x0=initial_params, bounds=bounds, constraints=constraints, method="SLSQP", options={"maxiter": 100})

            # Return parameters (use initial if optimization fails)
            if result.success:
                return tuple(result.x)
            return initial_params
        except:
            # If optimization fails, return initial parameters
            return initial_params

    def calculate_plr(self, df, optimize=True, adapt_to_regime=True):
        """Calculate the Optimal Path of Least Resistance (PLR) signals.
        Adapted for smaller datasets with simpler normalization.

        Parameters
        ----------
        df : pandas.DataFrame
            DataFrame with 'open', 'high', 'low', 'close' columns
        optimize : bool
            Whether to optimize parameters
        adapt_to_regime : bool
            Whether to adapt parameters to market regime

        Returns
        -------
        pandas.DataFrame
            Original dataframe with added PLR metrics and signals

        """
        # Make a copy to avoid modifying the original
        result = df.copy()

        # Initialize columns
        result["plr"] = np.nan
        result["plr_normalized"] = np.nan
        result["plr_zscore"] = np.nan
        result["volatility"] = np.nan
        result["regime"] = ""
        result["signal"] = ""
        result["signal_strength"] = np.nan
        result["confidence"] = np.nan

        # Need at least lookback_period + 1 rows to calculate
        min_required = max(3, self.lookback_period)
        if len(df) <= min_required:
            return result

        # Detect market regime using small dataset method
        market_regime = self.detect_regime_small_dataset(df["close"].values)

        # Get optimal parameters
        if optimize and len(df) >= min_required * 2:
            params = self.optimize_parameters(df, min_samples=min_required * 2)
            alpha, beta, gamma = params
        elif adapt_to_regime:
            # Use pre-defined regime-specific parameters
            regime_params = self.regime_adjustments[market_regime]
            alpha = regime_params["alpha"]
            beta = regime_params["beta"]
            gamma = regime_params["gamma"]
            self.signal_threshold = regime_params["threshold"]
        else:
            # Use default parameters
            alpha, beta, gamma = self.alpha, self.beta, self.gamma

        # Adaptive lookback for very small datasets
        actual_lookback = min(self.lookback_period, len(df) // 2)

        # Calculate PLR for each point after the lookback period
        for i in range(actual_lookback, len(df)):
            # Record regime
            result.loc[df.index[i], "regime"] = market_regime

            # Get data for lookback window
            window = slice(i - actual_lookback, i + 1)
            close_prices = df["close"].values[window]

            # Calculate volatility
            volatility = self.calculate_volatility(close_prices)
            result.loc[df.index[i], "volatility"] = volatility

            # Current candle for resistance calculation
            current_ohlc = (df["open"].values[i], df["high"].values[i], df["low"].values[i], df["close"].values[i])

            # Calculate components
            momentum = self.calculate_momentum(close_prices)
            per = self.calculate_path_efficiency(close_prices)
            crf = self.calculate_candlestick_resistance(current_ohlc)

            # Determine the signs
            momentum_sign = np.sign(momentum)
            net_direction_sign = np.sign(close_prices[-1] - close_prices[0])

            # Calculate PLR
            plr = alpha * momentum_sign * abs(momentum) + beta * net_direction_sign * per - gamma * crf

            # Normalize by volatility
            plr_normalized = plr / volatility if volatility > 0 else plr

            # Store the basic PLR results
            result.loc[df.index[i], "plr"] = plr
            result.loc[df.index[i], "plr_normalized"] = plr_normalized

        # Simplified Z-score calculation for small datasets
        # Use full sample stats for small datasets
        plr_values = result["plr_normalized"].dropna().values

        if len(plr_values) > 0:
            plr_mean = np.mean(plr_values)
            plr_std = np.std(plr_values)

            # Avoid division by zero
            if plr_std == 0:
                plr_std = 1.0

            # Calculate z-scores
            result.loc[~result["plr_normalized"].isna(), "plr_zscore"] = (result.loc[~result["plr_normalized"].isna(), "plr_normalized"] - plr_mean) / plr_std

        # Generate signals based on z-scores
        threshold = self.signal_threshold

        # Apply signals
        result.loc[result["plr_zscore"] > threshold, "signal"] = "LONG"
        result.loc[result["plr_zscore"] < -threshold, "signal"] = "SHORT"
        result.loc[(result["plr_zscore"] >= -threshold) & (result["plr_zscore"] <= threshold), "signal"] = "NEUTRAL"

        # Calculate signal strength (0-100%)
        result["signal_strength"] = (abs(result["plr_zscore"]) / 3) * 100
        result["signal_strength"] = result["signal_strength"].clip(0, 100)

        # Calculate statistical significance if enough data
        if len(plr_values) >= 5:
            _, p_value = stats.ttest_1samp(plr_values, 0)
            stat_conf = 1.0 - p_value if not np.isnan(p_value) else 0.5
        else:
            stat_conf = 0.5  # Not enough data for statistical testing

        # Calculate confidence for each signal
        for i in range(actual_lookback, len(df)):
            # Signal strength component
            signal_strength = result.loc[df.index[i], "signal_strength"] / 100

            # Regime confidence
            regime = result.loc[df.index[i], "regime"]
            signal = result.loc[df.index[i], "signal"]
            plr_zscore = result.loc[df.index[i], "plr_zscore"]

            if (regime == "trending" and ((signal == "LONG" and plr_zscore > 0) or (signal == "SHORT" and plr_zscore < 0))) or (regime == "reversal" and ((signal == "LONG" and plr_zscore < 0) or (signal == "SHORT" and plr_zscore > 0))):
                regime_conf = 0.8
            else:
                regime_conf = 0.5

            # Signal consistency (simpler version for small datasets)
            consistency_conf = 0.5
            if i >= actual_lookback + 1:
                # Check if previous value agrees with current
                prev_plr = result.loc[df.index[i - 1], "plr_normalized"]
                curr_plr = result.loc[df.index[i], "plr_normalized"]

                if np.sign(prev_plr) == np.sign(curr_plr):
                    consistency_conf = 0.7
                else:
                    consistency_conf = 0.3

            # Combined confidence score
            confidence = (stat_conf * 0.3 + signal_strength * 0.4 + regime_conf * 0.2 + consistency_conf * 0.1) * 100
            result.loc[df.index[i], "confidence"] = confidence.clip(0, 100)

        return result

    def get_latest_signal(self, result):
        """Extract the latest trading signal with important metadata.

        Parameters
        ----------
        result : pandas.DataFrame
            DataFrame with calculated PLR values and signals

        Returns
        -------
        dict
            Latest signal information

        """
        if "signal" not in result.columns or len(result) == 0:
            return {"signal": "NEUTRAL", "confidence": 0, "strength": 0, "regime": "unknown"}

        latest = result.iloc[-1]

        signal_info = {"timestamp": latest.name, "price": latest["close"], "signal": latest["signal"], "confidence": latest["confidence"], "strength": latest["signal_strength"], "regime": latest["regime"], "plr_value": latest["plr"], "plr_zscore": latest["plr_zscore"], "volatility": latest["volatility"]}

        return signal_info

    def plot_signals(self, result):
        """Plot the price chart with PLR signals.

        Parameters
        ----------
        result : pandas.DataFrame
            DataFrame with calculated PLR values and signals

        """
        # Create figure
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={"height_ratios": [2, 1]})

        # Plot price
        ax1.plot(result.index, result["close"], color="black", linewidth=1)
        ax1.set_title("Price Chart with PLR Signals")
        ax1.set_ylabel("Price")

        # Highlight regime periods
        regime_changes = result["regime"].ne(result["regime"].shift()).cumsum()
        for regime_id in regime_changes.unique():
            regime_data = result[regime_changes == regime_id]
            if len(regime_data) > 0:
                regime = regime_data["regime"].iloc[0]
                if regime == "trending":
                    color = "lightgreen"
                elif regime == "reversal":
                    color = "lightsalmon"
                else:
                    color = "lightgray"

                ax1.axvspan(regime_data.index[0], regime_data.index[-1], alpha=0.2, color=color)

        # Plot buy/sell signals
        for i in range(1, len(result)):
            if result["signal"].iloc[i] == "LONG" and result["signal"].iloc[i - 1] != "LONG":
                marker_size = 80 + result["confidence"].iloc[i] / 5  # Size based on confidence
                ax1.scatter(result.index[i], result["close"].iloc[i], color="green", marker="^", s=marker_size)
            elif result["signal"].iloc[i] == "SHORT" and result["signal"].iloc[i - 1] != "SHORT":
                marker_size = 80 + result["confidence"].iloc[i] / 5  # Size based on confidence
                ax1.scatter(result.index[i], result["close"].iloc[i], color="red", marker="v", s=marker_size)

        # PLR z-score indicator
        ax2.plot(result.index, result["plr_zscore"], color="purple", linewidth=1)
        ax2.axhline(y=self.signal_threshold, color="green", linestyle="--", alpha=0.5)
        ax2.axhline(y=-self.signal_threshold, color="red", linestyle="--", alpha=0.5)
        ax2.axhline(y=0, color="gray", linestyle="-", alpha=0.3)

        # Fill area between line and zero
        ax2.fill_between(result.index, result["plr_zscore"], 0, where=(result["plr_zscore"] > 0), color="green", alpha=0.3)
        ax2.fill_between(result.index, result["plr_zscore"], 0, where=(result["plr_zscore"] < 0), color="red", alpha=0.3)

        ax2.set_title("PLR Z-Score Indicator")
        ax2.set_ylabel("Z-Score")
        ax2.set_xlabel("Time")

        # Add legend for regimes
        from matplotlib.patches import Patch

        legend_elements = [Patch(facecolor="lightgreen", alpha=0.5, label="Trending Regime"), Patch(facecolor="lightsalmon", alpha=0.5, label="Reversal Regime"), Patch(facecolor="lightgray", alpha=0.5, label="Neutral Regime")]
        ax1.legend(handles=legend_elements, loc="upper left")

        plt.tight_layout()

        # Print latest signal
        latest_signal = self.get_latest_signal(result)
        print("===== LATEST SIGNAL =====")
        print(f"Signal: {latest_signal['signal']}")
        print(f"Confidence: {latest_signal['confidence']:.1f}%")
        print(f"Strength: {latest_signal['strength']:.1f}%")
        print(f"Market Regime: {latest_signal['regime']}")
        print(f"PLR Z-Score: {latest_signal['plr_zscore']:.2f}")

        plt.show()

        return latest_signal


def analyze_csv_data(csv_data=None, csv_file=None):
    """Analyze price data from either a CSV file or a CSV string.

    Parameters
    ----------
    csv_data : str, optional
        CSV data as a string
    csv_file : str, optional
        Path to a CSV file

    Returns
    -------
    tuple
        (result_df, latest_signal) - DataFrame with PLR results and latest signal info

    """
    # Read CSV data
    if csv_file:
        df = pd.read_csv(csv_file, parse_dates=["time"])
    elif csv_data:
        df = pd.read_csv(io.StringIO(csv_data), parse_dates=["time"])
    else:
        raise ValueError("Must provide either csv_data or csv_file")

    # Set time as index
    df.set_index("time", inplace=True)

    # Ensure column names are lowercase
    df.columns = [col.lower() for col in df.columns]

    # Verify we have all required columns
    required_cols = ["open", "high", "low", "close"]
    if not all(col in df.columns for col in required_cols):
        raise ValueError(f"CSV must contain columns: {', '.join(required_cols)}")

    # Create and apply the PLR signal generator
    # Use smaller lookback period due to limited data
    plr_signal = OptimalPLRSignal(lookback_period=2, optimization_window=4)

    # Calculate signals with adaptive parameters
    result = plr_signal.calculate_plr(df, optimize=True, adapt_to_regime=True)

    # Plot signals
    latest_signal = plr_signal.plot_signals(result)

    return result, latest_signal


# Run the analysis
if __name__ == "__main__":
    # You can analyze the data either from the CSV string or from a file
    results, signal = analyze_csv_data(csv_file="historical_price_data.csv")

    # Print full results
    print("\n===== DETAILED RESULTS =====")
    pd.set_option("display.max_columns", None)
    print(results[["open", "high", "low", "close", "regime", "plr", "plr_zscore", "signal", "confidence"]])
