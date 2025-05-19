"""Lottery Pattern Analysis and Prediction using HMM

This script:
- Loads historic UK lottery data from CSV
- Extracts features and analyzes number frequencies
- Fits a Hidden Markov Model to sequences of winning numbers
- Generates sample predictions for the next 5 draws

Author: You!
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Try to import CategoricalHMM, else fallback to MultinomialHMM
try:
    from hmmlearn.hmm import CategoricalHMM

    HMMClass = CategoricalHMM
    hmm_type = "categorical"
except ImportError:
    from hmmlearn.hmm import MultinomialHMM

    HMMClass = MultinomialHMM
    hmm_type = "multinomial"

# --- Load the CSV data ---
cols = [
    "DrawNumber",
    "Day",
    "DD",
    "MMM",
    "YYYY",
    "N1",
    "N2",
    "N3",
    "N4",
    "N5",
    "N6",
    "Bonus",
    "Jackpot",
    "Wins",
    "Machine",
    "Set",
]
df = pd.read_csv("lottery.csv", names=cols, skiprows=1)

df["Date"] = pd.to_datetime(
    df["DD"].astype(str) + df["MMM"].astype(str) + df["YYYY"].astype(str),
    format="%d%b%Y",
)

df["WeekNumber"] = df["Date"].dt.isocalendar().week
number_cols = ["N1", "N2", "N3", "N4", "N5", "N6"]
df[number_cols] = df[number_cols].astype(int)

print(df[["Date"] + number_cols].head())

# --- Frequency Analysis ---
all_numbers = np.concatenate([df[n].values for n in number_cols])
unique, counts = np.unique(all_numbers, return_counts=True)
freqs = pd.Series(counts, index=unique).sort_index()
print("Number Frequency Table (most common first):")
print(freqs.sort_values(ascending=False).head(10))

plt.figure(figsize=(12, 5))
freqs.plot(kind="bar")
plt.title("Frequency of Lottery Numbers")
plt.xlabel("Number")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

# --- Simple Transition Analysis ---
n1_transitions = pd.crosstab(df["N1"][:-1], df["N1"][1:])
print("\nTransition matrix for N1 (partial view):")
print(n1_transitions.iloc[:5, :5])

# --- HMM Modeling ---
flattened_numbers = df[number_cols].values.flatten().reshape(-1, 1) - 1
# Numbers must be zero-based for hmmlearn

n_unique = flattened_numbers.max() + 1

if hmm_type == "multinomial":
    # Must use n_trials=1 for categorical emulation in new hmmlearn
    model = HMMClass(
        n_components=10,
        n_iter=100,
        random_state=42,
        n_trials=1,
        n_features=n_unique,
    )
else:
    # CategoricalHMM doesn't need n_trials
    model = HMMClass(
        n_components=10,
        n_iter=100,
        random_state=42,
        n_features=n_unique,
    )

model.fit(flattened_numbers)

# --- Sample (predict) next 5 draws ---
n_draws = 5
n_numbers = 6
predictions = []
for _ in range(n_draws):
    seq, _ = model.sample(n_numbers)
    # Convert back to 1-based numbers
    nums = np.unique(seq.flatten() + 1)
    while len(nums) < n_numbers:
        extra, _ = model.sample(n_numbers - len(nums))
        nums = np.unique(np.concatenate([nums, extra.flatten() + 1]))
    nums = np.sort(nums)[:n_numbers]
    predictions.append(nums.tolist())

print("\nPredicted next 5 draws (for entertainment only!):")
for i, pred in enumerate(predictions, 1):
    print(f"Draw {i}: {pred}")

print(
    "\nDisclaimer: Lottery numbers are, by design, random. No statistical or AI approach can predict them better than chance. This script is a fun demonstration of feature engineering and HMMs applied to sequential data.\n",
)
