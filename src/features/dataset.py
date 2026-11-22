"""
dataset.py - shared data loading, targets and leak-free grouped train/test split.

Every model (yours and your teammates') should import from here so that
everyone trains and is evaluated on exactly the same rows.
"""
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ---------- paths ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = PROJECT_ROOT / "data" / "processed" / "pakistan_cars_clean.csv"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "reports" / "evidence"

# ---------- settings (change only here) ----------
RANDOM_STATE = 42
TEST_SIZE = 0.2
TARGET = "price_pkr"
CURRENT_YEAR = 2026

# Price bands for the classification task (PKR)
BAND_BINS = [0, 2_000_000, 5_000_000, 15_000_000, np.inf]
BAND_LABELS = ["Budget", "Mid-range", "Premium", "Luxury"]

# ---------- feature groups ----------
HIGH_CARD_COLS = ["make", "model", "variant"]          # -> TargetEncoder
LOW_CARD_COLS = ["fuel_type", "transmission"]          # -> OneHotEncoder
NUM_COLS = [
    "car_age", "mileage", "km_per_year", "engine_cc", "battery_kwh",
    "is_electric_unit", "has_auction_rating", "auction_rating_filled",
]
FEATURE_COLS = HIGH_CARD_COLS + LOW_CARD_COLS + NUM_COLS
GROUP_COLS = ["make", "model", "variant", "car_age", "mileage"]


@dataclass
class Split:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_reg_train: pd.Series   # price in PKR
    y_reg_test: pd.Series
    y_clf_train: pd.Series   # price band (str)
    y_clf_test: pd.Series


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["car_age"] = CURRENT_YEAR - df["model_year"]
    df["km_per_year"] = df["mileage"] / df["car_age"].clip(lower=1)
    return df


def load_data(path: Path = DATA_PATH) -> pd.DataFrame:
    if not Path(path).exists():
        raise FileNotFoundError(f"Dataset not found at: {path}")
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    df = add_features(df)
    missing = [c for c in FEATURE_COLS + [TARGET] if c not in df.columns]
    if missing:
        raise KeyError(f"Missing expected columns: {missing}")
    return df


def add_price_band(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["price_band"] = pd.cut(
        df[TARGET], bins=BAND_BINS, labels=BAND_LABELS
    ).astype(str)
    return df


def get_split() -> Split:
    """One stratified, group-aware 80/20 split used by BOTH tasks.

    Groups identical vehicle listings (make, model, variant, car_age, mileage)
    so relisted cars never leak across train and test sets.
    """
    df = add_price_band(load_data())
    X = df[FEATURE_COLS]
    groups = df[GROUP_COLS].astype(str).agg("|".join, axis=1)

    # 5 folds -> 80% train / 20% test, stratified by price_band with zero group overlap
    sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    train_idx, test_idx = next(sgkf.split(X, df["price_band"], groups=groups))

    return Split(
        X_train=X.iloc[train_idx], X_test=X.iloc[test_idx],
        y_reg_train=df.iloc[train_idx][TARGET], y_reg_test=df.iloc[test_idx][TARGET],
        y_clf_train=df.iloc[train_idx]["price_band"],
        y_clf_test=df.iloc[test_idx]["price_band"],
    )


if __name__ == "__main__":
    s = get_split()
    tr_g = set(s.X_train[GROUP_COLS].astype(str).agg("|".join, axis=1))
    te_g = s.X_test[GROUP_COLS].astype(str).agg("|".join, axis=1)
    print(f"Train: {s.X_train.shape} | Test: {s.X_test.shape}")
    print(f"Relisting overlap: {te_g.isin(tr_g).mean()*100:.2f}%")
    print("\nPrice band share (train):")
    print(s.y_clf_train.value_counts(normalize=True).round(3).to_string())