"""
SentinelShield AI - Data Preprocessing & Pipeline Module
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from typing import Tuple, Optional, Union
from pathlib import Path

from src.config import (
    FEATURE_COLS,
    TARGET_COL,
    SIMULATED_DATA_PATH,
    RAW_DATA_PATH
)


def load_dataset(filepath: Union[str, Path] = None) -> pd.DataFrame:
    """
    Load fraud transaction dataset from CSV file.
    Falls back to simulated dataset if raw dataset does not exist.
    """
    if filepath is None:
        if Path(RAW_DATA_PATH).exists():
            filepath = RAW_DATA_PATH
        elif Path(SIMULATED_DATA_PATH).exists():
            filepath = SIMULATED_DATA_PATH
        else:
            raise FileNotFoundError(f"No dataset found at {RAW_DATA_PATH} or {SIMULATED_DATA_PATH}")

    df = pd.read_csv(filepath)
    return df


def prepare_feature_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, Optional[pd.Series]]:
    """
    Extract and enforce the exact feature column ordering expected by the model.
    """
    # Ensure all required features exist
    missing = [col for col in FEATURE_COLS if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required feature columns: {missing}")

    X = df[FEATURE_COLS].copy()
    y = df[TARGET_COL].copy() if TARGET_COL in df.columns else None
    return X, y


def preprocess_and_split(
    filepath: Union[str, Path] = None,
    test_size: float = 0.2,
    random_state: int = 42,
    under_sample: bool = False
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    """
    Full pipeline to load data, optionally balance classes, scale features, and split.
    """
    df = load_dataset(filepath)

    if under_sample and TARGET_COL in df.columns:
        fraud = df[df[TARGET_COL] == 1]
        non_fraud = df[df[TARGET_COL] == 0].sample(n=len(fraud), random_state=random_state)
        df = pd.concat([fraud, non_fraud]).sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    X, y = prepare_feature_matrix(df)
    if y is None:
        raise ValueError("Target column 'Class' not found in dataset for supervised training.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    return X_train_scaled, X_test_scaled, y_train.values, y_test.values, scaler
