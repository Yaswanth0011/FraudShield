"""
FraudShield — Model Training & Evaluation Pipeline
"""

import argparse
import joblib
import pandas as pd
from pathlib import Path
from typing import Dict, Any

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
try:
    from xgboost import XGBClassifier
    HAS_XGBOOST = True
except ImportError:
    HAS_XGBOOST = False

from src.config import (
    MODEL_PATH,
    SCALER_PATH,
    FEATURE_COLS,
    TARGET_COL,
    SIMULATED_DATA_PATH,
    RAW_DATA_PATH
)
from src.data_preprocessing import preprocess_and_split
from src.evaluation import evaluate_model_performance


def train_baseline_models(X_train, y_train, random_state: int = 42) -> Dict[str, Any]:
    """
    Train baseline supervised classification models.
    Note: These baseline algorithms (Logistic Regression, Random Forest, XGBoost)
    are standard implementations from scikit-learn and XGBoost libraries.
    """
    models = {}

    # 1. Logistic Regression Baseline
    print("Training Logistic Regression...")
    lr = LogisticRegression(max_iter=1000, random_state=random_state, class_weight='balanced')
    lr.fit(X_train, y_train)
    models['Logistic Regression'] = lr

    # 2. Random Forest Classifier (Primary Ensemble)
    print("Training Random Forest Classifier...")
    rf = RandomForestClassifier(
        n_estimators=100,
        max_depth=12,
        random_state=random_state,
        n_jobs=-1,
        class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    models['Random Forest'] = rf

    # 3. XGBoost Classifier (if available)
    if HAS_XGBOOST:
        print("Training XGBoost Classifier...")
        xgb = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=6,
            eval_metric='logloss',
            random_state=random_state
        )
        xgb.fit(X_train, y_train)
        models['XGBoost'] = xgb

    return models


def run_training_pipeline(
    data_path: str = None,
    export_model_path: str = str(MODEL_PATH),
    export_scaler_path: str = str(SCALER_PATH)
):
    """
    End-to-end pipeline: load data -> preprocess & scale -> train models -> evaluate -> save artifacts.
    """
    print("=" * 60)
    print("🛡️ FraudShield — Training & Model Export Pipeline")
    print("=" * 60)

    # Preprocess & Split
    X_train, X_test, y_train, y_test, scaler = preprocess_and_split(
        filepath=data_path,
        test_size=0.2,
        random_state=42
    )

    print(f"Training set: {X_train.shape[0]} samples | Test set: {X_test.shape[0]} samples")
    print(f"Features: {len(FEATURE_COLS)} features {FEATURE_COLS[:3]} ... {FEATURE_COLS[-2:]}")

    # Train baselines
    models = train_baseline_models(X_train, y_train)

    # Evaluate each model
    print("\n--- Model Benchmark Results ---")
    for name, model in models.items():
        print(f"\nEvaluating: {name}")
        metrics = evaluate_model_performance(model, X_test, y_test)
        print(f"  Accuracy:  {metrics['accuracy']:.4f}")
        print(f"  Precision: {metrics['precision']:.4f}")
        print(f"  Recall:    {metrics['recall']:.4f}")
        print(f"  F1-Score:  {metrics['f1_score']:.4f}")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.4f}")
        print(f"  PR-AUC:    {metrics['pr_auc']:.4f}")

    # Save primary model (Random Forest) and scaler
    primary_model = models.get('Random Forest')
    Path(export_model_path).parent.mkdir(parents=True, exist_ok=True)
    Path(export_scaler_path).parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(primary_model, export_model_path)
    joblib.dump(scaler, export_scaler_path)

    print(f"\n✅ Successfully saved primary model to: {export_model_path}")
    print(f"✅ Successfully saved scaler to: {export_scaler_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train FraudShield Fraud Detection Models")
    parser.add_argument("--data", type=str, default=None, help="Path to creditcard.csv dataset")
    args = parser.parse_args()

    run_training_pipeline(data_path=args.data)
