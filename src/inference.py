"""
FraudShield — Inference & Risk Scoring Engine
"""

import os
import joblib
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Union, Optional
from pathlib import Path

from src.config import (
    FEATURE_COLS,
    HIGH_IMPACT_PCA_FEATURES,
    DEFAULT_DECISION_THRESHOLD,
    LOW_RISK_MAX,
    MEDIUM_RISK_MAX,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
    RISK_LEVEL_HIGH,
    ACTION_APPROVE,
    ACTION_MANUAL_REVIEW,
    ACTION_BLOCK,
    MODEL_PATH,
    SCALER_PATH
)


class FraudInferenceEngine:
    """
    Inference and risk classification engine for payment transactions.
    Supports 3-tier risk levels (LOW, MEDIUM, HIGH) and analyst action recommendations.
    """

    def __init__(
        self,
        model_path: Union[str, Path] = MODEL_PATH,
        scaler_path: Union[str, Path] = SCALER_PATH,
        model_instance: Any = None,
        scaler_instance: Any = None
    ):
        if model_instance is not None and scaler_instance is not None:
            self.model = model_instance
            self.scaler = scaler_instance
        else:
            self.model = self._load_artifact(model_path, "Model")
            self.scaler = self._load_artifact(scaler_path, "Scaler")

    def _load_artifact(self, filepath: Union[str, Path], name: str) -> Any:
        path = Path(filepath)
        if not path.exists():
            raise FileNotFoundError(f"{name} artifact not found at {path.resolve()}")
        return joblib.load(path)

    def extract_feature_vector(self, data: Union[pd.Series, dict, pd.DataFrame]) -> np.ndarray:
        """
        Extract features strictly in the canonical order required by the scaler & model.
        """
        if isinstance(data, pd.DataFrame):
            missing = [col for col in FEATURE_COLS if col not in data.columns]
            if missing:
                raise ValueError(f"Missing required features in input data: {missing}")
            return data[FEATURE_COLS].values

        elif isinstance(data, (pd.Series, dict)):
            row = []
            for col in FEATURE_COLS:
                if col not in data:
                    raise ValueError(f"Missing required feature: '{col}'")
                row.append(float(data[col]))
            return np.array([row])

        elif isinstance(data, np.ndarray):
            if data.ndim == 1:
                if len(data) != len(FEATURE_COLS):
                    raise ValueError(f"Expected {len(FEATURE_COLS)} features, got {len(data)}")
                return data.reshape(1, -1)
            elif data.ndim == 2:
                if data.shape[1] != len(FEATURE_COLS):
                    raise ValueError(f"Expected {len(FEATURE_COLS)} columns, got {data.shape[1]}")
                return data
            else:
                raise ValueError("NumPy array must be 1D or 2D.")
        else:
            raise TypeError(f"Unsupported data type for inference: {type(data)}")

    def score_transaction(
        self,
        transaction_data: Union[pd.Series, dict, np.ndarray],
        threshold: float = DEFAULT_DECISION_THRESHOLD
    ) -> Dict[str, Any]:
        """
        Execute full inference pipeline on a single transaction and generate rich risk analytics.
        Classifies risk into LOW (0-30%), MEDIUM (30-70%), HIGH (70-100%).
        """
        raw_vector = self.extract_feature_vector(transaction_data)
        scaled_vector = self.scaler.transform(raw_vector)

        # Get probability estimation
        if hasattr(self.model, "predict_proba"):
            prob_fraud = float(self.model.predict_proba(scaled_vector)[0][1])
        else:
            prob_fraud = float(self.model.predict(scaled_vector)[0])

        prob_pct = round(prob_fraud * 100, 1)
        is_flagged = bool(prob_fraud >= threshold)

        # 3-Tier Risk Level Classification (Change 2)
        if prob_fraud < LOW_RISK_MAX:
            risk_level = RISK_LEVEL_LOW
            recommendation = ACTION_APPROVE
            action_code = "APPROVE"
            analyst_guidance = "Transaction risk is minimal. Automatic approval recommended."
        elif prob_fraud < MEDIUM_RISK_MAX:
            risk_level = RISK_LEVEL_MEDIUM
            recommendation = ACTION_MANUAL_REVIEW
            action_code = "MANUAL_REVIEW"
            analyst_guidance = "Suspicious transaction pattern detected. Place on temporary hold and initiate manual analyst review / step-up verification."
        else:
            risk_level = RISK_LEVEL_HIGH
            recommendation = ACTION_BLOCK
            action_code = "BLOCK"
            analyst_guidance = "High anomaly confidence. Immediate decline, card freeze, and escalation to fraud operations required."

        # Prominent display string: e.g. "Risk Score: 82% — HIGH RISK"
        risk_score_display = f"Risk Score: {prob_pct:.0f}% — {risk_level} RISK"

        # Compute key PCA feature deviations for explainability
        anomaly_insights = self._analyze_feature_anomalies(raw_vector[0])

        return {
            "is_fraud": is_flagged,
            "prediction_label": "Fraudulent" if is_flagged else "Legitimate",
            "fraud_probability": prob_fraud,
            "fraud_probability_pct": prob_pct,
            "risk_score_display": risk_score_display,
            "risk_level": risk_level,
            "decision_threshold": threshold,
            "recommendation": recommendation,
            "action_code": action_code,
            "analyst_guidance": analyst_guidance,
            "anomaly_insights": anomaly_insights
        }

    def _analyze_feature_anomalies(self, raw_features: np.ndarray) -> List[Dict[str, Any]]:
        """
        Identify top PCA components showing anomalous values (extreme deviation from 0).
        """
        anomalies = []
        for feature_name in HIGH_IMPACT_PCA_FEATURES:
            idx = FEATURE_COLS.index(feature_name)
            val = float(raw_features[idx])
            abs_dev = abs(val)
            if abs_dev >= 2.0:
                severity = "High" if abs_dev >= 4.0 else "Medium"
                anomalies.append({
                    "feature": feature_name,
                    "value": round(val, 4),
                    "deviation_magnitude": round(abs_dev, 2),
                    "severity": severity
                })

        anomalies.sort(key=lambda x: x["deviation_magnitude"], reverse=True)
        return anomalies

    def score_batch(self, df: pd.DataFrame, threshold: float = DEFAULT_DECISION_THRESHOLD) -> pd.DataFrame:
        """
        Score a batch DataFrame and return dataframe enriched with risk predictions.
        """
        raw_matrix = self.extract_feature_vector(df)
        scaled_matrix = self.scaler.transform(raw_matrix)

        if hasattr(self.model, "predict_proba"):
            probs = self.model.predict_proba(scaled_matrix)[:, 1]
        else:
            probs = self.model.predict(scaled_matrix)

        result_df = df.copy()

        # Add transaction ID if not present
        if "Transaction_ID" not in result_df.columns:
            result_df["Transaction_ID"] = [f"TXN-{10001 + i}" for i in range(len(result_df))]

        result_df["Fraud_Probability"] = probs
        result_df["Risk_Score_Pct"] = (probs * 100).round(1)
        result_df["Predicted_Fraud"] = (probs >= threshold).astype(int)
        result_df["Prediction_Label"] = np.where(result_df["Predicted_Fraud"] == 1, "Fraudulent", "Legitimate")

        # 3-Tier Risk Levels (LOW, MEDIUM, HIGH)
        result_df["Risk_Level"] = np.where(
            probs < LOW_RISK_MAX,
            RISK_LEVEL_LOW,
            np.where(probs < MEDIUM_RISK_MAX, RISK_LEVEL_MEDIUM, RISK_LEVEL_HIGH)
        )

        result_df["Risk_Score_Display"] = [
            f"Risk Score: {p:.0f}% — {lvl} RISK"
            for p, lvl in zip(result_df["Risk_Score_Pct"], result_df["Risk_Level"])
        ]

        result_df["Recommended_Action"] = np.where(
            result_df["Risk_Level"] == RISK_LEVEL_LOW,
            ACTION_APPROVE,
            np.where(result_df["Risk_Level"] == RISK_LEVEL_MEDIUM, ACTION_MANUAL_REVIEW, ACTION_BLOCK)
        )

        return result_df
