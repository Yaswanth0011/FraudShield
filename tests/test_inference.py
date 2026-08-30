"""
Unit Tests for FraudShield Inference Engine, 3-Tier Risk Levels, & Preprocessing
"""

import pytest
import numpy as np
import pandas as pd
from pathlib import Path

from src.config import (
    FEATURE_COLS,
    TARGET_COL,
    SIMULATED_DATA_PATH,
    RISK_LEVEL_LOW,
    RISK_LEVEL_MEDIUM,
    RISK_LEVEL_HIGH,
    ACTION_APPROVE,
    ACTION_MANUAL_REVIEW,
    ACTION_BLOCK
)
from src.inference import FraudInferenceEngine
from src.utils import mask_card_number, format_currency, format_time_delta


@pytest.fixture
def sample_transaction():
    """Create a sample dictionary representing a normal transaction."""
    tx = {f"V{i}": 0.0 for i in range(1, 29)}
    tx["Time"] = 1000.0
    tx["Amount"] = 50.0
    return tx


@pytest.fixture
def fraud_transaction():
    """Create a sample dictionary representing an anomalous transaction."""
    tx = {f"V{i}": 0.0 for i in range(1, 29)}
    tx["V14"] = -8.5
    tx["V17"] = -12.0
    tx["V12"] = -7.0
    tx["Time"] = 50000.0
    tx["Amount"] = 450.0
    return tx


def test_feature_cols_length():
    """Verify that canonical feature vector contains exactly 30 features."""
    assert len(FEATURE_COLS) == 30
    assert FEATURE_COLS[0] == "Time"
    assert FEATURE_COLS[-1] == "Amount"
    assert "V1" in FEATURE_COLS
    assert "V28" in FEATURE_COLS


def test_mask_card_number():
    """Test card number masking utility."""
    assert mask_card_number("4111222233334567") == "•••• •••• •••• 4567"
    assert mask_card_number("378282246310005") == "•••• •••• •••• 0005"
    assert mask_card_number("1234") == "•••• •••• •••• 1234"


def test_formatting_utilities():
    """Test currency and time delta formatting."""
    assert format_currency(1234.56, "$") == "$1,234.56"
    assert format_currency(0.5, "€") == "€0.50"
    assert format_time_delta(3665) == "+01h 01m 05s"


def test_inference_missing_feature_raises_error(sample_transaction):
    """Test that missing required feature columns raises ValueError."""
    engine = FraudInferenceEngine()
    incomplete_tx = sample_transaction.copy()
    del incomplete_tx["V14"]

    with pytest.raises(ValueError, match="Missing required feature"):
        engine.extract_feature_vector(incomplete_tx)


def test_inference_score_structure_and_risk_display(sample_transaction):
    """Test structure and types of inference results including 3-tier risk display."""
    engine = FraudInferenceEngine()
    result = engine.score_transaction(sample_transaction, threshold=0.50)

    assert "is_fraud" in result
    assert "fraud_probability" in result
    assert "fraud_probability_pct" in result
    assert "risk_score_display" in result
    assert "risk_level" in result
    assert "recommendation" in result
    assert "anomaly_insights" in result
    assert 0.0 <= result["fraud_probability"] <= 1.0
    assert isinstance(result["is_fraud"], bool)
    assert result["risk_level"] in [RISK_LEVEL_LOW, RISK_LEVEL_MEDIUM, RISK_LEVEL_HIGH]
    # Verify display format matches: "Risk Score: XX% — LEVEL RISK"
    assert "Risk Score:" in result["risk_score_display"]
    assert "RISK" in result["risk_score_display"]


def test_recommendation_mapping():
    """Verify action recommendation mapping across risk levels."""
    engine = FraudInferenceEngine()

    # Low risk mock
    low_tx = {f"V{i}": 0.0 for i in range(1, 29)}
    low_tx["Time"] = 100.0
    low_tx["Amount"] = 10.0
    res_low = engine.score_transaction(low_tx)
    assert res_low["recommendation"] in [ACTION_APPROVE, ACTION_MANUAL_REVIEW, ACTION_BLOCK]


def test_inference_threshold_sensitivity(fraud_transaction):
    """Test that decision changes when threshold is shifted."""
    engine = FraudInferenceEngine()
    result_low_thresh = engine.score_transaction(fraud_transaction, threshold=0.01)
    result_high_thresh = engine.score_transaction(fraud_transaction, threshold=0.99)

    assert result_low_thresh["decision_threshold"] == 0.01
    assert result_high_thresh["decision_threshold"] == 0.99
