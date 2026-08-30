"""
SentinelShield AI - Configuration & Constants
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
ASSETS_DIR = BASE_DIR / "assets"

# File Paths
RAW_DATA_PATH = DATA_DIR / "creditcard.csv"
SIMULATED_DATA_PATH = DATA_DIR / "fraud_data_simulated.csv"
MODEL_PATH = MODELS_DIR / "fraud_model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"

# Feature Definitions
TIME_COL = "Time"
AMOUNT_COL = "Amount"
TARGET_COL = "Class"
PCA_FEATURE_COLS = [f"V{i}" for i in range(1, 29)]
FEATURE_COLS = [TIME_COL] + PCA_FEATURE_COLS + [AMOUNT_COL]

# Risk Scoring Configuration
DEFAULT_DECISION_THRESHOLD = 0.50
LOW_RISK_MAX = 0.30       # 0–30% -> LOW
MEDIUM_RISK_MAX = 0.70    # 30–70% -> MEDIUM
# 70–100% -> HIGH

# Risk Level Labels (Product Standard)
RISK_LEVEL_LOW = "LOW"
RISK_LEVEL_MEDIUM = "MEDIUM"
RISK_LEVEL_HIGH = "HIGH"

# Legacy aliases
RISK_LEVEL_MODERATE = RISK_LEVEL_MEDIUM
RISK_LEVEL_CRITICAL = RISK_LEVEL_HIGH

# Analyst Recommended Actions
ACTION_APPROVE = "APPROVE"
ACTION_MANUAL_REVIEW = "MANUAL REVIEW"
ACTION_BLOCK = "BLOCK & DECLINE"

# Key PCA features known for strong fraud correlation in European Cardholders dataset
HIGH_IMPACT_PCA_FEATURES = ["V14", "V17", "V12", "V10", "V4", "V11", "V16", "V18"]

# Synthetic Data Generation Presets
DEFAULT_MERCHANTS = [
    "Amazon Web Services",
    "Apple Store",
    "Walmart Supercenter",
    "Netflix Subscription",
    "Target Store",
    "Best Buy Electronics",
    "Uber Rides",
    "Airbnb Lodging",
    "Steam Games",
    "Shell Gas Station",
    "Starbucks Coffee",
    "Delta Air Lines"
]

DEFAULT_CITIES = [
    "New York, USA",
    "London, UK",
    "San Francisco, USA",
    "Frankfurt, Germany",
    "Tokyo, Japan",
    "Toronto, Canada",
    "Sydney, Australia",
    "Singapore",
    "Paris, France",
    "Mumbai, India"
]

DEFAULT_CARD_NETWORKS = ["Visa", "Mastercard", "American Express", "Discover"]
