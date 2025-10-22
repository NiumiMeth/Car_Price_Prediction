"""
Configuration file for Car Price Prediction Project
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ARTIFACTS_DIR = PROJECT_ROOT / "artifacts"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
SRC_DIR = PROJECT_ROOT / "src"
STREAMLIT_DIR = SRC_DIR / "streamlit"
MODELS_DIR = SRC_DIR / "models"
UTILS_DIR = SRC_DIR / "utils"

# Data files
RAW_DATA_FILE = "car.csv"
TRAIN_FILE = "train_final.csv"
TEST_FILE = "test_final.csv"

# Model artifacts
MODEL_FILE = "model.joblib"
LOG_SCALER_FILE = "log_scaler.joblib"
DIRECT_SCALER_FILE = "direct_scaler.joblib"
LOG_TRANSFORMER_FILE = "log_transformer.joblib"
FEATURE_ORDER_FILE = "feature_order.joblib"
MODEL_TARGET_MAPPING_FILE = "model_target_mapping.csv"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Feature engineering parameters
DATA_COLLECTION_YEAR = 2024  # Update based on your dataset
LOWER_POWER_THRESHOLD = 30
LOWER_MILEAGE_THRESHOLD = 500
UPPER_PRICE_PERCENTILE = 0.995
UPPER_FUEL_CONSUMPTION_PERCENTILE = 0.9998
UPPER_MILEAGE_PERCENTILE = 0.9998

# Categorical columns for one-hot encoding
CATEGORICAL_COLUMNS = ['brand', 'color', 'transmission_type', 'fuel_type']

# Columns for log transformation
LOG_SCALE_COLUMNS = ["power_kw", "fuel_consumption_g_km", "mileage_in_km"]

# Columns for direct scaling
DIRECT_SCALE_COLUMNS = ["ev_range_km", "vehicle_manufacturing_age", "vehicle_registration_age"]

# Passthrough columns (no scaling)
PASSTHROUGH_COLUMNS = ["reg_month_sin", "reg_month_cos", "model_target_enc", "mileage_per_year"]

# Valid fuel types
VALID_FUEL_TYPES = [
    "petrol", "diesel", "electric", "hybrid",
    "diesel_hybrid", "lpg", "cng", "hydrogen", "ethanol"
]

# Streamlit configuration
STREAMLIT_CONFIG = {
    "page_title": "Car Price Prediction",
    "page_icon": "🚗",
    "layout": "wide",
    "initial_sidebar_state": "expanded"
}

# Create directories if they don't exist
def create_directories():
    """Create necessary directories if they don't exist"""
    directories = [
        DATA_DIR,
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        ARTIFACTS_DIR,
        STREAMLIT_DIR,
        MODELS_DIR,
        UTILS_DIR
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

if __name__ == "__main__":
    create_directories()
