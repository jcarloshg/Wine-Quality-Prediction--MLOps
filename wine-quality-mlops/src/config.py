# src/config.py
import os
from pathlib import Path

# Project paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODELS_DIR = BASE_DIR / "models"

# MLflow configuration
MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000")
# MLFLOW_TRACKING_URI = "http://192.168.101.9:5000"
EXPERIMENT_NAME = "wine-quality-prediction"

# Data configuration
DATASET_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"
TARGET_COLUMN = "quality"
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model configuration
REGISTERED_MODEL_NAME = "wine-quality-model"

# API configuration
API_HOST = "0.0.0.0"
API_PORT = 5001

# Create directories
for dir_path in [RAW_DATA_DIR, PROCESSED_DATA_DIR, MODELS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)