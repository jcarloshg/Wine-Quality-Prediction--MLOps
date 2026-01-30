# Wine Quality Prediction - MLOps Project

## Overview

- This project implements an end-to-end MLOps pipeline for predicting red wine quality using chemical properties. 
- It demonstrates a complete machine learning lifecycle including experiment tracking, model registry, REST API deployment, and monitoring capabilities. 
- The system trains multiple regression models (Random Forest, Gradient Boosting, Ridge, ElasticNet) on the UCI Wine Quality dataset and automatically selects the best performer based on RMSE metrics. Built with MLflow, scikit-learn, and Flask, this project serves as a practical reference for implementing production-ready ML systems with Docker containerization and comprehensive testing.

---

## Table of Contents

- [Tech Stack](#tech-stack)
- [Key Features](#key-features)
- [MLOps Lifecycle](#mlops-lifecycle)
  - [Data Management](#data-management)
  - [Experiment Tracking](#experiment-tracking)
  - [Model Training](#model-training)
  - [Model Registry](#model-registry)
  - [Deployment](#deployment)
  - [Monitoring](#monitoring)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Running the Project](#running-the-project)
- [Project Structure](#project-structure)
- [API Documentation](#api-documentation)
- [Testing](#testing)
- [TODOs / Roadmap](#todos--roadmap)

---

## Tech Stack

### MLOps Framework
- **MLflow 2.16.2** - Experiment tracking, model registry, and model serving
- **SQLite** - MLflow backend storage (mlflow.db)

### Machine Learning
- **scikit-learn 1.4.0** - Model training and evaluation
- **Pandas 2.1.4** - Data manipulation
- **NumPy 1.26.3** - Numerical computations

### Visualization
- **Matplotlib 3.8.2** - Plot generation
- **Seaborn 0.13.0** - Statistical visualizations

### API & Deployment
- **Flask 3.0.0** - REST API framework
- **Gunicorn 21.2.0** - WSGI production server
- **Docker** - Containerization
- **Docker Compose** - Multi-container orchestration

### Testing & Utilities
- **pytest 7.4.3** - Testing framework
- **requests 2.31.0** - HTTP client
- **python-dotenv 1.0.0** - Environment configuration

---

## Key Features

- **Automated ML Pipeline**: End-to-end workflow from data loading to model deployment
- **Multi-Model Training**: Trains and compares 4 regression algorithms simultaneously
- **Experiment Tracking**: MLflow integration for reproducibility and comparison
- **Model Versioning**: Automatic model registration with champion/production aliases
- **REST API**: Production-ready Flask API with health checks and batch predictions
- **Docker Support**: Fully containerized application with docker-compose orchestration
- **Prediction Logging**: Tracks all predictions for drift detection and monitoring
- **Comprehensive Testing**: Unit tests, integration tests, and API tests included
- **Feature Engineering**: StandardScaler preprocessing with artifact persistence
- **Model Explainability**: Feature importance plots and prediction visualizations

---

## MLOps Lifecycle

### Data Management

**Tools Used**: Pandas, requests

**Implementation**: [src/data_loader.py](src/data_loader.py)

The `DataLoader` class handles:
- Downloads the UCI Wine Quality dataset (Red Wine)
- Source: https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/
- Validates data integrity (checks for missing values, empty data, target column)
- Provides dataset statistics (1,599 samples, 11 chemical features, quality scores 3-8)
- Saves raw data to `data/raw/winequality-red.csv`

**Features**:
- `fixed acidity`, `volatile acidity`, `citric acid`, `residual sugar`, `chlorides`
- `free sulfur dioxide`, `total sulfur dioxide`, `density`, `pH`, `sulphates`, `alcohol`

**Target**: `quality` (regression score 0-10)

---

### Experiment Tracking

**Tools Used**: MLflow Tracking Server

**Implementation**: [src/train.py](src/train.py), [src/config.py](src/config.py)

- **Tracking URI**: http://localhost:5000
- **Backend Store**: SQLite database (`mlflow.db`)
- **Artifact Store**: Local directory (`./mlruns/`)
- **Experiment Name**: `wine-quality-prediction`

**Logged Artifacts**:
- Model parameters (hyperparameters for each algorithm)
- Performance metrics: MSE, RMSE, MAE, R² (train and test sets)
- Trained model files (sklearn format)
- Prediction vs actual plots (PNG images)
- Feature importance charts (for tree-based models)

This enables complete reproducibility and comparison across all training runs.

---

### Model Training

**Tools Used**: scikit-learn, MLflow

**Implementation**: [src/train.py](src/train.py), [src/preprocessor.py](src/preprocessor.py)

**Data Preprocessing** ([preprocessor.py](src/preprocessor.py:1)):
1. **Train-Test Split**: 80-20 split (stratified on quality, random_state=42)
2. **Feature Scaling**: StandardScaler normalization for all 11 features
3. **Artifact Persistence**: Saves fitted scaler to `models/scaler.pkl`

**Models Trained**:

1. **Random Forest Regressor**
   - 100 estimators, max_depth=10, min_samples_split=5
   - Best for handling non-linear relationships

2. **Gradient Boosting Regressor**
   - 100 estimators, learning_rate=0.1, max_depth=5
   - Sequential ensemble for improved accuracy

3. **ElasticNet**
   - alpha=0.1, l1_ratio=0.5
   - Linear model with L1 + L2 regularization

4. **Ridge Regression**
   - alpha=1.0
   - Linear baseline with L2 regularization

All models are trained on the same preprocessed data and logged to MLflow for comparison.

---

### Model Registry

**Tools Used**: MLflow Model Registry

**Implementation**: [src/evaluate.py](src/evaluate.py)

**Workflow**:
1. **Best Model Selection**: Queries MLflow runs and selects model with lowest test RMSE
2. **Model Registration**: Registers the winning model to MLflow Registry
   - Registered name: `wine-quality-model`
   - Stores model metadata, source run, and version number
3. **Model Promotion**: Uses alias-based promotion strategy
   - Sets best model to **"champion"** alias (production-ready)
   - Alternative: Legacy stage-based promotion to "Production"

**Benefits**:
- Centralized model versioning
- Rollback capabilities
- Model lineage tracking
- Staged deployment support

**Access Pattern**:
```python
# Load champion model
model_uri = "models:/wine-quality-model@champion"
model = mlflow.sklearn.load_model(model_uri)
```

---

### Deployment

**Tools Used**: Flask, Gunicorn, Docker

**Implementation**: [src/predict.py](src/predict.py)

**REST API Server** (Port 5001):

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check with model status |
| `/model_info` | GET | Returns loaded model metadata and version |
| `/predict` | POST | Single wine quality prediction |
| `/predict_batch` | POST | Batch predictions for multiple samples |

**Example Request**:
```bash
curl -X POST http://localhost:5001/predict \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4,
    "volatile acidity": 0.7,
    "citric acid": 0.0,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }'
```

**Example Response**:
```json
{
  "prediction": 5.45,
  "quality_score": 5.45
}
```

**Model Loading Strategy**:
1. Attempts to load champion model from MLflow Registry
2. Falls back to latest version if champion unavailable
3. Loads fitted StandardScaler from `models/scaler.pkl`
4. Applies scaling transformation before prediction
5. Returns predictions with error handling

**Docker Deployment**:
- **Dockerfile**: Multi-stage build with Python 3.12-slim base
- **docker-compose.yml**: Orchestrates MLflow server + API
- **Ports Exposed**: 5000 (MLflow UI), 5001 (Prediction API)
- **Volume Mounts**: Persists data, models, mlruns, and database

```bash
docker-compose up --build
```

---

### Monitoring

**Tools Used**: Custom logging system, MLflow metrics

**Implementation**: [src/monitor.py](src/monitor.py)

**Capabilities**:

1. **Prediction Logging**
   - Logs all predictions with input features, timestamp, and actual values
   - Storage format: JSONL (newline-delimited JSON)
   - File: `predictions_log.json`

2. **Drift Detection**
   - Calculates statistical summaries on recent predictions
   - Tracks distribution changes over time
   - Enables data drift alerting

3. **Performance Tracking**
   - Computes MSE, RMSE, MAE, R² on logged predictions with ground truth
   - Monitors production model performance
   - Compares against baseline metrics from training

4. **Historical Analysis**
   - Retrieves all experiment metrics from MLflow
   - Trend analysis across model versions
   - Performance regression detection

**Usage**:
```python
from src.monitor import ModelMonitor

monitor = ModelMonitor()
monitor.log_prediction(features, prediction, actual_value)
metrics = monitor.get_performance_metrics()
drift_stats = monitor.detect_drift()
```

---

## Getting Started

### Prerequisites

- **Python 3.12+** installed
- **Docker & Docker Compose** (optional, for containerized deployment)
- **Git** (for cloning the repository)
- **Virtual environment tool** (venv, conda, or similar)

### Installation

1. **Clone the repository**:
```bash
git clone <repository-url>
cd wine-quality-mlops
```

2. **Create a virtual environment**:
```bash
python3.12 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**:
```bash
pip install -r requirements.txt
```

4. **Verify installation**:
```bash
python -c "import mlflow; import sklearn; import flask; print('All dependencies installed successfully!')"
```

---

### Running the Project

#### Option 1: Automated Pipeline (Recommended)

Run the complete pipeline using the automation script:

```bash
chmod +x run_experiments.sh
./run_experiments.sh
```

This script will:
1. Start the MLflow tracking server (http://localhost:5000)
2. Download and load the wine quality dataset
3. Preprocess data (split, scale, save artifacts)
4. Train all 4 models with experiment tracking
5. Register the best model to MLflow Registry
6. Start the prediction API server (http://localhost:5001)

**Access Points**:
- MLflow UI: http://localhost:5000
- Prediction API: http://localhost:5001
- API docs below for endpoint usage

---

#### Option 2: Manual Step-by-Step Execution

**Terminal 1 - Start MLflow Server**:
```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000
```

**Terminal 2 - Run ML Pipeline**:
```bash
# Step 1: Load data
python3.12 -m src.data_loader

# Step 2: Preprocess data
python3.12 -m src.preprocessor

# Step 3: Train models
python3.12 -m src.train

# Step 4: Register best model
python3.12 -m src.evaluate

# Step 5: Start prediction API
python3.12 -m src.predict
```

**Terminal 3 - Test API**:
```bash
chmod +x tests/test-api.sh
./tests/test-api.sh
```

---

#### Option 3: Docker Deployment

Run the entire stack with Docker Compose:

```bash
docker-compose up --build
```

This single command:
- Builds the Docker image
- Starts MLflow server on port 5000
- Runs the complete ML pipeline
- Launches the prediction API on port 5001
- Persists data, models, and MLflow artifacts

**Stop the services**:
```bash
docker-compose down
```

---

## Project Structure

```
wine-quality-mlops/
├── src/                          # Source code
│   ├── __init__.py
│   ├── config.py                 # Configuration management
│   ├── data_loader.py            # Data downloading and validation
│   ├── preprocessor.py           # Feature engineering and scaling
│   ├── train.py                  # Model training with MLflow
│   ├── evaluate.py               # Model registry and selection
│   ├── predict.py                # REST API and inference
│   └── monitor.py                # Monitoring and logging
│
├── data/                         # Data storage
│   ├── raw/                      # Original dataset
│   │   └── winequality-red.csv
│   └── processed/                # Preprocessed data
│       ├── train.csv
│       └── test.csv
│
├── models/                       # Saved artifacts
│   └── scaler.pkl                # Fitted StandardScaler
│
├── tests/                        # Test suite
│   ├── test_pipeline.py          # Unit tests (pytest)
│   ├── test-api.sh               # Shell-based API tests
│   └── api-tests.http            # REST Client tests
│
├── mlruns/                       # MLflow artifacts (auto-created)
├── mlflow.db                     # SQLite tracking database
├── predictions_log.json          # Prediction monitoring logs
│
├── docker/                       # Containerization
│   └── Dockerfile
├── Dockerfile                    # Main Docker image
├── docker-compose.yml            # Multi-container orchestration
│
├── requirements.txt              # Python dependencies
├── run_experiments.sh            # Automated pipeline script
├── start.sh                      # Docker startup script
├── .gitignore                    # Git exclusions
└── README.md                     # This file
```

---

## API Documentation

### Health Check
```bash
GET http://localhost:5001/health
```

**Response**:
```json
{
  "status": "healthy",
  "model_loaded": true,
  "model_version": "3"
}
```

---

### Model Information
```bash
GET http://localhost:5001/model_info
```

**Response**:
```json
{
  "model_name": "wine-quality-model",
  "model_version": "3",
  "model_stage": "champion",
  "algorithm": "RandomForestRegressor"
}
```

---

### Single Prediction
```bash
POST http://localhost:5001/predict
Content-Type: application/json

{
  "fixed acidity": 8.5,
  "volatile acidity": 0.6,
  "citric acid": 0.15,
  "residual sugar": 2.1,
  "chlorides": 0.08,
  "free sulfur dioxide": 15.0,
  "total sulfur dioxide": 40.0,
  "density": 0.998,
  "pH": 3.3,
  "sulphates": 0.65,
  "alcohol": 10.5
}
```

**Response**:
```json
{
  "prediction": 6.2,
  "quality_score": 6.2
}
```

---

### Batch Prediction
```bash
POST http://localhost:5001/predict_batch
Content-Type: application/json

{
  "samples": [
    {
      "fixed acidity": 7.0,
      "volatile acidity": 0.5,
      ...
    },
    {
      "fixed acidity": 8.0,
      "volatile acidity": 0.7,
      ...
    }
  ]
}
```

**Response**:
```json
{
  "predictions": [5.8, 6.1],
  "count": 2
}
```

---

## Testing

### Run Unit Tests
```bash
pytest tests/test_pipeline.py -v
```

**Test Coverage**:
- Data loading and validation
- Data preprocessing and splitting
- Model prediction with sample input

---

### Run API Integration Tests
```bash
chmod +x tests/test-api.sh
./tests/test-api.sh
```

**Test Scenarios**:
- Health check endpoint
- Model info retrieval
- Single predictions (low, medium, high quality wines)
- Batch predictions
- Error handling (missing features, invalid input)

---

### REST Client Tests (VS Code)
If using VS Code with the REST Client extension:

1. Open `tests/api-tests.http`
2. Click "Send Request" above each test case
3. View responses inline

---

## TODOs / Roadmap

### Short-term Improvements
- [ ] Add data versioning with DVC (Data Version Control)
- [ ] Implement A/B testing framework for model comparison
- [ ] Add Prometheus metrics export for monitoring dashboards
- [ ] Create CI/CD pipeline with GitHub Actions
- [ ] Add model performance alerting (email/Slack notifications)
- [ ] Implement feature drift detection with statistical tests
- [ ] Add model explainability with SHAP values

### Medium-term Features
- [ ] Deploy to cloud platform (AWS/GCP/Azure)
- [ ] Add Kubernetes deployment manifests
- [ ] Implement model retraining automation on drift detection
- [ ] Create web UI for predictions (React/Streamlit)
- [ ] Add authentication and rate limiting to API
- [ ] Implement model serving with MLflow deployments
- [ ] Add support for white wine quality prediction

### Long-term Goals
- [ ] Implement online learning for continuous model updates
- [ ] Add multi-model ensemble predictions
- [ ] Create automated hyperparameter tuning with Optuna
- [ ] Implement federated learning for privacy-preserving training
- [ ] Add GraphQL API alongside REST endpoints
- [ ] Build real-time prediction streaming with Kafka
- [ ] Create comprehensive MLOps monitoring dashboard

---

## Contributing

Contributions are welcome! Please follow these guidelines:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

---

## License

This project is intended for educational purposes. Dataset source: UCI Machine Learning Repository.

---

## Acknowledgments

- **Dataset**: [UCI Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/wine+quality)
- **MLflow**: For providing excellent MLOps tooling
- **scikit-learn**: For accessible machine learning algorithms

---

**Built with MLOps best practices for reproducible, production-ready ML systems.**
