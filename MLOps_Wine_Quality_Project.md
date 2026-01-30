# Complete MLOps Project: Wine Quality Prediction

## A Weekend Guide to MLOps with MLflow

---

## 📋 Project Overview

**Problem:** Predict wine quality (regression: score 0-10)  
**Dataset:** Wine Quality Dataset (~1,600 samples, 12 features)  
**Timeline:** One weekend (6-8 hours)  
**Difficulty:** Beginner-friendly

### What You'll Build

- Complete ML pipeline with experiment tracking
- Model versioning and registry
- REST API for model serving
- Basic monitoring dashboard

---

## 🎯 MLOps Lifecycle Phases

| Phase                      | Activities                       | Tools           | Time    |
| -------------------------- | -------------------------------- | --------------- | ------- |
| **1. Data Management**     | Download, validate, version data | Pandas, Python  | 30 min  |
| **2. Experiment Tracking** | Log experiments, compare models  | MLflow Tracking | 1 hour  |
| **3. Model Development**   | Train/tune multiple models       | Scikit-learn    | 2 hours |
| **4. Model Registry**      | Register and version models      | MLflow Registry | 45 min  |
| **5. Model Deployment**    | Serve via REST API               | MLflow Serving  | 1 hour  |
| **6. Monitoring**          | Track predictions, performance   | Custom + MLflow | 1 hour  |

---

## 📁 Project Structure

```
wine-quality-mlops/
│
├── data/
│   ├── raw/                        # Original dataset
│   │   └── winequality-red.csv
│   └── processed/                  # Preprocessed data
│       ├── train.csv
│       └── test.csv
│
├── src/
│   ├── __init__.py
│   ├── config.py                   # Configuration
│   ├── data_loader.py              # Data loading
│   ├── preprocessor.py             # Data preprocessing
│   ├── train.py                    # Model training
│   ├── evaluate.py                 # Model evaluation
│   ├── predict.py                  # Inference
│   └── monitor.py                  # Monitoring
│
├── notebooks/
│   └── 01_exploratory_analysis.ipynb
│
├── models/                         # Saved model artifacts
├── mlruns/                         # MLflow tracking (auto-created)
│
├── docker/
│   ├── Dockerfile
│   └── docker-compose.yml
│
├── tests/
│   └── test_pipeline.py
│
├── requirements.txt
├── setup.sh                        # Setup script
├── run_experiments.sh              # Run all experiments
└── README.md
```

---

## 🚀 Step-by-Step Implementation

---

## PART 1: Initial Setup (30 minutes)

### Step 1: Create Project Directory

```bash
mkdir wine-quality-mlops
cd wine-quality-mlops

# Create all directories
mkdir -p data/{raw,processed} src notebooks models tests docker
touch src/__init__.py
```

### Step 2: Create requirements.txt

```txt
# MLflow and tracking
mlflow==2.9.2
sqlalchemy==2.0.23

# ML libraries
scikit-learn==1.3.2
pandas==2.1.4
numpy==1.24.3

# Visualization
matplotlib==3.8.2
seaborn==0.13.0

# API serving
flask==3.0.0
gunicorn==21.2.0

# Utilities
python-dotenv==1.0.0
requests==2.31.0
pyyaml==6.0.1

# Testing
pytest==7.4.3
```

### Step 3: Setup Script

```bash
# setup.sh
#!/bin/bash

echo "🚀 Setting up Wine Quality MLOps Project..."

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# Create .env file
cat > .env << EOF
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_BACKEND_STORE_URI=sqlite:///mlflow.db
MLFLOW_ARTIFACT_ROOT=./mlruns
EOF

echo "✅ Setup complete! Activate environment with: source venv/bin/activate"
```

Make it executable and run:

```bash
chmod +x setup.sh
./setup.sh
source venv/bin/activate
```

---

## PART 2: Configuration (15 minutes)

### config.py

```python
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
```

---

## PART 3: Data Management (30 minutes)

### data_loader.py

```python
# src/data_loader.py
import pandas as pd
import urllib.request
from pathlib import Path
from src.config import DATASET_URL, RAW_DATA_DIR, TARGET_COLUMN

class DataLoader:
    """Handle data downloading and loading"""

    def __init__(self):
        self.raw_data_path = RAW_DATA_DIR / "winequality-red.csv"

    def download_data(self):
        """Download wine quality dataset"""
        if self.raw_data_path.exists():
            print(f"✅ Dataset already exists at {self.raw_data_path}")
            return self.raw_data_path

        print(f"📥 Downloading dataset from {DATASET_URL}...")
        urllib.request.urlretrieve(DATASET_URL, self.raw_data_path)
        print(f"✅ Dataset downloaded to {self.raw_data_path}")
        return self.raw_data_path

    def load_data(self):
        """Load dataset into pandas DataFrame"""
        if not self.raw_data_path.exists():
            self.download_data()

        df = pd.read_csv(self.raw_data_path, sep=';')
        print(f"\n📊 Dataset Info:")
        print(f"   Shape: {df.shape}")
        print(f"   Features: {df.columns.tolist()}")
        print(f"   Target range: {df[TARGET_COLUMN].min()} - {df[TARGET_COLUMN].max()}")
        print(f"   Missing values: {df.isnull().sum().sum()}")

        return df

    def validate_data(self, df):
        """Basic data validation"""
        assert not df.empty, "Dataset is empty"
        assert TARGET_COLUMN in df.columns, f"Target column '{TARGET_COLUMN}' not found"
        assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
        print("✅ Data validation passed")
        return True

if __name__ == "__main__":
    loader = DataLoader()
    df = loader.load_data()
    loader.validate_data(df)
    print(f"\n{df.head()}")
```

### preprocessor.py

```python
# src/preprocessor.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
from pathlib import Path

from src.config import (
    PROCESSED_DATA_DIR,
    TARGET_COLUMN,
    TEST_SIZE,
    RANDOM_STATE,
    MODELS_DIR
)

class DataPreprocessor:
    """Handle data preprocessing and feature engineering"""

    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = None

    def split_data(self, df):
        """Split data into train and test sets"""
        X = df.drop(columns=[TARGET_COLUMN])
        y = df[TARGET_COLUMN]

        self.feature_names = X.columns.tolist()

        X_train, X_test, y_train, y_test = train_test_split(
            X, y,
            test_size=TEST_SIZE,
            random_state=RANDOM_STATE
        )

        print(f"✅ Data split complete:")
        print(f"   Training samples: {len(X_train)}")
        print(f"   Test samples: {len(X_test)}")

        return X_train, X_test, y_train, y_test

    def scale_features(self, X_train, X_test):
        """Standardize features"""
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)

        # Convert back to DataFrame
        X_train_scaled = pd.DataFrame(
            X_train_scaled,
            columns=self.feature_names,
            index=X_train.index
        )
        X_test_scaled = pd.DataFrame(
            X_test_scaled,
            columns=self.feature_names,
            index=X_test.index
        )

        print("✅ Feature scaling complete")
        return X_train_scaled, X_test_scaled

    def save_processed_data(self, X_train, X_test, y_train, y_test):
        """Save processed data"""
        train_df = pd.concat([X_train, y_train], axis=1)
        test_df = pd.concat([X_test, y_test], axis=1)

        train_path = PROCESSED_DATA_DIR / "train.csv"
        test_path = PROCESSED_DATA_DIR / "test.csv"

        train_df.to_csv(train_path, index=False)
        test_df.to_csv(test_path, index=False)

        print(f"✅ Processed data saved:")
        print(f"   Train: {train_path}")
        print(f"   Test: {test_path}")

    def save_scaler(self):
        """Save the fitted scaler"""
        scaler_path = MODELS_DIR / "scaler.pkl"
        joblib.dump(self.scaler, scaler_path)
        print(f"✅ Scaler saved to {scaler_path}")
        return scaler_path

    def load_scaler(self):
        """Load a saved scaler"""
        scaler_path = MODELS_DIR / "scaler.pkl"
        self.scaler = joblib.load(scaler_path)
        return self.scaler

if __name__ == "__main__":
    from src.data_loader import DataLoader

    # Load data
    loader = DataLoader()
    df = loader.load_data()

    # Preprocess
    preprocessor = DataPreprocessor()
    X_train, X_test, y_train, y_test = preprocessor.split_data(df)
    X_train_scaled, X_test_scaled = preprocessor.scale_features(X_train, X_test)

    # Save
    preprocessor.save_processed_data(X_train_scaled, X_test_scaled, y_train, y_test)
    preprocessor.save_scaler()
```

---

## PART 4: Experiment Tracking & Model Training (2 hours)

### train.py

```python
# src/train.py
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    PROCESSED_DATA_DIR,
    RANDOM_STATE
)

class ModelTrainer:
    """Train and track ML models with MLflow"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        mlflow.set_experiment(EXPERIMENT_NAME)
        self.experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        print(f"🔬 Experiment: {EXPERIMENT_NAME}")
        print(f"📍 Tracking URI: {MLFLOW_TRACKING_URI}")

    def load_processed_data(self):
        """Load preprocessed train and test data"""
        train_df = pd.read_csv(PROCESSED_DATA_DIR / "train.csv")
        test_df = pd.read_csv(PROCESSED_DATA_DIR / "test.csv")

        X_train = train_df.drop(columns=['quality'])
        y_train = train_df['quality']
        X_test = test_df.drop(columns=['quality'])
        y_test = test_df['quality']

        return X_train, X_test, y_train, y_test

    def evaluate_model(self, y_true, y_pred):
        """Calculate evaluation metrics"""
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_true, y_pred)
        r2 = r2_score(y_true, y_pred)

        metrics = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'r2': r2
        }
        return metrics

    def plot_predictions(self, y_true, y_pred, model_name):
        """Create prediction plots"""
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))

        # Scatter plot
        axes[0].scatter(y_true, y_pred, alpha=0.5)
        axes[0].plot([y_true.min(), y_true.max()],
                     [y_true.min(), y_true.max()],
                     'r--', lw=2)
        axes[0].set_xlabel('Actual Quality')
        axes[0].set_ylabel('Predicted Quality')
        axes[0].set_title(f'{model_name}: Predictions vs Actual')

        # Residuals
        residuals = y_true - y_pred
        axes[1].hist(residuals, bins=30, edgecolor='black')
        axes[1].set_xlabel('Residuals')
        axes[1].set_ylabel('Frequency')
        axes[1].set_title(f'{model_name}: Residual Distribution')

        plt.tight_layout()

        # Save figure
        plot_path = f"/tmp/{model_name}_predictions.png"
        plt.savefig(plot_path)
        plt.close()

        return plot_path

    def train_model(self, model, model_name, params, X_train, y_train, X_test, y_test):
        """Train a single model with MLflow tracking"""

        with mlflow.start_run(run_name=model_name):
            print(f"\n🚀 Training {model_name}...")

            # Log parameters
            mlflow.log_params(params)
            mlflow.log_param("model_type", model_name)

            # Train model
            model.fit(X_train, y_train)

            # Make predictions
            y_train_pred = model.predict(X_train)
            y_test_pred = model.predict(X_test)

            # Evaluate
            train_metrics = self.evaluate_model(y_train, y_train_pred)
            test_metrics = self.evaluate_model(y_test, y_test_pred)

            # Log metrics
            for metric_name, value in train_metrics.items():
                mlflow.log_metric(f"train_{metric_name}", value)

            for metric_name, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric_name}", value)

            # Log model
            mlflow.sklearn.log_model(
                model,
                "model",
                registered_model_name=None  # We'll register the best one later
            )

            # Create and log plots
            plot_path = self.plot_predictions(y_test, y_test_pred, model_name)
            mlflow.log_artifact(plot_path, "plots")

            # Log feature importance if available
            if hasattr(model, 'feature_importances_'):
                importance_df = pd.DataFrame({
                    'feature': X_train.columns,
                    'importance': model.feature_importances_
                }).sort_values('importance', ascending=False)

                importance_path = f"/tmp/{model_name}_feature_importance.csv"
                importance_df.to_csv(importance_path, index=False)
                mlflow.log_artifact(importance_path, "feature_importance")

            print(f"✅ {model_name} complete!")
            print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
            print(f"   Test R²: {test_metrics['r2']:.4f}")

            return model, test_metrics

    def run_experiments(self):
        """Run multiple experiments with different models"""

        # Load data
        X_train, X_test, y_train, y_test = self.load_processed_data()

        # Define models to try
        models_config = {
            "Random Forest": {
                "model": RandomForestRegressor(random_state=RANDOM_STATE),
                "params": {
                    "n_estimators": 100,
                    "max_depth": 10,
                    "min_samples_split": 5,
                    "random_state": RANDOM_STATE
                }
            },
            "Gradient Boosting": {
                "model": GradientBoostingRegressor(random_state=RANDOM_STATE),
                "params": {
                    "n_estimators": 100,
                    "learning_rate": 0.1,
                    "max_depth": 5,
                    "random_state": RANDOM_STATE
                }
            },
            "ElasticNet": {
                "model": ElasticNet(random_state=RANDOM_STATE),
                "params": {
                    "alpha": 0.1,
                    "l1_ratio": 0.5,
                    "random_state": RANDOM_STATE
                }
            },
            "Ridge Regression": {
                "model": Ridge(random_state=RANDOM_STATE),
                "params": {
                    "alpha": 1.0,
                    "random_state": RANDOM_STATE
                }
            }
        }

        results = {}

        # Train each model
        for model_name, config in models_config.items():
            model = config["model"].set_params(**config["params"])
            _, metrics = self.train_model(
                model,
                model_name,
                config["params"],
                X_train,
                y_train,
                X_test,
                y_test
            )
            results[model_name] = metrics

        # Print summary
        print("\n" + "="*60)
        print("📊 EXPERIMENT SUMMARY")
        print("="*60)
        results_df = pd.DataFrame(results).T
        print(results_df.sort_values('rmse'))

        # Find best model
        best_model = min(results.items(), key=lambda x: x[1]['rmse'])
        print(f"\n🏆 Best Model: {best_model[0]}")
        print(f"   RMSE: {best_model[1]['rmse']:.4f}")
        print(f"   R²: {best_model[1]['r2']:.4f}")

        return results

if __name__ == "__main__":
    trainer = ModelTrainer()
    results = trainer.run_experiments()
```

---

## PART 5: Model Registry (30 minutes)

### evaluate.py

```python
# src/evaluate.py
import mlflow
import mlflow.sklearn
from mlflow.tracking import MlflowClient
import pandas as pd

from src.config import (
    MLFLOW_TRACKING_URI,
    EXPERIMENT_NAME,
    REGISTERED_MODEL_NAME
)

class ModelRegistry:
    """Manage model versioning and promotion"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

    def get_best_run(self, metric="test_rmse"):
        """Find the best run based on a metric"""
        runs = mlflow.search_runs(
            experiment_ids=[self.experiment.experiment_id],
            order_by=[f"metrics.{metric} ASC"],
            max_results=1
        )

        if runs.empty:
            print("❌ No runs found")
            return None

        best_run = runs.iloc[0]
        print(f"\n🏆 Best Run Found:")
        print(f"   Run ID: {best_run.run_id}")
        print(f"   Model: {best_run['tags.mlflow.runName']}")
        print(f"   Test RMSE: {best_run['metrics.test_rmse']:.4f}")
        print(f"   Test R²: {best_run['metrics.test_r2']:.4f}")

        return best_run

    def register_best_model(self, metric="test_rmse"):
        """Register the best model to the Model Registry"""

        best_run = self.get_best_run(metric)

        if best_run is None:
            return None

        # Register model
        model_uri = f"runs:/{best_run.run_id}/model"

        print(f"\n📝 Registering model to '{REGISTERED_MODEL_NAME}'...")

        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=REGISTERED_MODEL_NAME
        )

        print(f"✅ Model registered:")
        print(f"   Name: {REGISTERED_MODEL_NAME}")
        print(f"   Version: {model_version.version}")

        return model_version

    def promote_model_to_production(self, version=None):
        """Promote a model version to Production stage"""

        if version is None:
            # Get latest version
            latest_versions = self.client.get_latest_versions(
                REGISTERED_MODEL_NAME,
                stages=["None"]
            )
            if not latest_versions:
                print("❌ No model versions found")
                return None
            version = latest_versions[0].version

        print(f"\n🚀 Promoting model version {version} to Production...")

        # Transition to production
        self.client.transition_model_version_stage(
            name=REGISTERED_MODEL_NAME,
            version=version,
            stage="Production",
            archive_existing_versions=True
        )

        print(f"✅ Model version {version} is now in Production!")

        return version

    def list_registered_models(self):
        """List all registered models and their versions"""
        try:
            models = self.client.search_registered_models()

            if not models:
                print("📭 No registered models found")
                return

            print("\n📚 Registered Models:")
            print("="*60)

            for model in models:
                print(f"\nModel: {model.name}")
                for version in model.latest_versions:
                    print(f"  Version {version.version}: {version.current_stage}")

        except Exception as e:
            print(f"❌ Error listing models: {e}")

    def load_production_model(self):
        """Load the production model"""
        model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"

        print(f"\n📦 Loading production model...")
        model = mlflow.sklearn.load_model(model_uri)
        print(f"✅ Model loaded successfully")

        return model

if __name__ == "__main__":
    registry = ModelRegistry()

    # Register best model
    model_version = registry.register_best_model()

    # Promote to production
    if model_version:
        registry.promote_model_to_production(model_version.version)

    # List all models
    registry.list_registered_models()
```

---

## PART 6: Model Deployment (1 hour)

### predict.py

```python
# src/predict.py
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib

from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    MODELS_DIR,
    API_HOST,
    API_PORT
)

class ModelPredictor:
    """Handle model predictions"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.model = None
        self.scaler = None
        self.feature_names = None

    def load_production_model(self):
        """Load the production model and scaler"""
        try:
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/Production"
            self.model = mlflow.sklearn.load_model(model_uri)
            print(f"✅ Production model loaded")
        except Exception as e:
            print(f"⚠️  Could not load production model: {e}")
            print("   Loading latest version instead...")
            model_uri = f"models:/{REGISTERED_MODEL_NAME}/latest"
            self.model = mlflow.sklearn.load_model(model_uri)

        # Load scaler
        scaler_path = MODELS_DIR / "scaler.pkl"
        self.scaler = joblib.load(scaler_path)
        print(f"✅ Scaler loaded")

        return self.model

    def predict(self, features_dict):
        """Make prediction from feature dictionary"""

        if self.model is None:
            self.load_production_model()

        # Convert to DataFrame
        df = pd.DataFrame([features_dict])

        # Scale features
        df_scaled = self.scaler.transform(df)

        # Predict
        prediction = self.model.predict(df_scaled)[0]

        return float(prediction)

    def predict_batch(self, features_list):
        """Make predictions for multiple samples"""

        if self.model is None:
            self.load_production_model()

        # Convert to DataFrame
        df = pd.DataFrame(features_list)

        # Scale features
        df_scaled = self.scaler.transform(df)

        # Predict
        predictions = self.model.predict(df_scaled)

        return predictions.tolist()

# Flask API
app = Flask(__name__)
predictor = ModelPredictor()

@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': predictor.model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    """Single prediction endpoint"""
    try:
        data = request.get_json()

        # Make prediction
        prediction = predictor.predict(data)

        return jsonify({
            'prediction': prediction,
            'quality_score': round(prediction, 2)
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/predict_batch', methods=['POST'])
def predict_batch():
    """Batch prediction endpoint"""
    try:
        data = request.get_json()
        features_list = data.get('samples', [])

        # Make predictions
        predictions = predictor.predict_batch(features_list)

        return jsonify({
            'predictions': predictions,
            'count': len(predictions)
        })

    except Exception as e:
        return jsonify({
            'error': str(e)
        }), 400

@app.route('/model_info', methods=['GET'])
def model_info():
    """Get model information"""
    if predictor.model is None:
        predictor.load_production_model()

    return jsonify({
        'model_name': REGISTERED_MODEL_NAME,
        'model_type': str(type(predictor.model).__name__)
    })

def start_api():
    """Start the prediction API"""
    print(f"\n🚀 Starting Prediction API...")
    print(f"   Host: {API_HOST}")
    print(f"   Port: {API_PORT}")
    print(f"\n📡 Available endpoints:")
    print(f"   GET  /health")
    print(f"   POST /predict")
    print(f"   POST /predict_batch")
    print(f"   GET  /model_info")

    # Load model on startup
    predictor.load_production_model()

    # Run app
    app.run(host=API_HOST, port=API_PORT, debug=False)

if __name__ == "__main__":
    start_api()
```

---

## PART 7: Monitoring (1 hour)

### monitor.py

```python
# src/monitor.py
import mlflow
from mlflow.tracking import MlflowClient
import pandas as pd
import numpy as np
from datetime import datetime
import json
from pathlib import Path

from src.config import MLFLOW_TRACKING_URI, REGISTERED_MODEL_NAME

class ModelMonitor:
    """Monitor model performance and predictions"""

    def __init__(self):
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        self.client = MlflowClient()
        self.predictions_log = Path("predictions_log.json")

    def log_prediction(self, features, prediction, actual=None):
        """Log a prediction for monitoring"""

        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'features': features,
            'prediction': prediction,
            'actual': actual
        }

        # Append to log file
        with open(self.predictions_log, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def load_prediction_logs(self):
        """Load all prediction logs"""

        if not self.predictions_log.exists():
            return pd.DataFrame()

        logs = []
        with open(self.predictions_log, 'r') as f:
            for line in f:
                logs.append(json.loads(line))

        return pd.DataFrame(logs)

    def calculate_drift(self, recent_window=100):
        """Calculate prediction drift"""

        logs_df = self.load_prediction_logs()

        if logs_df.empty or len(logs_df) < recent_window:
            print("⚠️  Not enough data for drift detection")
            return None

        # Get recent predictions
        recent_preds = logs_df.tail(recent_window)['prediction']

        # Calculate statistics
        stats = {
            'mean': recent_preds.mean(),
            'std': recent_preds.std(),
            'min': recent_preds.min(),
            'max': recent_preds.max(),
            'count': len(recent_preds)
        }

        print(f"\n📊 Recent Predictions Statistics (last {recent_window}):")
        for key, value in stats.items():
            print(f"   {key}: {value:.4f}")

        return stats

    def calculate_performance_metrics(self):
        """Calculate model performance on logged predictions with actuals"""

        logs_df = self.load_prediction_logs()

        # Filter for entries with actual values
        with_actuals = logs_df[logs_df['actual'].notna()]

        if with_actuals.empty:
            print("⚠️  No actual values logged for performance calculation")
            return None

        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

        y_true = with_actuals['actual'].values
        y_pred = with_actuals['prediction'].values

        metrics = {
            'mse': mean_squared_error(y_true, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'samples': len(with_actuals)
        }

        print(f"\n📈 Production Performance Metrics:")
        for key, value in metrics.items():
            print(f"   {key}: {value:.4f}")

        return metrics

    def get_model_metrics_history(self):
        """Get historical metrics from all runs"""

        from src.config import EXPERIMENT_NAME
        experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)

        runs = mlflow.search_runs(
            experiment_ids=[experiment.experiment_id],
            order_by=["start_time DESC"]
        )

        if runs.empty:
            print("❌ No runs found")
            return None

        # Select relevant columns
        metrics_cols = [col for col in runs.columns if col.startswith('metrics.test_')]
        display_cols = ['run_id', 'tags.mlflow.runName', 'start_time'] + metrics_cols

        print(f"\n📊 Historical Model Performance:")
        print(runs[display_cols].head(10))

        return runs[display_cols]

if __name__ == "__main__":
    monitor = ModelMonitor()

    # Example: Log some predictions
    print("📝 Logging example predictions...")

    example_features = {
        'fixed acidity': 7.4,
        'volatile acidity': 0.7,
        'citric acid': 0.0,
        'residual sugar': 1.9,
        'chlorides': 0.076,
        'free sulfur dioxide': 11.0,
        'total sulfur dioxide': 34.0,
        'density': 0.9978,
        'pH': 3.51,
        'sulphates': 0.56,
        'alcohol': 9.4
    }

    # Simulate predictions with actual values
    for i in range(10):
        prediction = 5.0 + np.random.randn() * 0.5
        actual = 5 + np.random.choice([-1, 0, 1])
        monitor.log_prediction(example_features, prediction, actual)

    # Calculate metrics
    monitor.calculate_drift()
    monitor.calculate_performance_metrics()
    monitor.get_model_metrics_history()
```

---

## PART 8: Automation Scripts

### run_experiments.sh

```bash
#!/bin/bash

echo "🚀 Running Complete MLOps Pipeline"
echo "=================================="

# Activate virtual environment
source venv/bin/activate

# Step 1: Data Loading
echo -e "\n📥 Step 1: Loading Data..."
python -c "from src.data_loader import DataLoader; loader = DataLoader(); df = loader.load_data(); loader.validate_data(df)"

# Step 2: Data Preprocessing
echo -e "\n🔧 Step 2: Preprocessing Data..."
python src/preprocessor.py

# Step 3: Start MLflow UI in background
echo -e "\n🖥️  Step 3: Starting MLflow UI..."
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &
MLFLOW_PID=$!

sleep 5
echo "✅ MLflow UI running at http://localhost:5000"

# Step 4: Train Models
echo -e "\n🤖 Step 4: Training Models..."
python src/train.py

# Step 5: Register Best Model
echo -e "\n📝 Step 5: Registering Best Model..."
python src/evaluate.py

# Step 6: Start Prediction API
echo -e "\n🚀 Step 6: Starting Prediction API..."
echo "API will be available at http://localhost:5001"
echo "Press Ctrl+C to stop the API and MLflow UI"

# Trap Ctrl+C to clean up
trap "echo 'Stopping services...'; kill $MLFLOW_PID; exit" INT

# Start API (this will run in foreground)
python src/predict.py
```

Make it executable:

```bash
chmod +x run_experiments.sh
```

---

## PART 9: Docker Deployment (Optional)

### Dockerfile

```dockerfile
# docker/Dockerfile
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./src/
COPY data/ ./data/
COPY models/ ./models/
COPY .env .

# Expose ports
EXPOSE 5000 5001

# Run MLflow and API
CMD mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 & \
    python src/predict.py
```

### docker-compose.yml

```yaml
# docker/docker-compose.yml
version: "3.8"

services:
  mlflow:
    build:
      context: ..
      dockerfile: docker/Dockerfile
    ports:
      - "5000:5000" # MLflow UI
      - "5001:5001" # Prediction API
    volumes:
      - ../data:/app/data
      - ../models:/app/models
      - ../mlruns:/app/mlruns
      - ../mlflow.db:/app/mlflow.db
    environment:
      - MLFLOW_TRACKING_URI=http://localhost:5000
    command: >
      sh -c "mlflow server 
             --backend-store-uri sqlite:///mlflow.db 
             --default-artifact-root ./mlruns 
             --host 0.0.0.0 
             --port 5000 &
             sleep 5 &&
             python src/predict.py"
```

To run with Docker:

```bash
cd docker
docker-compose up --build
```

---

## PART 10: Testing

### test_pipeline.py

```python
# tests/test_pipeline.py
import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.data_loader import DataLoader
from src.preprocessor import DataPreprocessor
from src.predict import ModelPredictor

class TestDataPipeline:

    def test_data_loading(self):
        """Test data can be loaded"""
        loader = DataLoader()
        df = loader.load_data()

        assert not df.empty
        assert 'quality' in df.columns
        assert len(df) > 1000

    def test_data_preprocessing(self):
        """Test data preprocessing"""
        loader = DataLoader()
        df = loader.load_data()

        preprocessor = DataPreprocessor()
        X_train, X_test, y_train, y_test = preprocessor.split_data(df)

        assert len(X_train) > len(X_test)
        assert len(X_train) == len(y_train)
        assert len(X_test) == len(y_test)

    def test_prediction(self):
        """Test model prediction"""
        predictor = ModelPredictor()

        # Create sample features
        features = {
            'fixed acidity': 7.4,
            'volatile acidity': 0.7,
            'citric acid': 0.0,
            'residual sugar': 1.9,
            'chlorides': 0.076,
            'free sulfur dioxide': 11.0,
            'total sulfur dioxide': 34.0,
            'density': 0.9978,
            'pH': 3.51,
            'sulphates': 0.56,
            'alcohol': 9.4
        }

        try:
            prediction = predictor.predict(features)
            assert isinstance(prediction, float)
            assert 0 <= prediction <= 10
        except Exception as e:
            pytest.skip(f"Model not available: {e}")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
```

Run tests:

```bash
pytest tests/test_pipeline.py -v
```

---

## 🎯 Complete Execution Guide

### Day 1: Setup and Training (4-5 hours)

```bash
# 1. Clone/setup project (30 min)
./setup.sh
source venv/bin/activate

# 2. Download and preprocess data (15 min)
python src/data_loader.py
python src/preprocessor.py

# 3. Start MLflow UI (keep running)
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000

# In a new terminal:
# 4. Train models (30 min)
python src/train.py

# 5. View results at http://localhost:5000

# 6. Register best model (15 min)
python src/evaluate.py
```

### Day 2: Deployment and Monitoring (3-4 hours)

```bash
# 1. Start prediction API
python src/predict.py

# In a new terminal:
# 2. Test the API
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

# 3. Monitor predictions
python src/monitor.py

# 4. Run tests
pytest tests/ -v
```

---

## 🔍 Understanding Each Component

### MLflow Tracking

- **What**: Logs parameters, metrics, and artifacts
- **Why**: Compare experiments systematically
- **Where**: View at http://localhost:5000

### Model Registry

- **What**: Centralized model store with versioning
- **Why**: Manage model lifecycle (dev → staging → production)
- **How**: Promotes best performing models

### Model Serving

- **What**: REST API for predictions
- **Why**: Make models accessible to applications
- **Endpoint**: http://localhost:5001/predict

### Monitoring

- **What**: Track prediction quality over time
- **Why**: Detect model degradation
- **How**: Log predictions and compare with actuals

---

## 📚 Next Steps to Extend This Project

1. **Add CI/CD**:
   - GitHub Actions for automated testing
   - Automatic model retraining

2. **Improve Monitoring**:
   - Prometheus + Grafana dashboard
   - Data drift detection

3. **Scale Up**:
   - Kubernetes deployment
   - Model serving with TensorFlow Serving

4. **Add Features**:
   - A/B testing between model versions
   - Feature store integration
   - Automated hyperparameter tuning

5. **Documentation**:
   - API documentation with Swagger
   - Model cards for transparency

---

## 🐛 Troubleshooting

### MLflow UI won't start

```bash
# Check if port 5000 is in use
lsof -i :5000

# Use different port
mlflow server --host 0.0.0.0 --port 5050
```

### Model not found in registry

```bash
# List all models
python -c "from src.evaluate import ModelRegistry; ModelRegistry().list_registered_models()"

# Re-register the best model
python src/evaluate.py
```

### API prediction errors

```bash
# Check if scaler exists
ls models/scaler.pkl

# If not, rerun preprocessing
python src/preprocessor.py
```

---

## ✅ Success Criteria

After completing this project, you should be able to:

- ✅ Track experiments with MLflow
- ✅ Compare multiple ML models systematically
- ✅ Register and version models
- ✅ Deploy models as REST APIs
- ✅ Monitor model performance
- ✅ Understand the complete MLOps lifecycle

---

## 📖 Additional Resources

- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [Scikit-learn User Guide](https://scikit-learn.org/stable/user_guide.html)
- [Flask Quickstart](https://flask.palletsprojects.com/quickstart/)
- [Docker Documentation](https://docs.docker.com/)

---

**Happy Learning! 🚀**
