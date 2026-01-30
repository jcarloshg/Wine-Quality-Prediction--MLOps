# src/predict.py
from src.config import (
    MLFLOW_TRACKING_URI,
    REGISTERED_MODEL_NAME,
    MODELS_DIR,
    API_HOST,
    API_PORT
)
import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from flask import Flask, request, jsonify
import joblib

print(["="]*60)
print(f"🚀 Initializing Prediction Module...")
print(["="]*60)


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
