#!/bin/bash

echo "🚀 Running Complete MLOps Pipeline"
echo "=================================="

# Activate virtual environment
source venv/bin/activate

# Step 1: Data Loading
echo -e "\n📥 Step 1: Loading Data..."
python3.12 -c "from src.data_loader import DataLoader; loader = DataLoader(); df = loader.load_data(); loader.validate_data(df)"

# Step 2: Data Preprocessing
echo -e "\n🔧 Step 2: Preprocessing Data..."
python3.12 -m src.preprocessor

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
python3.12 -m src.train

# Step 5: Register Best Model
echo -e "\n📝 Step 5: Registering Best Model..."
python3.12 -m src.evaluate

# Step 6: Start Prediction API
echo -e "\n🚀 Step 6: Starting Prediction API..."
echo "API will be available at http://localhost:5001"
echo "Press Ctrl+C to stop the API and MLflow UI"

# Trap Ctrl+C to clean up
trap "echo 'Stopping services...'; kill $MLFLOW_PID; exit" INT

# Start API (this will run in foreground)
python3.12 -m src.predict