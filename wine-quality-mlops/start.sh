#!/bin/bash
set -e

echo "🚀 Starting MLflow server..."
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns \
    --host 0.0.0.0 \
    --port 5000 &

MLFLOW_PID=$!
echo "MLflow started with PID: $MLFLOW_PID"

echo "⏳ Waiting for MLflow to be ready..."
max_attempts=30
attempt=0

while [ $attempt -lt $max_attempts ]; do
    if curl -s http://localhost:5000/health > /dev/null 2>&1; then
        echo "✅ MLflow is ready!"
        break
    fi

    attempt=$((attempt + 1))
    echo "   Attempt $attempt/$max_attempts - MLflow not ready yet..."
    sleep 2
done

if [ $attempt -eq $max_attempts ]; then
    echo "❌ MLflow failed to start within 60 seconds"
    exit 1
fi

# # Step 4: Train Models
# echo -e "\n🤖 Step 4: Training Models..."
# python3.12 -m src.train

# # Step 5: Register Best Model
# echo -e "\n📝 Step 5: Registering Best Model..."
# python3.12 -m src.evaluate

echo ""
echo "🚀 Starting Prediction API..."
python3.12 -m src.predict

# Keep the script running
wait $MLFLOW_PID
