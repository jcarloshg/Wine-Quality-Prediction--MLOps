#!/bin/bash

# Wine Quality Prediction API Test Script
# Usage: ./test-api.sh

API_URL="http://localhost:5001"

echo "🧪 Testing Wine Quality Prediction API"
echo "========================================"

# 1. Health Check
echo -e "\n1️⃣  Health Check"
curl -X GET "${API_URL}/health" \
  -H "Content-Type: application/json" | jq

# 2. Model Info
echo -e "\n2️⃣  Model Info"
curl -X GET "${API_URL}/model_info" \
  -H "Content-Type: application/json" | jq

# 3. Single Prediction
echo -e "\n3️⃣  Single Prediction (High Quality Wine)"
curl -X POST "${API_URL}/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "fixed acidity": 7.4,
    "volatile acidity": 0.70,
    "citric acid": 0.00,
    "residual sugar": 1.9,
    "chlorides": 0.076,
    "free sulfur dioxide": 11.0,
    "total sulfur dioxide": 34.0,
    "density": 0.9978,
    "pH": 3.51,
    "sulphates": 0.56,
    "alcohol": 9.4
  }' | jq

# 4. Batch Prediction
echo -e "\n4️⃣  Batch Prediction (3 samples)"
curl -X POST "${API_URL}/predict_batch" \
  -H "Content-Type: application/json" \
  -d '{
    "samples": [
      {
        "fixed acidity": 7.4,
        "volatile acidity": 0.70,
        "citric acid": 0.00,
        "residual sugar": 1.9,
        "chlorides": 0.076,
        "free sulfur dioxide": 11.0,
        "total sulfur dioxide": 34.0,
        "density": 0.9978,
        "pH": 3.51,
        "sulphates": 0.56,
        "alcohol": 9.4
      },
      {
        "fixed acidity": 8.1,
        "volatile acidity": 0.56,
        "citric acid": 0.28,
        "residual sugar": 1.7,
        "chlorides": 0.368,
        "free sulfur dioxide": 16.0,
        "total sulfur dioxide": 56.0,
        "density": 0.9968,
        "pH": 3.11,
        "sulphates": 1.28,
        "alcohol": 9.3
      },
      {
        "fixed acidity": 11.2,
        "volatile acidity": 0.88,
        "citric acid": 0.40,
        "residual sugar": 2.5,
        "chlorides": 0.095,
        "free sulfur dioxide": 17.0,
        "total sulfur dioxide": 60.0,
        "density": 0.9980,
        "pH": 3.16,
        "sulphates": 0.58,
        "alcohol": 9.8
      }
    ]
  }' | jq

echo -e "\n✅ Tests completed!"
