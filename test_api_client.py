#!/usr/bin/env python3
"""
Test client for Wine Quality Prediction API
Tests all endpoints and demonstrates usage
"""

import requests
import json
import time
from typing import Dict, List

class WineQualityClient:
    """Client for interacting with Wine Quality Prediction API"""
    
    def __init__(self, base_url: str = "http://localhost:5001"):
        self.base_url = base_url
        self.session = requests.Session()
    
    def health_check(self) -> Dict:
        """Check API health status"""
        url = f"{self.base_url}/health"
        response = self.session.get(url)
        return response.json()
    
    def predict_single(self, features: Dict) -> Dict:
        """Make a single prediction"""
        url = f"{self.base_url}/predict"
        response = self.session.post(url, json=features)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Prediction failed: {response.text}")
    
    def predict_batch(self, samples: List[Dict]) -> Dict:
        """Make batch predictions"""
        url = f"{self.base_url}/predict_batch"
        data = {"samples": samples}
        response = self.session.post(url, json=data)
        
        if response.status_code == 200:
            return response.json()
        else:
            raise Exception(f"Batch prediction failed: {response.text}")
    
    def get_model_info(self) -> Dict:
        """Get model information"""
        url = f"{self.base_url}/model_info"
        response = self.session.get(url)
        return response.json()


def create_sample_features() -> Dict:
    """Create sample wine features"""
    return {
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


def create_batch_samples() -> List[Dict]:
    """Create multiple sample wines"""
    samples = [
        # Good quality wine
        {
            'fixed acidity': 7.8,
            'volatile acidity': 0.58,
            'citric acid': 0.02,
            'residual sugar': 2.0,
            'chlorides': 0.073,
            'free sulfur dioxide': 9.0,
            'total sulfur dioxide': 18.0,
            'density': 0.9968,
            'pH': 3.36,
            'sulphates': 0.57,
            'alcohol': 9.5
        },
        # Average quality wine
        {
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
        },
        # Poor quality wine
        {
            'fixed acidity': 11.2,
            'volatile acidity': 0.28,
            'citric acid': 0.56,
            'residual sugar': 1.9,
            'chlorides': 0.075,
            'free sulfur dioxide': 17.0,
            'total sulfur dioxide': 60.0,
            'density': 0.998,
            'pH': 3.16,
            'sulphates': 0.58,
            'alcohol': 9.8
        }
    ]
    return samples


def test_api():
    """Run comprehensive API tests"""
    
    print("🧪 Wine Quality API Test Suite")
    print("=" * 60)
    
    client = WineQualityClient()
    
    # Test 1: Health Check
    print("\n📋 Test 1: Health Check")
    print("-" * 60)
    try:
        health = client.health_check()
        print(f"✅ Status: {health['status']}")
        print(f"✅ Model Loaded: {health['model_loaded']}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
        return
    
    # Test 2: Model Info
    print("\n📋 Test 2: Model Information")
    print("-" * 60)
    try:
        info = client.get_model_info()
        print(f"✅ Model Name: {info['model_name']}")
        print(f"✅ Model Type: {info['model_type']}")
    except Exception as e:
        print(f"❌ Model info failed: {e}")
    
    # Test 3: Single Prediction
    print("\n📋 Test 3: Single Prediction")
    print("-" * 60)
    try:
        features = create_sample_features()
        print("Input features:")
        for key, value in list(features.items())[:3]:
            print(f"  {key}: {value}")
        print("  ...")
        
        result = client.predict_single(features)
        print(f"\n✅ Prediction: {result['prediction']:.2f}")
        print(f"✅ Quality Score: {result['quality_score']}")
    except Exception as e:
        print(f"❌ Single prediction failed: {e}")
    
    # Test 4: Batch Prediction
    print("\n📋 Test 4: Batch Prediction")
    print("-" * 60)
    try:
        samples = create_batch_samples()
        print(f"Testing {len(samples)} samples...")
        
        result = client.predict_batch(samples)
        print(f"\n✅ Predictions:")
        for i, pred in enumerate(result['predictions']):
            print(f"  Wine {i+1}: {pred:.2f}")
        print(f"\n✅ Total predictions: {result['count']}")
    except Exception as e:
        print(f"❌ Batch prediction failed: {e}")
    
    # Test 5: Performance Test
    print("\n📋 Test 5: Performance Test")
    print("-" * 60)
    try:
        features = create_sample_features()
        num_requests = 10
        
        start_time = time.time()
        for _ in range(num_requests):
            client.predict_single(features)
        end_time = time.time()
        
        total_time = end_time - start_time
        avg_time = total_time / num_requests
        
        print(f"✅ Total requests: {num_requests}")
        print(f"✅ Total time: {total_time:.2f}s")
        print(f"✅ Average latency: {avg_time*1000:.2f}ms")
        print(f"✅ Requests/second: {num_requests/total_time:.2f}")
    except Exception as e:
        print(f"❌ Performance test failed: {e}")
    
    # Test 6: Error Handling
    print("\n📋 Test 6: Error Handling")
    print("-" * 60)
    try:
        # Test with missing features
        invalid_features = {'fixed acidity': 7.4}
        result = client.predict_single(invalid_features)
        print("❌ Should have raised an error for missing features")
    except Exception as e:
        print(f"✅ Correctly handled invalid input: {str(e)[:50]}...")
    
    print("\n" + "=" * 60)
    print("🎉 API Testing Complete!")
    print("=" * 60)


def interactive_mode():
    """Interactive mode for manual testing"""
    
    print("\n🎮 Interactive Wine Quality Predictor")
    print("=" * 60)
    
    client = WineQualityClient()
    
    # Check if API is running
    try:
        health = client.health_check()
        if health['status'] != 'healthy':
            print("❌ API is not healthy. Please start the API first.")
            return
    except Exception as e:
        print(f"❌ Cannot connect to API: {e}")
        print("   Make sure the API is running at http://localhost:5001")
        return
    
    while True:
        print("\n" + "-" * 60)
        print("Options:")
        print("  1. Predict with default sample")
        print("  2. Predict with custom values")
        print("  3. Batch prediction with 3 samples")
        print("  4. View model info")
        print("  5. Exit")
        
        choice = input("\nEnter choice (1-5): ").strip()
        
        if choice == '1':
            features = create_sample_features()
            result = client.predict_single(features)
            print(f"\n🍷 Predicted Quality: {result['quality_score']}")
        
        elif choice == '2':
            print("\nEnter wine features (or press Enter for default):")
            features = {}
            feature_names = [
                'fixed acidity', 'volatile acidity', 'citric acid',
                'residual sugar', 'chlorides', 'free sulfur dioxide',
                'total sulfur dioxide', 'density', 'pH', 'sulphates', 'alcohol'
            ]
            
            defaults = create_sample_features()
            
            for name in feature_names:
                value = input(f"  {name} [{defaults[name]}]: ").strip()
                features[name] = float(value) if value else defaults[name]
            
            result = client.predict_single(features)
            print(f"\n🍷 Predicted Quality: {result['quality_score']}")
        
        elif choice == '3':
            samples = create_batch_samples()
            result = client.predict_batch(samples)
            print("\n🍷 Predictions:")
            for i, pred in enumerate(result['predictions']):
                quality = "Good" if pred >= 6 else "Average" if pred >= 5 else "Poor"
                print(f"  Wine {i+1}: {pred:.2f} ({quality})")
        
        elif choice == '4':
            info = client.get_model_info()
            print(f"\n📊 Model Information:")
            print(f"  Name: {info['model_name']}")
            print(f"  Type: {info['model_type']}")
        
        elif choice == '5':
            print("\n👋 Goodbye!")
            break
        
        else:
            print("❌ Invalid choice. Please enter 1-5.")


def main():
    """Main entry point"""
    
    import sys
    
    if len(sys.argv) > 1 and sys.argv[1] == '--interactive':
        interactive_mode()
    else:
        test_api()


if __name__ == "__main__":
    print("""
    ╔═══════════════════════════════════════════════════════════╗
    ║                                                           ║
    ║        Wine Quality Prediction API Test Client           ║
    ║                                                           ║
    ║  Usage:                                                   ║
    ║    python test_api_client.py              # Run tests    ║
    ║    python test_api_client.py --interactive  # Interactive║
    ║                                                           ║
    ╚═══════════════════════════════════════════════════════════╝
    """)
    
    main()
