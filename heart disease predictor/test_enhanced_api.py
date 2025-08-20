# test_enhanced_api.py
import requests
import json

def test_enhanced_api():
    """Test the enhanced heart disease prediction API"""
    
    # API endpoint
    url = "http://127.0.0.1:5000/api/predict"
    
    # Test data with all 13 features
    test_cases = [
        {
            "name": "Low Risk Patient",
            "data": {
                "age": 35,
                "sex": 0,  # Female
                "cp": 3,   # Asymptomatic
                "trestbps": 110,
                "chol": 180,
                "fbs": 0,  # Normal
                "restecg": 0,  # Normal
                "thalach": 180,
                "exang": 0,  # No exercise angina
                "oldpeak": 0.5,
                "slope": 0,  # Upsloping
                "ca": 0,   # No vessels
                "thal": 0  # Normal
            }
        },
        {
            "name": "High Risk Patient",
            "data": {
                "age": 65,
                "sex": 1,  # Male
                "cp": 0,   # Typical angina
                "trestbps": 160,
                "chol": 280,
                "fbs": 1,  # Elevated
                "restecg": 1,  # ST-T abnormality
                "thalach": 120,
                "exang": 1,  # Exercise angina
                "oldpeak": 2.5,
                "slope": 2,  # Downsloping
                "ca": 2,   # Two vessels
                "thal": 2  # Reversible defect
            }
        }
    ]
    
    print("Testing Enhanced Heart Disease Prediction API")
    print("=" * 50)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest Case {i}: {test_case['name']}")
        print("-" * 30)
        
        try:
            response = requests.post(url, json=test_case['data'], timeout=10)
            
            if response.status_code == 200:
                result = response.json()
                print(f"✅ Prediction: {result['result']}")
                print(f"   Probability: {result['probability']:.1%}")
                print(f"   Risk Level: {result['risk_info']['level']}")
                print(f"   Model Type: {result['model_info']['type']}")
                print("   Top Risk Factors:")
                for factor in result['feature_importance'][:3]:
                    print(f"     - {factor['display_name']}: {factor['value']}")
            else:
                print(f"❌ Error {response.status_code}: {response.text}")
                
        except requests.exceptions.RequestException as e:
            print(f"❌ Connection error: {e}")
    
    # Test health endpoint
    print(f"\n{'='*50}")
    print("Testing Health Endpoint")
    print("-" * 30)
    
    try:
        health_response = requests.get("http://127.0.0.1:5000/health", timeout=5)
        if health_response.status_code == 200:
            health_data = health_response.json()
            print(f"✅ Service Status: {health_data['status']}")
            print(f"   Model Loaded: {health_data['model_loaded']}")
            print(f"   Model Type: {health_data.get('model_type', 'N/A')}")
            print(f"   Test Accuracy: {health_data.get('model_info', {}).get('test_accuracy', 'N/A')}")
        else:
            print(f"❌ Health check failed: {health_response.status_code}")
    except requests.exceptions.RequestException as e:
        print(f"❌ Health check error: {e}")

if __name__ == "__main__":
    test_enhanced_api()
