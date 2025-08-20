# comprehensive_api_test.py
"""
Comprehensive API Test Suite
Tests all Flask endpoints and functionality
"""

import requests
import json
import time
import threading
import subprocess
import sys
import os
from pathlib import Path

class FlaskTestSuite:
    def __init__(self, base_url="http://127.0.0.1:5000"):
        self.base_url = base_url
        self.server_process = None
        self.tests_passed = 0
        self.tests_total = 0
    
    def start_flask_server(self):
        """Start Flask server in background"""
        try:
            python_path = "C:/Users/Rishant/Downloads/heart_Disease_Prediction-main/heart_Disease_Prediction-main/.venv/Scripts/python.exe"
            cmd = [python_path, "enhanced_app.py"]
            
            self.server_process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                cwd="c:/Users/Rishant/Downloads/heart_Disease_Prediction-main/heart_Disease_Prediction-main/heart disease predictor"
            )
            
            # Wait for server to start
            print("🔄 Starting Flask server...")
            time.sleep(3)
            
            # Test if server is responding
            for i in range(10):
                try:
                    response = requests.get(f"{self.base_url}/health", timeout=2)
                    if response.status_code == 200:
                        print("✅ Flask server started successfully")
                        return True
                except:
                    time.sleep(1)
            
            print("❌ Flask server failed to start")
            return False
            
        except Exception as e:
            print(f"❌ Error starting server: {e}")
            return False
    
    def stop_flask_server(self):
        """Stop Flask server"""
        if self.server_process:
            self.server_process.terminate()
            self.server_process.wait()
            print("🛑 Flask server stopped")
    
    def test_health_endpoint(self):
        """Test /health endpoint"""
        print("\n📊 Testing Health Endpoint")
        self.tests_total += 1
        
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                if data.get('status') == 'healthy':
                    print("✅ Health endpoint working")
                    self.tests_passed += 1
                else:
                    print(f"❌ Health endpoint unhealthy: {data}")
            else:
                print(f"❌ Health endpoint failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Health endpoint error: {e}")
    
    def test_model_info_endpoint(self):
        """Test /api/model-info endpoint"""
        print("\n📊 Testing Model Info Endpoint")
        self.tests_total += 1
        
        try:
            response = requests.get(f"{self.base_url}/api/model-info", timeout=5)
            
            if response.status_code == 200:
                data = response.json()
                required_fields = ['model_type', 'features', 'accuracy']
                
                if all(field in data for field in required_fields):
                    print(f"✅ Model info endpoint working")
                    print(f"   Model: {data.get('model_type')}")
                    print(f"   Accuracy: {data.get('accuracy', 'N/A')}")
                    self.tests_passed += 1
                else:
                    print(f"❌ Model info missing fields: {data}")
            else:
                print(f"❌ Model info failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Model info error: {e}")
    
    def test_prediction_endpoint(self):
        """Test /api/predict endpoint"""
        print("\n📊 Testing Prediction Endpoint")
        
        # Test 1: Valid prediction
        self.tests_total += 1
        try:
            test_data = {
                'age': 50, 'sex': 1, 'cp': 1, 'trestbps': 140, 'chol': 240,
                'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
                'oldpeak': 1.2, 'slope': 1, 'ca': 1, 'thal': 1
            }
            
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=test_data,
                timeout=5
            )
            
            if response.status_code == 200:
                data = response.json()
                if 'prediction' in data and 'probability' in data:
                    print(f"✅ Valid prediction: {data['prediction']} ({data['probability']:.1%} risk)")
                    self.tests_passed += 1
                else:
                    print(f"❌ Invalid prediction response: {data}")
            else:
                print(f"❌ Prediction failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Prediction error: {e}")
        
        # Test 2: Invalid input (missing fields)
        self.tests_total += 1
        try:
            invalid_data = {'age': 50, 'sex': 1}  # Missing fields
            
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=invalid_data,
                timeout=5
            )
            
            if response.status_code == 400:
                print("✅ Invalid input properly rejected")
                self.tests_passed += 1
            else:
                print(f"❌ Invalid input not rejected: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Invalid input test error: {e}")
        
        # Test 3: Invalid data types
        self.tests_total += 1
        try:
            invalid_data = {
                'age': 'fifty', 'sex': 1, 'cp': 1, 'trestbps': 140, 'chol': 240,
                'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
                'oldpeak': 1.2, 'slope': 1, 'ca': 1, 'thal': 1
            }
            
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=invalid_data,
                timeout=5
            )
            
            if response.status_code == 400:
                print("✅ Invalid data types properly rejected")
                self.tests_passed += 1
            else:
                print(f"❌ Invalid data types not rejected: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Invalid data types test error: {e}")
    
    def test_web_interface(self):
        """Test web interface endpoints"""
        print("\n📊 Testing Web Interface")
        
        # Test 1: Home page
        self.tests_total += 1
        try:
            response = requests.get(f"{self.base_url}/", timeout=5)
            
            if response.status_code == 200 and "Heart Disease Prediction" in response.text:
                print("✅ Home page loads correctly")
                self.tests_passed += 1
            else:
                print(f"❌ Home page failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Home page error: {e}")
        
        # Test 2: Form submission
        self.tests_total += 1
        try:
            form_data = {
                'age': '50', 'sex': '1', 'cp': '1', 'trestbps': '140', 'chol': '240',
                'fbs': '0', 'restecg': '0', 'thalach': '150', 'exang': '0',
                'oldpeak': '1.2', 'slope': '1', 'ca': '1', 'thal': '1'
            }
            
            response = requests.post(f"{self.base_url}/predict", data=form_data, timeout=5)
            
            if response.status_code == 200 and ("prediction" in response.text.lower() or "result" in response.text.lower()):
                print("✅ Form submission works")
                self.tests_passed += 1
            else:
                print(f"❌ Form submission failed: {response.status_code}")
                
        except Exception as e:
            print(f"❌ Form submission error: {e}")
    
    def run_all_tests(self):
        """Run all tests"""
        print("🚀 Running Comprehensive API Test Suite")
        print("=" * 60)
        
        # Start server
        if not self.start_flask_server():
            print("❌ Cannot start Flask server - aborting tests")
            return False
        
        try:
            # Run all tests
            self.test_health_endpoint()
            self.test_model_info_endpoint()
            self.test_prediction_endpoint()
            self.test_web_interface()
            
            # Results
            print("\n" + "=" * 60)
            print(f"API TEST RESULTS: {self.tests_passed}/{self.tests_total} tests passed")
            print(f"Success Rate: {(self.tests_passed/self.tests_total*100):.1f}%")
            
            if self.tests_passed == self.tests_total:
                print("🎉 All API tests passed!")
                return True
            else:
                print("⚠️  Some API tests failed")
                return False
                
        finally:
            # Always stop server
            self.stop_flask_server()

if __name__ == "__main__":
    # Change to correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    test_suite = FlaskTestSuite()
    success = test_suite.run_all_tests()
    exit(0 if success else 1)
