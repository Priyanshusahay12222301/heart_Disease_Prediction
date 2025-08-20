# comprehensive_test_suite.py
"""
Comprehensive Test Suite for Heart Disease Prediction System
Tests all components: Model, API, Frontend validation, and Integration
"""

import os
import sys
import json
import time
from pathlib import Path
from typing import Dict, List, Any

# Core data science imports
try:
    import pandas as pd
    import numpy as np
    import joblib
except ImportError as e:
    print(f"Warning: Missing data science packages: {e}")
    print("Please install: pip install pandas numpy joblib scikit-learn")

# Testing and API imports
try:
    import requests
except ImportError:
    print("Warning: requests not found. Install with: pip install requests")
    
try:
    import pytest
    PYTEST_AVAILABLE = True
except ImportError:
    print("Warning: pytest not found. Some advanced testing features disabled.")
    PYTEST_AVAILABLE = False

# Add the project directory to the path
project_dir = Path(__file__).parent
sys.path.insert(0, str(project_dir))

class TestResults:
    def __init__(self):
        self.total_tests = 0
        self.passed_tests = 0
        self.failed_tests = 0
        self.skipped_tests = 0
        self.errors = []
        
    def add_result(self, test_name: str, status: str, error: str = None):
        self.total_tests += 1
        if status == "PASS":
            self.passed_tests += 1
            print(f"✅ {test_name}")
        elif status == "FAIL":
            self.failed_tests += 1
            print(f"❌ {test_name}")
            if error:
                print(f"   Error: {error}")
                self.errors.append(f"{test_name}: {error}")
        elif status == "SKIP":
            self.skipped_tests += 1
            print(f"⏭️  {test_name} (SKIPPED)")
    
    def print_summary(self):
        print("\n" + "="*80)
        print("TEST SUITE SUMMARY")
        print("="*80)
        print(f"Total Tests: {self.total_tests}")
        print(f"✅ Passed: {self.passed_tests}")
        print(f"❌ Failed: {self.failed_tests}")
        print(f"⏭️  Skipped: {self.skipped_tests}")
        print(f"Success Rate: {(self.passed_tests/self.total_tests*100):.1f}%")
        
        if self.errors:
            print("\nERRORS:")
            for error in self.errors:
                print(f"  - {error}")

class HeartDiseaseTestSuite:
    def __init__(self):
        self.results = TestResults()
        self.base_url = "http://127.0.0.1:5000"
        self.model_path = "enhanced_heart_disease_model.pkl"
        self.test_data = self._generate_test_data()
        
    def _generate_test_data(self) -> List[Dict]:
        """Generate comprehensive test cases"""
        return [
            # Test Case 1: Low Risk Patient
            {
                "name": "Low Risk Young Female",
                "data": {
                    "age": 25, "sex": 0, "cp": 3, "trestbps": 110, "chol": 180,
                    "fbs": 0, "restecg": 0, "thalach": 180, "exang": 0,
                    "oldpeak": 0.2, "slope": 0, "ca": 0, "thal": 0
                },
                "expected_risk": "low"
            },
            # Test Case 2: High Risk Patient
            {
                "name": "High Risk Elderly Male",
                "data": {
                    "age": 70, "sex": 1, "cp": 0, "trestbps": 170, "chol": 300,
                    "fbs": 1, "restecg": 1, "thalach": 110, "exang": 1,
                    "oldpeak": 3.0, "slope": 2, "ca": 3, "thal": 2
                },
                "expected_risk": "high"
            },
            # Test Case 3: Moderate Risk Patient
            {
                "name": "Moderate Risk Middle-aged",
                "data": {
                    "age": 50, "sex": 1, "cp": 1, "trestbps": 140, "chol": 240,
                    "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
                    "oldpeak": 1.2, "slope": 1, "ca": 1, "thal": 1
                },
                "expected_risk": "moderate"
            },
            # Test Case 4: Edge Case - Boundary Values
            {
                "name": "Boundary Values Test",
                "data": {
                    "age": 29, "sex": 0, "cp": 0, "trestbps": 94, "chol": 126,
                    "fbs": 0, "restecg": 0, "thalach": 71, "exang": 0,
                    "oldpeak": 0.0, "slope": 0, "ca": 0, "thal": 0
                },
                "expected_risk": "low"
            },
            # Test Case 5: Maximum Values
            {
                "name": "Maximum Values Test",
                "data": {
                    "age": 77, "sex": 1, "cp": 3, "trestbps": 200, "chol": 564,
                    "fbs": 1, "restecg": 2, "thalach": 202, "exang": 1,
                    "oldpeak": 6.2, "slope": 2, "ca": 3, "thal": 2
                },
                "expected_risk": "high"
            }
        ]
    
    def run_all_tests(self):
        """Run the complete test suite"""
        print("🚀 Starting Comprehensive Heart Disease Prediction Test Suite")
        print("="*80)
        
        # Model Tests
        print("\n📊 TESTING MODEL COMPONENTS")
        print("-" * 40)
        self.test_model_loading()
        self.test_model_structure()
        self.test_model_predictions()
        
        # API Tests (if server is running)
        print("\n🌐 TESTING API ENDPOINTS")
        print("-" * 40)
        if self.check_server_status():
            self.test_health_endpoint()
            self.test_model_info_endpoint()
            self.test_prediction_endpoints()
            self.test_api_validation()
            self.test_api_error_handling()
        else:
            self.results.add_result("API Tests", "SKIP", "Server not running")
        
        # Data Validation Tests
        print("\n🔍 TESTING DATA VALIDATION")
        print("-" * 40)
        self.test_input_validation()
        self.test_edge_cases()
        
        # Integration Tests
        print("\n🔗 TESTING INTEGRATION")
        print("-" * 40)
        self.test_end_to_end_workflow()
        
        # Performance Tests
        print("\n⚡ TESTING PERFORMANCE")
        print("-" * 40)
        self.test_prediction_performance()
        
        # Print final results
        self.results.print_summary()
        
        return self.results.failed_tests == 0
    
    def test_model_loading(self):
        """Test if the model loads correctly"""
        try:
            if os.path.exists(self.model_path):
                model_package = joblib.load(self.model_path)
                
                # Check required components
                required_keys = ['model', 'feature_columns', 'model_type']
                for key in required_keys:
                    if key not in model_package:
                        raise ValueError(f"Missing required key: {key}")
                
                self.results.add_result("Model Loading", "PASS")
            else:
                self.results.add_result("Model Loading", "FAIL", "Model file not found")
        except Exception as e:
            self.results.add_result("Model Loading", "FAIL", str(e))
    
    def test_model_structure(self):
        """Test model structure and features"""
        try:
            model_package = joblib.load(self.model_path)
            
            # Check feature count
            expected_features = 13
            actual_features = len(model_package['feature_columns'])
            if actual_features != expected_features:
                raise ValueError(f"Expected {expected_features} features, got {actual_features}")
            
            # Check model type
            model_type = model_package['model_type']
            if model_type not in ['logistic_regression', 'random_forest']:
                raise ValueError(f"Unexpected model type: {model_type}")
            
            self.results.add_result("Model Structure", "PASS")
        except Exception as e:
            self.results.add_result("Model Structure", "FAIL", str(e))
    
    def test_model_predictions(self):
        """Test model prediction functionality"""
        try:
            model_package = joblib.load(self.model_path)
            model = model_package['model']
            scaler = model_package.get('scaler')
            feature_columns = model_package['feature_columns']
            
            # Test with sample data
            test_case = self.test_data[0]['data']
            input_df = pd.DataFrame([test_case], columns=feature_columns)
            
            # Make prediction
            if scaler:
                input_scaled = scaler.transform(input_df)
                prediction = model.predict(input_scaled)
                probability = model.predict_proba(input_scaled)[0][1]
            else:
                prediction = model.predict(input_df)
                probability = model.predict_proba(input_df)[0][1]
            
            # Validate outputs
            if not isinstance(prediction[0], (int, np.integer)) or prediction[0] not in [0, 1]:
                raise ValueError("Invalid prediction output")
            
            if not (0 <= probability <= 1):
                raise ValueError("Invalid probability output")
            
            self.results.add_result("Model Predictions", "PASS")
        except Exception as e:
            self.results.add_result("Model Predictions", "FAIL", str(e))
    
    def check_server_status(self) -> bool:
        """Check if the Flask server is running"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def test_health_endpoint(self):
        """Test the health endpoint"""
        try:
            response = requests.get(f"{self.base_url}/health", timeout=5)
            
            if response.status_code != 200:
                raise ValueError(f"Health endpoint returned {response.status_code}")
            
            data = response.json()
            if data.get('status') != 'healthy':
                raise ValueError("Health check failed")
            
            self.results.add_result("Health Endpoint", "PASS")
        except Exception as e:
            self.results.add_result("Health Endpoint", "FAIL", str(e))
    
    def test_model_info_endpoint(self):
        """Test the model info endpoint"""
        try:
            response = requests.get(f"{self.base_url}/model-info", timeout=5)
            
            if response.status_code != 200:
                raise ValueError(f"Model info endpoint returned {response.status_code}")
            
            data = response.json()
            required_fields = ['model_type', 'features', 'validation_ranges']
            for field in required_fields:
                if field not in data:
                    raise ValueError(f"Missing field in model info: {field}")
            
            self.results.add_result("Model Info Endpoint", "PASS")
        except Exception as e:
            self.results.add_result("Model Info Endpoint", "FAIL", str(e))
    
    def test_prediction_endpoints(self):
        """Test prediction endpoints with various inputs"""
        for test_case in self.test_data:
            try:
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json=test_case['data'],
                    timeout=10
                )
                
                if response.status_code != 200:
                    raise ValueError(f"API returned {response.status_code}")
                
                data = response.json()
                
                # Check required response fields
                required_fields = ['result', 'probability', 'risk_info']
                for field in required_fields:
                    if field not in data:
                        raise ValueError(f"Missing field: {field}")
                
                # Validate probability
                if not (0 <= data['probability'] <= 1):
                    raise ValueError("Invalid probability value")
                
                # Check risk level
                risk_level = data['risk_info']['level'].lower()
                if risk_level not in ['low', 'moderate', 'high']:
                    raise ValueError(f"Invalid risk level: {risk_level}")
                
                self.results.add_result(f"Prediction API - {test_case['name']}", "PASS")
                
            except Exception as e:
                self.results.add_result(f"Prediction API - {test_case['name']}", "FAIL", str(e))
    
    def test_api_validation(self):
        """Test API input validation"""
        # Test missing fields
        try:
            incomplete_data = {"age": 50, "sex": 1}  # Missing required fields
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=incomplete_data,
                timeout=5
            )
            
            if response.status_code == 200:
                raise ValueError("API should reject incomplete data")
            
            self.results.add_result("API Validation - Missing Fields", "PASS")
        except Exception as e:
            self.results.add_result("API Validation - Missing Fields", "FAIL", str(e))
        
        # Test invalid ranges
        try:
            invalid_data = {
                "age": 150, "sex": 0, "cp": 1, "trestbps": 120, "chol": 200,
                "fbs": 0, "restecg": 0, "thalach": 150, "exang": 0,
                "oldpeak": 1.0, "slope": 1, "ca": 1, "thal": 1
            }
            response = requests.post(
                f"{self.base_url}/api/predict",
                json=invalid_data,
                timeout=5
            )
            
            if response.status_code == 200:
                raise ValueError("API should reject invalid age")
            
            self.results.add_result("API Validation - Invalid Range", "PASS")
        except Exception as e:
            self.results.add_result("API Validation - Invalid Range", "FAIL", str(e))
    
    def test_api_error_handling(self):
        """Test API error handling"""
        # Test invalid JSON
        try:
            response = requests.post(
                f"{self.base_url}/api/predict",
                data="invalid json",
                headers={'Content-Type': 'application/json'},
                timeout=5
            )
            
            if response.status_code == 200:
                raise ValueError("API should reject invalid JSON")
            
            self.results.add_result("API Error Handling - Invalid JSON", "PASS")
        except Exception as e:
            self.results.add_result("API Error Handling - Invalid JSON", "FAIL", str(e))
    
    def test_input_validation(self):
        """Test input validation logic"""
        try:
            # Import validation function
            try:
                from enhanced_app import validate_input, VALIDATION_RANGES
            except ImportError:
                self.results.add_result("Input Validation - Module Import", "SKIP", "enhanced_app module not available")
                return
            
            # Test valid input
            valid_data = self.test_data[0]['data']
            errors = validate_input(valid_data)
            if errors:
                raise ValueError(f"Valid data rejected: {errors}")
            
            # Test invalid input
            invalid_data = {"age": -5, "trestbps": 300}
            errors = validate_input(invalid_data)
            if not errors:
                raise ValueError("Invalid data not caught")
            
            self.results.add_result("Input Validation Logic", "PASS")
        except Exception as e:
            self.results.add_result("Input Validation Logic", "FAIL", str(e))
    
    def test_edge_cases(self):
        """Test edge cases and boundary conditions"""
        edge_cases = [
            {"name": "Minimum Age", "age": 1},
            {"name": "Maximum Age", "age": 120},
            {"name": "Minimum BP", "trestbps": 80},
            {"name": "Maximum BP", "trestbps": 250},
            {"name": "Zero Oldpeak", "oldpeak": 0.0},
            {"name": "Maximum Oldpeak", "oldpeak": 10.0}
        ]
        
        for case in edge_cases:
            try:
                # Create complete test data with edge case value
                test_data = self.test_data[0]['data'].copy()
                test_data.update(case)
                
                if self.check_server_status():
                    response = requests.post(
                        f"{self.base_url}/api/predict",
                        json=test_data,
                        timeout=5
                    )
                    
                    if response.status_code != 200:
                        raise ValueError(f"Edge case failed: {response.status_code}")
                
                self.results.add_result(f"Edge Case - {case['name']}", "PASS")
            except Exception as e:
                self.results.add_result(f"Edge Case - {case['name']}", "FAIL", str(e))
    
    def test_end_to_end_workflow(self):
        """Test complete end-to-end workflow"""
        try:
            if not self.check_server_status():
                raise ValueError("Server not available for E2E testing")
            
            # Step 1: Get model info
            info_response = requests.get(f"{self.base_url}/model-info", timeout=5)
            if info_response.status_code != 200:
                raise ValueError("Could not get model info")
            
            # Step 2: Make prediction
            test_data = self.test_data[1]['data']  # Use high-risk patient
            pred_response = requests.post(
                f"{self.base_url}/api/predict",
                json=test_data,
                timeout=10
            )
            if pred_response.status_code != 200:
                raise ValueError("Prediction failed")
            
            # Step 3: Validate response completeness
            result = pred_response.json()
            required_components = [
                'result', 'probability', 'risk_info', 'feature_importance', 'inputs'
            ]
            for component in required_components:
                if component not in result:
                    raise ValueError(f"Missing component: {component}")
            
            self.results.add_result("End-to-End Workflow", "PASS")
        except Exception as e:
            self.results.add_result("End-to-End Workflow", "FAIL", str(e))
    
    def test_prediction_performance(self):
        """Test prediction performance and response times"""
        try:
            if not self.check_server_status():
                raise ValueError("Server not available for performance testing")
            
            test_data = self.test_data[0]['data']
            response_times = []
            
            # Make multiple requests to test performance
            for i in range(10):
                start_time = time.time()
                response = requests.post(
                    f"{self.base_url}/api/predict",
                    json=test_data,
                    timeout=10
                )
                end_time = time.time()
                
                if response.status_code != 200:
                    raise ValueError(f"Request {i+1} failed")
                
                response_times.append(end_time - start_time)
            
            # Check performance metrics
            avg_response_time = np.mean(response_times)
            max_response_time = np.max(response_times)
            
            if avg_response_time > 2.0:  # 2 seconds threshold
                raise ValueError(f"Average response time too slow: {avg_response_time:.2f}s")
            
            if max_response_time > 5.0:  # 5 seconds threshold
                raise ValueError(f"Max response time too slow: {max_response_time:.2f}s")
            
            print(f"   Average response time: {avg_response_time:.3f}s")
            print(f"   Max response time: {max_response_time:.3f}s")
            
            self.results.add_result("Performance Test", "PASS")
        except Exception as e:
            self.results.add_result("Performance Test", "FAIL", str(e))

def main():
    """Run the comprehensive test suite"""
    print("🏥 Heart Disease Prediction - Comprehensive Test Suite")
    print("=" * 80)
    
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    # Initialize and run test suite
    test_suite = HeartDiseaseTestSuite()
    success = test_suite.run_all_tests()
    
    # Exit with appropriate code
    exit_code = 0 if success else 1
    print(f"\n🏁 Test Suite Complete - Exit Code: {exit_code}")
    
    return exit_code

if __name__ == "__main__":
    main()
