# quick_test_suite.py
"""
Quick Test Suite - Tests that can run without Flask server
"""

import os
import sys
import joblib
import pandas as pd
import numpy as np
from pathlib import Path

def run_quick_tests():
    """Run basic tests that don't require the server"""
    print("🚀 Running Quick Test Suite (No Server Required)")
    print("=" * 60)
    
    tests_passed = 0
    tests_total = 0
    
    # Test 1: Model File Exists
    tests_total += 1
    if os.path.exists("enhanced_heart_disease_model.pkl"):
        print("✅ Model file exists")
        tests_passed += 1
    else:
        print("❌ Model file not found")
    
    # Test 2: Model Loading
    tests_total += 1
    try:
        model_package = joblib.load("enhanced_heart_disease_model.pkl")
        print("✅ Model loads successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Model loading failed: {e}")
        return
    
    # Test 3: Model Structure
    tests_total += 1
    try:
        required_keys = ['model', 'feature_columns', 'model_type']
        for key in required_keys:
            if key not in model_package:
                raise ValueError(f"Missing key: {key}")
        print("✅ Model structure is valid")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Model structure invalid: {e}")
    
    # Test 4: Feature Count
    tests_total += 1
    if len(model_package['feature_columns']) == 13:
        print("✅ Correct number of features (13)")
        tests_passed += 1
    else:
        print(f"❌ Wrong feature count: {len(model_package['feature_columns'])}")
    
    # Test 5: Model Prediction
    tests_total += 1
    try:
        model = model_package['model']
        scaler = model_package.get('scaler')
        feature_columns = model_package['feature_columns']
        
        # Sample test data
        test_data = {
            'age': 50, 'sex': 1, 'cp': 1, 'trestbps': 140, 'chol': 240,
            'fbs': 0, 'restecg': 0, 'thalach': 150, 'exang': 0,
            'oldpeak': 1.2, 'slope': 1, 'ca': 1, 'thal': 1
        }
        
        input_df = pd.DataFrame([test_data], columns=feature_columns)
        
        if scaler:
            input_scaled = scaler.transform(input_df)
            prediction = model.predict(input_scaled)
            probability = model.predict_proba(input_scaled)[0][1]
        else:
            prediction = model.predict(input_df)
            probability = model.predict_proba(input_df)[0][1]
        
        if 0 <= probability <= 1 and prediction[0] in [0, 1]:
            print(f"✅ Model prediction works (Risk: {probability:.1%})")
            tests_passed += 1
        else:
            print("❌ Invalid prediction output")
            
    except Exception as e:
        print(f"❌ Model prediction failed: {e}")
    
    # Test 6: Enhanced App Import
    tests_total += 1
    try:
        import enhanced_app
        print("✅ Enhanced app imports successfully")
        tests_passed += 1
    except Exception as e:
        print(f"❌ Enhanced app import failed: {e}")
    
    # Summary
    print("\n" + "=" * 60)
    print(f"QUICK TEST RESULTS: {tests_passed}/{tests_total} tests passed")
    print(f"Success Rate: {(tests_passed/tests_total*100):.1f}%")
    
    if tests_passed == tests_total:
        print("🎉 All quick tests passed!")
        return True
    else:
        print("⚠️  Some tests failed")
        return False

if __name__ == "__main__":
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = run_quick_tests()
    exit(0 if success else 1)
