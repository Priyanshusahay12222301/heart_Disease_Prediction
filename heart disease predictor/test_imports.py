# test_imports.py
"""
Simple script to test all imports and dependencies
"""

import sys
import os
from pathlib import Path

def test_imports():
    """Test all required imports for the heart disease prediction system"""
    print("🔍 Testing all imports...")
    print("=" * 50)
    
    success_count = 0
    total_count = 0
    
    # Test core Python imports
    print("\n📦 Core Python modules:")
    core_modules = ['os', 'sys', 'json', 'time', 'pathlib']
    for module in core_modules:
        total_count += 1
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    # Test data science imports
    print("\n🔬 Data Science modules:")
    ds_modules = ['pandas', 'numpy', 'joblib', 'sklearn']
    for module in ds_modules:
        total_count += 1
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    # Test web/API imports
    print("\n🌐 Web/API modules:")
    web_modules = ['flask', 'requests']
    for module in web_modules:
        total_count += 1
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    # Test testing modules
    print("\n🧪 Testing modules:")
    test_modules = ['pytest']
    for module in test_modules:
        total_count += 1
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    # Test project modules
    print("\n📁 Project modules:")
    project_modules = ['comprehensive_test_suite', 'enhanced_app']
    for module in project_modules:
        total_count += 1
        try:
            __import__(module)
            print(f"✅ {module}")
            success_count += 1
        except ImportError as e:
            print(f"❌ {module}: {e}")
    
    # Summary
    print("\n" + "=" * 50)
    print(f"Import Test Results: {success_count}/{total_count} successful")
    print(f"Success Rate: {(success_count/total_count*100):.1f}%")
    
    if success_count == total_count:
        print("🎉 All imports successful! System ready to run.")
        return True
    else:
        print("⚠️  Some imports failed. Install missing packages.")
        return False

if __name__ == "__main__":
    # Change to the correct directory
    script_dir = Path(__file__).parent
    os.chdir(script_dir)
    
    success = test_imports()
    exit(0 if success else 1)
