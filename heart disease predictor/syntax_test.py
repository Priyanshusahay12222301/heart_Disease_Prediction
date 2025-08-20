# syntax_test.py
"""
Test script to verify all files have correct syntax
"""

def test_syntax():
    print("Testing syntax for all Python files...")
    
    try:
        # Test enhanced_app.py
        import enhanced_app
        print("✅ enhanced_app.py - No syntax errors")
        
        # Test enhanced_retrain_model.py
        import enhanced_retrain_model
        print("✅ enhanced_retrain_model.py - No syntax errors")
        
        # Test test_enhanced_api.py
        import test_enhanced_api
        print("✅ test_enhanced_api.py - No syntax errors")
        
        # Test Flask app creation
        app = enhanced_app.app
        print("✅ Flask app initializes correctly")
        print(f"   Model loaded: {enhanced_app.model is not None}")
        print(f"   Available routes: {len(list(app.url_map.iter_rules()))}")
        
        print("\n🎉 All syntax tests passed!")
        return True
        
    except Exception as e:
        print(f"❌ Syntax error found: {e}")
        return False

if __name__ == "__main__":
    test_syntax()
