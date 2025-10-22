#!/usr/bin/env python3
"""
Test script for the Car Price Prediction Streamlit app
"""

import sys
import os
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.append(str(project_root))

def test_imports():
    """Test if all required modules can be imported"""
    try:
        from src.utils.data_analyzer import clean_and_analyze_data, validate_input, get_price_insights
        import src.streamlit.config as config
        print("SUCCESS: All imports successful")
        return True
    except ImportError as e:
        print(f"ERROR: Import error: {e}")
        return False

def test_data_analysis():
    """Test data analysis functionality"""
    try:
        from src.utils.data_analyzer import clean_and_analyze_data
        
        data_path = project_root / "data" / "raw" / "car.csv"
        if data_path.exists():
            analysis = clean_and_analyze_data(str(data_path))
            print(f"SUCCESS: Data analysis successful")
            print(f"   - Found {len(analysis['brands'])} brands")
            print(f"   - Found {len(analysis['colors'])} colors")
            print(f"   - Year range: {analysis['year_range']}")
            return True
        else:
            print("WARNING: Data file not found, using default values")
            return True
    except Exception as e:
        print(f"ERROR: Data analysis error: {e}")
        return False

def test_validation():
    """Test input validation"""
    try:
        from src.utils.data_analyzer import validate_input
        
        # Test valid input
        is_valid, error = validate_input(
            "audi", "a4", 2020, "black", "Manual", "petrol",
            150, 150.0, 50000, 0
        )
        
        if is_valid:
            print("SUCCESS: Input validation working")
        else:
            print(f"ERROR: Validation error: {error}")
            return False
        
        # Test invalid input
        is_valid, error = validate_input(
            "", "a4", 2020, "black", "Manual", "petrol",
            150, 150.0, 50000, 0
        )
        
        if not is_valid:
            print("SUCCESS: Error detection working")
        else:
            print("ERROR: Should have detected empty brand")
            return False
        
        return True
    except Exception as e:
        print(f"ERROR: Validation test error: {e}")
        return False

def test_price_insights():
    """Test price insights functionality"""
    try:
        from src.utils.data_analyzer import get_price_insights
        
        # Test different price ranges
        insights = get_price_insights(3000)  # This should be Budget
        if insights['category'] == 'Budget':
            print("SUCCESS: Price insights working")
        else:
            print(f"ERROR: Price insights not working correctly. Got {insights['category']} for price 3000")
            return False
        
        return True
    except Exception as e:
        print(f"ERROR: Price insights test error: {e}")
        return False

def test_config():
    """Test configuration loading"""
    try:
        import src.streamlit.config as config
        
        if config.APP_TITLE and config.PRIMARY_COLOR:
            print("SUCCESS: Configuration loading successful")
            return True
        else:
            print("ERROR: Configuration not loaded properly")
            return False
    except Exception as e:
        print(f"ERROR: Configuration test error: {e}")
        return False

def main():
    """Run all tests"""
    print("Testing Car Price Prediction App")
    print("=" * 50)
    
    tests = [
        ("Import Test", test_imports),
        ("Data Analysis Test", test_data_analysis),
        ("Validation Test", test_validation),
        ("Price Insights Test", test_price_insights),
        ("Configuration Test", test_config)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n{test_name}...")
        if test_func():
            passed += 1
        else:
            print(f"FAILED: {test_name}")
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("SUCCESS: All tests passed! The app should work correctly.")
        return 0
    else:
        print("WARNING: Some tests failed. Please check the errors above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
