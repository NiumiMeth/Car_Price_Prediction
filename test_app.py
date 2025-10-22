"""
Test script for the car price prediction app
"""

import sys
from pathlib import Path

def test_imports():
    """Test if all required modules can be imported"""
    print("Testing imports...")
    
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
    except ImportError as e:
        print(f"❌ Streamlit import failed: {e}")
        return False
    
    try:
        import pandas as pd
        print("✅ Pandas imported successfully")
    except ImportError as e:
        print(f"❌ Pandas import failed: {e}")
        return False
    
    try:
        import numpy as np
        print("✅ NumPy imported successfully")
    except ImportError as e:
        print(f"❌ NumPy import failed: {e}")
        return False
    
    try:
        import joblib
        print("✅ Joblib imported successfully")
    except ImportError as e:
        print(f"❌ Joblib import failed: {e}")
        return False
    
    try:
        from xgboost import XGBRegressor
        print("✅ XGBoost imported successfully")
    except ImportError as e:
        print(f"❌ XGBoost import failed: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✅ Matplotlib imported successfully")
    except ImportError as e:
        print(f"❌ Matplotlib import failed: {e}")
        return False
    
    try:
        import seaborn as sns
        print("✅ Seaborn imported successfully")
    except ImportError as e:
        print(f"❌ Seaborn import failed: {e}")
        return False
    
    return True

def test_data_files():
    """Test if data files exist"""
    print("\nTesting data files...")
    
    data_files = [
        "data/raw/car.csv",
        "data/processed/train_final.csv",
        "data/processed/test_final.csv"
    ]
    
    all_exist = True
    for file_path in data_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False
    
    return all_exist

def test_app_files():
    """Test if app files exist"""
    print("\nTesting app files...")
    
    required_files = [
        "app.py",
        "train_model.py",
        "requirements.txt"
    ]
    
    all_exist = True
    for file_path in required_files:
        if Path(file_path).exists():
            print(f"✅ {file_path} exists")
        else:
            print(f"❌ {file_path} not found")
            all_exist = False
    
    return all_exist

def main():
    """Run all tests"""
    print("🧪 Running Car Price Prediction App Tests")
    print("=" * 50)
    
    tests = [
        test_imports,
        test_data_files,
        test_app_files
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
        print()
    
    print("=" * 50)
    print(f"📊 Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The app is ready to run.")
        print("🚀 Run the app with: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please fix the issues before running the app.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)