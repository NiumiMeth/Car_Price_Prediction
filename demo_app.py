#!/usr/bin/env python3
"""
Demo script to run the Car Price Prediction Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit application with demo data"""
    
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Change to project directory
    os.chdir(project_root)
    
    # Check if data exists
    data_dir = project_root / "data" / "raw"
    if not (data_dir / "car.csv").exists():
        print("❌ Raw data file not found!")
        print("Please ensure car.csv is in the data/raw/ directory")
        return
    
    # Check if processed data exists
    processed_dir = project_root / "data" / "processed"
    if not (processed_dir / "train_final.csv").exists():
        print("⚠️  Processed data not found!")
        print("The app will use default values for dropdowns")
        print("To get full functionality, run the data cleaning notebook first")
    
    # Check if model artifacts exist
    artifacts_dir = project_root / "artifacts"
    if not (artifacts_dir / "model.joblib").exists():
        print("⚠️  Model artifacts not found!")
        print("The app will show an error message")
        print("To get full functionality, run the model training notebook first")
    
    # Run Streamlit app
    print("🚀 Starting Car Price Prediction App...")
    print("📍 App will be available at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "src/streamlit/app.py",
            "--server.port=8501",
            "--server.address=0.0.0.0",
            "--theme.base=light",
            "--theme.primaryColor=#667eea",
            "--theme.backgroundColor=#ffffff",
            "--theme.secondaryBackgroundColor=#f0f2f6",
            "--theme.textColor=#262730"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        return 1

if __name__ == "__main__":
    main()
