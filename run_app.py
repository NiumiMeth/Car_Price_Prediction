#!/usr/bin/env python3
"""
Simple script to run the Car Price Prediction Streamlit app
"""

import subprocess
import sys
import os
from pathlib import Path

def main():
    """Run the Streamlit application"""
    
    # Get project root directory
    project_root = Path(__file__).parent
    
    # Change to project directory
    os.chdir(project_root)
    
    # Check if artifacts directory exists
    artifacts_dir = project_root / "artifacts"
    if not artifacts_dir.exists():
        print("❌ Artifacts directory not found!")
        print("Please run the model training notebook first to generate the required artifacts.")
        print("Run: jupyter notebook notebooks/02_model_training.ipynb")
        sys.exit(1)
    
    # Check if required model files exist
    required_files = [
        "model.joblib",
        "log_scaler.joblib", 
        "direct_scaler.joblib",
        "feature_order.joblib",
        "model_target_mapping.csv"
    ]
    
    missing_files = []
    for file in required_files:
        if not (artifacts_dir / file).exists():
            missing_files.append(file)
    
    if missing_files:
        print("❌ Missing required model files:")
        for file in missing_files:
            print(f"   - {file}")
        print("\nPlease run the model training notebook first.")
        print("Run: jupyter notebook notebooks/02_model_training.ipynb")
        sys.exit(1)
    
    # Check if processed data exists
    processed_data_dir = project_root / "data" / "processed"
    if not (processed_data_dir / "train_final.csv").exists():
        print("❌ Processed training data not found!")
        print("Please run the data cleaning notebook first.")
        print("Run: jupyter notebook notebooks/01_data_cleaning.ipynb")
        sys.exit(1)
    
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
            "--server.address=0.0.0.0"
        ])
    except KeyboardInterrupt:
        print("\n👋 App stopped by user")
    except Exception as e:
        print(f"❌ Error running app: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
