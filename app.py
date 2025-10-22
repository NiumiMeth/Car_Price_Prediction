"""
Car Price Prediction App - Main Entry Point
Run this file to start the application
"""
import subprocess
import sys
from pathlib import Path

def main():
    """Main entry point for the car price prediction app"""
    print("🚗 Starting Car Price Prediction App...")
    print("📍 Access the app at: http://localhost:8501")
    print("🛑 Press Ctrl+C to stop the app")
    print("-" * 50)
    
    try:
        # Run the enhanced Streamlit app
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "enhanced_car_app.py", "--server.port", "8501"
        ])
    except KeyboardInterrupt:
        print("\n🛑 App stopped by user")
    except Exception as e:
        print(f"❌ Error starting app: {e}")

if __name__ == "__main__":
    main()
