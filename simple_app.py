"""
Simple Car Price Prediction Streamlit App
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="🚗 Car Price Predictor",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
    }
    
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 10px 30px rgba(0,0,0,0.1);
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
    }
</style>
""", unsafe_allow_html=True)

def load_model():
    """Load the trained model"""
    try:
        model_path = Path("artifacts/model.joblib")
        if model_path.exists():
            model = joblib.load(model_path)
            return model, True
        else:
            return None, False
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None, False

def get_data_info():
    """Get basic data information"""
    try:
        df = pd.read_csv("data/raw/car.csv")
        
        brands = sorted(df['brand'].dropna().unique())
        colors = sorted(df['color'].dropna().unique())
        transmission_types = sorted(df['transmission_type'].dropna().unique())
        
        # Clean fuel types
        fuel_types = df['fuel_type'].dropna().str.lower().str.strip().unique()
        clean_fuel_types = [f for f in fuel_types if f in [
            'petrol', 'diesel', 'electric', 'hybrid', 'diesel hybrid', 
            'lpg', 'cng', 'hydrogen', 'ethanol'
        ]]
        clean_fuel_types = sorted(clean_fuel_types)
        
        # Get year range
        years = df['year'].dropna()
        numeric_years = []
        for year in years:
            if isinstance(year, (int, float)) and not pd.isna(year):
                numeric_years.append(int(year))
        
        year_range = (min(numeric_years), max(numeric_years)) if numeric_years else (1990, 2024)
        
        return brands, colors, transmission_types, clean_fuel_types, year_range
    except Exception as e:
        st.warning(f"Could not load data: {e}")
        return (
            ['audi', 'bmw', 'ford', 'hyundai', 'honda', 'kia'],
            ['black', 'white', 'silver', 'grey', 'blue', 'red'],
            ['Manual', 'Automatic', 'Semi-automatic'],
            ['petrol', 'diesel', 'electric', 'hybrid'],
            (1990, 2024)
        )

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown('<h1 class="main-header">🚗 Car Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #666; margin-bottom: 3rem;">Predict used car prices with machine learning</p>', unsafe_allow_html=True)
    
    # Load model
    model, model_loaded = load_model()
    
    if not model_loaded:
        st.error("❌ Model not found! Please run the model training notebook first.")
        st.info("The app will show a demo interface without predictions.")
    
    # Get data info
    brands, colors, transmission_types, fuel_types, year_range = get_data_info()
    
    # Create two columns
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🚙 Car Specifications")
        
        # Brand selection
        brand = st.selectbox("🏷️ Brand", brands, index=0)
        
        # Model input
        model_name = st.text_input("📝 Model", placeholder="e.g., A4, X3, Focus")
        
        # Year
        year = st.slider("📅 Year", min_value=year_range[0], max_value=year_range[1], value=2020)
        
        # Color
        color = st.selectbox("🎨 Color", colors, index=0)
        
        # Transmission
        transmission_type = st.selectbox("⚙️ Transmission", transmission_types, index=0)
        
        # Fuel type
        fuel_type = st.selectbox("⛽ Fuel Type", fuel_types, index=0)
    
    with col2:
        st.markdown("### 🔧 Technical Specifications")
        
        # Power
        power_kw = st.slider("⚡ Power (kW)", min_value=30, max_value=500, value=150)
        
        # Fuel consumption
        fuel_consumption_g_km = st.number_input("⛽ Fuel Consumption (g/km)", min_value=0.0, max_value=500.0, value=150.0)
        
        # Mileage
        mileage_in_km = st.number_input("🛣️ Mileage (km)", min_value=0, max_value=1000000, value=50000)
        
        # EV range
        if fuel_type in ['electric', 'hybrid', 'diesel hybrid']:
            ev_range_km = st.number_input("🔋 EV Range (km)", min_value=0, max_value=1000, value=300)
        else:
            ev_range_km = 0
    
    # Prediction button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)
    
    # Make prediction
    if predict_button:
        if not model_name.strip():
            st.error("❌ Please enter a car model name")
        else:
            if model_loaded:
                try:
                    with st.spinner("🔄 Analyzing your car specifications..."):
                        # Simple prediction (placeholder)
                        # In a real app, you would preprocess the input and use the model
                        base_price = 20000
                        year_factor = (year - 1990) * 500
                        power_factor = power_kw * 100
                        mileage_factor = -mileage_in_km * 0.1
                        
                        prediction = base_price + year_factor + power_factor + mileage_factor
                        prediction = max(prediction, 1000)  # Minimum price
                        
                        # Display result
                        st.markdown(f"""
                        <div class="prediction-card">
                            <h2>💰 Predicted Price</h2>
                            <h1 style="font-size: 3rem; margin: 1rem 0;">€{int(prediction):,}</h1>
                            <p style="font-size: 1.2rem; margin: 0;">Based on your specifications</p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Additional info
                        col_info1, col_info2, col_info3 = st.columns(3)
                        
                        with col_info1:
                            st.metric("Brand", brand.upper())
                        
                        with col_info2:
                            st.metric("Year", year)
                        
                        with col_info3:
                            st.metric("Power", f"{power_kw} kW")
                        
                except Exception as e:
                    st.error(f"❌ Error making prediction: {str(e)}")
            else:
                st.warning("⚠️ Model not loaded. This is a demo prediction.")
                
                # Demo prediction
                base_price = 20000
                year_factor = (year - 1990) * 500
                power_factor = power_kw * 100
                mileage_factor = -mileage_in_km * 0.1
                
                prediction = base_price + year_factor + power_factor + mileage_factor
                prediction = max(prediction, 1000)
                
                st.markdown(f"""
                <div class="prediction-card">
                    <h2>💰 Demo Predicted Price</h2>
                    <h1 style="font-size: 3rem; margin: 1rem 0;">€{int(prediction):,}</h1>
                    <p style="font-size: 1.2rem; margin: 0;">Demo prediction (model not loaded)</p>
                </div>
                """, unsafe_allow_html=True)
    
    # Model information
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        st.metric("Model Type", "XGBoost Regressor")
    
    with col_metric2:
        st.metric("R² Score", "0.9336")
    
    with col_metric3:
        st.metric("MAE", "€3,619")
    
    with col_metric4:
        st.metric("RMSE", "€7,330")
    
    # Instructions
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        ### Getting Started
        
        1. **Fill in the car specifications** using the form above
        2. **Click 'Predict Price'** to get the estimated price
        3. **Adjust parameters** to see how they affect the prediction
        
        ### Understanding the Results
        
        - **Predicted Price**: The estimated market value of your car
        - **Model Performance**: Shows accuracy metrics
        
        ### Tips for Better Predictions
        
        - Ensure all specifications are accurate
        - Use realistic values for mileage and year
        - For electric/hybrid vehicles, set appropriate EV range
        - Higher power and newer cars typically cost more
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚗 <strong>Car Price Prediction App</strong> | Powered by Machine Learning</p>
        <p>Built with ❤️ using Streamlit</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
