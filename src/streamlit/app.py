"""
Streamlit Web Application for Car Price Prediction
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.append(str(project_root))

from config import (
    ARTIFACTS_DIR, 
    MODEL_FILE, 
    LOG_SCALER_FILE, 
    DIRECT_SCALER_FILE,
    FEATURE_ORDER_FILE,
    MODEL_TARGET_MAPPING_FILE,
    STREAMLIT_CONFIG
)

# Page configuration
st.set_page_config(
    page_title=STREAMLIT_CONFIG["page_title"],
    page_icon=STREAMLIT_CONFIG["page_icon"],
    layout=STREAMLIT_CONFIG["layout"],
    initial_sidebar_state=STREAMLIT_CONFIG["initial_sidebar_state"]
)

@st.cache_data
def load_model_and_preprocessors():
    """Load the trained model and preprocessing artifacts"""
    try:
        model = joblib.load(ARTIFACTS_DIR / MODEL_FILE)
        log_scaler = joblib.load(ARTIFACTS_DIR / LOG_SCALER_FILE)
        direct_scaler = joblib.load(ARTIFACTS_DIR / DIRECT_SCALER_FILE)
        feature_order = joblib.load(ARTIFACTS_DIR / FEATURE_ORDER_FILE)
        model_target_mapping = pd.read_csv(ARTIFACTS_DIR / MODEL_TARGET_MAPPING_FILE)
        
        return model, log_scaler, direct_scaler, feature_order, model_target_mapping
    except FileNotFoundError as e:
        st.error(f"Model files not found: {e}")
        st.info("Please run the model training notebook first to generate the required artifacts.")
        return None, None, None, None, None

def preprocess_input(brand, model, year, color, transmission_type, fuel_type, 
                    power_kw, fuel_consumption_g_km, mileage_in_km, ev_range_km):
    """Preprocess user input for prediction"""
    
    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'brand': [brand.lower()],
        'model': [f"{brand.lower()}_{model.lower()}"],
        'year': [year],
        'color': [color.lower()],
        'transmission_type': [transmission_type.lower()],
        'fuel_type': [fuel_type.lower()],
        'power_kw': [power_kw],
        'fuel_consumption_g_km': [fuel_consumption_g_km],
        'mileage_in_km': [mileage_in_km],
        'ev_range_km': [ev_range_km]
    })
    
    # Calculate derived features
    data_collection_year = 2024  # Update based on your dataset
    input_data['vehicle_manufacturing_age'] = data_collection_year - input_data['year']
    input_data['vehicle_registration_age'] = data_collection_year - input_data['year']  # Simplified
    input_data['mileage_per_year'] = input_data['mileage_in_km'] / input_data['vehicle_registration_age']
    input_data['reg_month_sin'] = np.sin(2 * np.pi * 6 / 12)  # Default to June
    input_data['reg_month_cos'] = np.cos(2 * np.pi * 6 / 12)
    
    # One-hot encoding for categorical variables
    categorical_columns = ['brand', 'color', 'transmission_type', 'fuel_type']
    input_encoded = pd.get_dummies(input_data, columns=categorical_columns, drop_first=True, dtype=int)
    
    # Add model target encoding (simplified)
    input_encoded['model_target_enc'] = 25000  # Default value
    
    # Ensure all required columns are present
    feature_order = joblib.load(ARTIFACTS_DIR / FEATURE_ORDER_FILE)
    for col in feature_order:
        if col not in input_encoded.columns:
            input_encoded[col] = 0
    
    # Reorder columns to match training data
    input_encoded = input_encoded[feature_order]
    
    return input_encoded

def main():
    """Main Streamlit application"""
    
    # Header
    st.title("🚗 Car Price Prediction")
    st.markdown("Predict the price of used cars using machine learning")
    
    # Load model and preprocessors
    model, log_scaler, direct_scaler, feature_order, model_target_mapping = load_model_and_preprocessors()
    
    if model is None:
        st.stop()
    
    # Sidebar for input
    st.sidebar.header("Car Specifications")
    
    # Brand selection
    brands = ['audi', 'bmw', 'ford', 'hyundai', 'honda', 'kia', 'mercedes', 'toyota', 'volkswagen']
    brand = st.sidebar.selectbox("Brand", brands)
    
    # Model input
    model_name = st.sidebar.text_input("Model", placeholder="e.g., a4, x3, focus")
    
    # Year
    year = st.sidebar.number_input("Year", min_value=1990, max_value=2024, value=2020)
    
    # Color
    colors = ['black', 'white', 'silver', 'grey', 'blue', 'red', 'green', 'brown', 'gold', 'orange']
    color = st.sidebar.selectbox("Color", colors)
    
    # Transmission type
    transmission_types = ['manual', 'automatic', 'semi-automatic']
    transmission_type = st.sidebar.selectbox("Transmission", transmission_types)
    
    # Fuel type
    fuel_types = ['petrol', 'diesel', 'electric', 'hybrid', 'diesel_hybrid', 'lpg', 'cng', 'hydrogen', 'ethanol']
    fuel_type = st.sidebar.selectbox("Fuel Type", fuel_types)
    
    # Power
    power_kw = st.sidebar.number_input("Power (kW)", min_value=30, max_value=1000, value=150)
    
    # Fuel consumption
    fuel_consumption_g_km = st.sidebar.number_input("Fuel Consumption (g/km)", min_value=0, max_value=500, value=150)
    
    # Mileage
    mileage_in_km = st.sidebar.number_input("Mileage (km)", min_value=0, max_value=500000, value=50000)
    
    # EV range (for electric/hybrid)
    ev_range_km = st.sidebar.number_input("EV Range (km)", min_value=0, max_value=1000, value=0)
    
    # Prediction button
    if st.sidebar.button("Predict Price", type="primary"):
        try:
            # Preprocess input
            input_processed = preprocess_input(
                brand, model_name, year, color, transmission_type, fuel_type,
                power_kw, fuel_consumption_g_km, mileage_in_km, ev_range_km
            )
            
            # Make prediction
            prediction = model.predict(input_processed)[0]
            
            # Display result
            st.success(f"Predicted Price: €{prediction:,.2f}")
            
            # Additional info
            st.info(f"Model: {brand.upper()} {model_name.upper()}")
            st.info(f"Year: {year}")
            st.info(f"Power: {power_kw} kW")
            st.info(f"Mileage: {mileage_in_km:,} km")
            
        except Exception as e:
            st.error(f"Error making prediction: {str(e)}")
    
    # Main content area
    st.markdown("---")
    
    # Model information
    st.header("Model Information")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Model Type", "XGBoost Regressor")
    
    with col2:
        st.metric("R² Score", "0.9336")
    
    with col3:
        st.metric("MAE", "€3,619")
    
    # Feature importance (placeholder)
    st.header("Feature Importance")
    st.info("Feature importance analysis will be displayed here after model training.")
    
    # Instructions
    st.header("How to Use")
    st.markdown("""
    1. **Fill in the car specifications** using the sidebar
    2. **Click 'Predict Price'** to get the estimated price
    3. **Adjust parameters** to see how they affect the prediction
    
    ### Tips for Better Predictions:
    - Ensure all specifications are accurate
    - Use realistic values for mileage and year
    - For electric/hybrid vehicles, set appropriate EV range
    """)

if __name__ == "__main__":
    main()
