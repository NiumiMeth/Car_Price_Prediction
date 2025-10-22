"""
🚗 Car Price Prediction - Streamlit Web Application
A beautiful and intuitive interface for predicting used car prices
"""

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import sys
from pathlib import Path
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, date

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
# Import data analyzer functions
import sys
sys.path.append(str(project_root / "src" / "utils"))
from data_analyzer import clean_and_analyze_data, get_popular_models, validate_input, get_price_insights

# Import config
sys.path.append(str(project_root / "src" / "streamlit"))
import config as app_config

# Page configuration
st.set_page_config(
    page_title=app_config.APP_TITLE,
    page_icon=app_config.APP_ICON,
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        'Get Help': 'https://github.com/NiumiMeth/Car_Price_Prediction',
        'Report a bug': "https://github.com/NiumiMeth/Car_Price_Prediction/issues",
        'About': "# Car Price Prediction App\nPredict used car prices with machine learning!"
    }
)

# Custom CSS for better styling
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
    
    .sub-header {
        font-size: 1.2rem;
        text-align: center;
        color: #666;
        margin-bottom: 3rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
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
    
    .stSelectbox > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .stNumberInput > div > div > input {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
    
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 25px;
        padding: 0.5rem 2rem;
        font-weight: bold;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 5px 15px rgba(0,0,0,0.2);
    }
    
    .info-box {
        background-color: #e3f2fd;
        border-left: 4px solid #2196f3;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .success-box {
        background-color: #e8f5e8;
        border-left: 4px solid #4caf50;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
    
    .warning-box {
        background-color: #fff3e0;
        border-left: 4px solid #ff9800;
        padding: 1rem;
        margin: 1rem 0;
        border-radius: 0 8px 8px 0;
    }
</style>
""", unsafe_allow_html=True)

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
        st.error(f"❌ Model files not found: {e}")
        st.info("Please run the model training notebook first to generate the required artifacts.")
        return None, None, None, None, None

@st.cache_data
def get_unique_values():
    """Get unique values for dropdowns from the raw data"""
    try:
        data_path = project_root / "data" / "raw" / "car.csv"
        analysis = clean_and_analyze_data(str(data_path))
        
        return (
            analysis['brands'],
            analysis['colors'],
            analysis['transmission_types'],
            analysis['fuel_types'],
            analysis['year_range'],
            analysis['power_range'],
            analysis['mileage_range']
        )
    except Exception as e:
        st.warning(f"Could not load data for dropdowns: {e}")
        # Return default values
        return (
            ['audi', 'bmw', 'ford', 'hyundai', 'honda', 'kia', 'mercedes', 'toyota', 'volkswagen'],
            ['black', 'white', 'silver', 'grey', 'blue', 'red', 'green', 'brown', 'gold', 'orange'],
            ['Manual', 'Automatic', 'Semi-automatic'],
            ['petrol', 'diesel', 'electric', 'hybrid', 'diesel hybrid', 'lpg', 'cng', 'hydrogen', 'ethanol'],
            (1990, 2024),
            (30, 500),
            (0, 1000000)
        )

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

def create_feature_importance_chart(model, feature_names, top_n=10):
    """Create a feature importance chart"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': feature_names,
            'importance': importances
        }).sort_values('importance', ascending=True).tail(top_n)
        
        fig = px.bar(
            feature_importance, 
            x='importance', 
            y='feature',
            orientation='h',
            title=f"Top {top_n} Most Important Features",
            color='importance',
            color_continuous_scale='viridis'
        )
        fig.update_layout(
            height=400,
            showlegend=False,
            title_x=0.5
        )
        return fig
    return None

def main():
    """Main Streamlit application"""
    
    # Header
    st.markdown(f'<h1 class="main-header">{app_config.APP_TITLE}</h1>', unsafe_allow_html=True)
    st.markdown(f'<p class="sub-header">{app_config.APP_SUBTITLE}</p>', unsafe_allow_html=True)
    
    # Load model and preprocessors
    model, log_scaler, direct_scaler, feature_order, model_target_mapping = load_model_and_preprocessors()
    
    if model is None:
        st.stop()
    
    # Get unique values for dropdowns
    brands, colors, transmission_types, fuel_types, year_range, power_range, mileage_range = get_unique_values()
    
    # Create two columns for layout
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 🚙 Car Specifications")
        
        # Brand selection with search
        brand = st.selectbox(
            "🏷️ Brand",
            brands,
            index=brands.index(app_config.DEFAULT_BRAND) if app_config.DEFAULT_BRAND in brands else 0,
            help=app_config.HELP_TEXTS['brand']
        )
        
        # Model input with placeholder
        model_name = st.text_input(
            "📝 Model",
            placeholder="e.g., A4, X3, Focus, Civic",
            help=app_config.HELP_TEXTS['model']
        )
        
        # Year with slider
        year = st.slider(
            "📅 Year",
            min_value=year_range[0],
            max_value=year_range[1],
            value=app_config.DEFAULT_YEAR,
            help=app_config.HELP_TEXTS['year']
        )
        
        # Color selection
        color = st.selectbox(
            "🎨 Color",
            colors,
            index=colors.index(app_config.DEFAULT_COLOR) if app_config.DEFAULT_COLOR in colors else 0,
            help=app_config.HELP_TEXTS['color']
        )
        
        # Transmission type
        transmission_type = st.selectbox(
            "⚙️ Transmission",
            transmission_types,
            index=transmission_types.index(app_config.DEFAULT_TRANSMISSION) if app_config.DEFAULT_TRANSMISSION in transmission_types else 0,
            help=app_config.HELP_TEXTS['transmission']
        )
        
        # Fuel type
        fuel_type = st.selectbox(
            "⛽ Fuel Type",
            fuel_types,
            index=fuel_types.index(app_config.DEFAULT_FUEL_TYPE) if app_config.DEFAULT_FUEL_TYPE in fuel_types else 0,
            help=app_config.HELP_TEXTS['fuel_type']
        )
    
    with col2:
        st.markdown("### 🔧 Technical Specifications")
        
        # Power with range slider
        power_kw = st.slider(
            "⚡ Power (kW)",
            min_value=int(power_range[0]),
            max_value=int(power_range[1]),
            value=app_config.DEFAULT_POWER,
            step=5,
            help=app_config.HELP_TEXTS['power']
        )
        
        # Fuel consumption
        fuel_consumption_g_km = st.number_input(
            "⛽ Fuel Consumption (g/km)",
            min_value=0.0,
            max_value=500.0,
            value=150.0,
            step=1.0,
            help=app_config.HELP_TEXTS['fuel_consumption']
        )
        
        # Mileage
        mileage_in_km = st.number_input(
            "🛣️ Mileage (km)",
            min_value=int(mileage_range[0]),
            max_value=int(mileage_range[1]),
            value=app_config.DEFAULT_MILEAGE,
            step=1000,
            help=app_config.HELP_TEXTS['mileage']
        )
        
        # EV range (conditional on fuel type)
        if fuel_type in ['electric', 'hybrid', 'diesel hybrid']:
            ev_range_km = st.number_input(
                "🔋 EV Range (km)",
                min_value=0,
                max_value=1000,
                value=300 if fuel_type == 'electric' else 50,
                step=10,
                help=app_config.HELP_TEXTS['ev_range']
            )
        else:
            ev_range_km = 0
        
        # Additional info box
        st.markdown("""
        <div class="info-box">
            <strong>💡 Tips for better predictions:</strong><br>
            • Ensure all specifications are accurate<br>
            • Use realistic values for mileage and year<br>
            • For electric/hybrid vehicles, set appropriate EV range<br>
            • Higher power and newer cars typically cost more
        </div>
        """, unsafe_allow_html=True)
    
    # Prediction button
    st.markdown("---")
    
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    
    with col_btn2:
        predict_button = st.button(
            "🔮 Predict Price",
            type="primary",
            use_container_width=True
        )
    
    # Make prediction
    if predict_button:
        # Validate input
        is_valid, error_message = validate_input(
            brand, model_name, year, color, transmission_type, fuel_type,
            power_kw, fuel_consumption_g_km, mileage_in_km, ev_range_km
        )
        
        if not is_valid:
            st.error(f"❌ {error_message}")
        else:
            try:
                with st.spinner("🔄 Analyzing your car specifications..."):
                    # Preprocess input
                    input_processed = preprocess_input(
                        brand, model_name, year, color, transmission_type, fuel_type,
                        power_kw, fuel_consumption_g_km, mileage_in_km, ev_range_km
                    )
                    
                    # Make prediction
                    prediction = model.predict(input_processed)[0]
                    
                    # Display result with beautiful styling
                    st.markdown("""
                    <div class="prediction-card">
                        <h2>💰 Predicted Price</h2>
                        <h1 style="font-size: 3rem; margin: 1rem 0;">€{:,}</h1>
                        <p style="font-size: 1.2rem; margin: 0;">Based on your specifications</p>
                    </div>
                    """.format(int(prediction)), unsafe_allow_html=True)
                    
                    # Additional info
                    col_info1, col_info2, col_info3 = st.columns(3)
                    
                    with col_info1:
                        st.metric("Brand", brand.upper())
                    
                    with col_info2:
                        st.metric("Year", year)
                    
                    with col_info3:
                        st.metric("Power", f"{power_kw} kW")
                    
                    # Get price insights
                    price_insights = get_price_insights(prediction)
                    
                    st.markdown(f"""
                    <div class="success-box">
                        <strong>{price_insights['emoji']} Price Category:</strong> {price_insights['category']} Vehicle<br>
                        <strong>Description:</strong> {price_insights['description']}<br>
                        <strong>Price Range:</strong> {price_insights['price_range']}
                    </div>
                    """, unsafe_allow_html=True)
                    
            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check your input values and try again.")
    
    # Model information section
    st.markdown("---")
    st.markdown("### 📊 Model Information")
    
    col_metric1, col_metric2, col_metric3, col_metric4 = st.columns(4)
    
    with col_metric1:
        st.metric("Model Type", app_config.MODEL_METRICS['model_type'])
    
    with col_metric2:
        st.metric("R² Score", f"{app_config.MODEL_METRICS['r2_score']:.4f}")
    
    with col_metric3:
        st.metric("MAE", f"€{app_config.MODEL_METRICS['mae']:,}")
    
    with col_metric4:
        st.metric("RMSE", f"€{app_config.MODEL_METRICS['rmse']:,}")
    
    # Feature importance chart
    if hasattr(model, 'feature_importances_'):
        st.markdown("### 🎯 Feature Importance")
        feature_names = feature_order if feature_order else []
        importance_chart = create_feature_importance_chart(model, feature_names)
        if importance_chart:
            st.plotly_chart(importance_chart, use_container_width=True)
    
    # Instructions and help
    with st.expander("📖 How to Use This App"):
        st.markdown("""
        ### Getting Started
        
        1. **Fill in the car specifications** using the form above
        2. **Click 'Predict Price'** to get the estimated price
        3. **Adjust parameters** to see how they affect the prediction
        
        ### Understanding the Results
        
        - **Predicted Price**: The estimated market value of your car
        - **Price Category**: Budget, Mid-range, Premium, or Luxury
        - **Feature Importance**: Shows which factors most influence the price
        
        ### Tips for Better Predictions
        
        - Ensure all specifications are accurate
        - Use realistic values for mileage and year
        - For electric/hybrid vehicles, set appropriate EV range
        - Higher power and newer cars typically cost more
        - Lower mileage generally increases value
        
        ### Model Performance
        
        This model achieves:
        - **93.36% accuracy** (R² Score)
        - **€3,619 average error** (MAE)
        - **€7,330 root mean square error** (RMSE)
        """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🚗 <strong>Car Price Prediction App</strong> | Powered by Machine Learning</p>
        <p>Built with ❤️ using Streamlit and XGBoost</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()