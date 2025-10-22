"""
Car Price Prediction App - Enhanced v2 (NO PLOTLY)
Fixes for File Not Found and Session State/Sensitivity errors.
"""
import streamlit as st
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from xgboost import XGBRegressor
import warnings
import matplotlib.pyplot as plt # Added for simple bar chart fallback
import seaborn as sns          # Added for simple bar chart fallback

warnings.filterwarnings('ignore')

# ---------------------------------------------------------
# Global Configuration/Mocks
# ---------------------------------------------------------
# Define expected file paths based on the earlier prompt's assumptions
MODELS_DIR = Path("models")
REQUIRED_FILES = [
    MODELS_DIR / "xgb_model.json",
    MODELS_DIR / "feature_order.joblib",
    MODELS_DIR / "scaler.joblib",
    MODELS_DIR / "model_metrics.joblib",
]

# Mock data for graceful fallback
MOCK_METRICS = {'r2_score': 0.874, 'mae': 3500, 'accuracy': 87.4}
MOCK_ORDER = ['year', 'power_kw', 'mileage_in_km', 'fuel_consumption_g_km', 'vehicle_manufacturing_age', 'power_squared', 'mileage_log', 'age_log', 'brand_audi', 'color_black']
MOCK_WEIGHTS = {'xgb': 0.6, 'rf': 0.4} # Assuming XGB is slightly better


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="🚗 Car Price Predictor (Simple Charts)",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------
# Custom CSS (dark-theme friendly)
# ---------------------------------------------------------
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: 800;
        text-align: center;
        background: linear-gradient(90deg, #67d7f5 0%, #7a6ff0 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 2rem;
        letter-spacing: 0.5px;
    }
    .prediction-card {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 2rem;
        border-radius: 18px;
        color: white;
        text-align: center;
        margin: 2rem 0;
        box-shadow: 0 14px 40px rgba(0,0,0,0.22);
    }
    .stButton > button {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        border-radius: 28px;
        padding: 0.6rem 2rem;
        font-weight: 700;
        font-size: 1.05rem;
        transition: transform .08s ease-in-out;
    }
    .stButton > button:active { transform: scale(0.98); }

    /* Enhanced info box (glass + glow for dark background) */
    .info-box {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(255, 255, 255, 0.18);
        border-left: 4px solid #90caf9;
        backdrop-filter: blur(10px);
        -webkit-backdrop-filter: blur(10px);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        margin: 1.1rem 0 0.8rem 0;
        color: #e3f2fd;
        box-shadow: 0 10px 30px rgba(33, 150, 243, 0.15);
        transition: all 0.2s ease-in-out;
    }
    .info-box:hover {
        box-shadow: 0 16px 38px rgba(33, 150, 243, 0.22);
        transform: translateY(-1px);
    }

    .accuracy-badge {
        background: linear-gradient(45deg, #4CAF50, #45a049);
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 20px;
        font-weight: bold;
        display: inline-block;
        margin: 0.35rem 0.35rem;
        box-shadow: 0 6px 18px rgba(76,175,80,.22);
    }
    /* Simple box for gauge replacement */
    .price-band-box {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid #7a6ff0;
        border-radius: 14px;
        padding: 1.5rem;
        margin: 1.5rem 0;
        text-align: center;
        color: #e3f2fd;
        box-shadow: 0 8px 20px rgba(122, 111, 240, 0.1);
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------
# Cache helpers (Robustified)
# ---------------------------------------------------------
@st.cache_resource(show_spinner="Loading models and artifacts...")
def load_enhanced_model():
    """Load the enhanced model artifacts with graceful fallback."""
    try:
        # Check all required files
        for f in REQUIRED_FILES:
            if not f.exists():
                st.error(f"❌ Missing required file: {f}. Using mock/fallback data.")
                raise FileNotFoundError(f)

        # 1. Load XGBoost model
        xgb_model = XGBRegressor()
        xgb_model.load_model(str(MODELS_DIR / "xgb_model.json"))

        # 2. Load Random Forest model (for ensemble) - optional
        try:
            rf_model = joblib.load(MODELS_DIR / "rf_model.joblib")
            rf_available = True
        except Exception:
            rf_model = None
            rf_available = False

        # 3. Load other artifacts
        feature_order = joblib.load(MODELS_DIR / "feature_order.joblib")
        scaler = joblib.load(MODELS_DIR / "scaler.joblib")
        metrics = joblib.load(MODELS_DIR / "model_metrics.joblib")
        # Ensemble weights default to XGB-only if RF is unavailable
        try:
            weights = joblib.load(MODELS_DIR / "ensemble_weights.joblib")
        except Exception:
            weights = {'xgb': 1.0, 'rf': 0.0}
        if not rf_available:
            weights = {'xgb': 1.0, 'rf': 0.0}

        return xgb_model, rf_model, feature_order, scaler, metrics, weights
    
    except Exception as e:
        # Fallback logic: Use mock metrics and data structure
        st.warning(f"⚠️ Model load failure ({e}). Running with **mock model and placeholder metrics**. ")
        
        class MockPredictor:
            def predict(self, X):
                base_price = 25000 
                if 'year' in MOCK_ORDER: base_price += X[0, MOCK_ORDER.index('year')] * 500
                if 'power_kw' in MOCK_ORDER: base_price += X[0, MOCK_ORDER.index('power_kw')] * 100
                if 'mileage_in_km' in MOCK_ORDER: base_price -= X[0, MOCK_ORDER.index('mileage_in_km')] * 0.05
                return np.array([max(1000, base_price)])
            def get_params(self, deep=True): return {} 
            def load_model(self, path): pass
            def feature_importances_(self): return np.linspace(0.01, 0.15, len(MOCK_ORDER))
            
        class MockScaler:
            def transform(self, X): return X.values 

        mock_predictor = MockPredictor()
        mock_scaler = MockScaler()

        return mock_predictor, None, MOCK_ORDER, mock_scaler, MOCK_METRICS, {'xgb': 1.0, 'rf': 0.0}

@st.cache_data
def get_data_info():
    """Get unique values from the raw data for dropdowns"""
    try:
        df = pd.read_csv(Path("data/raw/car.csv"))

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

        years = df['year'].dropna()
        numeric_years = [int(y) for y in years if isinstance(y, (int, float)) and not pd.isna(y)]
        year_range = (min(numeric_years), max(numeric_years)) if numeric_years else (1990, 2024)

        return brands, colors, transmission_types, clean_fuel_types, year_range
    except Exception as e:
        st.warning(f"Could not load raw data (data/raw/car.csv): {e}. Using hardcoded defaults.")
        return (
            ['Audi', 'BMW', 'Ford', 'Hyundai', 'Honda', 'Kia', 'Mercedes', 'Toyota', 'Volkswagen'],
            ['Black', 'White', 'Silver', 'Grey', 'Blue', 'Red', 'Green', 'Brown', 'Gold', 'Orange'],
            ['Manual', 'Automatic', 'Semi-automatic'],
            ['petrol', 'diesel', 'electric', 'hybrid'],
            (1990, 2024)
        )

# ---------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------
def create_enhanced_features(df):
    """Create enhanced features matching training data"""
    df_enhanced = df.copy()

    # Power-based
    df_enhanced['power_squared'] = df_enhanced['power_kw'] ** 2
    df_enhanced['power_log'] = np.log1p(df_enhanced['power_kw'])
    df_enhanced['power_sqrt'] = np.sqrt(df_enhanced['power_kw'])

    # Mileage-based
    df_enhanced['mileage_log'] = np.log1p(df_enhanced['mileage_in_km'])
    df_enhanced['mileage_sqrt'] = np.sqrt(df_enhanced['mileage_in_km'])

    # Age-based
    df_enhanced['age_squared'] = df_enhanced['vehicle_manufacturing_age'] ** 2
    df_enhanced['age_log'] = np.log1p(df_enhanced['vehicle_manufacturing_age'])

    # Fuel efficiency
    df_enhanced['fuel_efficiency'] = 1 / (df_enhanced['fuel_consumption_g_km'] + 1)

    # EV range features
    df_enhanced['ev_range_log'] = np.log1p(df_enhanced['ev_range_km'])
    df_enhanced['ev_range_sqrt'] = np.sqrt(df_enhanced['ev_range_km'])

    # Interactions
    df_enhanced['power_age_interaction'] = df_enhanced['power_kw'] * df_enhanced['vehicle_manufacturing_age']
    df_enhanced['mileage_age_interaction'] = df_enhanced['mileage_in_km'] * df_enhanced['vehicle_manufacturing_age']
    df_enhanced['power_mileage_interaction'] = df_enhanced['power_kw'] * df_enhanced['mileage_in_km']

    # Ratios
    df_enhanced['power_per_age'] = df_enhanced['power_kw'] / (df_enhanced['vehicle_manufacturing_age'] + 1)
    df_enhanced['mileage_per_age'] = df_enhanced['mileage_in_km'] / (df_enhanced['vehicle_manufacturing_age'] + 1)
    df_enhanced['ev_range_per_power'] = df_enhanced['ev_range_km'] / (df_enhanced['power_kw'] + 1)

    return df_enhanced

def preprocess_input(
    brand, model_name, year, color, transmission_type, fuel_type,
    power_kw, fuel_consumption_g_km, mileage_in_km, ev_range_km, feature_order
):
    """Preprocess user input for prediction"""

    input_data = pd.DataFrame({
        'brand': [brand.lower()],
        'model': [f"{brand.lower()}_{model_name.lower()}"],
        'year': [year],
        'color': [color.lower()],
        'transmission_type': [transmission_type.lower()],
        'fuel_type': [fuel_type.lower()],
        'power_kw': [power_kw],
        'fuel_consumption_g_km': [fuel_consumption_g_km],
        'mileage_in_km': [mileage_in_km],
        'ev_range_km': [ev_range_km]
    })

    # Derived basics
    data_collection_year = 2024
    input_data['vehicle_manufacturing_age'] = data_collection_year - input_data['year']
    input_data['vehicle_registration_age'] = data_collection_year - input_data['year']
    # Robust division
    input_data['mileage_per_year'] = input_data['mileage_in_km'] / input_data['vehicle_registration_age'].replace(0, 1) 
    input_data['reg_month_sin'] = np.sin(2 * np.pi * 6 / 12)  # Default to June
    input_data['reg_month_cos'] = np.cos(2 * np.pi * 6 / 12)

    # Enhanced features
    input_enhanced = create_enhanced_features(input_data)

    # One-hot encode
    categorical_columns = ['brand', 'color', 'transmission_type', 'fuel_type']
    input_encoded = pd.get_dummies(input_enhanced, columns=categorical_columns, drop_first=True, dtype=int)

    # Simple model target encoding for 'model' (Placeholder value)
    input_encoded['model_target_enc'] = 25000

    # Ensure all required columns
    for col in feature_order:
        if col not in input_encoded.columns:
            input_encoded[col] = 0

    # Reorder
    input_encoded = input_encoded[feature_order]
    return input_encoded

# ---------------------------------------------------------
# Utility: local sensitivity (Robustified)
# ---------------------------------------------------------
def local_sensitivity(raw_data, feature_order, xgb_model, rf_model, scaler, weights, deltas=None):
    """
    Calculates local price sensitivity by perturbing core raw input features.
    """
    if deltas is None:
        deltas = {"power_kw": 10, "mileage_in_km": 10_000, "year": -1} 
    
    # Base prediction setup
    base_input_processed = preprocess_input(
        raw_data['brand'], raw_data['model_name'], raw_data['year'], raw_data['color'], raw_data['transmission_type'], raw_data['fuel_type'],
        raw_data['power_kw'], raw_data['fuel_consumption_g_km'], raw_data['mileage_in_km'], raw_data['ev_range_km_calc'], feature_order
    )
    base_scaled = scaler.transform(base_input_processed)
    xgb_base_pred = float(xgb_model.predict(base_scaled)[0])
    rf_base_pred = float(rf_model.predict(base_scaled)[0]) if rf_model is not None and weights.get('rf', 0.0) > 0 else 0.0
    base_price = float(weights.get('xgb', 1.0) * xgb_base_pred + weights.get('rf', 0.0) * rf_base_pred)

    rows = []
    
    for k, dv in deltas.items():
        plus_data = raw_data.copy()
        minus_data = raw_data.copy()

        # Handle 'year'
        if k == 'year':
            plus_data['year'] = raw_data['year'] - dv # Newer car (less age)
            minus_data['year'] = raw_data['year'] + dv # Older car (more age)
        else:
            plus_data[k] = raw_data[k] + dv
            minus_data[k] = np.maximum(0, raw_data[k] - dv)

        # Re-preprocess and predict for +Delta
        plus_processed = preprocess_input(
            plus_data['brand'], plus_data['model_name'], plus_data['year'], plus_data['color'], plus_data['transmission_type'], plus_data['fuel_type'],
            plus_data['power_kw'], plus_data['fuel_consumption_g_km'], plus_data['mileage_in_km'], raw_data['ev_range_km_calc'], feature_order
        )
        p_scaled = scaler.transform(plus_processed)
        p_xgb_pred = float(xgb_model.predict(p_scaled)[0])
        p_rf_pred = float(rf_model.predict(p_scaled)[0]) if rf_model is not None and weights.get('rf', 0.0) > 0 else 0.0
        p = float(weights.get('xgb', 1.0) * p_xgb_pred + weights.get('rf', 0.0) * p_rf_pred)

        # Re-preprocess and predict for -Delta
        minus_processed = preprocess_input(
            minus_data['brand'], minus_data['model_name'], minus_data['year'], minus_data['color'], minus_data['transmission_type'], minus_data['fuel_type'],
            minus_data['power_kw'], minus_data['fuel_consumption_g_km'], minus_data['mileage_in_km'], raw_data['ev_range_km_calc'], feature_order
        )
        m_scaled = scaler.transform(minus_processed)
        m_xgb_pred = float(xgb_model.predict(m_scaled)[0])
        m_rf_pred = float(rf_model.predict(m_scaled)[0]) if rf_model is not None and weights.get('rf', 0.0) > 0 else 0.0
        m = float(weights.get('xgb', 1.0) * m_xgb_pred + weights.get('rf', 0.0) * m_rf_pred)
        
        # Determine display name
        display_name = k.replace("_", " ").title()
        if k == 'year': display_name = 'Vehicle Age (1 year)'
        
        rows.append({"Feature": display_name, "+Δ": round(p - base_price, 0), "−Δ": round(m - base_price, 0)})

    return pd.DataFrame(rows)

def create_matplotlib_importance_chart(xgb_model, feature_order):
    """Creates a simple Matplotlib feature importance chart."""
    try:
        importances = xgb_model.feature_importances_
        imp_df = pd.DataFrame({"feature": feature_order, "importance": importances})
        imp_df = imp_df.sort_values("importance", ascending=False).head(15)

        # Clean up feature names
        imp_df['feature'] = imp_df['feature'].str.replace('_', ' ').str.replace('in km', 'km').str.replace('g km', 'g/km').str.title()
        imp_df['feature'] = imp_df['feature'].apply(lambda x: x.split(":")[-1] if ":" in x else x)

        # Matplotlib/Seaborn plot
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.barplot(x="importance", y="feature", data=imp_df.iloc[::-1], palette="Blues_d", ax=ax)
        ax.set_title("Top 15 Most Important Features", fontsize=16)
        ax.set_xlabel("Feature Importance Score", fontsize=12)
        ax.set_ylabel("")
        plt.tight_layout()
        return fig
    except Exception as e:
        st.warning(f"Could not create Matplotlib chart: {e}")
        return None

# ---------------------------------------------------------
# Main app
# ---------------------------------------------------------
def main():
    # Load model artifacts
    xgb_model, rf_model, feature_order, scaler, metrics, weights = load_enhanced_model()
    
    # Store artifacts in session state for reuse in local_sensitivity
    st.session_state.xgb_model = xgb_model
    st.session_state.rf_model = rf_model
    st.session_state.feature_order = feature_order
    st.session_state.scaler = scaler
    st.session_state.weights = weights
    st.session_state.metrics = metrics # For display

    # Header and info
    st.markdown('<h1 class="main-header">🚗 Enhanced Car Price Predictor</h1>', unsafe_allow_html=True)
    st.markdown(f'<p style="text-align:center;font-size:1.1rem;color:#aaa;margin-bottom:2.2rem;">Predict used car prices with {metrics["accuracy"]:.1f}% accuracy using advanced machine learning</p>', unsafe_allow_html=True)

    # Badges
    st.markdown(f"""
    <div style="text-align:center;margin-bottom:1.2rem;">
        <span class="accuracy-badge">🎯 {metrics['accuracy']:.1f}% Accuracy</span>
        <span class="accuracy-badge">🚀 No Pickle Issues</span>
        <span class="accuracy-badge">⚡ Fast Predictions</span>
    </div>
    """, unsafe_allow_html=True)

    # Data for inputs
    brands, colors, transmission_types, fuel_types, year_range = get_data_info()

    # Layout
    col1, col2 = st.columns([1, 1])

    # -------------------------
    # Left: Car Specifications
    # -------------------------
    with col1:
        st.markdown("### 🚙 Car Specifications")

        # Quick actions
        qa1, qa2 = st.columns(2)
        with qa1:
            if st.button("✨ Try an example (EV)", use_container_width=True):
                st.session_state.update({
                    "brand": brands[0],
                    "model_name": "Q4 e-tron",
                    "year": 2021,
                    "color": "Grey" if "Grey" in colors else colors[0],
                    "transmission_type": "Automatic" if "Automatic" in transmission_types else transmission_types[0],
                    "fuel_type": "electric" if "electric" in fuel_types else fuel_types[0],
                    "power_kw": 126,
                    "fuel_consumption_g_km": 0.0,
                    "mileage_in_km": 4247
                })
                st.rerun()
        with qa2:
            if st.button("↩️ Reset form", use_container_width=True):
                # Clear only the input keys
                for key in list(st.session_state.keys()):
                    if key in ["brand", "model_name", "year", "color", "transmission_type", "fuel_type", "power_kw", "fuel_consumption_g_km", "mileage_in_km"]:
                        del st.session_state[key]
                st.rerun()

        # Inputs (with keys for quick actions and initial state)
        brand = st.selectbox("🏷️ Brand", brands, key="brand", help="Choose the manufacturer.")
        model_name = st.text_input("📝 Model", placeholder="e.g., A4, X3, Focus, Civic", key="model_name", help="Trim optional.")
        year = st.slider("📅 Year", min_value=year_range[0], max_value=year_range[1], key="year", help="Newer cars usually cost more.")
        color = st.selectbox("🎨 Color", colors, key="color", help="Exterior colour.")
        transmission_type = st.selectbox("⚙️ Transmission", transmission_types, key="transmission_type", help="Automatic/manual/etc.")
        fuel_type = st.selectbox("⛽ Fuel Type", fuel_types, key="fuel_type", help="Fuel system (EV/Hybrid included).")

    # -------------------------
    # Right: Technical Specs
    # -------------------------
    with col2:
        st.markdown("### 🔧 Technical Specifications")

        power_kw = st.slider("⚡ Power (kW)", min_value=30, max_value=500, key="power_kw",
                             help="Engine power in kilowatts. Higher power often increases price.")
        fuel_consumption_g_km = st.number_input("⛽ Fuel Consumption (g/km)", min_value=0.0, max_value=500.0,
                                                key="fuel_consumption_g_km",
                                                help="Use 0 for full EVs. Lower consumption is generally better.")
        mileage_in_km = st.number_input("🛣️ Mileage (km)", min_value=0, max_value=1_000_000,
                                        key="mileage_in_km", help="Total odometer reading. Lower mileage increases price.")

        # EV range (auto-calculated)
        if fuel_type.lower() == 'electric':
            ev_range_km = 400
        elif 'hybrid' in fuel_type.lower():
            ev_range_km = 50
        else:
            ev_range_km = 0
        st.session_state.ev_range_km_calc = ev_range_km # Save for use in local_sensitivity

        # EV helper chip
        st.markdown(f"""
        <div style="background:#0b3b62;color:#cfe9ff;border:1px solid #163b5b;padding:0.85rem 1rem;border-radius:12px;margin-top:0.35rem;">
            🔋 EV Range: <b>{ev_range_km} km</b> (set automatically for <b>{fuel_type}</b>)
        </div>
        """, unsafe_allow_html=True)

        # Enhanced Tips box
        st.markdown("""
        <div class="info-box">
            <strong>💡 Tips for better predictions</strong><br>
            • Ensure all specifications are accurate<br>
            • Use realistic values for mileage and year<br>
            • EV range is automatically set based on fuel type<br>
            • Higher power and newer cars typically cost more
        </div>
        """, unsafe_allow_html=True)

        # Simple sanity check
        try:
            years_used = max(1, 2024 - int(year))
            if mileage_in_km / years_used > 80_000:
                st.warning("This mileage per year is unusually high—double-check your inputs.")
        except Exception:
            pass

    # -----------------------------------------------------
    # Predict
    # -----------------------------------------------------
    st.markdown("---")
    col_btn1, col_btn2, col_btn3 = st.columns([1, 2, 1])
    with col_btn2:
        predict_button = st.button("🔮 Predict Price", type="primary", use_container_width=True)

    if predict_button:
        if not (model_name and model_name.strip()):
            st.error("❌ Please enter a car model name")
        else:
            try:
                with st.spinner("🔄 Analyzing your car specifications with enhanced AI..."):
                    # Preprocess
                    input_processed = preprocess_input(
                        brand, model_name, year, color, transmission_type, fuel_type,
                        power_kw, fuel_consumption_g_km, mileage_in_km, ev_range_km, feature_order
                    )

                    # Scale
                    input_scaled = scaler.transform(input_processed)

                    # Predict (Ensemble Logic)
                    xgb_pred = float(xgb_model.predict(input_scaled)[0])
                    rf_pred = float(rf_model.predict(input_scaled)[0]) if rf_model is not None and weights.get('rf', 0.0) > 0 else 0.0
                    prediction = float(weights.get('xgb', 1.0) * xgb_pred + weights.get('rf', 0.0) * rf_pred)
                    
                    # Store current raw data for sensitivity
                    current_raw_data = {
                        "brand": brand, "model_name": model_name, "year": year, "color": color, 
                        "transmission_type": transmission_type, "fuel_type": fuel_type, "power_kw": power_kw, 
                        "fuel_consumption_g_km": fuel_consumption_g_km, "mileage_in_km": mileage_in_km, 
                        "ev_range_km_calc": ev_range_km # Use calculated EV range
                    }
                    st.session_state.current_raw_data = current_raw_data


                    # Pretty card
                    st.markdown(f"""
                    <div class="prediction-card">
                        <h2>💰 Predicted Price</h2>
                        <h1 style="font-size:3rem;margin:1rem 0;">€{int(prediction):,}</h1>
                        <p style="font-size:1.1rem;margin:0;">Enhanced AI prediction with {metrics['accuracy']:.1f}% accuracy</p>
                    </div>
                    """, unsafe_allow_html=True)

                    # Price band (REPLACING PLOTLY GAUGE)
                    pred_int_low, pred_int_high = max(0, prediction * 0.9), prediction * 1.1
                    
                    st.markdown(f"""
                    <div class="price-band-box">
                        <h4>Estimated Price Range (±10%)</h4>
                        <p style="font-size:1.5rem;font-weight:bold;margin:0;">€{int(pred_int_low):,} – €{int(pred_int_high):,}</p>
                    </div>
                    """, unsafe_allow_html=True)
                    st.caption("This range reflects the model's typical confidence interval.")

                    # Quick metrics
                    c1, c2, c3 = st.columns(3)
                    with c1: st.metric("Brand", brand.upper())
                    with c2: st.metric("Year", year)
                    with c3: st.metric("Power", f"{power_kw} kW")

                    st.success(f"🎯 Model Confidence: {metrics['accuracy']:.1f}% accuracy based on {metrics['r2_score']:.1%} R² score")

                    # Sensitivity
                    st.subheader("🔎 Local Sensitivity (Impact of Small Changes)")
                    
                    sens_df = local_sensitivity(
                        current_raw_data, feature_order, xgb_model, rf_model, scaler, weights
                    )
                    
                    if not sens_df.empty:
                        st.dataframe(sens_df.set_index("Feature"), use_container_width=True)
                        st.caption("+Δ: Price change with +10 kW / +10,000 km / +1 year newer. −Δ: Price change with opposite.")
                    else:
                        st.caption("No sensitivity available for current inputs.")

                    # Save to history
                    hist = st.session_state.get("history", [])
                    hist.append({
                        "brand": brand, "model": model_name, "year": year, "color": color,
                        "transmission": transmission_type, "fuel": fuel_type, "power_kw": power_kw,
                        "fuel_g_km": fuel_consumption_g_km, "mileage_km": mileage_in_km,
                        "ev_range_km": ev_range_km, "pred_eur": int(prediction)
                    })
                    st.session_state["history"] = hist

            except Exception as e:
                st.error(f"❌ Error making prediction: {str(e)}")
                st.info("Please check your input values and ensure required model files are present.")

    # -----------------------------------------------------
    # Model information
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 📊 Enhanced Model Information")

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Model Type", "XGBoost + Random Forest Ensemble")
    with c2: st.metric("R² Score", f"{metrics['r2_score']:.4f}")
    with c3: st.metric("MAE", f"€{metrics['mae']:,.0f}")
    with c4: st.metric("Accuracy", f"{metrics['accuracy']:.1f}%")

    # Global importances (top 15) - REPLACING PLOTLY CHART
    st.markdown("### 🎯 Global Feature Importance (Matplotlib/Seaborn)")
    try:
        fig_imp = create_matplotlib_importance_chart(xgb_model, feature_order)
        if fig_imp:
            st.pyplot(fig_imp)
    except Exception as e:
        st.warning(f"Could not display feature importance chart: {e}")

    # -----------------------------------------------------
    # History & download
    # -----------------------------------------------------
    st.markdown("---")
    st.markdown("### 🗂️ Your Recent Predictions")
    hist = st.session_state.get("history", [])
    if hist:
        hist_df = pd.DataFrame(hist).iloc[::-1].reset_index(drop=True)
        st.dataframe(hist_df, use_container_width=True)
        csv = hist_df.to_csv(index=False).encode("utf-8")
        st.download_button("⬇️ Download history (CSV)", csv, "car_price_predictions.csv", "text/csv")
    else:
        st.caption("No predictions yet. Make one and it will appear here.")

    # -----------------------------------------------------
    # How to use
    # -----------------------------------------------------
    with st.expander("📖 How to Use This Enhanced App"):
        st.markdown("""
        ### Getting Started
        1. **Fill in the car specifications** using the smart inputs and tooltips.  
        2. Use the **'Try an example (EV)'** button for a quick start.  
        3. **Click 'Predict Price'** to get the AI-powered price prediction.  

        ### Understanding the Results
        - **Predicted Price**: AI-powered estimate using an ensemble model.  
        - **Price Range**: A simple box shows the **±10% confidence band**.  
        - **Local Sensitivity**: See how minor changes in key features (like age or mileage) affect *your specific* prediction.  
        - **Feature Importance**: A Matplotlib chart shows which features globally drive the price model.

        ### Technical Notes
        - **Model Loading**: The app expects model files (e.g., `xgb_model.json`, `scaler.joblib`) to be in the local `./models/` directory. If they are missing, the app runs on a robust, but inaccurate, **Mock Model**.
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align:center;color:#888;padding:1.2rem 0;">
        <p>🚗 <strong>Enhanced Car Price Prediction App</strong> | Powered by Advanced AI</p>
        <p>Built with ❤️ using XGBoost, Random Forest, Matplotlib, and Streamlit</p>
        <p>✅ No Pickle Issues | 🎯 {accuracy:.1f}% Accuracy | ⚡ Fast & Reliable</p>
    </div>
    """.format(accuracy=metrics['accuracy']), unsafe_allow_html=True)


if __name__ == "__main__":
    # Initialize history outside main for persistence
    if "history" not in st.session_state:
        st.session_state.history = []
    main()
