"""
Streamlit app configuration
"""

# App settings
APP_TITLE = "🚗 Car Price Predictor"
APP_SUBTITLE = "Predict used car prices with advanced machine learning algorithms"
APP_ICON = "🚗"

# Color schemes
PRIMARY_COLOR = "#667eea"
SECONDARY_COLOR = "#764ba2"
SUCCESS_COLOR = "#4caf50"
WARNING_COLOR = "#ff9800"
ERROR_COLOR = "#f44336"
INFO_COLOR = "#2196f3"

# Layout settings
SIDEBAR_WIDTH = 300
MAIN_COLUMN_WIDTH = 600

# Default values
DEFAULT_BRAND = "audi"
DEFAULT_MODEL = "a4"
DEFAULT_YEAR = 2020
DEFAULT_COLOR = "black"
DEFAULT_TRANSMISSION = "Manual"
DEFAULT_FUEL_TYPE = "petrol"
DEFAULT_POWER = 150
DEFAULT_FUEL_CONSUMPTION = 150.0
DEFAULT_MILEAGE = 50000
DEFAULT_EV_RANGE = 0

# Validation ranges
MIN_YEAR = 1950
MAX_YEAR = 2024
MIN_POWER = 30
MAX_POWER = 1000
MIN_FUEL_CONSUMPTION = 0
MAX_FUEL_CONSUMPTION = 1000
MIN_MILEAGE = 0
MAX_MILEAGE = 2000000
MIN_EV_RANGE = 0
MAX_EV_RANGE = 1000

# Price categories
PRICE_CATEGORIES = {
    'budget': {'min': 0, 'max': 5000, 'emoji': '🟢', 'color': '#4caf50'},
    'economy': {'min': 5000, 'max': 15000, 'emoji': '🟡', 'color': '#ff9800'},
    'mid_range': {'min': 15000, 'max': 30000, 'emoji': '🟠', 'color': '#ff5722'},
    'premium': {'min': 30000, 'max': 60000, 'emoji': '🔴', 'color': '#f44336'},
    'luxury': {'min': 60000, 'max': float('inf'), 'emoji': '💎', 'color': '#9c27b0'}
}

# Model performance metrics
MODEL_METRICS = {
    'model_type': 'XGBoost Regressor',
    'r2_score': 0.9336,
    'mae': 3619,
    'rmse': 7330,
    'cv_score': 0.9284
}

# Feature importance settings
TOP_FEATURES = 10
FEATURE_CHART_HEIGHT = 400

# Help text
HELP_TEXTS = {
    'brand': 'Select the car brand from the dropdown',
    'model': 'Enter the car model name (e.g., A4, X3, Focus)',
    'year': 'Select the manufacturing year',
    'color': 'Select the car color',
    'transmission': 'Select the transmission type',
    'fuel_type': 'Select the fuel type',
    'power': 'Engine power in kilowatts (kW)',
    'fuel_consumption': 'Fuel consumption in grams per kilometer (g/km)',
    'mileage': 'Total mileage in kilometers (km)',
    'ev_range': 'Electric vehicle range in kilometers (km) - only for electric/hybrid vehicles'
}

# Tips for better predictions
PREDICTION_TIPS = [
    "Ensure all specifications are accurate",
    "Use realistic values for mileage and year",
    "For electric/hybrid vehicles, set appropriate EV range",
    "Higher power and newer cars typically cost more",
    "Lower mileage generally increases value",
    "Consider market trends and seasonal variations"
]
