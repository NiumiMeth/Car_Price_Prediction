"""
Data utility functions for car price prediction
"""

import pandas as pd
import numpy as np
import re
from typing import List, Dict, Any, Optional

def clean_brand_name(name: str) -> str:
    """Clean and standardize brand names"""
    if pd.isna(name):
        return np.nan
    return str(name).lower().strip()

def clean_model_name(name: str) -> str:
    """Clean and standardize model names"""
    if pd.isna(name):
        return np.nan
    
    name = str(name).lower().strip()
    name = name.split("/")[0]  # Remove everything after /
    name = re.sub(r"\s+", "_", name)  # Replace spaces with underscores
    return name

def clean_year(value: Any) -> Optional[int]:
    """Clean and validate year values"""
    if pd.isna(value):
        return np.nan
    
    val = str(value).strip()
    
    if val.isdigit():
        year = int(val)
        if 1950 <= year <= 2025:
            return year
    
    # Extract year from string
    match = re.search(r"(\d{4})", val)
    if match:
        year = int(match.group(1))
        if 1950 <= year <= 2025:
            return year
    
    return np.nan

def clean_price(price: Any) -> Optional[float]:
    """Clean and validate price values"""
    if pd.isna(price):
        return np.nan
    
    # Convert to numeric, coercing errors to NaN
    return pd.to_numeric(price, errors="coerce")

def clean_fuel_consumption(value: Any) -> Optional[float]:
    """Clean fuel consumption values"""
    if pd.isna(value):
        return np.nan
    
    try:
        val = str(value).lower().strip()
        # Extract numeric value
        num = re.findall(r"[\d,.]+", val)
        if num:
            return float(num[0].replace(",", "."))
    except:
        pass
    
    return np.nan

def remove_outliers_iqr(df: pd.DataFrame, column: str, factor: float = 1.5) -> pd.DataFrame:
    """Remove outliers using IQR method"""
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - factor * IQR
    upper_bound = Q3 + factor * IQR
    
    return df[(df[column] >= lower_bound) & (df[column] <= upper_bound)]

def remove_outliers_percentile(df: pd.DataFrame, column: str, upper_percentile: float = 0.99) -> pd.DataFrame:
    """Remove outliers using percentile method"""
    upper_cap = df[column].quantile(upper_percentile)
    return df[df[column] <= upper_cap]

def create_derived_features(df: pd.DataFrame, data_collection_year: int = 2024) -> pd.DataFrame:
    """Create derived features for the dataset"""
    df = df.copy()
    
    # Vehicle manufacturing age
    df['vehicle_manufacturing_age'] = data_collection_year - df['year'].astype(int)
    
    # Registration month and year (simplified)
    df['registration_month'] = 6  # Default to June
    df['registration_year'] = df['year']
    
    # Vehicle registration age
    df['vehicle_registration_age'] = data_collection_year - df['registration_year']
    
    # Cyclical encoding for registration month
    df['reg_month_sin'] = np.sin(2 * np.pi * df['registration_month'] / 12)
    df['reg_month_cos'] = np.cos(2 * np.pi * df['registration_month'] / 12)
    
    # Mileage per year
    df['mileage_per_year'] = df.apply(
        lambda row: row['mileage_in_km'] / row['vehicle_registration_age'] 
        if row['vehicle_registration_age'] > 0 else row['mileage_in_km'],
        axis=1
    )
    df['mileage_per_year'] = df['mileage_per_year'].round(2)
    
    return df

def get_feature_importance(model, feature_names: List[str], top_n: int = 15) -> pd.DataFrame:
    """Get feature importance from trained model"""
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
    else:
        # For models without feature_importances_ attribute
        return pd.DataFrame()
    
    feature_importance = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    return feature_importance.head(top_n)

def validate_input_data(data: Dict[str, Any]) -> Dict[str, Any]:
    """Validate input data for prediction"""
    required_fields = [
        'brand', 'model', 'year', 'color', 'transmission_type', 
        'fuel_type', 'power_kw', 'fuel_consumption_g_km', 'mileage_in_km'
    ]
    
    missing_fields = [field for field in required_fields if field not in data or pd.isna(data[field])]
    
    if missing_fields:
        raise ValueError(f"Missing required fields: {missing_fields}")
    
    # Validate year
    if not (1990 <= data['year'] <= 2024):
        raise ValueError("Year must be between 1990 and 2024")
    
    # Validate power
    if data['power_kw'] < 30 or data['power_kw'] > 1000:
        raise ValueError("Power must be between 30 and 1000 kW")
    
    # Validate mileage
    if data['mileage_in_km'] < 0 or data['mileage_in_km'] > 1000000:
        raise ValueError("Mileage must be between 0 and 1,000,000 km")
    
    return data
