"""
Data analysis utilities for the car price prediction app
"""

import pandas as pd
import numpy as np
from typing import Tuple, List, Dict, Any
import re

def clean_and_analyze_data(file_path: str) -> Dict[str, Any]:
    """
    Clean and analyze the raw car data to extract unique values for dropdowns
    """
    try:
        df = pd.read_csv(file_path)
        
        # Clean brand names
        brands = df['brand'].dropna().str.lower().str.strip().unique()
        brands = sorted([b for b in brands if isinstance(b, str) and len(b) > 0])
        
        # Clean color names
        colors = df['color'].dropna().str.lower().str.strip().unique()
        colors = sorted([c for c in colors if isinstance(c, str) and len(c) > 0])
        
        # Clean transmission types
        transmission_types = df['transmission_type'].dropna().str.strip().unique()
        transmission_types = sorted([t for t in transmission_types if isinstance(t, str) and len(t) > 0])
        
        # Clean fuel types - more robust cleaning
        fuel_types = df['fuel_type'].dropna().str.lower().str.strip().unique()
        clean_fuel_types = []
        
        for fuel in fuel_types:
            if isinstance(fuel, str):
                # Remove any non-alphabetic characters and extra spaces
                clean_fuel = re.sub(r'[^a-z\s]', '', fuel).strip()
                if clean_fuel in ['petrol', 'diesel', 'electric', 'hybrid', 'diesel hybrid', 
                                'lpg', 'cng', 'hydrogen', 'ethanol', 'gasoline']:
                    clean_fuel_types.append(clean_fuel)
        
        clean_fuel_types = sorted(list(set(clean_fuel_types)))
        
        # Clean year data
        years = df['year'].dropna()
        numeric_years = []
        for year in years:
            if isinstance(year, (int, float)) and not pd.isna(year):
                numeric_years.append(int(year))
            elif isinstance(year, str):
                # Try to extract year from string
                year_match = re.search(r'(\d{4})', str(year))
                if year_match:
                    year_val = int(year_match.group(1))
                    if 1950 <= year_val <= 2024:
                        numeric_years.append(year_val)
        
        year_range = (min(numeric_years), max(numeric_years)) if numeric_years else (1990, 2024)
        
        # Clean power data
        power_values = df['power_kw'].dropna()
        numeric_power = []
        for power in power_values:
            if isinstance(power, (int, float)) and not pd.isna(power):
                numeric_power.append(float(power))
            elif isinstance(power, str):
                # Try to extract numeric value
                power_match = re.search(r'(\d+(?:\.\d+)?)', str(power))
                if power_match:
                    numeric_power.append(float(power_match.group(1)))
        
        power_range = (min(numeric_power), max(numeric_power)) if numeric_power else (30, 500)
        
        # Clean mileage data
        mileage_values = df['mileage_in_km'].dropna()
        numeric_mileage = []
        for mileage in mileage_values:
            if isinstance(mileage, (int, float)) and not pd.isna(mileage):
                numeric_mileage.append(float(mileage))
            elif isinstance(mileage, str):
                # Try to extract numeric value
                mileage_match = re.search(r'(\d+(?:\.\d+)?)', str(mileage))
                if mileage_match:
                    numeric_mileage.append(float(mileage_match.group(1)))
        
        mileage_range = (min(numeric_mileage), max(numeric_mileage)) if numeric_mileage else (0, 1000000)
        
        return {
            'brands': brands,
            'colors': colors,
            'transmission_types': transmission_types,
            'fuel_types': clean_fuel_types,
            'year_range': year_range,
            'power_range': power_range,
            'mileage_range': mileage_range,
            'total_records': len(df),
            'clean_records': len(df.dropna())
        }
        
    except Exception as e:
        print(f"Error analyzing data: {e}")
        # Return default values
        return {
            'brands': ['audi', 'bmw', 'ford', 'hyundai', 'honda', 'kia', 'mercedes', 'toyota', 'volkswagen'],
            'colors': ['black', 'white', 'silver', 'grey', 'blue', 'red', 'green', 'brown', 'gold', 'orange'],
            'transmission_types': ['Manual', 'Automatic', 'Semi-automatic'],
            'fuel_types': ['petrol', 'diesel', 'electric', 'hybrid', 'diesel hybrid', 'lpg', 'cng', 'hydrogen', 'ethanol'],
            'year_range': (1990, 2024),
            'power_range': (30, 500),
            'mileage_range': (0, 1000000),
            'total_records': 0,
            'clean_records': 0
        }

def get_popular_models(brand: str, file_path: str, top_n: int = 10) -> List[str]:
    """
    Get popular models for a specific brand
    """
    try:
        df = pd.read_csv(file_path)
        brand_models = df[df['brand'].str.lower() == brand.lower()]['model'].dropna().unique()
        return sorted(brand_models)[:top_n]
    except:
        return []

def validate_input(brand: str, model: str, year: int, color: str, 
                  transmission_type: str, fuel_type: str, power_kw: float, 
                  fuel_consumption_g_km: float, mileage_in_km: float, 
                  ev_range_km: float) -> Tuple[bool, str]:
    """
    Validate user input for car specifications
    """
    errors = []
    
    if not brand or not brand.strip():
        errors.append("Brand is required")
    
    if not model or not model.strip():
        errors.append("Model is required")
    
    if year < 1950 or year > 2024:
        errors.append("Year must be between 1950 and 2024")
    
    if power_kw < 30 or power_kw > 1000:
        errors.append("Power must be between 30 and 1000 kW")
    
    if fuel_consumption_g_km < 0 or fuel_consumption_g_km > 1000:
        errors.append("Fuel consumption must be between 0 and 1000 g/km")
    
    if mileage_in_km < 0 or mileage_in_km > 2000000:
        errors.append("Mileage must be between 0 and 2,000,000 km")
    
    if ev_range_km < 0 or ev_range_km > 1000:
        errors.append("EV range must be between 0 and 1000 km")
    
    # Validate fuel type and EV range consistency
    if fuel_type in ['electric', 'hybrid', 'diesel hybrid'] and ev_range_km == 0:
        errors.append(f"EV range should be specified for {fuel_type} vehicles")
    
    if fuel_type not in ['electric', 'hybrid', 'diesel hybrid'] and ev_range_km > 0:
        errors.append(f"EV range should be 0 for {fuel_type} vehicles")
    
    if errors:
        return False, "; ".join(errors)
    
    return True, ""

def get_price_insights(prediction: float) -> Dict[str, Any]:
    """
    Get insights about the predicted price
    """
    if prediction < 5000:
        category = "Budget"
        emoji = "GREEN"
        description = "Affordable entry-level vehicle"
    elif prediction < 15000:
        category = "Economy"
        emoji = "YELLOW"
        description = "Good value for money"
    elif prediction < 30000:
        category = "Mid-range"
        emoji = "ORANGE"
        description = "Well-equipped mid-market vehicle"
    elif prediction < 60000:
        category = "Premium"
        emoji = "RED"
        description = "High-end vehicle with premium features"
    else:
        category = "Luxury"
        emoji = "DIAMOND"
        description = "Luxury vehicle with exceptional features"
    
    return {
        'category': category,
        'emoji': emoji,
        'description': description,
        'price_range': f"€{int(prediction * 0.8):,} - €{int(prediction * 1.2):,}"
    }
