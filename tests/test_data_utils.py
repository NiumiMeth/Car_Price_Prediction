"""
Tests for data utility functions
"""

import pytest
import pandas as pd
import numpy as np
from src.utils.data_utils import (
    clean_brand_name, clean_model_name, clean_year, 
    clean_price, clean_fuel_consumption, remove_outliers_iqr
)

def test_clean_brand_name():
    """Test brand name cleaning"""
    assert clean_brand_name("BMW") == "bmw"
    assert clean_brand_name("  Audi  ") == "audi"
    assert pd.isna(clean_brand_name(np.nan))

def test_clean_model_name():
    """Test model name cleaning"""
    assert clean_model_name("A4 Avant") == "a4_avant"
    assert clean_model_name("X3/SUV") == "x3"
    assert pd.isna(clean_model_name(np.nan))

def test_clean_year():
    """Test year cleaning"""
    assert clean_year(2020) == 2020
    assert clean_year("2020") == 2020
    assert clean_year("Car from 2020") == 2020
    assert pd.isna(clean_year(1800))  # Invalid year
    assert pd.isna(clean_year("invalid"))

def test_clean_price():
    """Test price cleaning"""
    assert clean_price(25000) == 25000.0
    assert clean_price("25000") == 25000.0
    assert pd.isna(clean_price("invalid"))

def test_clean_fuel_consumption():
    """Test fuel consumption cleaning"""
    assert clean_fuel_consumption("150 g/km") == 150.0
    assert clean_fuel_consumption("150,5 g/km") == 150.5
    assert pd.isna(clean_fuel_consumption("invalid"))

def test_remove_outliers_iqr():
    """Test outlier removal using IQR method"""
    # Create test data with outliers
    data = pd.DataFrame({
        'value': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100]  # 100 is an outlier
    })
    
    result = remove_outliers_iqr(data, 'value')
    assert len(result) < len(data)  # Should remove outliers
    assert result['value'].max() < 100  # Outlier should be removed
