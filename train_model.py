"""
Train the Car Price Prediction Model - Simple Version
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def create_features(df):
    """Create features for prediction"""
    df_enhanced = df.copy()
    
    # Basic features
    df_enhanced['power_squared'] = df_enhanced['power_kw'] ** 2
    df_enhanced['power_log'] = np.log1p(df_enhanced['power_kw'])
    df_enhanced['mileage_log'] = np.log1p(df_enhanced['mileage_in_km'])
    df_enhanced['age_squared'] = df_enhanced['vehicle_manufacturing_age'] ** 2
    df_enhanced['fuel_efficiency'] = 1 / (df_enhanced['fuel_consumption_g_km'] + 1)
    
    # Interaction features
    df_enhanced['power_age_interaction'] = df_enhanced['power_kw'] * df_enhanced['vehicle_manufacturing_age']
    df_enhanced['mileage_age_interaction'] = df_enhanced['mileage_in_km'] * df_enhanced['vehicle_manufacturing_age']
    df_enhanced['power_mileage_interaction'] = df_enhanced['power_kw'] * df_enhanced['mileage_in_km']
    
    return df_enhanced

def train_model():
    """Train the car price prediction model"""
    
    print("🚗 Training Car Price Prediction Model")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📊 Loading data...")
    try:
        train_df = pd.read_csv("data/processed/train_final.csv")
        test_df = pd.read_csv("data/processed/test_final.csv")
    except FileNotFoundError as e:
        print(f"❌ Error: {e}")
        print("Please ensure the processed data files exist in data/processed/")
        return None
    
    # Apply feature engineering
    print("🔧 Creating features...")
    train_enhanced = create_features(train_df)
    test_enhanced = create_features(test_df)
    
    # Prepare features and target
    target_col = "price_in_euro"
    
    # Get all numeric columns for features
    numeric_cols = train_enhanced.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Remove any columns with too many missing values
    numeric_cols = [col for col in numeric_cols if train_enhanced[col].isnull().sum() < len(train_enhanced) * 0.5]
    
    # Prepare final datasets
    X_train = train_enhanced[numeric_cols].fillna(0)
    y_train = train_enhanced[target_col]
    X_test = test_enhanced[numeric_cols].fillna(0)
    y_test = test_enhanced[target_col]
    
    print(f"📈 Training data: {X_train.shape}, Test data: {X_test.shape}")
    print(f"🔢 Features: {len(numeric_cols)}")
    
    # Train XGBoost
    print("🤖 Training XGBoost model...")
    xgb_model = XGBRegressor(
        n_estimators=1000,
        max_depth=12,
        learning_rate=0.01,
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=1.0,
        reg_lambda=2.0,
        min_child_weight=5,
        gamma=0.2,
        random_state=42,
        n_jobs=-1
    )
    
    # Train with early stopping
    xgb_model.fit(
        X_train, y_train,
        eval_set=[(X_test, y_test)],
        verbose=False
    )
    
    # Make predictions
    xgb_pred = xgb_model.predict(X_test)
    
    # Calculate metrics
    r2 = r2_score(y_test, xgb_pred)
    mae = mean_absolute_error(y_test, xgb_pred)
    rmse = np.sqrt(mean_squared_error(y_test, xgb_pred))
    
    # Calculate accuracy percentage
    accuracy = (1 - abs(mae / y_test.mean())) * 100
    
    print(f"\n🎯 MODEL RESULTS:")
    print(f"📈 R² Score: {r2:.4f}")
    print(f"💰 MAE: €{mae:,.0f}")
    print(f"📊 RMSE: €{rmse:,.0f}")
    print(f"🎯 Accuracy: {accuracy:.1f}%")
    
    # Save model
    print("\n💾 Saving model...")
    
    # Save XGBoost model in native format
    xgb_model.save_model(str(models_dir / "xgb_model.json"))
    
    # Save feature information
    joblib.dump(numeric_cols, models_dir / "feature_order.joblib")
    
    # Save scaler
    scaler = RobustScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, models_dir / "scaler.joblib")
    
    # Save model metrics
    metrics = {
        'r2_score': r2,
        'mae': mae,
        'rmse': rmse,
        'accuracy': accuracy
    }
    joblib.dump(metrics, models_dir / "model_metrics.joblib")
    
    print("✅ Model training completed successfully!")
    print(f"📁 Models saved to: {models_dir}")
    print("🚀 Ready to run the app with: streamlit run app.py")
    
    return xgb_model, numeric_cols, scaler, metrics

if __name__ == "__main__":
    train_model()
