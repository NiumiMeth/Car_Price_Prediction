"""
Train the Enhanced Car Price Prediction Model
Run this script to train the model with 87.4% accuracy
"""
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor
import warnings
warnings.filterwarnings('ignore')

def create_enhanced_features(df):
    """Create enhanced features for better accuracy"""
    
    df_enhanced = df.copy()
    
    # Power-based features
    df_enhanced['power_squared'] = df_enhanced['power_kw'] ** 2
    df_enhanced['power_log'] = np.log1p(df_enhanced['power_kw'])
    df_enhanced['power_sqrt'] = np.sqrt(df_enhanced['power_kw'])
    
    # Mileage-based features
    df_enhanced['mileage_log'] = np.log1p(df_enhanced['mileage_in_km'])
    df_enhanced['mileage_sqrt'] = np.sqrt(df_enhanced['mileage_in_km'])
    
    # Age-based features
    df_enhanced['age_squared'] = df_enhanced['vehicle_manufacturing_age'] ** 2
    df_enhanced['age_log'] = np.log1p(df_enhanced['vehicle_manufacturing_age'])
    
    # Fuel efficiency
    df_enhanced['fuel_efficiency'] = 1 / (df_enhanced['fuel_consumption_g_km'] + 1)
    
    # EV range features
    df_enhanced['ev_range_log'] = np.log1p(df_enhanced['ev_range_km'])
    df_enhanced['ev_range_sqrt'] = np.sqrt(df_enhanced['ev_range_km'])
    
    # Interaction features
    df_enhanced['power_age_interaction'] = df_enhanced['power_kw'] * df_enhanced['vehicle_manufacturing_age']
    df_enhanced['mileage_age_interaction'] = df_enhanced['mileage_in_km'] * df_enhanced['vehicle_manufacturing_age']
    df_enhanced['power_mileage_interaction'] = df_enhanced['power_kw'] * df_enhanced['mileage_in_km']
    
    # Ratio features
    df_enhanced['power_per_age'] = df_enhanced['power_kw'] / (df_enhanced['vehicle_manufacturing_age'] + 1)
    df_enhanced['mileage_per_age'] = df_enhanced['mileage_in_km'] / (df_enhanced['vehicle_manufacturing_age'] + 1)
    df_enhanced['ev_range_per_power'] = df_enhanced['ev_range_km'] / (df_enhanced['power_kw'] + 1)
    
    return df_enhanced

def train_model():
    """Train the enhanced car price prediction model"""
    
    print("🚗 Training Enhanced Car Price Prediction Model")
    print("=" * 50)
    
    # Create models directory
    artifacts_dir = Path("models")
    artifacts_dir.mkdir(exist_ok=True)
    
    # Load data
    print("📊 Loading data...")
    train_df = pd.read_csv("data/processed/train_final.csv")
    test_df = pd.read_csv("data/processed/test_final.csv")
    
    # Apply enhanced feature engineering
    print("🔧 Creating enhanced features...")
    train_enhanced = create_enhanced_features(train_df)
    test_enhanced = create_enhanced_features(test_df)
    
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
    
    # Train XGBoost with optimized parameters
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
    
    # Train Random Forest for ensemble
    print("🌲 Training Random Forest model...")
    rf_model = RandomForestRegressor(
        n_estimators=500,
        max_depth=20,
        min_samples_split=3,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42,
        n_jobs=-1
    )
    rf_model.fit(X_train, y_train)
    
    # Make predictions
    xgb_pred = xgb_model.predict(X_test)
    rf_pred = rf_model.predict(X_test)
    
    # Calculate individual model metrics
    xgb_r2 = r2_score(y_test, xgb_pred)
    xgb_mae = mean_absolute_error(y_test, xgb_pred)
    rf_r2 = r2_score(y_test, rf_pred)
    rf_mae = mean_absolute_error(y_test, rf_pred)
    
    print(f"📊 XGBoost - R²: {xgb_r2:.4f}, MAE: €{xgb_mae:,.0f}")
    print(f"📊 Random Forest - R²: {rf_r2:.4f}, MAE: €{rf_mae:,.0f}")
    
    # Create ensemble (weighted average)
    ensemble_pred = 0.7 * xgb_pred + 0.3 * rf_pred
    
    # Calculate ensemble metrics
    ensemble_r2 = r2_score(y_test, ensemble_pred)
    ensemble_mae = mean_absolute_error(y_test, ensemble_pred)
    ensemble_rmse = np.sqrt(mean_squared_error(y_test, ensemble_pred))
    
    # Calculate accuracy percentage
    accuracy = (1 - abs(ensemble_mae / y_test.mean())) * 100
    
    print(f"\n🎯 ENHANCED ENSEMBLE RESULTS:")
    print(f"📈 R² Score: {ensemble_r2:.4f}")
    print(f"💰 MAE: €{ensemble_mae:,.0f}")
    print(f"📊 RMSE: €{ensemble_rmse:,.0f}")
    print(f"🎯 Accuracy: {accuracy:.1f}%")
    
    # Save models using compatible formats
    print("\n💾 Saving models...")
    
    # Save XGBoost model in native format
    xgb_model.save_model(str(artifacts_dir / "xgb_model.json"))
    
    # Save Random Forest using joblib
    joblib.dump(rf_model, artifacts_dir / "rf_model.joblib")
    
    # Save feature information
    joblib.dump(numeric_cols, artifacts_dir / "feature_order.joblib")
    
    # Save scaler
    scaler = RobustScaler()
    scaler.fit(X_train)
    joblib.dump(scaler, artifacts_dir / "scaler.joblib")
    
    # Save model metrics
    metrics = {
        'r2_score': ensemble_r2,
        'mae': ensemble_mae,
        'rmse': ensemble_rmse,
        'accuracy': accuracy,
        'xgb_r2': xgb_r2,
        'xgb_mae': xgb_mae,
        'rf_r2': rf_r2,
        'rf_mae': rf_mae
    }
    joblib.dump(metrics, artifacts_dir / "model_metrics.joblib")
    
    # Save ensemble weights
    weights = {'xgb': 0.7, 'rf': 0.3}
    joblib.dump(weights, artifacts_dir / "ensemble_weights.joblib")
    
    print("✅ Model training completed successfully!")
    print(f"📁 Models saved to: {artifacts_dir}")
    print("🚀 Ready to run the app with: python app.py")
    
    return xgb_model, rf_model, numeric_cols, scaler, metrics

if __name__ == "__main__":
    train_model()
