"""
Model utility functions for car price prediction
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score
from typing import Dict, Any, Tuple, List
import matplotlib.pyplot as plt
import seaborn as sns

def evaluate_model(y_true: np.ndarray, y_pred: np.ndarray, model_name: str = "Model") -> Dict[str, float]:
    """Evaluate model performance with multiple metrics"""
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    
    results = {
        "model": model_name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }
    
    return results

def print_model_results(results: Dict[str, float]) -> None:
    """Print model evaluation results in a formatted way"""
    print(f"{results['model']} Results:")
    print(f"MAE  : {results['MAE']:.2f}")
    print(f"RMSE : {results['RMSE']:.2f}")
    print(f"R²   : {results['R2']:.4f}")

def cross_validate_model(model, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> Dict[str, float]:
    """Perform cross-validation on a model"""
    # MAE scoring
    mae_scores = cross_val_score(model, X, y, cv=cv, scoring='neg_mean_absolute_error')
    mae_mean = -mae_scores.mean()
    mae_std = mae_scores.std()
    
    # R2 scoring
    r2_scores = cross_val_score(model, X, y, cv=cv, scoring='r2')
    r2_mean = r2_scores.mean()
    r2_std = r2_scores.std()
    
    return {
        "MAE_mean": mae_mean,
        "MAE_std": mae_std,
        "R2_mean": r2_mean,
        "R2_std": r2_std
    }

def plot_feature_importance(model, feature_names: List[str], top_n: int = 15, 
                          figsize: Tuple[int, int] = (10, 6)) -> None:
    """Plot feature importance from trained model"""
    if not hasattr(model, 'feature_importances_'):
        print("Model does not have feature_importances_ attribute")
        return
    
    importances = model.feature_importances_
    feature_importance = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    plt.figure(figsize=figsize)
    sns.barplot(x=feature_importance[:top_n], y=feature_importance.index[:top_n])
    plt.title(f"Top {top_n} Feature Importances")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.show()

def plot_predictions_vs_actual(y_true: np.ndarray, y_pred: np.ndarray, 
                              model_name: str = "Model", figsize: Tuple[int, int] = (8, 6)) -> None:
    """Plot predictions vs actual values"""
    plt.figure(figsize=figsize)
    plt.scatter(y_true, y_pred, alpha=0.5)
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2)
    plt.xlabel('Actual Price (€)')
    plt.ylabel('Predicted Price (€)')
    plt.title(f'{model_name}: Predictions vs Actual')
    plt.grid(True, alpha=0.3)
    plt.show()

def plot_residuals(y_true: np.ndarray, y_pred: np.ndarray, 
                  model_name: str = "Model", figsize: Tuple[int, int] = (8, 6)) -> None:
    """Plot residuals (prediction errors)"""
    residuals = y_true - y_pred
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Residuals vs predictions
    ax1.scatter(y_pred, residuals, alpha=0.5)
    ax1.axhline(y=0, color='r', linestyle='--')
    ax1.set_xlabel('Predicted Price (€)')
    ax1.set_ylabel('Residuals (€)')
    ax1.set_title(f'{model_name}: Residuals vs Predictions')
    ax1.grid(True, alpha=0.3)
    
    # Histogram of residuals
    ax2.hist(residuals, bins=50, alpha=0.7, edgecolor='black')
    ax2.set_xlabel('Residuals (€)')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'{model_name}: Distribution of Residuals')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def save_model_artifacts(model, preprocessor, feature_order: List[str], 
                        model_target_mapping: pd.DataFrame, artifacts_dir: str) -> None:
    """Save all model artifacts to disk"""
    artifacts_path = Path(artifacts_dir)
    artifacts_path.mkdir(exist_ok=True)
    
    # Save model
    joblib.dump(model, artifacts_path / "model.joblib")
    
    # Save preprocessor components
    if hasattr(preprocessor, 'named_transformers_'):
        if 'log_scale' in preprocessor.named_transformers_:
            log_pipeline = preprocessor.named_transformers_['log_scale']
            joblib.dump(log_pipeline.named_steps['scaler'], artifacts_path / "log_scaler.joblib")
            joblib.dump(log_pipeline.named_steps['log'], artifacts_path / "log_transformer.joblib")
        
        if 'direct_scale' in preprocessor.named_transformers_:
            direct_scaler = preprocessor.named_transformers_['direct_scale']
            joblib.dump(direct_scaler, artifacts_path / "direct_scaler.joblib")
    
    # Save feature order and mappings
    joblib.dump(feature_order, artifacts_path / "feature_order.joblib")
    model_target_mapping.to_csv(artifacts_path / "model_target_mapping.csv", index=False)
    
    print(f"Model artifacts saved to {artifacts_dir}")

def load_model_artifacts(artifacts_dir: str) -> Tuple[Any, Any, Any, List[str], pd.DataFrame]:
    """Load all model artifacts from disk"""
    artifacts_path = Path(artifacts_dir)
    
    # Load model
    model = joblib.load(artifacts_path / "model.joblib")
    
    # Load preprocessors
    log_scaler = joblib.load(artifacts_path / "log_scaler.joblib")
    direct_scaler = joblib.load(artifacts_path / "direct_scaler.joblib")
    log_transformer = joblib.load(artifacts_path / "log_transformer.joblib")
    
    # Load feature order and mappings
    feature_order = joblib.load(artifacts_path / "feature_order.joblib")
    model_target_mapping = pd.read_csv(artifacts_path / "model_target_mapping.csv")
    
    return model, log_scaler, direct_scaler, log_transformer, feature_order, model_target_mapping

def compare_models(results_list: List[Dict[str, float]]) -> pd.DataFrame:
    """Compare multiple models and return a comparison DataFrame"""
    comparison_df = pd.DataFrame(results_list)
    comparison_df = comparison_df.sort_values('MAE', ascending=True)
    return comparison_df

def get_model_summary(model) -> Dict[str, Any]:
    """Get a summary of model parameters and attributes"""
    summary = {
        "model_type": type(model).__name__,
        "parameters": model.get_params() if hasattr(model, 'get_params') else {}
    }
    
    if hasattr(model, 'n_estimators'):
        summary["n_estimators"] = model.n_estimators
    if hasattr(model, 'max_depth'):
        summary["max_depth"] = model.max_depth
    if hasattr(model, 'learning_rate'):
        summary["learning_rate"] = model.learning_rate
    
    return summary
