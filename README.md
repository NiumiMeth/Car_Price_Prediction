# 🚗 Car Price Prediction App

A machine learning-powered web application for predicting used car prices using XGBoost regression.

## 🚀 Quick Start

1. **Install Dependencies**
   ```bash
   pip install pandas numpy scikit-learn xgboost plotly streamlit joblib
   ```

2. **Train the Model** (if not already done)
   ```bash
   python train_model.py
   ```

3. **Run the Application**
   ```bash
   streamlit run car_price_app.py
   ```

4. **Access the App**
   Open your browser and go to: `http://localhost:8501`

## 📊 Model Performance

- **R² Score**: 0.9254 (92.54% accuracy)
- **MAE**: €3,902 (Mean Absolute Error)
- **RMSE**: €7,705 (Root Mean Square Error)

## 🎯 Features

- **Real Predictions**: Uses trained XGBoost model (not demo prices)
- **Interactive UI**: Beautiful, responsive interface
- **Comprehensive Input**: Brand, model, year, color, transmission, fuel type, power, consumption, mileage, EV range
- **Model Metrics**: Display of model performance
- **Data Validation**: Input validation and error handling

## 📁 Project Structure

```
Car_Price_Prediction/
├── car_price_app.py          # Main Streamlit application
├── train_model.py            # Model training script
├── config.py                 # Configuration file
├── data/
│   ├── raw/
│   │   └── car.csv          # Raw dataset
│   └── processed/
│       ├── train_final.csv  # Training data
│       └── test_final.csv   # Test data
├── artifacts/               # Model artifacts
│   ├── model.joblib         # Trained model
│   ├── scaler.joblib        # Data scaler
│   ├── feature_order.joblib # Feature order
│   └── model_metrics.joblib # Performance metrics
└── notebooks/               # Jupyter notebooks
    ├── 01_data_cleaning.ipynb
    └── 02_model_training.ipynb
```

## 🔧 How to Use

1. **Fill in Car Specifications**:
   - Select brand, model, year, color
   - Choose transmission and fuel type
   - Enter power, fuel consumption, mileage
   - Set EV range (for electric/hybrid vehicles)

2. **Get Prediction**:
   - Click "🔮 Predict Price" button
   - View the predicted price in euros
   - See additional car information

3. **Understand Results**:
   - Predicted price is based on real ML model
   - Model performance metrics are displayed
   - Tips for better predictions are provided

## 🎨 UI Features

- **Gradient Design**: Modern, attractive interface
- **Responsive Layout**: Works on different screen sizes
- **Interactive Elements**: Sliders, dropdowns, number inputs
- **Real-time Validation**: Input validation and error messages
- **Performance Metrics**: Model accuracy and error metrics

## 📈 Model Details

- **Algorithm**: XGBoost Regressor
- **Features**: 61 engineered features including:
  - Categorical: brand, model, color, transmission, fuel type
  - Numerical: year, power, consumption, mileage, EV range
  - Derived: vehicle age, mileage per year, seasonal features
- **Preprocessing**: Standard scaling, one-hot encoding
- **Training**: 79,566 samples, 19,892 test samples

## 🛠️ Technical Stack

- **Frontend**: Streamlit
- **ML**: XGBoost, scikit-learn
- **Data**: Pandas, NumPy
- **Visualization**: Plotly
- **Deployment**: Local Streamlit server

## 📝 Notes

- The app uses real trained models, not demo predictions
- All predictions are based on the actual dataset and trained model
- Model artifacts are automatically loaded when the app starts
- Input validation ensures realistic predictions

## 🚀 Future Enhancements

- Add more car brands and models
- Implement feature importance visualization
- Add price range confidence intervals
- Include market trend analysis
- Add data export functionality