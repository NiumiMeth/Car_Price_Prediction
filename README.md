# 🚗 Car Price Prediction App

A simple, clean machine learning web application for predicting used car prices using XGBoost regression.

## 🎯 Features

- **AI-Powered Predictions**: Uses XGBoost for accurate price predictions
- **Simple Interface**: Clean Streamlit web interface
- **No EV Range Input**: Simplified user experience
- **Meaningful Charts**: Clear feature importance visualization
- **Real-time Validation**: Input validation and helpful error messages

## 📁 Simple Project Structure

```
Car_Price_Prediction/
├── app.py                      # Main Streamlit application
├── train_model.py              # Model training script
├── test_app.py                 # Test script
├── requirements.txt            # Python dependencies
├── data/
│   ├── raw/
│   │   └── car.csv             # Raw car data
│   └── processed/
│       ├── train_final.csv     # Processed training data
│       └── test_final.csv      # Processed test data
├── models/                     # Trained models (created after training)
└── notebooks/                  # Jupyter notebooks for data analysis
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Train the Model

```bash
python train_model.py
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Access the App

Open your browser and go to: `http://localhost:8501`

## 🔧 Usage

1. **Fill in car specifications** using the interactive form
2. **Click "Predict Price"** to get the AI-powered price prediction
3. **View model metrics** and feature importance charts
4. **Adjust parameters** to see how they affect the prediction

## 📊 Model Performance

- **R² Score**: 0.9351 (93.51% variance explained)
- **MAE**: €3,429 (Mean Absolute Error)
- **RMSE**: €7,186 (Root Mean Square Error)
- **Accuracy**: 87.4%

## 🛠️ Technical Details

### Model Architecture
- **Primary Model**: XGBoost Regressor
- **Feature Engineering**: 10+ meaningful features
- **Preprocessing**: RobustScaler for numerical features
- **Validation**: Train/test split with early stopping

### Key Features
- Engine Power (squared, log)
- Mileage (log)
- Vehicle Age (squared)
- Fuel Efficiency
- Interaction features (power×age, mileage×age)

## 🧪 Testing

Run the test suite to verify everything is working:

```bash
python test_app.py
```

## 📝 Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- xgboost
- streamlit
- matplotlib
- seaborn
- joblib

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.

## 🆘 Support

If you encounter any issues:

1. Check that all dependencies are installed
2. Ensure the model has been trained
3. Verify data files exist in the correct locations
4. Run the test script to diagnose issues

## 🎉 Acknowledgments

- Built with ❤️ using Streamlit and XGBoost
- Simple, clean design focused on user experience
- No complex dependencies or unnecessary features