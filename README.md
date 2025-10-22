# 🚗 Car Price Prediction Project

A comprehensive machine learning project for predicting used car prices using various regression algorithms. This project includes data cleaning, feature engineering, model training, and a Streamlit web application for interactive price predictions.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Data](#data)
- [Models](#models)
- [Results](#results)
- [Streamlit App](#streamlit-app)
- [Contributing](#contributing)
- [License](#license)

## 🎯 Project Overview

This project aims to predict used car prices based on various features such as:
- Vehicle specifications (brand, model, year, power, fuel type)
- Physical characteristics (color, transmission type)
- Usage data (mileage, fuel consumption)
- Market factors (registration date, vehicle age)

The project uses advanced machine learning techniques including feature engineering, target encoding, and hyperparameter optimization to achieve high prediction accuracy.

## ✨ Features

- **Comprehensive Data Cleaning**: Handles missing values, outliers, and data inconsistencies
- **Advanced Feature Engineering**: Creates meaningful features like vehicle age, mileage per year, and cyclical encoding
- **Multiple ML Models**: Implements Decision Tree, Random Forest, and XGBoost regressors
- **Hyperparameter Optimization**: Uses RandomizedSearchCV for optimal model tuning
- **Interactive Web App**: Streamlit-based interface for easy price predictions
- **Modular Code Structure**: Well-organized codebase for maintainability

## 📁 Project Structure

```
Car_Price_Prediction/
├── data/
│   ├── raw/                    # Original dataset
│   └── processed/              # Cleaned and processed data
├── notebooks/
│   ├── 01_data_cleaning.ipynb  # Data cleaning and preprocessing
│   └── 02_model_training.ipynb # Model training and evaluation
├── src/
│   ├── streamlit/              # Streamlit web application
│   ├── models/                 # Model definitions and utilities
│   └── utils/                  # Helper functions and utilities
├── artifacts/                  # Trained models and preprocessors
├── docs/                       # Documentation
├── tests/                      # Unit tests
├── requirements.txt            # Python dependencies
├── .gitignore                  # Git ignore rules
└── README.md                   # Project documentation
```

## 🚀 Installation

### Prerequisites

- Python 3.8 or higher
- pip (Python package installer)

### Setup

1. **Clone the repository**
   ```bash
   git clone <your-repository-url>
   cd Car_Price_Prediction
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv venv
   
   # On Windows
   venv\Scripts\activate
   
   # On macOS/Linux
   source venv/bin/activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download the dataset**
   - Place your car dataset CSV file in `data/raw/`
   - Update the file path in the notebooks if needed

## 📊 Usage

### Running the Notebooks

1. **Data Cleaning and Preprocessing**
   ```bash
   jupyter notebook notebooks/01_data_cleaning.ipynb
   ```

2. **Model Training**
   ```bash
   jupyter notebook notebooks/02_model_training.ipynb
   ```

### Running the Streamlit App

```bash
streamlit run src/streamlit/app.py
```

The app will be available at `http://localhost:8501`

## 📈 Data

The project uses a comprehensive car dataset with the following features:

- **Basic Info**: Brand, Model, Year, Color
- **Technical Specs**: Power (kW), Transmission Type, Fuel Type
- **Usage Data**: Mileage, Fuel Consumption, EV Range
- **Market Info**: Price, Registration Date

### Data Processing Steps

1. **Data Cleaning**
   - Remove duplicates
   - Handle missing values
   - Clean and standardize text data
   - Remove outliers

2. **Feature Engineering**
   - Calculate vehicle age
   - Create mileage per year feature
   - Cyclical encoding for time features
   - Target encoding for high-cardinality categorical variables

3. **Data Preprocessing**
   - One-hot encoding for categorical variables
   - Log transformation for skewed features
   - Standard scaling for numerical features

## 🤖 Models

The project implements and compares multiple regression models:

### 1. Decision Tree Regressor
- **Parameters**: max_depth=20, min_samples_split=10
- **Performance**: MAE: 4351.59, R²: 0.8970

### 2. Random Forest Regressor
- **Parameters**: n_estimators=200, max_depth=30, min_samples_split=10
- **Performance**: MAE: 3527.89, R²: 0.9334

### 3. XGBoost Regressor (Best Model)
- **Optimized Parameters**: 
  - n_estimators: 800
  - learning_rate: 0.05
  - max_depth: 10
  - subsample: 1.0
  - colsample_bytree: 0.8
  - reg_alpha: 1
  - reg_lambda: 3
- **Performance**: MAE: 3619.21, R²: 0.9336

## 📊 Results

The XGBoost model achieved the best performance with:
- **Mean Absolute Error (MAE)**: €3,619
- **Root Mean Square Error (RMSE)**: €7,330
- **R² Score**: 0.9336

### Feature Importance

The most important features for price prediction are:
1. Vehicle manufacturing age
2. Power (kW)
3. Mileage per year
4. Brand (target encoded)
5. Fuel consumption

## 🌐 Streamlit App

The Streamlit application provides an interactive interface for:
- Input car specifications
- Real-time price predictions
- Model performance visualization
- Feature importance analysis

### App Features

- **User-friendly Interface**: Easy-to-use form for car specifications
- **Real-time Predictions**: Instant price estimates
- **Model Comparison**: Side-by-side comparison of different models
- **Data Visualization**: Interactive charts and graphs

## 🧪 Testing

Run the test suite:

```bash
pytest tests/
```

## 📝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Dataset source: [Your dataset source]
- Libraries: pandas, scikit-learn, xgboost, streamlit
- Inspiration: Used car market analysis

## 📞 Contact

- **Your Name** - [your.email@example.com]
- **Project Link**: [https://github.com/yourusername/Car_Price_Prediction](https://github.com/yourusername/Car_Price_Prediction)

---

⭐ If you found this project helpful, please give it a star!
