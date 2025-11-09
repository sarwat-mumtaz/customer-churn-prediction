# 📊 Customer Churn Prediction System

A machine learning-powered web application to predict customer churn for telecom companies. Built with Python, Scikit-learn, and Streamlit.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-1.0+-orange.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.0+-red.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## 🎯 Project Overview

Customer churn prediction is crucial for telecom companies to identify customers likely to leave and take proactive retention measures. This project uses machine learning to predict churn based on customer demographics, service usage, and billing information.

**Live Demo:** [Link to your Streamlit app if deployed]

## ✨ Features

- 🎨 Interactive web interface with Streamlit
- 🔮 Real-time churn probability predictions
- 📊 Model performance: **ROC-AUC Score of 0.8285**
- 📈 Feature importance analysis
- 🤖 Multiple ML algorithms comparison
- 💾 Trained model ready for deployment

## 🚀 Quick Start

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/customer-churn-prediction.git
cd customer-churn-prediction

# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Run the Streamlit app
streamlit run app.py
```

Open your browser and navigate to `http://localhost:8501`

## 📁 Project Structure
```
customer-churn-prediction/
│
├── data/
│   └── WA_Fn-UseC_-Telco-Customer-Churn.csv
│
├── models/
│   ├── churn_model.pkl
│   └── scaler.pkl
│
├── notebooks/
│   └── Train_model.ipynb
│
├── app.py
├── requirements.txt
├── README.md
└── .gitignore
```

## 📊 Dataset Information

**Telco Customer Churn Dataset**
- 7,043 customers
- 21 features
- Target: Churn (Yes/No)

### Key Features:
- **Demographics:** Gender, Senior Citizen, Partner, Dependents
- **Services:** Phone, Internet, Online Security, Tech Support, etc.
- **Account:** Contract type, Payment method, Tenure
- **Billing:** Monthly Charges, Total Charges

## 🤖 Machine Learning Pipeline

### Data Preprocessing
- Handled missing values
- Encoded categorical variables
- Standardized numerical features
- Train-test split (80-20)

### Models Trained & Performance

| Model | Accuracy | Notes |
|-------|----------|-------|
| Logistic Regression | 78.54% | Baseline model |
| Decision Tree | 71.71% | Prone to overfitting |
| **Random Forest** | **78.75%** | ✅ **Best performer** |

### Hyperparameter Tuning

Used GridSearchCV for Random Forest:

**Best Parameters:**
- `n_estimators`: 300
- `max_depth`: 10
- `min_samples_split`: 2

### Model Evaluation

- **ROC-AUC Score:** 0.8285
- **Precision (No Churn):** 0.83
- **Recall (No Churn):** 0.90
- **F1-Score:** 0.79

## 💻 Usage

### Web Application

1. Launch the app:
```bash
streamlit run app.py
```

2. Enter customer information in the sidebar
3. Click "Predict Churn"
4. View prediction and probability

### Python Script
```python
import joblib
import pandas as pd

# Load model and scaler
model = joblib.load('models/churn_model.pkl')
scaler = joblib.load('models/scaler.pkl')

# Prepare customer data
customer_data = {
    'gender': 1,
    'SeniorCitizen': 0,
    'Partner': 1,
    'tenure': 12,
    'MonthlyCharges': 65.5,
    # ... add all features
}

# Make prediction
df = pd.DataFrame([customer_data])
scaled_data = scaler.transform(df)
prediction = model.predict(scaled_data)
probability = model.predict_proba(scaled_data)

print(f"Churn: {prediction[0]}")
print(f"Probability: {probability[0][1]:.2%}")
```

## 📈 Top Features Influencing Churn

1. Contract Type
2. Tenure
3. Total Charges
4. Monthly Charges
5. Internet Service Type
6. Online Security
7. Tech Support
8. Payment Method
9. Paperless Billing
10. Multiple Lines

## 🛠️ Technologies Used

- **Python 3.8+**
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Scikit-learn** - ML algorithms
- **Matplotlib & Seaborn** - Visualization
- **Streamlit** - Web app framework
- **Joblib** - Model serialization

## 📦 Dependencies
```txt
pandas==1.5.3
numpy==1.24.3
scikit-learn==1.3.0
matplotlib==3.7.1
seaborn==0.12.2
streamlit==1.25.0
joblib==1.3.1
imbalanced-learn==0.11.0
```

## 🔮 Future Enhancements

- [ ] Implement SMOTE for class imbalance
- [ ] Add XGBoost and LightGBM models
- [ ] Deploy to cloud (AWS/Heroku/Streamlit Cloud)
- [ ] Create REST API with FastAPI
- [ ] Add batch prediction feature
- [ ] Implement MLOps pipeline
- [ ] Add model monitoring dashboard


---

⭐ **If you found this project helpful, please give it a star!** ⭐

Made with ❤️ and ☕
