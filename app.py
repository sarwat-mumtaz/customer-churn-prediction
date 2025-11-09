import streamlit as st
import pandas as pd
import joblib
import os

st.set_page_config(page_title="Churn Prediction", page_icon="🔮")

# Load model and scaler
if not os.path.exists('churn_model.pkl') or not os.path.exists('scaler.pkl'):
    st.error("❌ Model files not found!")
    st.stop()

try:
    model = joblib.load('churn_model.pkl')
    scaler = joblib.load('scaler.pkl')
except:
    st.error("❌ Error loading model")
    st.stop()

st.title("🔮 Will Customer Leave?")
st.write("Answer 3 simple questions")
st.write("")

# Just 3 questions
tenure = st.slider("How many months have they been a customer?", 0, 72, 12)
MonthlyCharges = st.slider("How much do they pay per month? ($)", 20, 120, 70)
Contract = st.radio("What's their contract?", ["Month-to-month", "1 Year", "2 Years"])

st.write("")

if st.button("🔍 Check Risk", use_container_width=True, type="primary"):
    try:
        # Map contract names
        contract_map = {"Month-to-month": "Month-to-month", "1 Year": "One year", "2 Years": "Two year"}
        mapped_contract = contract_map[Contract]
        
        TotalCharges = MonthlyCharges * tenure
        
        # Create data with defaults
        input_data = pd.DataFrame({
            'gender': [1], 'SeniorCitizen': [0], 'Partner': [0], 'Dependents': [0],
            'tenure': [tenure], 'PhoneService': [1], 'MultipleLines': ['No'],
            'InternetService': ['Fiber optic'], 'OnlineSecurity': ['No'],
            'OnlineBackup': ['No'], 'DeviceProtection': ['No'], 'TechSupport': ['No'],
            'StreamingTV': ['No'], 'StreamingMovies': ['No'], 'Contract': [mapped_contract],
            'PaperlessBilling': [1], 'PaymentMethod': ['Electronic check'],
            'MonthlyCharges': [MonthlyCharges], 'TotalCharges': [TotalCharges]
        })
        
        # One-hot encode
        input_data = pd.get_dummies(input_data, columns=[
            'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
            'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
            'Contract', 'PaymentMethod'
        ])
        
        # Match columns
        expected_cols = scaler.feature_names_in_
        for col in expected_cols:
            if col not in input_data.columns:
                input_data[col] = 0
        input_data = input_data[expected_cols]
        
        # Predict
        input_scaled = scaler.transform(input_data)
        prediction = model.predict(input_scaled)
        probability = model.predict_proba(input_scaled)[0, 1]
        
        st.write("")
        st.write("---")
        st.write("")
        
        # Show result
        if prediction[0] == 1:
            st.error("# ⚠️ HIGH RISK")
            st.write("### Customer likely to leave")
        else:
            st.success("# ✅ SAFE")
            st.write("### Customer likely to stay")
        
        st.write("")
        st.metric("Risk Level", f"{round(probability * 100)}%")
        st.progress(float(probability))
            
    except Exception as e:
        st.error(f"Error: {e}")