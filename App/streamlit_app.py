import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Load saved models, scaler, and weights
gb = joblib.load('gb_model.pkl')
xgb = joblib.load('xgb_model.pkl')
lgb = joblib.load('lgb_model.pkl')
scaler = joblib.load('scaler.pkl')
weights = joblib.load('ensemble_weights.pkl')
w_gb = weights['w_gb']
w_xgb = weights['w_xgb']
w_lgb = weights['w_lgb']

# Streamlit app title
st.title("Insurance Charges Prediction App")
st.write("Enter the details below to predict insurance charges using an ensemble of machine learning models.")

# Create input fields for user
st.header("User Input")
age = st.slider("Age", min_value=18, max_value=100, value=45)
bmi = st.slider("BMI", min_value=15.0, max_value=50.0, value=30.36, step=0.1)
children = st.slider("Number of Children", min_value=0, max_value=10, value=0)
sex = st.selectbox("Sex", options=["Male", "Female"])
smoker = st.selectbox("Smoker", options=["Yes", "No"])
region = st.selectbox("Region", options=["Northeast", "Northwest", "Southeast", "Southwest"])

# Convert inputs to model-compatible format
sex = 1 if sex == "Male" else 0
smoker = 1 if smoker == "Yes" else 0
region_northwest = 1 if region == "Northwest" else 0
region_southeast = 1 if region == "Southeast" else 0
region_southwest = 1 if region == "Southwest" else 0

# Create a dictionary for the new person
new_person = {
    'age': age,
    'bmi': bmi,
    'children': children,
    'sex': sex,
    'smoker': smoker,
    'region_northwest': region_northwest,
    'region_southeast': region_southeast,
    'region_southwest': region_southwest
}

# Convert to DataFrame and ensure correct column order
feature_columns = ['age', 'bmi', 'children', 'sex', 'smoker', 'region_northwest', 'region_southeast', 'region_southwest']
new_X = pd.DataFrame([new_person])
new_X = new_X[feature_columns]  # Reorder columns to match training data

# Predict button
if st.button("Predict"):
    # Make predictions with each model
    log_gb = gb.predict(new_X)
    log_xgb = xgb.predict(new_X)
    log_lgb = lgb.predict(new_X)

    # Combine predictions with ensemble weights
    log_ensemble = w_gb * log_gb + w_xgb * log_xgb + w_lgb * log_lgb

    # Convert back to original scale
    ensemble_prediction = np.exp(log_ensemble)[0]

    # Display result
    st.header("Prediction Result")
    st.write(f"Estimated Insurance Charges: **${ensemble_prediction:.2f}**")
