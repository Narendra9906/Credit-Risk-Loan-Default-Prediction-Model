import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle
import shap
import time
import datetime

# Set Streamlit page configuration
st.set_page_config(page_title="Credit Risk Prediction", layout="wide")

# --- 1. Load Model and Artifacts ---

@st.cache_resource
def load_model_and_artifacts():
    """
    Load the trained model, the feature engineering function,
    and the SHAP explainer.
    """
    try:
        model = xgb.XGBClassifier()
        model.load_model('loan_default_model.json')
        
        with open('loan_artifacts.pkl', 'rb') as f:
            artifacts = pickle.load(f)
            
        return model, artifacts
    except FileNotFoundError:
        return None, None

model, artifacts = load_model_and_artifacts()

if model is None or artifacts is None:
    st.error("Model or artifacts not found. Please run `preprocess.py` and `train_model.py` first.")
    st.stop()

# Extract artifacts
base_features = artifacts['base_features']
engineer_features_fn = artifacts['engineer_features_fn']
label_encoders = artifacts['label_encoders']
explainer = artifacts['explainer']
engineered_features_list = artifacts['engineered_features_list']

# --- 2. Streamlit UI Layout ---
st.title("💳 Credit Risk & Loan Default Prediction Model")
st.markdown("""
This dashboard delivers real-time risk scoring for new loan applications 
using an XGBoost model trained on Lending Club data.
""")

col1, col2 = st.columns([1, 1.5])

# --- 3. Input Form in the Sidebar ---
with st.sidebar:
    st.header("👤 Borrower Application Details")
    
    # Create input fields based on the *base features*
    # We use the cleaned data columns as our input guide
    
    inputs = {}
    
    inputs['loan_amnt'] = st.number_input("Loan Amount ($)", min_value=1000, max_value=50000, value=10000, step=500)
    inputs['term'] = st.selectbox("Loan Term (Months)", [36, 60])
    inputs['int_rate'] = st.slider("Interest Rate (%)", min_value=5.0, max_value=35.0, value=11.5, step=0.1)
    
    inputs['annual_inc'] = st.number_input("Annual Income ($)", min_value=10000, max_value=1000000, value=75000, step=1000)
    inputs['dti'] = st.slider("Debt-to-Income (DTI) Ratio", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
    
    inputs['fico_range_low'] = st.slider("FICO Score (Low)", min_value=600, max_value=850, value=690, step=1)
    inputs['fico_range_high'] = st.slider("FICO Score (High)", min_value=600, max_value=850, value=694, step=1)

    inputs['home_ownership'] = st.selectbox("Home Ownership", ['MORTGAGE', 'RENT', 'OWN', 'ANY'])
    inputs['emp_length'] = st.slider("Employment Length (Years)", min_value=0, max_value=10, value=5, step=1)
    
    inputs['purpose'] = st.selectbox("Loan Purpose", ['debt_consolidation', 'credit_card', 'home_improvement', 'other'])
    inputs['verification_status'] = st.selectbox("Verification Status", ['Verified', 'Source Verified', 'Not Verified'])
    
    inputs['earliest_cr_line'] = st.date_input("Earliest Credit Line Date", datetime.date(2005, 1, 1))
    
    inputs['open_acc'] = st.number_input("Open Accounts", min_value=0, max_value=50, value=10, step=1)
    inputs['total_acc'] = st.number_input("Total Accounts", min_value=0, max_value=100, value=25, step=1)
    inputs['revol_bal'] = st.number_input("Revolving Balance ($)", min_value=0, max_value=200000, value=15000, step=100)
    inputs['revol_util'] = st.slider("Revolving Utilization (%)", min_value=0.0, max_value=150.0, value=45.0, step=0.1)
    
    inputs['mort_acc'] = st.number_input("Number of Mortgage Accounts", min_value=0, max_value=10, value=1, step=1)
    
    # Add other required base features (with defaults)
    for col in base_features:
        if col not in inputs:
            inputs[col] = 0 # Use 0 or median as a simple default for unseen features in the form
            
    # Handle categorical defaults
    inputs['grade'] = 'B'
    inputs['sub_grade'] = 'B2'
    inputs['initial_list_status'] = 'w'
    inputs['application_type'] = 'Individual'

    predict_button = st.button("Calculate Risk Score", type="primary")

# --- 4. Prediction and SHAP Logic ---
if predict_button:
    # 1. Convert inputs to DataFrame
    input_df = pd.DataFrame([inputs])
    
    # 2. Pre-process inputs
    # Convert date
    input_df['earliest_cr_line'] = pd.to_datetime(input_df['earliest_cr_line'])
    
    # Apply label encoders
    for col, le in label_encoders.items():
        if col in input_df.columns:
            # Handle unseen labels by mapping them to a known label (e.g., the first one)
            input_df[col] = input_df[col].apply(lambda x: x if x in le.classes_ else le.classes_[0])
            input_df[col] = le.transform(input_df[col])

    # 3. Apply the *exact same* feature engineering
    start_time = time.time()
    input_df_engineered = engineer_features_fn(input_df)
    
    # 4. Ensure columns are in the correct order for XGBoost
    input_df_engineered = input_df_engineered[engineered_features_list]
    
    # 5. Make prediction
    pred_proba = model.predict_proba(input_df_engineered)[0]
    prediction = model.predict(input_df_engineered)[0]
    end_time = time.time()
    
    risk_score = pred_proba[1] # Probability of default
    
    # 6. Display results
    st.header("📊 Prediction Results")
    
    with col1:
        st.metric(label="Prediction Speed", value=f"{(end_time - start_time) * 1000:.0f} ms")
        
        if prediction == 0:
            st.success("Loan Status: **APPROVED** (Low Risk)")
        else:
            st.error("Loan Status: **DENIED** (High Risk)")
        
        st.subheader(f"Default Risk Score: {risk_score * 100:.2f}%")
        st.progress(risk_score)
        st.caption("This is the model's predicted probability that the borrower will default.")

    with col2:
        st.subheader("Key Risk Drivers (SHAP Analysis)")
        st.markdown("This chart shows *why* the model made this prediction.")
        
        with st.spinner("Calculating SHAP explanation..."):
            shap_value = explainer.shap_values(input_df_engineered.iloc[0])
            p = shap.force_plot(
                base_value=explainer.expected_value,
                shap_values=shap_value,
                features=input_df_engineered.iloc[0],
                feature_names=engineered_features_list,
                matplotlib=False,
                show=False
            )
            st.components.v1.html(p.data, height=200, scrolling=True)
        st.caption("Features pushing the score higher (red) increase the risk of default. Features pushing it lower (blue) decrease the risk.")

else:
    st.info("Please enter borrower details in the sidebar and click 'Calculate Risk Score'.")
