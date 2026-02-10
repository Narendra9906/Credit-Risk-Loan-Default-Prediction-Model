# app.py

import streamlit as st
import joblib
import pandas as pd

model = joblib.load("loan_model.pkl")

st.title("Loan Default Risk Predictor")

st.write("Enter applicant details:")

income = st.number_input("Annual Income")
loan_amount = st.number_input("Loan Amount")
credit_score = st.number_input("Credit Score")

if st.button("Predict Risk"):
    input_data = pd.DataFrame([[income, loan_amount, credit_score]],
                              columns=["income", "loan_amount", "credit_score"])

    prediction = model.predict(input_data)[0]

    if prediction == 1:
        st.error("High Risk Applicant")
    else:
        st.success("Low Risk Applicant")
