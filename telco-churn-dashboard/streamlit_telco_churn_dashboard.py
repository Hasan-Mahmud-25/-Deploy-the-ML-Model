# streamlit_telco_churn_app.py

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt

# -------------------------
# Load saved objects
# -------------------------
final_model = joblib.load("final_model.pkl")
scaler = joblib.load("scaler.pkl")
encoders = joblib.load("encoders.pkl")
X_train_columns = joblib.load("X_train_columns.pkl")  # Columns order used in training
num_cols = ['tenure', 'MonthlyCharges', 'TotalCharges', 'loyalty_score', 'lifetime_value']

# -------------------------
# Helper Functions
# -------------------------
def preprocess_input(input_df):
    input_df = input_df.copy()
    
    for col in encoders:
        if col in input_df.columns:
            input_df[col] = input_df[col].astype(str)
            valid_classes = encoders[col].classes_
            input_df[col] = input_df[col].apply(lambda x: x if x in valid_classes else valid_classes[0])
            input_df[col] = encoders[col].transform(input_df[col])
        else:
            input_df[col] = 0

    for col in X_train_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    input_df = input_df[X_train_columns]
    input_df[num_cols] = scaler.transform(input_df[num_cols])

    return input_df

def predict_churn(processed_df):
    return final_model.predict_proba(processed_df)[:,1]

# -------------------------
# Streamlit App Layout
# -------------------------
st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title("📊 Telco Customer Churn Dashboard")
st.markdown("---")

tab1, tab2, tab3 = st.tabs(["Single Customer Prediction", "Batch Prediction", "Executive Dashboard"])

# -------------------------
# Tab 1: Single Customer Prediction
# -------------------------
with tab1:
    st.header("Single Customer Input")
    
    with st.form("single_customer_form"):
        col1, col2 = st.columns(2)
        with col1:
            gender = st.selectbox("Gender", ["Male", "Female"])
            SeniorCitizen = st.selectbox("Senior Citizen", [0,1])
            Partner = st.selectbox("Partner", ["Yes","No"])
            Dependents = st.selectbox("Dependents", ["Yes","No"])
            tenure = st.number_input("Tenure (months)", min_value=0, value=12)
            PhoneService = st.selectbox("Phone Service", ["Yes","No"])
            MultipleLines = st.selectbox("Multiple Lines", ["Yes","No","No phone service"])
            InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        with col2:
            OnlineSecurity = st.selectbox("Online Security", ["Yes","No","No internet service"])
            OnlineBackup = st.selectbox("Online Backup", ["Yes","No","No internet service"])
            DeviceProtection = st.selectbox("Device Protection", ["Yes","No","No internet service"])
            TechSupport = st.selectbox("Tech Support", ["Yes","No","No internet service"])
            StreamingTV = st.selectbox("Streaming TV", ["Yes","No","No internet service"])
            StreamingMovies = st.selectbox("Streaming Movies", ["Yes","No","No internet service"])
            Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
            PaperlessBilling = st.selectbox("Paperless Billing", ["Yes","No"])
            PaymentMethod = st.selectbox("Payment Method", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
            MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=70.0)
            TotalCharges = st.number_input("Total Charges", min_value=0.0, value=850.0)
        
        loyalty_score = tenure / (TotalCharges + 1e-5)
        lifetime_value = MonthlyCharges * tenure

        submitted = st.form_submit_button("Predict Churn")
    
    if submitted:
        input_data = pd.DataFrame([{
            "gender": gender, "SeniorCitizen": SeniorCitizen, "Partner": Partner, "Dependents": Dependents,
            "tenure": tenure, "PhoneService": PhoneService, "MultipleLines": MultipleLines, "InternetService": InternetService,
            "OnlineSecurity": OnlineSecurity, "OnlineBackup": OnlineBackup, "DeviceProtection": DeviceProtection, "TechSupport": TechSupport,
            "StreamingTV": StreamingTV, "StreamingMovies": StreamingMovies, "Contract": Contract, "PaperlessBilling": PaperlessBilling,
            "PaymentMethod": PaymentMethod, "MonthlyCharges": MonthlyCharges, "TotalCharges": TotalCharges,
            "loyalty_score": loyalty_score, "lifetime_value": lifetime_value
        }])
        processed = preprocess_input(input_data)
        prob = predict_churn(processed)[0]

        st.metric("Churn Probability", f"{prob:.2%}", delta_color="inverse" if prob >= 0.5 else "normal")

# -------------------------
# Tab 2: Batch Prediction
# -------------------------
with tab2:
    st.header("Batch Customer Prediction")
    uploaded_file = st.file_uploader("Upload CSV for batch prediction", type="csv")
    
    batch_df = None
    if uploaded_file:
        batch_df = pd.read_csv(uploaded_file)
        batch_processed = preprocess_input(batch_df)
        churn_probs = predict_churn(batch_processed)
        batch_df['Churn_Prob'] = churn_probs
        batch_df['High_Risk'] = np.where(churn_probs >= 0.5, "Yes","No")
        
        # Styled table
        def highlight_risk(val):
            color = 'red' if val == "Yes" else 'green'
            return f'background-color: {color}; color: white; font-weight: bold;'
        
        st.dataframe(batch_df.style.applymap(highlight_risk, subset=['High_Risk']), height=400)
        st.download_button("Download Predictions", batch_df.to_csv(index=False), "churn_predictions.csv", "text/csv")

# -------------------------
# Tab 3: Executive Dashboard
# -------------------------
with tab3:
    st.header("Executive Dashboard")

    if batch_df is not None:
        st.subheader("Prediction Values and Risk")

        # Create a styled dataframe
        def highlight_risk(val):
            color = 'red' if val == 'Yes' else 'green'
            return f'background-color: {color}; color: white; font-weight: bold'

        styled_df = batch_df[['Churn_Prob', 'High_Risk']].style.applymap(highlight_risk, subset=['High_Risk'])
        st.dataframe(styled_df, height=400)

        st.subheader("Churn Probability Distribution")
        fig1, ax1 = plt.subplots(figsize=(8,4))
        batch_df['Churn_Prob'].hist(bins=20, ax=ax1, color='#1f77b4', edgecolor='black')
        ax1.set_xlabel("Churn Probability")
        ax1.set_ylabel("Number of Customers")
        ax1.set_title("Distribution of Churn Probability")
        st.pyplot(fig1, bbox_inches='tight')
        plt.clf()

        st.subheader("High-Risk vs Low-Risk Customers")
        fig2, ax2 = plt.subplots(figsize=(6,4))
        batch_df['High_Risk'].value_counts().plot(kind='bar', color=['#ff7f0e','#2ca02c'], ax=ax2, edgecolor='black')
        ax2.set_xlabel("Risk Category")
        ax2.set_ylabel("Number of Customers")
        ax2.set_title("High-Risk vs Low-Risk Customers")
        st.pyplot(fig2, bbox_inches='tight')
        plt.clf()
    else:
        st.info("Please upload a CSV file in the Batch Prediction tab to see Executive Dashboard visuals.")

