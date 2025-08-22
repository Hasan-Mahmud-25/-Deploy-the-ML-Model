import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------
# Load models
# --------------------------
# Get the directory where this script lives
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Build absolute paths to each model
cat_model_path = os.path.join(BASE_DIR, "../models/catboost_churn_model.pkl")
xgb_model_path = os.path.join(BASE_DIR, "../models/xgb_churn_model.pkl")
lgb_model_path = os.path.join(BASE_DIR, "../models/lgb_churn_model.pkl")

# Load models
cat_model = joblib.load(cat_model_path)
xgb_model = joblib.load(xgb_model_path)
lgb_model = joblib.load(lgb_model_path)

# --------------------------
# Features used in training (exclude user_id)
# --------------------------
features = [
    'purchase_number_max','inter_session_time_mean','session_count_ratio',
    'customer_value_month_lag1','purchase_number_sum','inter_purchase_time_mean',
    'purchase_number_mean','purchase_recency_mean','session_recency_max',
    'purchase_revenue_month_lag1','session_recency_mean','customer_value_month_lag2',
    'customer_value_month_lag3','purchase_revenue_month_lag2','purchase_revenue_sum'
]

# --------------------------
# UI Title
# --------------------------
st.title("🎯 E-Commerce Customer Churn Prediction")
st.markdown("Predict churn probability for an individual customer and visualize top drivers.")

# --------------------------
# Sidebar: User Input
# --------------------------
st.sidebar.header("Customer Input Features")
input_data = {}
for feature in features:
    input_data[feature] = st.sidebar.number_input(feature, value=0.0)

# --------------------------
# Prepare DataFrame
# --------------------------
input_df = pd.DataFrame([input_data])


# --------------------------
# Predictions with explanations
# --------------------------
st.subheader("🔮 Predicted Churn Probability per Model")
predictions = {}
for name, model in models.items():
    prob = model.predict_proba(input_df)[:, 1][0]
    predictions[name] = prob
    
    # Display metric
    st.metric(label=f"{name} Prediction", value=f"{prob:.4f}")
    
    # Add explanation / tooltip-like text
    st.markdown(f"<small>ℹ️ Probability of churn: {prob:.1%} → Higher values indicate higher risk of this customer leaving.</small>", unsafe_allow_html=True)

# --------------------------
# Business Insights
# --------------------------
st.subheader("💡 Business Insights")
st.markdown(f"""
- Low cart activity, high time since last session, low purchase revenue → higher risk.
- Recommended action: send a personalized email, offer, or promotion to retain customer.
""")

# --------------------------
# Feature Importance Plots
# --------------------------
st.subheader("📊 Feature Importance Comparison")
fig, axes = plt.subplots(1, 3, figsize=(18, 6))

for i, (name, model) in enumerate(models.items()):
    if hasattr(model, "get_feature_importance"):  # CatBoost
        imp = model.get_feature_importance()
    else:  # XGBoost / LightGBM
        imp = model.feature_importances_
    
    importance_df = pd.DataFrame({
        "feature": features,
        "importance": imp
    }).sort_values(by="importance", ascending=True)
    
    axes[i].barh(importance_df["feature"], importance_df["importance"], color=sns.color_palette("Set2"))
    axes[i].set_title(name)
    axes[i].tick_params(axis='y', labelsize=10)

plt.tight_layout()
st.pyplot(fig)

