# streamlit_walmart_dashboard.py
import streamlit as st
import pandas as pd
import joblib
import matplotlib.pyplot as plt

# ------------------------
# Load saved models
# ------------------------
xgb_model = joblib.load('xgb_walmart_model.pkl')
cat_model = joblib.load('catboost_walmart_model.pkl')
lgbm_model = joblib.load('lgbm_walmart_model.pkl')

# ------------------------
# Page config
# ------------------------
st.set_page_config(page_title="🛒 Walmart Weekly Sales Prediction", layout="wide")

# ------------------------
# Title
# ------------------------
st.title("🛒 Walmart Weekly Sales Prediction Dashboard")
st.markdown("Predict weekly sales for a given store, department, and week data with visual insights.")

# ------------------------
# Layout: Inputs left, graphs right
# ------------------------
input_col, graph_col = st.columns([1,2])

with input_col:
    st.header("Enter Store Details")
    
    # ---- Model Selection ----
    model_choice = st.selectbox("Select Model 🔧", ["XGBoost", "CatBoost", "LightGBM"], help="Choose the ML model for prediction")
    
    store = st.number_input("Store ID (?)", min_value=1, value=1, help="Unique identifier for the store")
    dept = st.number_input("Department ID (?)", min_value=1, value=1, help="Unique identifier for the department")
    store_type = st.selectbox("Store Type (?)", ["A","B","C"], help="Categorical type of the store")
    is_holiday = st.selectbox("Holiday? (?)", ["No","Yes"], help="Whether the current week is a holiday")
    size = st.number_input("Store Size (sq ft) (?)", value=150000, help="Physical size of the store in square feet")
    temperature = st.number_input("Temperature (°F) (?)", value=60.0, help="Average temperature during the week")
    fuel_price = st.number_input("Fuel Price ($) (?)", value=3.5, help="Fuel price during the week")
    CPI = st.number_input("Consumer Price Index (CPI) (?)", value=210.0, help="Economic indicator affecting sales")
    unemployment = st.number_input("Unemployment Rate (%) (?)", value=7.0, help="Local unemployment rate")
    
    st.subheader("Markdown Discounts")
    markdown1 = st.number_input("MarkDown1 ($) (?)", value=0.0, help="Promotional markdown 1")
    markdown2 = st.number_input("MarkDown2 ($) (?)", value=0.0, help="Promotional markdown 2")
    markdown3 = st.number_input("MarkDown3 ($) (?)", value=0.0, help="Promotional markdown 3")
    markdown4 = st.number_input("MarkDown4 ($) (?)", value=0.0, help="Promotional markdown 4")
    markdown5 = st.number_input("MarkDown5 ($) (?)", value=0.0, help="Promotional markdown 5")
    
    st.subheader("Date & Lag Features")
    year = st.number_input("Year (?)", min_value=2010, max_value=2025, value=2012, help="Year of prediction")
    month = st.number_input("Month (?)", min_value=1, max_value=12, value=1, help="Month of prediction")
    week = st.number_input("Week (?)", min_value=1, max_value=53, value=1, help="Week number of prediction")
    day = st.number_input("Day (?)", min_value=1, max_value=31, value=1, help="Day of the week")
    lag_1 = st.number_input("Previous Week Sales (Lag 1) (?)", value=20000.0, help="Sales from last week")
    lag_2 = st.number_input("2 Weeks Ago Sales (Lag 2) (?)", value=21000.0, help="Sales from 2 weeks ago")
    lag_3 = st.number_input("3 Weeks Ago Sales (Lag 3) (?)", value=22000.0, help="Sales from 3 weeks ago")
    rolling_mean_4 = st.number_input("4-Week Avg Sales (?)", value=21500.0, help="Rolling mean of past 4 weeks")
    rolling_std_4 = st.number_input("4-Week Sales Std Dev (?)", value=1500.0, help="Rolling standard deviation of past 4 weeks")

# ------------------------
# Prepare input
# ------------------------
holiday_val = 1 if is_holiday=="Yes" else 0
type_val = {'A':0,'B':1,'C':2}[store_type]

input_df = pd.DataFrame([{
    'Store': store,
    'Dept': dept,
    'Type': type_val,
    'IsHoliday': holiday_val,
    'Size': size,
    'Temperature': temperature,
    'Fuel_Price': fuel_price,
    'CPI': CPI,
    'Unemployment': unemployment,
    'MarkDown1': markdown1,
    'MarkDown2': markdown2,
    'MarkDown3': markdown3,
    'MarkDown4': markdown4,
    'MarkDown5': markdown5,
    'Year': year,
    'Month': month,
    'Week': week,
    'Day': day,
    'Lag_1': lag_1,
    'Lag_2': lag_2,
    'Lag_3': lag_3,
    'Rolling_Mean_4': rolling_mean_4,
    'Rolling_Std_4': rolling_std_4
}])

# ------------------------
# Prediction & Visuals
# ------------------------
with graph_col:
    if st.button("Predict Weekly Sales 💵"):
        # ---- Choose model based on selection ----
        if model_choice == "XGBoost":
            model = xgb_model
        elif model_choice == "CatBoost":
            model = cat_model
        else:
            model = lgbm_model
        
        pred = model.predict(input_df)[0]
        st.success(f"📈 Predicted Weekly Sales ({model_choice}): ${pred:,.2f}")
        
        # ---- 1️⃣ Recent Sales Trends ----
        st.subheader("Recent Sales Trends")
        fig1, ax1 = plt.subplots()
        ax1.bar(["Lag1","Lag2","Lag3","4-Week Avg"], [lag_1, lag_2, lag_3, rolling_mean_4], color='skyblue')
        ax1.set_ylabel("Sales ($)")
        ax1.set_title("Sales Comparison (Past Weeks)")
        for i, val in enumerate([lag_1, lag_2, lag_3, rolling_mean_4]):
            ax1.text(i, val + 200, f"${val:,.0f}", ha='center')
        st.pyplot(fig1)
        
        # ---- 2️⃣ Predicted vs Lag Trend ----
        st.subheader("Predicted vs Previous Sales")
        fig2, ax2 = plt.subplots()
        weeks = ["Lag3","Lag2","Lag1","Predicted"]
        sales = [lag_3, lag_2, lag_1, pred]
        ax2.plot(weeks, sales, marker='o', color='green')
        ax2.set_ylabel("Sales ($)")
        ax2.set_title("Trend Analysis")
        for i, val in enumerate(sales):
            ax2.text(i, val + 200, f"${val:,.0f}", ha='center')
        st.pyplot(fig2)
        
 