# app.py

import logging
logging.getLogger("streamlit").setLevel(logging.ERROR)  # Suppress that annoying warning

import streamlit as st
import pandas as pd
import smartbiz_utils as utils  # Make sure smartbiz_utils.py is in the same folder

st.set_page_config(page_title="SmartBiz Dashboard", layout="wide")
st.title("📊 SmartBiz Dashboard")

uploaded_file = st.file_uploader("Upload your CSV or Excel business dataset", type=["csv", "xlsx"])

if uploaded_file:
    df = utils.load_data(uploaded_file)

    st.sidebar.header("🔍 Filter Data")
    start_date = st.sidebar.date_input("Start Date", df["Date"].min())
    end_date = st.sidebar.date_input("End Date", df["Date"].max())
    region = st.sidebar.multiselect("Select Region", df["Region"].unique(), default=df["Region"].unique())

    df = df[(df["Date"] >= pd.to_datetime(start_date)) & (df["Date"] <= pd.to_datetime(end_date))]
    df = df[df["Region"].isin(region)]

    # KPIs
    kpi1, kpi2, kpi3 = utils.generate_kpis(df)
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Revenue", f"${kpi1:,.2f}")
    col2.metric("Total Profit", f"${kpi2:,.2f}")
    col3.metric("Total Orders", kpi3)

    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("📈 Monthly Revenue Trend")
        fig = utils.plot_sales_trend(df)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏆 Top Customers")
        fig = utils.plot_top_customers(df)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📦 Sales by Product Category")
    fig = utils.plot_category_pie(df)
    st.plotly_chart(fig, use_container_width=True)

    st.download_button("📥 Download Cleaned Data", data=df.to_csv(index=False), file_name="cleaned_data.csv", mime="text/csv")

else:
    st.info("👆 Please upload a CSV or Excel file to begin.")
