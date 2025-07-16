# smartbiz_utils.py

import pandas as pd
import plotly.express as px

def load_data(file):
    if file.name.endswith(".csv"):
        df = pd.read_csv(file)
    else:
        df = pd.read_excel(file)

    df.columns = df.columns.str.strip()
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df.dropna(subset=["Date"], inplace=True)
    df["Revenue"] = pd.to_numeric(df["Revenue"], errors="coerce")
    df["Profit"] = pd.to_numeric(df["Profit"], errors="coerce")
    df.dropna(subset=["Revenue", "Profit"], inplace=True)
    df = df[df["Revenue"] >= 0]
    return df

def generate_kpis(df):
    total_revenue = df["Revenue"].sum()
    total_profit = df["Profit"].sum()
    total_orders = len(df)
    return total_revenue, total_profit, total_orders

def plot_sales_trend(df):
    trend = df.groupby(df["Date"].dt.to_period("M")).agg({"Revenue": "sum"}).reset_index()
    trend["Date"] = trend["Date"].astype(str)
    fig = px.line(trend, x="Date", y="Revenue", markers=True, title="Monthly Revenue")
    return fig

def plot_top_customers(df):
    top = df.groupby("Customer")["Revenue"].sum().nlargest(5).reset_index()
    fig = px.bar(top, x="Customer", y="Revenue", title="Top 5 Customers")
    return fig

def plot_category_pie(df):
    cat = df.groupby("Category")["Revenue"].sum().reset_index()
    fig = px.pie(cat, names="Category", values="Revenue", title="Sales by Category", hole=0.4, width=700,height=500)
    return fig
