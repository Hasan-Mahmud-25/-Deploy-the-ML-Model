# ==============================
# all_in_one_clv_dashboard_final.py
# Streamlit CLV Dashboard with Tooltips
# ==============================

import streamlit as st
import pandas as pd
import plotly.express as px

# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(page_title="CLV Dashboard", layout="wide")
st.title("🚀 Customer Lifetime Value (CLV) Dashboard")
st.markdown("This dashboard helps business owners visualize predicted CLV, targets, segments, and top customers.")

# -----------------------------
# Load CSVs
# -----------------------------
clv_dashboard = pd.read_csv("building_a_predictive_model_for_customer_lifetime_value_CLV/clv_dashboard_full.csv")
top_vips = pd.read_csv("building_a_predictive_model_for_customer_lifetime_value_CLV/top_vips.csv")
growth_targets = pd.read_csv("building_a_predictive_model_for_customer_lifetime_value_CLV/growth_targets.csv")
segments = pd.read_csv("building_a_predictive_model_for_customer_lifetime_value_CLV/clv_segments_sample.csv")
all_features = pd.read_csv("building_a_predictive_model_for_customer_lifetime_value_CLV/all_models_top25_business_terms.csv")

# -----------------------------
# Clean column names
# -----------------------------
for df in [clv_dashboard, top_vips, growth_targets, segments, all_features]:
    df.columns = df.columns.str.strip()

# -----------------------------
# Merge segments
# -----------------------------
clv_dashboard = clv_dashboard.merge(
    segments[['user_id','segment']], on='user_id', how='left'
)
clv_dashboard['VIP'] = clv_dashboard['user_id'].isin(top_vips['user_id'])

# -----------------------------
# Sidebar filters
# -----------------------------
st.sidebar.header("Filters (for business users)")
st.sidebar.markdown("Select customer segments, VIP status, or CLV decile to filter the dashboard.")

segment_filter = st.sidebar.multiselect(
    "Select Segment",
    options=clv_dashboard['segment'].dropna().unique(),
    default=clv_dashboard['segment'].dropna().unique(),
    help="Filter customers by segment (VIP, Growth, At-Risk, Low Value)."
)
vip_filter = st.sidebar.selectbox(
    "VIP Users Only?", options=['All', True, False], index=0,
    help="Show all users, only VIPs, or only non-VIPs."
)
clv_decile_filter = st.sidebar.multiselect(
    "Select CLV Decile",
    options=sorted(clv_dashboard['clv_decile'].dropna().unique()),
    default=sorted(clv_dashboard['clv_decile'].dropna().unique()),
    help="Filter customers by CLV decile (1=lowest, 10=highest)."
)

# Apply filters
df_filtered = clv_dashboard[clv_dashboard['segment'].isin(segment_filter)]
if vip_filter != 'All':
    df_filtered = df_filtered[df_filtered['VIP'] == vip_filter]
df_filtered = df_filtered[df_filtered['clv_decile'].isin(clv_decile_filter)]

# -----------------------------
# KPI Cards
# -----------------------------
st.subheader("📊 Key Metrics")
col1, col2, col3 = st.columns(3)
total_pred_clv = df_filtered['predicted_clv_avg'].sum()
total_target = df_filtered['target_customer_value'].sum()
col1.metric("Total Predicted CLV", f"${total_pred_clv:,.0f}", help="Sum of predicted CLV for filtered customers.")
col2.metric("Total Target CLV", f"${total_target:,.0f}", help="Sum of target CLV for filtered customers.")
col3.metric("CLV vs Target", f"${total_pred_clv - total_target:,.0f}", help="Difference between predicted and target CLV.")

# -----------------------------
# Bar Chart: Predicted vs Target per segment
# -----------------------------
st.subheader("📊 Predicted vs Target CLV by Segment")
st.markdown("This chart shows the sum of predicted CLV vs target for each segment. Useful to see which segments are over/under-performing.")
bar_df = df_filtered.groupby('segment')[['predicted_clv_avg','target_customer_value']].sum().reset_index()
fig_bar = px.bar(
    bar_df,
    x='segment',
    y=['predicted_clv_avg','target_customer_value'],
    barmode='group',
    labels={'value':'CLV ($)', 'segment':'Segment'},
    title="Predicted CLV vs Target per Segment"
)
st.plotly_chart(fig_bar, use_container_width=True)

# -----------------------------
# Pie Chart: VIP vs Non-VIP
# -----------------------------
st.subheader("🥇 CLV Distribution: VIP vs Non-VIP")
st.markdown("Visualize what fraction of total predicted CLV comes from VIP customers vs others.")
fig_pie = px.pie(
    df_filtered,
    names='VIP',
    values='predicted_clv_avg',
    title="CLV Contribution by VIP Status"
)
st.plotly_chart(fig_pie, use_container_width=True)

# -----------------------------
# Top Features / Insights
# -----------------------------
st.subheader("🔑 Top Features Across Models")
st.markdown("Shows which features contribute most to predicted CLV in business-friendly terms.")
top_features_chart = all_features.groupby('business_metric')['importance'].mean().sort_values(ascending=True)
fig_feat = px.bar(
    top_features_chart,
    x='importance',
    y=top_features_chart.index,
    orientation='h',
    title="Average Feature Importance (Business Terms)"
)
st.plotly_chart(fig_feat, use_container_width=True)

# -----------------------------
# User-level Table
# -----------------------------
st.subheader("👤 User-level CLV Details")
st.markdown("Table shows individual customers with predicted CLV, target CLV, segment, VIP status, and CLV decile. Sort to identify top customers for action.")
st.dataframe(df_filtered[[
    'user_id','predicted_clv_avg','target_customer_value','segment','VIP','clv_decile'
]].sort_values(by='predicted_clv_avg', ascending=False))
