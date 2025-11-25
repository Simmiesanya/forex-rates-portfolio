import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

st.title("FX Rates Dashboard")

# DB connection (same as before)
conn_str = st.secrets["POSTGRES_CONN_STR"]
engine = create_engine(conn_str)

# Fetch full data for analyses
@st.cache_data(ttl=3600)  # Cache for 1 hour to reduce DB hits
def load_data():
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, currency, buying_rate, central_rate, selling_rate
            FROM fx_vault.fx_rates_daily
            ORDER BY date DESC, currency
        """), conn)
    df['date'] = pd.to_datetime(df['date'])  # Ensure date is datetime
    return df

df = load_data()

# Sidebar for user inputs
st.sidebar.header("Analysis Options")
currencies = sorted(df['currency'].unique())
selected_currencies = st.sidebar.multiselect("Select Currencies", currencies, default=['USD', 'EUR', 'GBP'])
time_window = st.sidebar.slider("Time Window for Trends (Days)", 7, 365, 30)

# Filter data based on selections for trends
latest_date = df['date'].max()
df_trends = df[(df['date'] >= latest_date - pd.Timedelta(days=time_window)) & (df['currency'].isin(selected_currencies))]
df_trends = df_trends.sort_values(['currency', 'date'])

# 1. Latest Date Table: All rows for max date, no empty rate columns, formatted date, beautified
st.subheader("Rates for Latest Available Date")
with engine.connect() as conn:
    max_date_df = pd.read_sql(text("""
        SELECT date, currency, central_rate FROM fx_vault.fx_rates_daily
        WHERE date = (SELECT MAX(date) FROM fx_vault.fx_rates_daily)
    """), conn)

max_date_df['date'] = pd.to_datetime(max_date_df['date']).dt.strftime('%Y-%m-%d')  # Format date as YYYY-MM-DD
# Drop rows with any NaN in rates (to avoid empty columns/rows)
max_date_df = max_date_df.dropna(subset=['central_rate'])
# Beautify: Use styled dataframe with colors/background
styled_df = max_date_df.style.background_gradient(cmap='viridis', subset=['central_rate']).format(precision=2)
st.dataframe(styled_df)

if df_trends.empty:
    st.warning("No data available for the selected currencies and time window.")
else:
    # 2. Time Series Trend Chart (Simple Plotly line with hover)
    st.subheader("Time Series Trends")
    fig_trend = px.line(df_trends, x='date', y='central_rate', color='currency',
                        title="Central Rates Over Time",
                        labels={'central_rate': 'Central Rate', 'date': 'Date'},
                        markers=True)
    fig_trend.update_layout(xaxis_title="Date", yaxis_title="Rate", hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)

    # 3. Daily % Change Bar Chart (Green positive, red negative)
    st.subheader("Daily Percentage Changes")
    df_changes = df_trends.copy()
    df_changes['pct_change'] = df_changes.groupby('currency')['central_rate'].pct_change() * 100
    # Create color column: Green for positive, red for negative
    df_changes['color'] = np.where(df_changes['pct_change'] > 0, 'Positive (Green)', 'Negative (Red)')
    fig_change = px.bar(df_changes, x='date', y='pct_change', color='color',
                        title="Daily % Change in Central Rates",
                        labels={'pct_change': '% Change', 'date': 'Date'},
                        color_discrete_map={'Positive (Green)': 'green', 'Negative (Red)': 'red'})
    fig_change.update_layout(xaxis_title="Date", yaxis_title="% Change", barmode='group',
                             template="plotly_dark", showlegend=False)  # Hide legend since colors are self-explanatory
    st.plotly_chart(fig_change, use_container_width=True)

    # 4. Volatility Analysis (Fixed: Ensure window_size handles short data; added min data check)
    st.subheader("Volatility Analysis (Rolling Std Dev)")
    window_size = min(30, len(df_trends) // len(selected_currencies))  # Adjust if data is short
    if window_size >= 2:  # Need at least 2 points for std dev
        df_vol = df_trends.copy()
        df_vol['volatility'] = df_vol.groupby('currency')['central_rate'].rolling(window=window_size).std().reset_index(0, drop=True)
        fig_vol = px.line(df_vol, x='date', y='volatility', color='currency',
                          title=f"{window_size}-Day Rolling Volatility",
                          labels={'volatility': 'Volatility (Std Dev)', 'date': 'Date'})
        fig_vol.update_layout(xaxis_title="Date", yaxis_title="Volatility", template="plotly_dark")
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.write("Insufficient data points for volatility calculation (need at least 2 days).")

    # 5. Moving Averages (SMA and EMA)
    st.subheader("Moving Averages (Simple & Exponential)")
    df_ma = df_trends.copy()
    for curr in selected_currencies:
        df_curr = df_ma[df_ma['currency'] == curr]
        df_ma.loc[df_ma['currency'] == curr, 'sma_10'] = df_curr['central_rate'].rolling(window=10).mean()
        df_ma.loc[df_ma['currency'] == curr, 'ema_10'] = df_curr['central_rate'].ewm(span=10, adjust=False).mean()

    fig_ma = make_subplots(rows=1, cols=1, shared_xaxes=True)
    for curr in selected_currencies:
        df_curr = df_ma[df_ma['currency'] == curr]
        fig_ma.add_trace(go.Scatter(x=df_curr['date'], y=df_curr['central_rate'], name=f"{curr} Rate", mode='lines'), row=1, col=1)
        fig_ma.add_trace(go.Scatter(x=df_curr['date'], y=df_curr['sma_10'], name=f"{curr} SMA-10", mode='lines', line=dict(dash='dot')), row=1, col=1)
        fig_ma.add_trace(go.Scatter(x=df_curr['date'], y=df_curr['ema_10'], name=f"{curr} EMA-10", mode='lines', line=dict(dash='dash')), row=1, col=1)

    fig_ma.update_layout(title="Rates with 10-Day SMA & EMA", xaxis_title="Date", yaxis_title="Rate", hovermode="x unified",
                         template="plotly_dark")
    st.plotly_chart(fig_ma, use_container_width=True)

    # 7. Box Plot for Rate Distribution
    st.subheader("Rate Distribution Box Plot")
    fig_box = px.box(df_trends, x='currency', y='central_rate',
                     title="Distribution of Central Rates by Currency",
                     labels={'central_rate': 'Central Rate'})
    fig_box.update_layout(template="plotly_dark")
    st.plotly_chart(fig_box, use_container_width=True)

    # 8. All-Time Lowest and Highest
    st.subheader("All-Time Low & High Rates")
    all_time = df.groupby('currency')['central_rate'].agg(['min', 'max']).reset_index()
    all_time.columns = ['Currency', 'All-Time Low', 'All-Time High']
    st.dataframe(all_time.style.format({'All-Time Low': '{:.2f}', 'All-Time High': '{:.2f}'}))

    # 9. Averages (Past 30 Days & Last 365 Days)
    st.subheader("Average Rates")
    df_30 = df[df['date'] >= latest_date - pd.Timedelta(days=30)].groupby('currency')['central_rate'].mean().reset_index()
    df_365 = df[df['date'] >= latest_date - pd.Timedelta(days=365)].groupby('currency')['central_rate'].mean().reset_index()
    averages = df_30.merge(df_365, on='currency', suffixes=('_30_days', '_365_days'))
    averages.columns = ['Currency', 'Monthly Avg (30 Days)', 'Yearly Avg (365 Days)']
    st.dataframe(averages.style.format({'Monthly Avg (30 Days)': '{:.2f}', 'Yearly Avg (365 Days)': '{:.2f}'}))

# Manual refresh button
if st.button("Refresh Data"):
    st.rerun()

# Footer with designer and data sources (not too obvious, placed at the bottom)
st.markdown("""
---
**Designed by:** Sanya Similoluwa  
**Data sources:** https://www.cbn.gov.ng/rates/ExchRateByCurrency.html for backfill and fixer.io for daily FX rates
""")