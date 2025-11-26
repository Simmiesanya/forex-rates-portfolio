import streamlit as st
import pandas as pd
from sqlalchemy import create_engine, text
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np
st.title("FX Rates Dashboard")
# NEW: Friendly note right after title
st.markdown(
    """
    <div style="background-color: #0E1117; padding: 12px; border-radius: 8px; border-left: 5px solid #1f77b4; margin: 20px 0;">
    <p style="margin:0; color:#8A9BA8; font-size:15px;">
    ℹ️ <strong>Data is automatically refreshed daily by 12pm.</strong>
    </p>
    </div>
    """,
    unsafe_allow_html=True
)
# Manual refresh
if st.button("Refresh Data"):
    st.rerun()
# DB connection
conn_str = st.secrets["POSTGRES_CONN_STR"]
engine = create_engine(conn_str)
# Fetch full data once and cache
@st.cache_data(ttl=3600)
def load_data():
    with engine.connect() as conn:
        df = pd.read_sql(text("""
            SELECT date, currency, buying_rate, central_rate, selling_rate
            FROM vault.fx_rates_daily
            ORDER BY date DESC, currency
        """), conn)
    df['date'] = pd.to_datetime(df['date'])
    return df
df = load_data()
# Sidebar
st.sidebar.header("Analysis Options")
currencies = sorted(df['currency'].unique())
selected_currencies = st.sidebar.multiselect("Select Currencies", currencies, default=['USD', 'EUR', 'GBP'])
time_window = st.sidebar.slider("Time Window for Trends (Days)", 7, 365, 30)
# Filter for trends
latest_date = df['date'].max()
df_trends = df[
    (df['date'] >= latest_date - pd.Timedelta(days=time_window)) &
    (df['currency'].isin(selected_currencies))
].sort_values(['currency', 'date'])
# 1. Latest Date Table
with engine.connect() as conn:
    max_date_df = pd.read_sql(text("""
        SELECT date, currency, central_rate AS rate
        FROM vault.fx_rates_daily
        WHERE date = (SELECT MAX(date) FROM vault.fx_rates_daily)
    """), conn)
max_date_df['date'] = pd.to_datetime(max_date_df['date']).dt.strftime('%Y-%m-%d')
max_date_df = max_date_df.dropna(subset=['rate'])
order = ['USD', 'GBP', 'EUR', 'CAD', 'CHF', 'CNY', 'DKK', 'JPY', 'ZAR']
max_date_df = max_date_df[max_date_df['currency'].isin(order)]
max_date_df = max_date_df.sort_values(by='currency', key=lambda x: pd.Categorical(x, categories=order, ordered=True))
max_date_df = max_date_df.reset_index(drop=True)
max_date_df['currency'] = '1 ' + max_date_df['currency']
latest_date_str = max_date_df['date'].iloc[0]
st.subheader(f"Latest Rates: {latest_date_str}")
max_date_df = max_date_df.drop(columns=['date'])
max_date_df = max_date_df.rename(columns={'rate': 'Naira Rate'})
def alternate_rows(row):
    if row.name % 2 == 0:
        bg_color = '#333333' # light gray/black
    else:
        bg_color = '#1a1a1a' # deep grey
    return [f'background-color: {bg_color}; color: #ffffff'] * len(row)
styled_df = max_date_df.style.apply(alternate_rows, axis=1).format({'Naira Rate': '{:.2f}'}).set_properties(subset=['Naira Rate'], **{'text-align': 'left'}).hide(axis='index')
st.dataframe(styled_df, use_container_width=False, width=400, hide_index=True)
# All the rest of your awesome charts (unchanged)
if df_trends.empty:
    st.warning("No data available for the selected currencies and time window.")
else:
    # 2. Time Series Trend Chart
    st.subheader("Time Series Trends")
    fig_trend = px.line(df_trends, x='date', y='central_rate', color='currency',
                        title="Central Rates Over Time", markers=True,
                        labels={'central_rate': 'Central Rate', 'date': 'Date'})
    fig_trend.update_layout(xaxis_title="Date", yaxis_title="Rate", hovermode="x unified")
    st.plotly_chart(fig_trend, use_container_width=True)
    # 5. Moving Averages
    st.subheader("Moving Averages (Simple & Exponential)")
    df_ma = df_trends.copy()
    for curr in selected_currencies:
        mask = df_ma['currency'] == curr
        df_ma.loc[mask, 'sma_10'] = df_ma.loc[mask, 'central_rate'].rolling(10).mean()
        df_ma.loc[mask, 'ema_10'] = df_ma.loc[mask, 'central_rate'].ewm(span=10, adjust=False).mean()
    fig_ma = make_subplots()
    for curr in selected_currencies:
        d = df_ma[df_ma['currency'] == curr]
        fig_ma.add_trace(go.Scatter(x=d['date'], y=d['central_rate'], name=f"{curr} Rate", mode='lines'))
        fig_ma.add_trace(go.Scatter(x=d['date'], y=d['sma_10'], name=f"{curr} SMA-10", line=dict(dash='dot')))
        fig_ma.add_trace(go.Scatter(x=d['date'], y=d['ema_10'], name=f"{curr} EMA-10", line=dict(dash='dash')))
    fig_ma.update_layout(title="Rates with 10-Day SMA & EMA", template="plotly_dark", hovermode="x unified")
    st.plotly_chart(fig_ma, use_container_width=True)
    # 3. Daily % Change Bar Chart
    st.subheader("Daily Percentage Changes")
    selected_currency_change = st.selectbox("Select Currency for % Changes", currencies, index=currencies.index('USD') if 'USD' in currencies else 0)
    time_window_change = st.slider("Time Window for % Changes (Days)", 7, 365, 30)
    df_changes = df[
        (df['date'] >= latest_date - pd.Timedelta(days=time_window_change)) &
        (df['currency'] == selected_currency_change)
    ].sort_values('date')
    df_changes['pct_change'] = df_changes['central_rate'].pct_change() * 100
    df_changes['color'] = np.where(df_changes['pct_change'] > 0, 'Positive (Green)', 'Negative (Red)')
    fig_change = px.bar(df_changes, x='date', y='pct_change', color='color',
                        color_discrete_map={'Positive (Green)': 'green', 'Negative (Red)': 'red'},
                        labels={'pct_change': '% Change', 'date': 'Date'})
    fig_change.update_layout(showlegend=False, template="plotly_dark")
    st.plotly_chart(fig_change, use_container_width=True)
    # 4. Volatility
    st.subheader("Volatility Analysis (Rolling Std Dev)")
    df_vol = df[
        (df['date'] >= latest_date - pd.Timedelta(days=time_window_change)) &
        (df['currency'] == selected_currency_change)
    ].sort_values('date')
    window_size = min(30, len(df_vol))
    if window_size >= 2:
        df_vol['volatility'] = df_vol['central_rate'].rolling(window=window_size).std()
        fig_vol = px.line(df_vol, x='date', y='volatility', color='currency',
                          title=f"{window_size}-Day Rolling Volatility")
        fig_vol.update_layout(template="plotly_dark")
        st.plotly_chart(fig_vol, use_container_width=True)
    else:
        st.write("Insufficient data for volatility calculation.")
    # 7. Averages
    st.subheader("Average Rates")
    df_30 = df[df['date'] >= latest_date - pd.Timedelta(days=30)].groupby('currency')['central_rate'].mean().reset_index(name='Monthly Avg (30 Days)')
    df_365 = df[df['date'] >= latest_date - pd.Timedelta(days=365)].groupby('currency')['central_rate'].mean().reset_index(name='Yearly Avg (365 Days)')
    averages = df_30.merge(df_365, on='currency')
    st.dataframe(averages.style.format({'Monthly Avg (30 Days)': '{:.2f}', 'Yearly Avg (365 Days)': '{:.2f}'}), use_container_width=True)
# =============================================================================
# NEW SECTION 1: Independent Historical Table with Filters
# =============================================================================
st.markdown("---")
st.subheader("Historical Rates Explorer")
col1, col2 = st.columns([1, 3])
with col1:
    selected_currency_hist = st.selectbox("Choose Currency", options=["All"] + list(currencies), index=0)
with col2:
    selected_date_hist = st.date_input("Choose Date", value=latest_date.date(), min_value=df['date'].min().date(), max_value=latest_date.date())
# Filter data
df_hist = df.copy()
if selected_currency_hist != "All":
    df_hist = df_hist[df_hist['currency'] == selected_currency_hist]
df_hist = df_hist[df_hist['date'].dt.date == selected_date_hist]
if df_hist.empty:
    st.info("No data available for the selected date and currency.")
else:
    df_hist_display = df_hist[['date', 'currency', 'central_rate']].copy()
    df_hist_display['date'] = df_hist_display['date'].dt.strftime('%Y-%m-%d')
    df_hist_display = df_hist_display.sort_values('currency')
    st.dataframe(df_hist_display.style.format({
        'central_rate': '{:.2f}'
    }), use_container_width=True)
# =============================================================================
# NEW SECTION 2: All-Time Low & High with Dates (Separate Tables)
# =============================================================================
st.markdown("---")
st.subheader("All-Time Records")
# All-Time Low
lows = df.loc[df.groupby('currency')['central_rate'].idxmin()][['currency', 'date', 'central_rate']]
lows = lows.rename(columns={'central_rate': 'All-Time Low Rate', 'date': 'Date of Low'})
lows['Date of Low'] = lows['Date of Low'].dt.strftime('%Y-%m-%d')
lows = lows[['currency', 'Date of Low', 'All-Time Low Rate']].sort_values('currency').reset_index(drop=True)
# All-Time High
highs = df.loc[df.groupby('currency')['central_rate'].idxmax()][['currency', 'date', 'central_rate']]
highs = highs.rename(columns={'central_rate': 'All-Time High Rate', 'date': 'Date of High'})
highs['Date of High'] = highs['Date of High'].dt.strftime('%Y-%m-%d')
highs = highs[['currency', 'Date of High', 'All-Time High Rate']].sort_values('currency').reset_index(drop=True)
col_low, col_high = st.columns(2)
with col_low:
    st.write("**All-Time Lowest Central Rates**")
    st.dataframe(lows.style.format({'All-Time Low Rate': '{:.2f}'}), use_container_width=True)
with col_high:
    st.write("**All-Time Highest Central Rates**")
    st.dataframe(highs.style.format({'All-Time High Rate': '{:.2f}'}), use_container_width=True)
# Footer
st.markdown("""
---
**Designed by:** Sanya Similoluwa
**Data sources:** https://www.cbn.gov.ng/rates/ExchRateByCurrency.html (backfill) • fixer.io (daily rates)
""")