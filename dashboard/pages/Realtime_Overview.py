"""
Real-time Overview Dashboard Page
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import pandas as pd
from datetime import datetime, timedelta
import sys
sys.path.append('..')
from components.charts import create_revenue_timeseries, create_bar_chart
from components.metrics import display_kpi_row, format_currency, format_number

st.set_page_config(page_title="Real-time Overview", page_icon="📊", layout="wide")

# API Base URL
API_BASE_URL = "http://localhost:5000/api"

# Header
st.title("📊 Real-time Analytics Dashboard")
st.markdown("---")

# Refresh controls
col1, col2, col3 = st.columns([2, 1, 1])
with col1:
    st.subheader("Live Metrics")
with col2:
    # auto_refresh = st.checkbox("Auto-refresh (30s)", value=False) # Tạm thời vô hiệu hóa do time.sleep()
    pass
with col3:
    if st.button("🔄 Refresh Now"):
        st.cache_data.clear()
        st.rerun()

# Auto refresh logic (time.sleep) đã bị xóa vì nó làm treo ứng dụng.
# @st.cache_data(ttl=30) sẽ tự động làm mới dữ liệu sau 30 giây
# khi người dùng nhấn nút "Refresh Now".

# Fetch real-time data
@st.cache_data(ttl=30)
def fetch_realtime_revenue():
    try:
        response = requests.get(f"{API_BASE_URL}/realtime/revenue?minutes=15")
        response.raise_for_status()  # Báo lỗi nếu request thất bại
        data = response.json()
        if data.get('success'):
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi gọi API Doanh thu: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=30)
def fetch_top_products():
    try:
        response = requests.get(f"{API_BASE_URL}/realtime/products?limit=100")
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi gọi API Sản phẩm: {e}")
        return pd.DataFrame()

df_revenue = fetch_realtime_revenue()
df_products = fetch_top_products()

# KPI Cards
if not df_revenue.empty:
    total_revenue = df_revenue['TotalRevenue'].sum()
    total_transactions = df_revenue['TransactionCount'].sum()
    unique_customers = df_revenue['UniqueCustomers'].sum()
    avg_order_value = total_revenue / total_transactions if total_transactions > 0 else 0
    
    metrics = [
        {
            'title': '💰 Total Revenue (15min)',
            'value': format_currency(total_revenue),
            'delta': '+5.2%'
        },
        {
            'title': '🧾 Transactions',
            'value': format_number(total_transactions),
            'delta': '+12'
        },
        {
            'title': '👥 Unique Customers',
            'value': format_number(unique_customers),
            'delta': '+8'
        },
        {
            'title': '📊 Avg Order Value',
            'value': format_currency(avg_order_value),
            'delta': '+£2.30'
        }
    ]
    
    display_kpi_row(metrics)
else:
    st.warning("⚠️ No real-time data available")

st.markdown("---")

# Main Charts
if not df_revenue.empty:
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📈 Revenue Timeline")
        
        # Prepare data
        df_plot = df_revenue.copy()
        # Sửa lỗi logic: Cột bây giờ là 'WindowStart', không phải 'window'
        if 'WindowStart' in df_plot.columns:
            # Sửa lỗi: Chỉ định đơn vị là 'ms' (milliseconds)
            df_plot['WindowStart'] = pd.to_datetime(df_plot['WindowStart'], unit='ms')
            df_plot = df_plot.sort_values('WindowStart')
        
        fig = create_revenue_timeseries(
            df_plot,
            'WindowStart',
            'TotalRevenue',
            title='Revenue per Minute (Last 15 minutes)'
        )
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        st.subheader("🌍 Revenue by Country")
        
        country_revenue = df_revenue.groupby('Country')['TotalRevenue'].sum().reset_index()
        country_revenue = country_revenue.sort_values('TotalRevenue', ascending=False).head(10)
        
        fig = create_bar_chart(
            country_revenue,
            'Country',
            'TotalRevenue',
            title='Top 10 Countries',
            orientation='v'
        )
        st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Top Products Section
st.subheader("🏆 Top Selling Products (Last 10 minutes)")

if not df_products.empty:
    # Aggregate by product
    product_agg = df_products.groupby(['StockCode', 'Description']).agg({
        'TotalQuantity': 'sum',
        'TotalRevenue': 'sum',
        'TransactionCount': 'sum'
    }).reset_index()
    
    top_10 = product_agg.nlargest(10, 'TotalRevenue')
    
    # Format for display
    display_df = top_10.copy()
    display_df['TotalRevenue'] = display_df['TotalRevenue'].apply(lambda x: f"£{x:,.2f}")
    display_df['TotalQuantity'] = display_df['TotalQuantity'].apply(lambda x: f"{int(x):,}")
    display_df['TransactionCount'] = display_df['TransactionCount'].apply(lambda x: f"{int(x):,}")
    
    display_df.columns = ['Product Code', 'Description', 'Quantity Sold', 'Revenue', 'Transactions']
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "Product Code": st.column_config.TextColumn("Product Code", width="small"),
            "Description": st.column_config.TextColumn("Description", width="large"),
        }
    )
    
    # Visualization
    col1, col2 = st.columns(2)
    
    with col1:
        fig = px.pie(
            top_10,
            names='Description',
            values='TotalRevenue',
            title='Revenue Distribution - Top 10 Products',
            hole=0.4
        )
        fig.update_traces(textposition='inside', textinfo='percent')
        st.plotly_chart(fig, width='stretch')
    
    with col2:
        fig = px.bar(
            top_10,
            x='TotalQuantity',
            y='Description',
            orientation='h',
            title='Quantity Sold - Top 10 Products',
            color='TotalQuantity',
            color_continuous_scale='Blues'
        )
        fig.update_layout(showlegend=False, yaxis={'categoryorder':'total ascending'})
        st.plotly_chart(fig, width='stretch')
else:
    st.info("No product data available")

st.markdown("---")

# World Map (if country data available)
if not df_revenue.empty and 'Country' in df_revenue.columns:
    st.subheader("🗺️ Global Revenue Distribution")
    
    country_totals = df_revenue.groupby('Country').agg({
        'TotalRevenue': 'sum',
        'TransactionCount': 'sum',
        'UniqueCustomers': 'sum'
    }).reset_index()
    
    fig = px.choropleth(
        country_totals,
        locations='Country',
        locationmode='country names',
        color='TotalRevenue',
        hover_name='Country',
        hover_data={
            'TotalRevenue': ':,.2f',
            'TransactionCount': ':,',
            'UniqueCustomers': ':,'
        },
        color_continuous_scale='Blues',
        title='Revenue by Country'
    )
    
    fig.update_layout(
        geo=dict(
            showframe=False,
            showcoastlines=True,
            projection_type='natural earth'
        ),
        height=500
    )
    
    st.plotly_chart(fig, width='stretch')

# Footer
st.markdown("---")
st.caption(f"Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
