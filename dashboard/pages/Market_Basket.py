"""
Market Basket Analysis Dashboard Page
"""
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
import requests
import pandas as pd
import networkx as nx
from datetime import datetime
import sys
sys.path.append('..')

st.set_page_config(page_title="Market Basket Analysis", page_icon="🛒", layout="wide")

# API Base URL
API_BASE_URL = "http://localhost:5000/api"

# Header
st.title("🛒 Market Basket Analysis")
st.markdown("Discover product associations and cross-selling opportunities using FP-Growth Algorithm")
st.markdown("---")

# Fetch association rules
@st.cache_data(ttl=3600)
def fetch_association_rules():
    try:
        # Lấy các tham số từ API, limit=1000
        response = requests.get(f"{API_BASE_URL}/association/rules?limit=1000&min_lift=1")
        response.raise_for_status()
        data = response.json()
        if data.get('success'):
            return pd.DataFrame(data['data'])
        return pd.DataFrame()
    except requests.exceptions.RequestException as e:
        st.error(f"Lỗi khi gọi API Association Rules: {e}")
        return pd.DataFrame()

df_rules = fetch_association_rules()

if df_rules.empty:
    st.error("❌ No association rules available. Please run the batch processing job first.")
    st.stop()

# Sidebar filters
st.sidebar.header("⚙️ Filter Settings")

min_confidence = st.sidebar.slider(
    "Minimum Confidence",
    min_value=0.0,
    max_value=1.0,
    value=0.3,
    step=0.05,
    help="Probability that consequent is purchased given antecedent"
)

min_lift = st.sidebar.slider(
    "Minimum Lift",
    min_value=1.0,
    max_value=10.0,
    value=2.0,
    step=0.5,
    help="How much more likely items are bought together vs independently"
)

min_support = st.sidebar.slider(
    "Minimum Support",
    min_value=0.0,
    max_value=0.1,
    value=0.01,
    step=0.005,
    help="Frequency of the itemset in all transactions"
)

top_n = st.sidebar.selectbox(
    "Show Top N Rules",
    [10, 20, 50, 100, 200],
    index=1
)

# Filter rules
filtered_rules = df_rules[
    (df_rules['confidence'] >= min_confidence) & 
    (df_rules['lift'] >= min_lift) &
    (df_rules['support'] >= min_support)
].head(top_n)

# Summary metrics
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("📊 Total Rules Found", f"{len(filtered_rules):,}")

with col2:
    avg_confidence = filtered_rules['confidence'].mean()
    st.metric("🎯 Avg Confidence", f"{avg_confidence:.1%}")

with col3:
    avg_lift = filtered_rules['lift'].mean()
    st.metric("📈 Avg Lift", f"{avg_lift:.2f}x")

with col4:
    avg_support = filtered_rules['support'].mean()
    st.metric("💡 Avg Support", f"{avg_support:.3f}")

st.markdown("---")

# Rules Table
st.subheader("📋 Association Rules")

if not filtered_rules.empty:
    # Prepare display dataframe
    display_df = filtered_rules.copy()
    
    # Handle list columns
    if 'antecedent_str' in display_df.columns:
        display_df['Antecedent'] = display_df['antecedent_str']
    else:
        display_df['Antecedent'] = display_df['antecedent'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
    
    if 'consequent_str' in display_df.columns:
        display_df['Consequent'] = display_df['consequent_str']
    else:
        display_df['Consequent'] = display_df['consequent'].apply(
            lambda x: ', '.join(x) if isinstance(x, list) else str(x)
        )
    
    display_df['Support'] = display_df['support'].apply(lambda x: f"{x:.3f}")
    display_df['Confidence'] = display_df['confidence'].apply(lambda x: f"{x:.1%}")
    display_df['Lift'] = display_df['lift'].apply(lambda x: f"{x:.2f}")
    
    display_df = display_df[['Antecedent', 'Consequent', 'Support', 'Confidence', 'Lift']]
    display_df.columns = ['If Customer Buys →', 'Then Also Buys →', 'Support', 'Confidence (%)', 'Lift']
    
    st.dataframe(
        display_df,
        width='stretch',
        hide_index=True,
        column_config={
            "If Customer Buys →": st.column_config.TextColumn(width="large"),
            "Then Also Buys →": st.column_config.TextColumn(width="large"),
        }
    )
    
    # Download button
    csv = display_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Rules",
        data=csv,
        file_name="association_rules.csv",
        mime="text/csv"
    )
else:
    st.warning("No rules match the current filters")

st.markdown("---")

# Visualizations
st.subheader("📊 Rules Visualization")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### Confidence vs Lift Scatter Plot")
    
    fig = px.scatter(
        filtered_rules,
        x='confidence',
        y='lift',
        size='support',
        color='lift',
        hover_data=['antecedent_str', 'consequent_str'] if 'antecedent_str' in filtered_rules.columns else [],
        labels={
            'confidence': 'Confidence',
            'lift': 'Lift',
            'support': 'Support'
        },
        color_continuous_scale='Viridis'
    )
    
    fig.update_layout(
        xaxis=dict(title="Confidence", showgrid=True, gridcolor='#333333', color='#E0E0E0'),
        yaxis=dict(title="Lift", showgrid=True, gridcolor='#333333', color='#E0E0E0'),
        height=400,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
    )
    
    st.plotly_chart(fig, width='stretch')

with col2:
    st.markdown("#### Lift Distribution")
    
    fig = px.histogram(
        filtered_rules,
        x='lift',
        nbins=30,
        labels={'lift': 'Lift', 'count': 'Frequency'},
        color_discrete_sequence=['#F72585']
    )
    fig.update_traces(marker_line_width=1, marker_line_color="white", opacity=0.8)
    
    fig.update_layout(
        xaxis=dict(title="Lift", showgrid=False, gridcolor='#333333', color='#E0E0E0'),
        yaxis=dict(title="Number of Rules", showgrid=True, gridcolor='#333333', color='#E0E0E0'),
        height=400,
        showlegend=False,
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0')
    )
    
    st.plotly_chart(fig, width='stretch')

st.markdown("---")

# Product Search
st.subheader("🔍 Find Related Products")

search_product = st.text_input(
    "Enter Product Code to find frequently bought together items:",
    placeholder="e.g., 22423"
)

if search_product:
    # Search in antecedent
    related_rules = filtered_rules[
        filtered_rules['antecedent_str'].str.contains(search_product, case=False, na=False) |
        filtered_rules['consequent_str'].str.contains(search_product, case=False, na=False)
    ].copy()
    
    if not related_rules.empty:
        st.success(f"Found {len(related_rules)} related rules for product '{search_product}'")
        
        # Show related products
        related_display = related_rules.copy()
        related_display['From'] = related_display['antecedent_str']
        related_display['To'] = related_display['consequent_str']
        related_display['Confidence'] = related_display['confidence'].apply(lambda x: f"{x:.1%}")
        related_display['Lift'] = related_display['lift'].apply(lambda x: f"{x:.2f}")
        
        st.dataframe(
            related_display[['From', 'To', 'Confidence', 'Lift']],
            width='stretch',
            hide_index=True
        )
        
        # Network graph
        st.markdown("#### 🕸️ Product Association Network")
        
        # Create network graph
        G = nx.DiGraph()
        
        for _, row in related_rules.head(20).iterrows():
            antecedent = row['antecedent_str'][:30]  # Truncate for readability
            consequent = row['consequent_str'][:30]
            lift = row['lift']
            
            G.add_edge(antecedent, consequent, weight=lift)
        
        # Get positions
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Create edges
        edge_trace = []
        for edge in G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_trace.append(
                go.Scatter(
                    x=[x0, x1, None],
                    y=[y0, y1, None],
                    mode='lines',
                    line=dict(width=1, color='#888'),
                    hoverinfo='none',
                    showlegend=False
                )
            )
        
        # Create nodes
        node_x = []
        node_y = []
        node_text = []
        
        for node in G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
        
        node_trace = go.Scatter(
            x=node_x,
            y=node_y,
            mode='markers+text',
            text=node_text,
            textposition="top center",
            hoverinfo='text',
            marker=dict(
                size=20,
                color='#1f77b4',
                line=dict(width=2, color='white')
            ),
            showlegend=False
        )
        
        # Create figure
        fig = go.Figure(data=edge_trace + [node_trace])
        
        fig.update_layout(
            showlegend=False,
            hovermode='closest',
            margin=dict(b=0,l=0,r=0,t=0),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            height=500,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#E0E0E0')
        )
        # Update node colors
        node_trace.marker.color = '#F72585'
        
        st.plotly_chart(fig, width='stretch')
        
    else:
        st.info(f"No association rules found for product '{search_product}'")

st.markdown("---")

# Top Product Pairs
st.subheader("🏆 Top 10 Strongest Rules (Lift & Confidence)")

# 1. Lấy Top 10 rules
top_pairs = filtered_rules.nlargest(10, 'lift').copy()

# 2. Tạo tên Rule dễ đọc hơn cho biểu đồ
# Cắt ngắn tên sản phẩm nếu quá dài để biểu đồ không bị vỡ
top_pairs['Rule_Name'] = (
    top_pairs['antecedent_str'].apply(lambda x: x[:25] + '...' if len(str(x))>25 else str(x)) +
    " ➡ " +
    top_pairs['consequent_str'].apply(lambda x: x[:25] + '...' if len(str(x))>25 else str(x))
)

# 3. Sắp xếp để hiển thị cái mạnh nhất lên đầu
top_pairs = top_pairs.sort_values('lift', ascending=True)

# 4. Vẽ biểu đồ Horizontal Bar
fig = px.bar(
    top_pairs,
    x='lift',
    y='Rule_Name',
    color='confidence',  # Màu sắc thể hiện Confidence
    orientation='h',  # Biểu đồ ngang
    text='lift',  # Hiển thị giá trị Lift ngay trên cột
    title='Top Rules by Lift (Length) and Confidence (Color)',
    labels={
        'lift': 'Lift (How much more likely?)',
        'Rule_Name': 'Product Rule (If -> Then)',
        'confidence': 'Confidence'
    },
    color_continuous_scale='Viridis'  # Thang màu nổi bật
)

fig.update_traces(
    texttemplate='%{text:.2f}x',  # Format số Lift (ví dụ: 5.20x)
    textposition='outside'
)

fig.update_layout(
    height=500,
    plot_bgcolor='rgba(0,0,0,0)',
    paper_bgcolor='rgba(0,0,0,0)',
    font=dict(color='#E0E0E0'),
    xaxis=dict(showgrid=True, gridcolor='#333333'),
    yaxis=dict(showgrid=False),
    coloraxis_colorbar=dict(title="Confidence")
)

st.plotly_chart(fig, use_container_width=True)

# Insights
st.markdown("---")
st.subheader("💡 Key Insights")
if not filtered_rules.empty:
    strongest_rule = filtered_rules.nlargest(1, 'lift').iloc[0]

    col1, col2 = st.columns(2)

    with col1:
        st.info(f"""
        **🎯 Strongest Association:**
        - **If customers buy:** {strongest_rule['antecedent_str']}
        - **They also buy:** {strongest_rule['consequent_str']}
        - **Confidence:** {strongest_rule['confidence']:.1%}
        - **Lift:** {strongest_rule['lift']:.2f}x more likely
        """)

    with col2:
        st.success(f"""
        **📊 Recommendations:**
        - Place these products near each other in store
        - Create bundle promotions
        - Use for cross-selling in e-commerce
        - Include in "Frequently Bought Together" sections
        """)
else:
    # Thông báo nếu không tìm thấy quy tắc nào
    st.info("No rules found matching the current filters.")
# Footer
st.markdown("---")
st.caption(f"Analysis based on FP-Growth algorithm | Last updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
