"""
Reusable Chart Components
"""
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd


def create_revenue_timeseries(df: pd.DataFrame, x_col: str, y_col: str, 
                              title: str = "Revenue Over Time"):
    """Create revenue time series chart"""
    # Đổi sang px.bar (Biểu đồ thanh)
    fig = px.bar(
        df,
        x=x_col,
        y=y_col,
        title=title,
        labels={x_col: 'Time', y_col: 'Revenue (£)'}
    )
    
    fig.update_traces(
        marker_color='#00B4D8',  # Màu Cyan sáng (nổi bật trên nền tối)
        marker_line_width=0,
        hovertemplate='<b>Time:</b> %{x}<br><b>Revenue:</b> £%{y:,.2f}<extra></extra>'
    )
    
    fig.update_layout(
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',  # Nền trong suốt
        paper_bgcolor='rgba(0,0,0,0)',  # Nền trong suốt
        font=dict(size=12, color='#E0E0E0'),  # Chữ màu trắng sáng
        xaxis=dict(
            showgrid=True,
            gridcolor='#333333',  # Lưới màu xám tối
            linecolor='#555555'
        ),
        yaxis=dict(
            showgrid=True,
            gridcolor='#333333',
            linecolor='#555555'
        )
    )
    
    return fig


def create_bar_chart(df: pd.DataFrame, x_col: str, y_col: str, 
                     title: str, orientation: str = 'v'):
    """Create bar chart"""
    fig = px.bar(
        df,
        x=x_col if orientation == 'v' else y_col,
        y=y_col if orientation == 'v' else x_col,
        title=title,
        orientation=orientation,
        color=y_col if orientation == 'v' else x_col,
        color_continuous_scale='Viridis'  # Thang màu xanh-vàng rực rỡ
    )
    
    fig.update_layout(
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        font=dict(color='#E0E0E0'),  # Chữ trắng
        showlegend=False,
        xaxis=dict(showgrid=False, gridcolor='#333333'),
        yaxis=dict(showgrid=True, gridcolor='#333333')
    )
    
    return fig


def create_forecast_chart(df: pd.DataFrame, date_col: str, 
                         forecast_col: str, lower_col: str, upper_col: str,
                         title: str = "Demand Forecast"):
    """Create forecast chart with confidence interval"""
    fig = go.Figure()

    # 1. Upper bound (Vẽ trước để làm nền)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[upper_col],
        mode='lines',
        line=dict(width=0),
        showlegend=False,
        hoverinfo='skip'
    ))
    
    # 2. Lower bound with fill (Tô màu vùng tin cậy)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[lower_col],
        mode='lines',
        name='Uncertainty Range (95%)',
        fill='tonexty',
        fillcolor='rgba(255, 255, 255, 0.1)',  # Màu trắng mờ nhẹ, sang trọng trên nền tối
        line=dict(width=0)
    ))

    # 3. Forecast line (Vẽ sau cùng để nổi lên trên)
    fig.add_trace(go.Scatter(
        x=df[date_col],
        y=df[forecast_col],
        mode='lines+markers',  # Thêm markers để thấy rõ điểm dữ liệu
        name='Forecast Trend',
        line=dict(color='#FF9F1C', width=3),  # Màu Cam Neon nổi bật
        marker=dict(size=6, color='#FF9F1C')
    ))
    
    fig.update_layout(
        title=dict(text=title, font=dict(size=20, color='#E0E0E0')),
        xaxis=dict(
            title="Date",
            showgrid=True,
            gridcolor='#333333',  # Lưới xám tối
            color='#E0E0E0'
        ),
        yaxis=dict(
            title="Quantity",
            showgrid=True,
            gridcolor='#333333',
            color='#E0E0E0'
        ),
        hovermode='x unified',
        plot_bgcolor='rgba(0,0,0,0)',  # Nền trong suốt
        paper_bgcolor='rgba(0,0,0,0)',  # Nền trong suốt
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(color='#E0E0E0')
        )
    )
    
    return fig


def create_scatter_plot(df: pd.DataFrame, x_col: str, y_col: str,
                       size_col: str = None, color_col: str = None,
                       title: str = "Scatter Plot"):
    """Create scatter plot"""
    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        size=size_col,
        color=color_col,
        title=title,
        hover_data=df.columns.tolist()
    )
    
    fig.update_layout(
        plot_bgcolor='white',
        paper_bgcolor='white'
    )
    
    return fig


def create_pie_chart(df: pd.DataFrame, names_col: str, values_col: str,
                    title: str = "Distribution"):
    """Create pie chart"""
    fig = px.pie(
        df,
        names=names_col,
        values=values_col,
        title=title,
        hole=0.4
    )
    
    fig.update_traces(
        textposition='inside',
        textinfo='percent+label'
    )
    
    return fig


def create_heatmap(df: pd.DataFrame, x_col: str, y_col: str, z_col: str,
                  title: str = "Heatmap"):
    """Create heatmap"""
    pivot = df.pivot(index=y_col, columns=x_col, values=z_col)
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values,
        x=pivot.columns,
        y=pivot.index,
        colorscale='Blues'
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col
    )
    
    return fig


def create_gauge_chart(value: float, title: str, 
                      max_value: float = 100, 
                      thresholds: dict = None):
    """Create gauge chart"""
    if thresholds is None:
        thresholds = {'low': 30, 'medium': 70}
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=value,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': title},
        gauge={
            'axis': {'range': [None, max_value]},
            'bar': {'color': "darkblue"},
            'steps': [
                {'range': [0, thresholds['low']], 'color': "lightgray"},
                {'range': [thresholds['low'], thresholds['medium']], 'color': "gray"},
                {'range': [thresholds['medium'], max_value], 'color': "darkgray"}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': max_value * 0.9
            }
        }
    ))
    
    return fig
