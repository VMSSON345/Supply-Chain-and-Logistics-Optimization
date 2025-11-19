"""
Streamlit Dashboard - Main App (Home Page)
"""
import streamlit as st

# Page config
st.set_page_config(
    page_title="Retail Analytics Dashboard",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Function to load CSS
def load_css(file_name):
    try:
        with open(file_name) as f:
            st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)
    except FileNotFoundError:
        st.warning("CSS file not found. Please check path.")

# Load custom CSS from file
load_css("dashboard/assets/style.css")

# Main header
st.markdown('<h1 class="main-header">🛒 Retail Analytics Dashboard</h1>', 
            unsafe_allow_html=True)
st.markdown("---")

# Home page content
st.header("Welcome to the Supply Chain & Logistics Dashboard")
st.info("""
**Điều hướng (Navigation):**

Sử dụng thanh bên (sidebar) ở bên trái để truy cập các trang khác nhau của hệ thống:

- 📊 **Real-time Overview**: Theo dõi doanh thu, giao dịch, và sản phẩm bán chạy nhất trong 15 phút qua.

- 📈 **Demand Forecasting**: Xem dự báo nhu cầu cho 30 ngày tới cho từng sản phẩm.

- 🛒 **Market Basket**: Phân tích các luật kết hợp để tìm ra sản phẩm nào thường được mua cùng nhau.

- 📦 **Inventory Optimization**: Xem các cảnh báo tồn kho và khuyến nghị về mức tồn kho an toàn (safety stock).
""")

st.markdown("---")

# Sidebar
st.sidebar.title("Giới thiệu")
st.sidebar.info("""
Hệ thống này phân tích dữ liệu bán lẻ bằng kiến trúc Big Data (Kafka, Spark, Elasticsearch) để cung cấp thông tin chi tiết về:

- **Tốc độ (Speed Layer)**: Dữ liệu streaming real-time.

- **Batch (Batch Layer)**: Phân tích sâu (Dự báo, Phân khúc).

- **Phục vụ (Serving Layer)**: Lưu trữ và truy vấn kết quả.
""")

st.sidebar.title("⚙️ System Status")
st.sidebar.markdown("---")
st.sidebar.success("✅ Kafka: Running")
st.sidebar.success("✅ Spark: Active")
st.sidebar.success("✅ Elasticsearch: Connected")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🛒 Retail Analytics Dashboard | Powered by Spark, Kafka & Elasticsearch</p>
</div>
""", unsafe_allow_html=True)


# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>🛒 Retail Analytics Dashboard | Powered by Spark, Kafka & Elasticsearch</p>
</div>
""", unsafe_allow_html=True)
