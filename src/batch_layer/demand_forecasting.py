"""
Demand Forecasting using Prophet (Time Series) - Distributed & Hybrid Strategy
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, max as _max, current_date, date_sub, lit
from pyspark.sql.types import StructType, StructField, StringType, DateType, DoubleType, FloatType
import pandas as pd
import numpy as np  # [NEW] Import numpy cho Log/Exp transform
from prophet import Prophet
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DemandForecaster:
    """Dự báo nhu cầu sản phẩm sử dụng Prophet chạy song song trên Spark"""
    
    def __init__(self, spark: SparkSession):
        self.spark = spark
    
    def prepare_timeseries_data(self, df):
        """
        Chuẩn bị dữ liệu time series: aggregate theo ngày và sản phẩm
        """
        logger.info("Preparing time series data...")
        from pyspark.sql.functions import to_date
        
        # Aggregate daily sales
        daily_sales = df.groupBy(
            'StockCode',
            to_date('InvoiceDate').alias('Date')
        ).agg(
            _sum('Quantity').alias('Quantity'),
            _sum(col('Quantity') * col('UnitPrice')).alias('Revenue')
        )
        
        return daily_sales

    def forecast_all_products(self, daily_sales_df, top_n=100, periods=30):
        """
        Dự báo song song sử dụng Spark Pandas UDF với chiến lược Hybrid
        """
        logger.info(f"Forecasting demand for top {top_n} products using Distributed Spark (Hybrid Strategy)...")
        
        # 1. Lọc lấy Top N sản phẩm
        top_products_df = daily_sales_df.groupBy('StockCode') \
            .agg(_sum('Quantity').alias('TotalQuantity')) \
            .orderBy(col('TotalQuantity').desc()) \
            .limit(top_n) \
            .select('StockCode')
        
        # Join lại
        data_candidate = daily_sales_df.join(top_products_df, on='StockCode', how='inner')

        # Lọc bỏ "Zombie Products" (30 ngày không bán)
        max_date = data_candidate.agg(_max('Date')).collect()[0][0]
        active_cutoff = date_sub(lit(max_date), 30)  # 30 ngày trước ngày cuối cùng
        
        active_products = data_candidate.groupBy('StockCode') \
            .agg(_max('Date').alias('LastSaleDate')) \
            .filter(col('LastSaleDate') >= active_cutoff) \
            .select('StockCode')
            
        filtered_data = data_candidate.join(active_products, on='StockCode', how='inner')
        
        # 2. Định nghĩa Schema
        result_schema = StructType([
            StructField("StockCode", StringType()),
            StructField("ds", DateType()),
            StructField("yhat", FloatType()),
            StructField("yhat_lower", FloatType()),
            StructField("yhat_upper", FloatType())
        ])

        # 3. Định nghĩa hàm huấn luyện (UDF) với logic HYBRID
        def forecast_store_item(history_pd: pd.DataFrame) -> pd.DataFrame:
            # Kiểm tra dữ liệu tối thiểu
            if len(history_pd) < 10:  # Giảm yêu cầu dữ liệu tối thiểu
                return pd.DataFrame(columns=['StockCode', 'ds', 'yhat', 'yhat_lower', 'yhat_upper'])

            stock_code = history_pd['StockCode'].iloc[0]
            
            # --- BƯỚC 1: Phân tích đặc tính dữ liệu (Data Profiling) ---
            # Điền ngày trống để tính toán độ thưa thớt chính xác
            history_pd['Date'] = pd.to_datetime(history_pd['Date'])
            full_date_range = pd.date_range(start=history_pd['Date'].min(), end=history_pd['Date'].max())
            df_filled = history_pd.set_index('Date').reindex(full_date_range).fillna({'Quantity': 0}).rename_axis('ds').reset_index()
            
            # Tính tỷ lệ ngày có bán hàng (Non-zero ratio)
            non_zero_days = df_filled[df_filled['Quantity'] > 0].shape[0]
            total_days = df_filled.shape[0]
            sparsity_ratio = 1.0 - (non_zero_days / total_days)
            
            # Quyết định chiến lược: Nếu > 60% ngày là số 0 -> Hàng bán ngắt quãng (Intermittent)
            is_intermittent = sparsity_ratio > 0.6
            
            df_train = df_filled.copy()
            df_train.rename(columns={'Quantity': 'y'}, inplace=True)

            # --- BƯỚC 2: Áp dụng chiến lược ---
            if is_intermittent:
                # === CHIẾN LƯỢC A: HÀNG BÁN CHẬM/NGẮT QUÃNG ===
                # Không cắt Outlier (vì đỉnh cao là tín hiệu quan trọng)
                # Không Log Transform (để giữ nguyên biên độ dao động)
                # Dùng seasonality_mode='multiplicative' để bắt các đỉnh cao
                
                m = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    changepoint_prior_scale=0.5,  # Tăng thật cao để bắt kịp các thay đổi đột ngột
                    seasonality_mode='multiplicative'  # Quan trọng cho dữ liệu có đỉnh nhọn
                )
            else:
                # === CHIẾN LƯỢC B: HÀNG BÁN ĐỀU/NHANH (High Volume) ===
                # Áp dụng tất cả tối ưu hóa để làm mượt đường dự báo
                
                # 1. Cắt Outlier
                Q1 = df_train['y'].quantile(0.25)
                Q3 = df_train['y'].quantile(0.75)
                upper_limit = Q3 + 1.5 * (Q3 - Q1)
                df_train['y'] = df_train['y'].clip(upper=upper_limit)
                
                # 2. Log Transform
                df_train['y'] = np.log1p(df_train['y'])
                
                m = Prophet(
                    daily_seasonality=False,
                    weekly_seasonality=True,
                    yearly_seasonality=True,
                    changepoint_prior_scale=0.05,  # Giữ thấp để đường trend mượt
                    seasonality_prior_scale=10.0,
                    holidays_prior_scale=20.0,
                    seasonality_mode='additive'
                )

            # Thêm ngày lễ (chung cho cả 2)
            m.add_country_holidays(country_name='UK')

            try:
                m.fit(df_train)
                future = m.make_future_dataframe(periods=periods)
                forecast = m.predict(future)
                
                # Lấy kết quả
                results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                
                # Nếu là chiến lược B (có Log), cần nghịch đảo lại
                if not is_intermittent:
                    for col_name in ['yhat', 'yhat_lower', 'yhat_upper']:
                        results[col_name] = np.expm1(results[col_name])
                
                # Clean up số âm
                for col_name in ['yhat', 'yhat_lower', 'yhat_upper']:
                    results[col_name] = results[col_name].clip(lower=0)
                
                results['StockCode'] = stock_code
                
                return results[['StockCode', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            except Exception as e:
                # Fallback nếu lỗi
                return pd.DataFrame(columns=['StockCode', 'ds', 'yhat', 'yhat_lower', 'yhat_upper'])

        # 4. Chạy Parallel: ApplyInPandas
        # Spark sẽ chia dữ liệu theo StockCode và gửi đến các workers
        forecasts_spark = filtered_data.groupBy('StockCode') \
            .applyInPandas(forecast_store_item, schema=result_schema)

        logger.info("Hybrid forecasting calculation submitted to Spark Cluster...")
        
        return forecasts_spark.toPandas()

    def calculate_safety_stock(self, forecast_df, service_level=0.95):
        """
        Tính toán safety stock (Adaptive)
        """
        from scipy import stats
        
        z_score = stats.norm.ppf(service_level)
        
        # Group by product
        safety_stock = forecast_df.groupby('StockCode').agg({
            'yhat': 'mean',
            'yhat_upper': 'max',
            'yhat_lower': 'min'
        }).reset_index()
        
        # [NEW FORMULA] Công thức Safety Stock thích ứng
        # std_demand: Ước lượng độ lệch chuẩn cầu dựa trên khoảng tin cậy của Prophet
        safety_stock['std_demand'] = (safety_stock['yhat_upper'] - safety_stock['yhat_lower']) / 4
        
        def adaptive_reorder_point(row):
            base_demand = row['yhat']
            safety_buffer = z_score * row['std_demand']
            
            # Với hàng bán chậm, Upper Bound quan trọng hơn Mean
            if base_demand < 1 or row['std_demand'] > base_demand:
                return (row['yhat_upper'] * 0.8) + safety_buffer
            else:
                return base_demand + safety_buffer

        safety_stock['safety_stock'] = z_score * safety_stock['std_demand']
        safety_stock['reorder_point'] = safety_stock.apply(adaptive_reorder_point, axis=1)
        
        return safety_stock


def run_demand_forecasting_job(spark, input_path, output_path, es_config):
    """Job chính để chạy demand forecasting"""
    logger.info("="*60)
    logger.info("Starting Demand Forecasting Job (Hybrid Strategy)")
    logger.info("="*60)
    
    # 1. Load data
    logger.info(f"Loading data from {input_path}")
    if input_path.endswith('.csv'):
        df = spark.read.csv(input_path, header=True, inferSchema=True)
    else:
        df = spark.read.parquet(input_path)
    
    # 2. Clean data
    from pyspark.sql.functions import to_timestamp
    df_clean = df.filter(
        (col('Quantity') > 0) &
        (col('UnitPrice') > 0)
    ).withColumn('InvoiceDate', to_timestamp('InvoiceDate'))
    
    # 3. Create forecaster
    forecaster = DemandForecaster(spark)
    
    # 4. Prepare time series data
    daily_sales = forecaster.prepare_timeseries_data(df_clean)
    
    # 5. Forecast for top products (Distributed)
    forecasts = forecaster.forecast_all_products(
        daily_sales,
        top_n=50,
        periods=30
    )
    
    # 6. Calculate safety stock
    safety_stock = forecaster.calculate_safety_stock(forecasts, service_level=0.95)
    
    # 7. Save results
    forecasts_path = f"{output_path}/demand_forecasts.csv"
    safety_stock_path = f"{output_path}/safety_stock.csv"
    
    logger.info(f"Saving forecasts to {forecasts_path}")
    forecasts.to_csv(forecasts_path, index=False)
    
    logger.info(f"Saving safety stock to {safety_stock_path}")
    safety_stock.to_csv(safety_stock_path, index=False)
    
    # 8. Save to Elasticsearch
    if es_config:
        logger.info("Saving forecasts to Elasticsearch...")
        from elasticsearch import Elasticsearch, helpers
        es = Elasticsearch([es_config['host']])
        
        # Prepare forecast documents
        forecast_actions = [
            {
                "_index": "retail_demand_forecasts",
                "_id": f"{row['StockCode']}_{row['ds'].strftime('%Y-%m-%d')}",
                "_source": {
                    'StockCode': row['StockCode'],
                    'Date': row['ds'].isoformat(),
                    'ForecastQuantity': float(row['yhat']),
                    'LowerBound': float(row['yhat_lower']),
                    'UpperBound': float(row['yhat_upper'])
                }
            }
            for _, row in forecasts.iterrows()
        ]
        
        helpers.bulk(es, forecast_actions)
        logger.info(f"Indexed {len(forecast_actions)} forecast records")
        
        # Prepare safety stock documents
        safety_actions = [
            {
                "_index": "retail_safety_stock",
                "_id": row['StockCode'],
                "_source": row.to_dict()
            }
            for _, row in safety_stock.iterrows()
        ]
        
        helpers.bulk(es, safety_actions)
        logger.info(f"Indexed {len(safety_actions)} safety stock records")
    
    logger.info("="*60)
    logger.info("Demand Forecasting Job Completed")
    logger.info("="*60)
    
    return forecasts, safety_stock


if __name__ == "__main__":
    spark = SparkSession.builder \
        .appName("RetailDemandForecasting") \
        .config("spark.driver.memory", "4g") \
        .config("spark.executor.memory", "4g") \
        .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
        .getOrCreate()
    
    INPUT_PATH = "/opt/spark-data/raw/online_retail.csv"
    OUTPUT_PATH = "/opt/spark-data/processed"
    ES_CONFIG = {'host': 'http://elasticsearch:9200'}
    
    run_demand_forecasting_job(spark, INPUT_PATH, OUTPUT_PATH, ES_CONFIG)
    
    spark.stop()
