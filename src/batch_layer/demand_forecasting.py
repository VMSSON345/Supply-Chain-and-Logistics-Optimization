"""
Demand Forecasting using Prophet (Time Series) - Distributed & Optimized Version
"""
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sum as _sum, current_date
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
        Dự báo song song sử dụng Spark Pandas UDF với Advanced Preprocessing
        """
        logger.info(f"Forecasting demand for top {top_n} products using Distributed Spark (Optimized)...")
        
        # 1. Lọc lấy Top N sản phẩm
        top_products_df = daily_sales_df.groupBy('StockCode') \
            .agg(_sum('Quantity').alias('TotalQuantity')) \
            .orderBy(col('TotalQuantity').desc()) \
            .limit(top_n) \
            .select('StockCode')
        
        # Join lại để chỉ lấy dữ liệu của top products
        filtered_data = daily_sales_df.join(top_products_df, on='StockCode', how='inner')
        
        # 2. Định nghĩa Schema kết quả trả về
        result_schema = StructType([
            StructField("StockCode", StringType()),
            StructField("ds", DateType()),
            StructField("yhat", FloatType()),
            StructField("yhat_lower", FloatType()),
            StructField("yhat_upper", FloatType())
        ])

        # 3. Định nghĩa hàm huấn luyện (UDF) chứa logic TỐI ƯU HÓA
        def forecast_store_item(history_pd: pd.DataFrame) -> pd.DataFrame:
            # Kiểm tra dữ liệu tối thiểu
            if len(history_pd) < 20:
                return pd.DataFrame(columns=['StockCode', 'ds', 'yhat', 'yhat_lower', 'yhat_upper'])

            # Lấy StockCode hiện tại
            stock_code = history_pd['StockCode'].iloc[0]
            
            # [OPTIMIZATION 1] Lấp đầy ngày trống (Gap Filling)
            history_pd['Date'] = pd.to_datetime(history_pd['Date'])
            full_date_range = pd.date_range(start=history_pd['Date'].min(), end=history_pd['Date'].max())
            
            # Reindex: Tạo các dòng ngày còn thiếu và điền Quantity = 0
            df_opt = history_pd.set_index('Date').reindex(full_date_range).fillna({'Quantity': 0}).rename_axis('ds').reset_index()
            
            # [OPTIMIZATION 2] Xử lý Outlier bằng IQR (Cắt ngọn)
            Q1 = df_opt['Quantity'].quantile(0.25)
            Q3 = df_opt['Quantity'].quantile(0.75)
            IQR = Q3 - Q1
            upper_limit = Q3 + 1.5 * IQR
            
            # Clip dữ liệu: Giới hạn giá trị trần, không xóa dòng
            df_opt['y'] = df_opt['Quantity'].clip(upper=upper_limit)

            # [OPTIMIZATION 3] Log Transform (Giảm phương sai)
            # Sử dụng log1p (log(1+x)) để xử lý số 0 tốt hơn
            df_opt['y'] = np.log1p(df_opt['y'])

            # Chuẩn bị dataframe cho Prophet
            df_prophet = df_opt[['ds', 'y']].sort_values('ds')

            # [OPTIMIZATION 4] Cấu hình Model tối ưu
            m = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=True,
                changepoint_prior_scale=0.05,  # Giảm để tránh overfit với nhiễu
                seasonality_prior_scale=10.0,
                holidays_prior_scale=20.0,     # Tăng độ nhạy với ngày lễ
                seasonality_mode='additive'
            )
            
            # Thêm ngày lễ UK
            m.add_country_holidays(country_name='UK')

            try:
                m.fit(df_prophet)
                future = m.make_future_dataframe(periods=periods)
                forecast = m.predict(future)
                
                # Lấy kết quả
                results = forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail(periods)
                
                # [OPTIMIZATION 5] Nghịch đảo (Inverse Transform)
                # Chuyển từ Log Scale về số thực bằng expm1 (exp(x)-1)
                for col_name in ['yhat', 'yhat_lower', 'yhat_upper']:
                    results[col_name] = np.expm1(results[col_name]).clip(lower=0)
                
                results['StockCode'] = stock_code
                
                return results[['StockCode', 'ds', 'yhat', 'yhat_lower', 'yhat_upper']]
            except Exception as e:
                # Fallback nếu lỗi
                return pd.DataFrame(columns=['StockCode', 'ds', 'yhat', 'yhat_lower', 'yhat_upper'])

        # 4. Chạy Parallel: ApplyInPandas
        # Spark sẽ chia dữ liệu theo StockCode và gửi đến các workers
        forecasts_spark = filtered_data.groupBy('StockCode') \
            .applyInPandas(forecast_store_item, schema=result_schema)

        logger.info("Optimized forecasting calculation submitted to Spark Cluster...")
        
        return forecasts_spark.toPandas()

    def calculate_safety_stock(self, forecast_df, service_level=0.95):
        """
        Tính toán safety stock dựa trên forecast
        """
        from scipy import stats
        
        z_score = stats.norm.ppf(service_level)
        
        # Group by product và tính safety stock
        safety_stock = forecast_df.groupby('StockCode').agg({
            'yhat': 'mean',
            'yhat_upper': 'max',
            'yhat_lower': 'min'
        }).reset_index()
        
        # Safety stock formula
        # Khoảng tin cậy (upper - lower) phản ánh sự không chắc chắn của mô hình
        safety_stock['std_demand'] = (safety_stock['yhat_upper'] - safety_stock['yhat_lower']) / 4
        safety_stock['safety_stock'] = z_score * safety_stock['std_demand']
        safety_stock['reorder_point'] = safety_stock['yhat'] + safety_stock['safety_stock']
        
        return safety_stock


def run_demand_forecasting_job(spark, input_path, output_path, es_config):
    """Job chính để chạy demand forecasting"""
    logger.info("="*60)
    logger.info("Starting Demand Forecasting Job (Advanced Optimized)")
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
