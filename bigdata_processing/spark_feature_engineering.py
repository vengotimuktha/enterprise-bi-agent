from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lag, round, avg, to_timestamp
from pyspark.sql.window import Window
import datetime

# Initialize Spark
spark = SparkSession.builder \
    .appName("Enterprise Feature Engineering") \
    .getOrCreate()

# Paths
input_path = "bigdata_processing/output/combined_csv/"
output_path = "bigdata_processing/output/engineered_data/"
log_path = "logs/pipeline_log.csv"

# Read CSV
df = spark.read.option("header", True).csv(input_path)

# Safe print for debugging columns
print("[DEBUG] Original columns (escaped):", [c.encode('unicode_escape').decode('utf-8') for c in df.columns])

# Rename columns with invalid characters
df = df.withColumnRenamed("Close\ufffd", "Close") \
       .withColumnRenamed("Adj Close\ufffd", "Adj_Close")

# Cast numeric columns
numeric_cols = ["Open", "High", "Low", "Close", "Adj_Close", "Volume"]
for col_name in numeric_cols:
    df = df.withColumn(col_name, col(col_name).cast("double"))

# Define rolling window
window_spec = Window.partitionBy("company").orderBy("Date")

# Feature engineering
df = df \
    .withColumn("daily_change", round(col("Close") - col("Open"), 2)) \
    .withColumn("daily_range", round(col("High") - col("Low"), 2)) \
    .withColumn("volatility_percent", round((col("High") - col("Low")) / col("Open") * 100, 2)) \
    .withColumn("7day_moving_avg", round(avg("Close").over(window_spec.rowsBetween(-6, 0)), 2)) \
    .withColumn("cumulative_return", round((col("Close") / lag("Close", 1).over(window_spec)) - 1, 4))

# Save as Parquet
df.write.mode("overwrite").parquet(output_path)

# Logging
log_df = spark.createDataFrame([{
    "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "step": "Feature Engineering",
    "status": "Success",
    "output_path": output_path
}])
log_df.write.mode("append").option("header", True).csv(log_path)

print("[SUCCESS] Feature engineering completed and saved to:", output_path)

# Stop Spark
spark.stop()
