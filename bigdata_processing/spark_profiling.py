import os
from pyspark.sql import SparkSession
from datetime import datetime

# === Setingup ===
spark = SparkSession.builder \
    .appName("Enterprise Data Profiling") \
    .master("local[*]") \
    .getOrCreate()

input_path = "output/cleaned.parquet"
summary_path = "profile/summary.parquet"
log_path = "logs/pipeline_log.csv"

os.makedirs("profile", exist_ok=True)
os.makedirs("logs", exist_ok=True)

log_time = datetime.now().isoformat()
log_entries = []

def log_step(step, status):
    log_entries.append((log_time, step, status))

try:
    # === Loading Cleaned Data ===
    df = spark.read.parquet(input_path)
    row_count = df.count()
    log_step("Load Cleaned Parquet", f"Success ({row_count} rows)")

    # ===  Generating Summary ===
    summary = df.summary("count", "mean", "stddev", "min", "max")
    summary.write.mode("overwrite").parquet(summary_path)
    log_step("Profile Summary", "Saved to profile/summary.parquet")

except Exception as e:
    log_step("Exception", str(e))
    raise

# === Writing Pipeline Log ===
log_df = spark.createDataFrame(log_entries, ["timestamp", "step", "status"])
log_df.coalesce(1).write.mode("append").option("header", True).csv(log_path)

print("[SUCCESS] Profiling completed. Summary written.")
spark.stop()
