import os
import sys
import io
from pyspark.sql import SparkSession
from pyspark.sql.functions import input_file_name, lit, regexp_extract
from datetime import datetime

# Creating Spark session
spark = SparkSession.builder \
    .appName("Enterprise Data Ingestion") \
    .master("local[*]") \
    .getOrCreate()

# Paths
csv_folder = os.path.abspath("data/csv/")
output_folder = os.path.abspath("bigdata_processing/output/")
log_path = os.path.abspath("logs/ingestion_log.txt")

# Ensuring output and log folders exist
os.makedirs(output_folder, exist_ok=True)
os.makedirs(os.path.dirname(log_path), exist_ok=True)

def log(message):
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    with open(log_path, "a") as log_file:
        log_file.write(f"[{timestamp}] {message}\n")

# Starting logging
log(f"Starting ingestion from folder: {csv_folder}")

# Checking files
if not os.path.exists(csv_folder):
    log(f"[ERROR] Folder does not exist: {csv_folder}")
    exit(1)

files = os.listdir(csv_folder)
if not any(file.endswith(".csv") for file in files):
    log("[ERROR] No CSV files found.")
    exit(1)

try:
    # Reading all CSVs
    df = spark.read.option("header", True).csv(f"{csv_folder}/*.csv") \
        .withColumn("file_path", input_file_name()) \
        .withColumn("company", regexp_extract("file_path", r"\/(\w+)[\-_]", 1)) \
        .withColumn("ingested_at", lit(datetime.now().isoformat()))

    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
    df.show(5, truncate=False)

    # Saving as CSV and Parquet
    df.coalesce(1).write.mode("overwrite").option("header", True).csv(os.path.join(output_folder, "combined_csv"))
    df.write.mode("overwrite").parquet(os.path.join(output_folder, "combined.parquet"))

    log("[SUCCESS] Ingestion completed and saved to output folder.")

except Exception as e:
    log(f"[FAILURE] Ingestion error: {str(e)}")
    raise

spark.stop()
