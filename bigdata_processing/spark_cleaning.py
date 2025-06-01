import os
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, lit
from datetime import datetime
from pyspark.sql.functions import input_file_name, current_timestamp, monotonically_increasing_id, lit
# === Creating Spark Session ===
spark = SparkSession.builder \
    .appName("Enterprise Data Cleaning") \
    .master("local[*]") \
    .getOrCreate()

# === Defining Paths ===
input_path = "bigdata_processing/output/combined_csv"
output_path = "output/cleaned.parquet"
log_path = "logs/cleaning_log.csv"

os.makedirs("output", exist_ok=True)
os.makedirs("logs", exist_ok=True)

# === Starting Log ===
log_entries = []
log_time = datetime.now().isoformat()

def log_step(step, status):
    log_entries.append((log_time, step, status))

try:
    # ===  Loading Input CSV ===
    df = spark.read.option("header", True).csv(input_path)
    # Cleaning column names by removing unwanted characters
    df = df.toDF(*[col.strip().replace("�", "") for col in df.columns])
    log_step("Read CSV", "Success")

    # Adding enrichment columns
    df = df.withColumn("record_id", monotonically_increasing_id()) \
       .withColumn("ingested_at", current_timestamp()) \
       .withColumn("source_file", input_file_name()) \
       .withColumn("company", lit("enterprise_bi"))

    # === Droping fully null rows ===
    df = df.dropna(how="all")
    log_step("Drop fully null rows", "Success")

    # === Filling partial missing values if any ===
    df = df.fillna({
        "Open": "0",
        "High": "0",
        "Low": "0",
        "Close": "0",
        "Volume": "0"
    })
    log_step("Fill partial nulls", "Success")

    # === Cast types  ===
    numeric_fields = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col_name in numeric_fields:
        if col_name in df.columns:
            df = df.withColumn(col_name, col(col_name).cast("double"))
    log_step("Cast types", "Success")

    # ===  Saving to Parquet ===
    df.write.mode("overwrite").parquet(output_path)
    log_step("Write cleaned.parquet", "Success")

except Exception as e:
    log_step("Exception", str(e))
    raise

# ===  Writing log to CSV ===
log_df = spark.createDataFrame(log_entries, ["timestamp", "step", "status"])
log_df.coalesce(1).write.mode("overwrite").option("header", True).csv(log_path)

# === Finalizing ===
print("[SUCCESS] Data cleaned and saved to:", output_path)
spark.stop()
