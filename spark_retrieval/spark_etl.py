from pyspark.sql import SparkSession
from datetime import datetime
import os
import shutil

os.environ["PYSPARK_PYTHON"] = r"C:\Users\mukth\Desktop\enterprise_bi_agent\env\Scripts\python.exe"
# Create SparkSession
spark = SparkSession.builder \
    .appName("BI Agent MLflow Analytics ETL") \
    .getOrCreate()

# Simulated log data (in real case, extract from mlruns or query logs)
data = [
    ("What is elleftr?", "gpt-3.5-turbo", 120, "2024-05-01 10:22:15"),
    ("Define vera-elleftr", "gpt-3.5-turbo", 138, "2024-05-01 10:23:02"),
    ("Meaning of elptr", "gpt-4", 165, "2024-05-01 10:24:19"),
]

columns = ["question", "model", "response_length", "timestamp"]

# Convert to DataFrame
df = spark.createDataFrame(data, columns)

# Convert timestamp column to proper type
df = df.withColumn("timestamp", df["timestamp"].cast("timestamp"))

# Show preview
df.show()

# Save as CSV
output_path = "data/analytics_output"
os.makedirs(output_path, exist_ok=True)
df.write.mode("overwrite").option("header", "true").csv(f"{output_path}/analytics.csv")
# Merge part-files into a single CSV
output_dir = f"{output_path}/analytics.csv"
final_csv_path = f"{output_path}/analytics_final.csv"

if os.path.exists(output_dir):
    with open(final_csv_path, "wb") as fout:
        for filename in sorted(os.listdir(output_dir)):
            if filename.startswith("part-") and filename.endswith(".csv"):
                with open(os.path.join(output_dir, filename), "rb") as f:
                    shutil.copyfileobj(f, fout)
    print("✅ Merged CSV saved to:", final_csv_path)
else:
    print("❌ Output directory not found:", output_dir)

# Optional: Save as Parquet (for Power BI or advanced dashboards)
df.write.mode("overwrite").parquet(f"{output_path}/analytics.parquet")

print("ETL complete. Output saved to:", output_path)
