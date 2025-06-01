import os
import json
import fitz  # PyMuPDF
import pandas as pd
from datetime import datetime

# Input and output paths
pdf_dir = "data/pdf"
metadata_csv_path = "data/spark_pdf_metadata.csv"
chunks_json_path = "data/spark_pdf_chunks.json"

# Storage
metadata_records = []
chunk_records = []

# Iterating through all PDFs
for filename in os.listdir(pdf_dir):
    if filename.endswith(".pdf"):
        file_path = os.path.join(pdf_dir, filename)
        try:
            doc = fitz.open(file_path)
            metadata = doc.metadata or {}

            # --- Store metadata ---
            metadata_records.append({
                "file_name": filename,
                "title": metadata.get("title", ""),
                "author": metadata.get("author", ""),
                "subject": metadata.get("subject", ""),
                "creation_date": metadata.get("creationDate", ""),
                "mod_date": metadata.get("modDate", ""),
                "page_count": doc.page_count,
                "extracted_at": datetime.now().isoformat()
            })

            # --- Storing chunked text by page ---
            for page_number in range(doc.page_count):
                page = doc.load_page(page_number)
                text = page.get_text().strip()
                if text:
                    chunk_records.append({
                        "file_name": filename,
                        "page": page_number + 1,
                        "text": text,
                        "source": f"{filename}#page={page_number + 1}"
                    })

            doc.close()

        except Exception as e:
            print(f"[ERROR] Failed to process {filename}: {e}")

# Saving metadata to CSV
pd.DataFrame(metadata_records).to_csv(metadata_csv_path, index=False)
print(f"[SUCCESS] PDF metadata saved to {metadata_csv_path}")

# Saving chunks to JSON
with open(chunks_json_path, "w", encoding="utf-8") as f:
    json.dump(chunk_records, f, ensure_ascii=False, indent=2)
print(f"[SUCCESS] PDF chunks saved to {chunks_json_path}")
