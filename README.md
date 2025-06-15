# Enterprise BI Agent – Big Data Powered RAG System

![License](https://img.shields.io/badge/license-MIT-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Platform](https://img.shields.io/badge/platform-GCP-lightgrey)
![RAG](https://img.shields.io/badge/RAG-LangChain-blue?logo=langchain)
![LLM](https://img.shields.io/badge/LLM-OpenAI-informational?logo=openai)
![ETL](https://img.shields.io/badge/Data%20Pipeline-Spark-orange?logo=apachespark)

---

##  Overview

**Enterprise BI Agent** is a scalable, production-ready Retrieval-Augmented Generation (RAG) system designed for intelligent Q&A over enterprise PDFs and CSVs. It uses **Apache Spark** for data profiling, **LangChain + OpenAI** for retrieval and generation, and **FastAPI** as a secure backend — all deployed via **Google Cloud Run**.

The system is optimized for structured and unstructured documents, supporting real-time semantic search with metadata indexing.

---

## Preview

> _Semantic Q&A over documents + API demo screenshot here (insert later)_

<!-- Replace with actual image when ready -->
![Preview Placeholder](https://github.com/vengotimuktha/enterprise-bi-agent/blob/main/docs/demo-placeholder.png)

---

##  Project Structure

```bash
enterprise_bi_agent/
├── spark_etl.py                  # Full Spark ETL pipeline
├── spark_cleaning.py            # Structured/unstructured cleansing
├── spark_feature_engineering.py # Feature extraction
├── pdf_metadata_extractor.py    # Extracts PDF text + metadata
├── build_faiss_retriever.py     # OpenAI + FAISS indexing
├── rag_pipeline.py              # LangChain RAG pipeline
├── main.py                      # FastAPI entry point
├── rag_service.py               # Modular RAG API logic
├── config.py, schemas.py        # Config and Pydantic schemas
├── frontend/                    # React/Streamlit POC frontend
├── data/, vectorstores/         # Document input + FAISS DB

---
```
##  Key Features

### 🔹 Spark-Based Document Pipeline

- Batch ingestion of PDF and CSV files  
- Cleansing, deduplication, null handling  
- Metadata profiling (rows, columns, stats)  
- Feature extraction from PDF metadata  

### 🔹 Retrieval-Augmented Generation (RAG)

- Chunking via `RecursiveCharacterTextSplitter`  
- Embeddings via `text-embedding-ada-002`  
- Indexed with FAISS  
- RAG pipeline using `gpt-3.5-turbo`  

### 🔹 FastAPI Backend

- REST endpoints for:
  - `/upload` – File ingestion  
  - `/index` – FAISS vector build  
  - `/query` – Answer generation  
- OpenAPI docs at `localhost:8000/docs`  
- Clean MVC structure with schema validation  

### 🔹 GCP Cloud Deployment

- Dockerized app deployed on **Cloud Run**  
- File storage via **Google Cloud Storage (GCS)**  
- Config management via `.env` and `config.py`  

---

## Installation & Setup

# 1. Clone repo
git clone https://github.com/vengotimuktha/enterprise-bi-agent.git
cd enterprise-bi-agent

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate     # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set OpenAI API Key
echo "OPENAI_API_KEY=your-key-here" > .env

---

## Run the App Locally

# Start FastAPI backend
uvicorn main:app --reload
- Open http://localhost:8000/docs for Swagger API.

---

##  Index & Query Documents
 
# Build vector store
python build_faiss_retriever.py

# Run semantic Q&A
python rag_pipeline.py

---

##  GCP Cloud Deployment

**Prerequisites:** Docker, GCP CLI, Cloud Run + Artifact Registry enabled


# 1. Build Docker image
docker build -t enterprise-bi-agent .

# 2. Tag image
docker tag enterprise-bi-agent gcr.io/YOUR_PROJECT_ID/enterprise-bi-agent

# 3. Push to Artifact Registry
docker push gcr.io/YOUR_PROJECT_ID/enterprise-bi-agent

# 4. Deploy to Cloud Run
gcloud run deploy enterprise-bi-agent \
  --image gcr.io/YOUR_PROJECT_ID/enterprise-bi-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated

---

##  Example Query (via API)

# Request:

json
Copy code
POST /query
{
  "query": "Summarize all payment terms from recent PDFs",
  "top_k": 3
}

# Sample Response:

json
Copy code
{
  "answer": "The most common payment terms include Net 30, due on receipt, and early-pay discounts..."
}

---

##  Future Enhancements

-  Migrate FAISS to ChromaDB or Weaviate  
-  Add GitHub Actions for CI/CD  
-  Support for HuggingFace/BGE embeddings  
-  Integrate LangSmith tracing & observability  
-  Add Firebase/Auth0 for user authentication  

---

##  Acknowledgments

Built as part of an enterprise-grade AI challenge with inspiration from:

- [LangChain](https://github.com/langchain-ai/langchain)  
- [Apache Spark](https://spark.apache.org/)  
- [OpenAI](https://platform.openai.com/)  
- [FAISS](https://github.com/facebookresearch/faiss)  

---

##  License

This project is licensed under the [MIT License](LICENSE).

