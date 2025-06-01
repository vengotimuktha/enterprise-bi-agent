# Enterprise BI Agent – Big Data Powered RAG System

![License](https://img.shields.io/badge/license-MIT-blue)
![Build](https://img.shields.io/badge/build-passing-brightgreen)
![Python](https://img.shields.io/badge/python-3.11-blue)
![Platform](https://img.shields.io/badge/platform-GCP-lightgrey)

##  Overview

**Enterprise BI Agent** is a scalable, production-ready Retrieval-Augmented Generation (RAG) system built for intelligent document analysis and Q&A. It supports massive document pipelines using **Apache Spark**, **FAISS**, and **LangChain + OpenAI** — all served through a **FastAPI** backend and deployed on **Google Cloud Platform (GCP)**.

The system supports PDF & CSV ingestion, Spark-based cleaning/profiling, and real-time Q&A with semantically enriched answers.

---

##  Project Structure
enterprise_bi_agent/

- `spark_etl.py` # End-to-end ETL with Spark
- `spark_cleaning.py` # Cleansing of structured/unstructured data
- `spark_feature_engineering.py` # Feature extraction from PDF/CSV
- spark_profiling.py # Metadata profiling (row count, schema stats)

- `pdf_metadata_extractor.py` # Extract PDF metadata & text
- `build_faiss_retriever.py` # Vector indexer using OpenAI + FAISS
- `rag_pipeline.py` # Core LangChain RAG logic

- `main.py` # FastAPI entry point
- `rag_service.py` # Modular service layer for RAG
- `upload_handler.py` # File upload + chunking logic
- `config.py` # Central config/env loader
- `schemas.py`,`models.py` # Pydantic schema and internal models

- `frontend/` # React/HTML (or Streamlit for POC)
- `data/`, `vectorstores/` # Document store and FAISS DB

  
---

##  Features

###  Spark-Based Data Pipeline
- Batch PDF/CSV ingestion
- Cleansing, deduplication, null handling
- Profiling: schema info, stats, row completeness
- Feature extraction (e.g., title, author, topic from metadata)

###  Retrieval-Augmented Generation (RAG)
- Text split via `RecursiveCharacterTextSplitter`
- Embedding via OpenAI (`text-embedding-ada-002`)
- Vector DB with FAISS
- Retrieval + generation using `gpt-3.5-turbo`

###  FastAPI Backend
- Modular REST APIs for:
  - File upload
  - Document indexing
  - RAG query-answering
- Pydantic schemas and clean MVC-style routing

###  GCP Cloud Deployment
- Hosted via **GCP Cloud Run** for containerized inference
- File storage via **GCS** (Google Cloud Storage)
- Configured using `.env` and `config.py`

---

##  Installation

```bash
# 1. Clone repository
git clone https://github.com/vengotimuktha/enterprise-bi-agent.git
cd enterprise-bi-agent

# 2. Setup virtual environment
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Set your OpenAI key
echo "OPENAI_API_KEY=your-key-here" > .env

```

Usage
Local Server (FastAPI)

uvicorn main:app --reload
Visit: http://localhost:8000/docs to access Swagger UI for interactive testing.

## Index & Query Documents

# Index documents
python build_faiss_retriever.py

# Start query pipeline
python rag_pipeline.py

## GCP Deployment Guide
Prerequisite: Docker + GCP account with Cloud Run, GCS setup.

 Deploy to GCP Cloud Run

 1. Build Docker image
docker build -t enterprise-bi-agent .

 2. Tag for Google Artifact Registry
docker tag enterprise-bi-agent gcr.io/YOUR_PROJECT_ID/enterprise-bi-agent

 3. Push image
docker push gcr.io/YOUR_PROJECT_ID/enterprise-bi-agent

 4. Deploy to Cloud Run
gcloud run deploy enterprise-bi-agent \
  --image gcr.io/YOUR_PROJECT_ID/enterprise-bi-agent \
  --platform managed \
  --region us-central1 \
  --allow-unauthenticated
  
## Future Enhancements
 Vector DB migration to ChromaDB or Weaviate

 CI/CD via GitHub Actions + Cloud Build

 Support for custom embeddings (e.g., BGE or HuggingFace)

 LangChain + LangSmith tracing

 Add user authentication (Firebase/Auth0)

## Acknowledgments
Built as part of an enterprise AI challenge, with references to:

LangChain

Apache Spark

OpenAI

FAISS

## License
This project is licensed under the MIT License. See LICENSE file for details.


---







