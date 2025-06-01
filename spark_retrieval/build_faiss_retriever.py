import json
import os
import datetime
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS, Chroma
from langchain.schema import Document
from dotenv import load_dotenv

# Load environment
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# File paths
chunk_path = "data/spark_pdf_chunks.json"
faiss_path = "spark_vectorstores/spark_faiss_index"
chroma_path = "spark_vectorstores/spark_chroma_index"
USE_CHROMA = True  # Set to True for ChromaDB, False for FAISS

# Load chunks
with open(chunk_path, "r", encoding="utf-8") as file:
    chunks = json.load(file)

# Prepare documents
documents = []
for item in chunks:
    text = item["text"]
    metadata = {
        "source": item["source"],
        "page": item["page"],
        "title": item["file_name"],
        "chunk_id": str(item.get("chunk_id", "")),
        "ingested_at": item.get("ingested_at", datetime.datetime.now().isoformat())
    }
    documents.append(Document(page_content=text, metadata=metadata))

# Initialize embeddings
print("[INFO] Embedding PDF chunks...")
embedding_model = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Build vector store
if USE_CHROMA:
    print("[INFO] Creating Chroma index...")
    os.makedirs(chroma_path, exist_ok=True)
    vector_store = Chroma.from_documents(documents, embedding_model, persist_directory=chroma_path)
    vector_store.persist()
    print(f"[SUCCESS] Chroma vector store saved to: {chroma_path}")
else:
    print("[INFO] Creating FAISS index...")
    vector_store = FAISS.from_documents(documents, embedding_model)
    vector_store.save_local(faiss_path)
    print(f"[SUCCESS] FAISS vector store saved to: {faiss_path}")
