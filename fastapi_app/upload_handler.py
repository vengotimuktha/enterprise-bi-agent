import os
import pdfplumber
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from fastapi_app.config import OPENAI_API_KEY, VECTOR_STORE_TYPE
from datetime import datetime

VECTORSTORE_DIR = "spark_vectorstores"

def extract_text_from_pdf(pdf_path: str) -> str:
    text = ""
    with pdfplumber.open(pdf_path) as pdf:
        for page in pdf.pages:
            content = page.extract_text()
            if content:
                text += content + "\n"
    return text

def process_and_store_pdf(pdf_path: str) -> str:
    raw_text = extract_text_from_pdf(pdf_path)

    if not raw_text.strip():
        raise ValueError("No text could be extracted from the uploaded PDF.")

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_text(raw_text)

    # Fixed debug print: 'texts' is a list of strings, not Documents
    for i, chunk in enumerate(texts[:5]):
        print(f"\n=== Chunk {i} Preview ===")
        print(chunk[:300])  # Just use the string directly

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    index_dir = os.path.join(VECTORSTORE_DIR, f"index_{timestamp}")
    os.makedirs(index_dir, exist_ok=True)

    if VECTOR_STORE_TYPE == "faiss":
        db = FAISS.from_texts(texts, embeddings)
        db.save_local(index_dir)
    else:
        raise ValueError("Unsupported VECTOR_STORE_TYPE")

    return index_dir
