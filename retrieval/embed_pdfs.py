import os
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Loading API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Path to PDF folder
pdf_folder = "data/pdf"

# Loading all PDFs
documents = []
for filename in os.listdir(pdf_folder):
    if filename.endswith(".pdf"):
        pdf_path = os.path.join(pdf_folder, filename)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

# Spliting documents into chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)
docs = text_splitter.split_documents(documents)

# Initializing embeddings
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key)

# Creating FAISS vectorstore
vectorstore = FAISS.from_documents(docs, embeddings)

# Saving the FAISS index
faiss_folder = "faiss_store"
if not os.path.exists(faiss_folder):
    os.makedirs(faiss_folder)

vectorstore.save_local(faiss_folder)

print(f"Embedded {len(docs)} chunks from PDF files and saved to FAISS.")
