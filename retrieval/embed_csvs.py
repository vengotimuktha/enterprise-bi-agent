import os
import pandas as pd
from langchain_community.document_loaders import DataFrameLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from dotenv import load_dotenv

# Loading API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Path to CSV folder
csv_folder = "data/csv"

# Loading all CSVs
documents = []
for filename in os.listdir(csv_folder):
    if filename.endswith(".csv"):
        csv_path = os.path.join(csv_folder, filename)
        df = pd.read_csv(csv_path, encoding="ISO-8859-1")

        # Convert dataframe rows to "text documents"
        loader = DataFrameLoader(df, page_content_column=df.columns[0])  # Take first column as text
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
faiss_folder = "faiss_store_csv"
if not os.path.exists(faiss_folder):
    os.makedirs(faiss_folder)

vectorstore.save_local(faiss_folder)

print(f"Embedded {len(docs)} chunks from CSV files and saved to FAISS.")
