import os
import csv
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.retrievers import BaseRetriever
from typing import List
from dotenv import load_dotenv
from langchain_core.documents import Document
from pydantic import Field
from colorama import Fore, Style, init

# Initialize colorama
init(autoreset=True)

# Loading API key
load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")

# Loading PDF embeddings
pdf_vectorstore = FAISS.load_local(
    "faiss_store",
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    allow_dangerous_deserialization=True
)

# Loading CSV embeddings
csv_vectorstore = FAISS.load_local(
    "faiss_store_csv",
    OpenAIEmbeddings(openai_api_key=openai_api_key),
    allow_dangerous_deserialization=True
)

# Combining both vectorstores
all_vectorstores = [pdf_vectorstore, csv_vectorstore]

# Building Hybrid Retriever
class HybridRetriever(BaseRetriever):
    stores: List = Field(default_factory=list)

    def __init__(self, stores):
        super().__init__()
        object.__setattr__(self, 'stores', stores)

    def get_relevant_documents(self, query: str) -> List[Document]:
        all_docs = []
        for store in self.stores:
            docs = store.similarity_search(query, k=4)  # Top 4 from each source
            all_docs.extend(docs)
        return all_docs

    async def aget_relevant_documents(self, query: str) -> List[Document]:
        return self.get_relevant_documents(query)

retriever = HybridRetriever(all_vectorstores)

# Building QA Chain
llm = ChatOpenAI(openai_api_key=openai_api_key, model_name="gpt-3.5-turbo")

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=retriever,
    return_source_documents=True
)

# Creating logs folder if not exist
if not os.path.exists("logs"):
    os.makedirs("logs")

log_file_path = "logs/qa_logs.csv"

# Creating log file with headers if not already
if not os.path.exists(log_file_path):
    with open(log_file_path, mode='w', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow(["Question", "Answer", "Sources"])

# CLI Query Loop
print("\nEnterprise GPT Agent is ready. Ask a question (type 'exit' to quit):\n")

while True:
    user_query = input("Ask: ")
    if user_query.lower() == "exit":
        break

    response = qa_chain.invoke({"query": user_query})

    # Extracting Answer and Sources
    answer_text = response.get("result", "No answer found.")
    sources = response.get("source_documents", [])

    source_files = []
    for doc in sources:
        if hasattr(doc.metadata, 'get'):
            source_name = doc.metadata.get("source", "Unknown")
            if source_name != "Unknown":
                source_files.append(source_name)

    source_files = list(set(source_files))  # Deduplicate

    # Printing beautifully with colors
    print(Fore.GREEN + "\nAnswer:\n")
    print(Style.RESET_ALL + answer_text)

    print(Fore.BLUE + "\nSources:")
    if source_files:
        for src in source_files:
            print(f"- {src}")
    else:
        print("- No sources found.")
    print(Style.RESET_ALL)

    print("\n" + "-" * 60 + "\n")

    # Save to logs
    with open(log_file_path, mode='a', newline='', encoding='utf-8') as file:
        writer = csv.writer(file)
        writer.writerow([user_query, answer_text, "; ".join(source_files)])
