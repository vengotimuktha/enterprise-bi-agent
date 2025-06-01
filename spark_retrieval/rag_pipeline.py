import os
import datetime
from dotenv import load_dotenv

from langchain_community.vectorstores import FAISS
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

# -------------------------
# Loading environment
# -------------------------
load_dotenv()

api_key = os.getenv("OPENAI_API_KEY")

if not api_key:
    raise ValueError("OPENAI_API_KEY not found in .env or environment variables.")

# -------------------------
# Loading FAISS Vector Store (trusted)
# -------------------------
print("[INFO] Loading FAISS vector store...")

embedding_model = OpenAIEmbeddings(openai_api_key=api_key)

db = FAISS.load_local(
    "spark_vectorstores/spark_faiss_index",
    embeddings=embedding_model,
    allow_dangerous_deserialization=True  
)

# -------------------------
# Seting up retriever
# -------------------------
retriever = db.as_retriever(search_kwargs={"k": 5})

# -------------------------
# Prompt Template
# -------------------------
prompt_template = """
You are an intelligent business assistant.
Answer the question using only the provided context from PDF reports.
If the answer is not in the context, say "I don't know".

Context:
{context}

Question:
{question}
"""

prompt = PromptTemplate(
    input_variables=["context", "question"],
    template=prompt_template
)

# -------------------------
# Language model
# -------------------------
llm = ChatOpenAI(
    temperature=0.2,
    model="gpt-3.5-turbo",
    openai_api_key=api_key
)

# -------------------------
# RAG Pipeline
# -------------------------
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# -------------------------
# Sample Query
# -------------------------
query = "What are Amazon's sustainability goals in 2023?"
print(f"\n[USER QUESTION] {query}\n")

result = qa_chain({"query": query})
print("[ANSWER]", result["result"])

# -------------------------
# Showing sources
# -------------------------
print("\n[SOURCE DOCUMENTS]")
for doc in result["source_documents"]:
    print(f"→ {doc.metadata.get('source')} (Page {doc.metadata.get('page', 'N/A')})")
