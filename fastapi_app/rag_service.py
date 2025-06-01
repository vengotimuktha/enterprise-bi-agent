import os
import re
from datetime import datetime
import mlflow
#from langchain_core.vectorstores import VectorStoreRetriever
#from langchain_community.vectorstores.utils import load_local_vectorstore
from langchain_community.vectorstores import FAISS

from langchain_community.embeddings import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from openai import AuthenticationError

from fastapi_app.config import OPENAI_API_KEY, VECTOR_STORE_TYPE

# Hybrid logic imports
from utils.question_classifier import is_numeric_question
from utils.table_extractor import extract_tables_from_pdf
from utils.table_query import query_tables_with_question

mlflow.set_tracking_uri("file:./mlruns")
mlflow.set_experiment("enterprise_bi_queries")
VECTORSTORE_DIR = "spark_vectorstores"

# === Helper functions ===

def _get_latest_index_path() -> str:
    idxs = [d for d in os.listdir(VECTORSTORE_DIR) if d.startswith("index_")]
    if not idxs:
        raise FileNotFoundError("No FAISS index found. Please upload a PDF first.")
    return os.path.join(VECTORSTORE_DIR, sorted(idxs)[-1])

def _extract_target_word(question: str) -> str:
    match = re.search(r'\b(?:define|what does|meaning of|explain|describe)\b.*?\b([\w\-]+)\b', question.lower())
    if match:
        return match.group(1)
    return ""

def _augment_question(question: str) -> str:
    word = _extract_target_word(question)
    if word:
        variations = [word, f"-{word}", f"vera-{word}"]
        appended = " ".join(f"(related term: {v})" for v in variations)
        return f"{question.strip()} {appended}"
    return question

# === Main handler ===

def query_pdf(pdf_path: str, question: str):
    try:
        # Case 1: Use table logic for numeric
        if is_numeric_question(question):
            tables = extract_tables_from_pdf(pdf_path)
            return {
                "answer": query_tables_with_question(tables, question),
                "source": "structured_table"
            }

        # Case 2: RAG-style
        index_path = _get_latest_index_path()

        print(f"[INFO] Loading FAISS index from: {index_path}")
        print(f"[INFO] Allowing dangerous deserialization: True")

        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

        if VECTOR_STORE_TYPE == "faiss":
            db = FAISS.load_local(
                folder_path=index_path,
                embeddings=embeddings,
                allow_dangerous_deserialization=True
            )

        elif VECTOR_STORE_TYPE == "chroma":
            from langchain_community.vectorstores import Chroma
            db = Chroma(persist_directory=index_path, embedding_function=embeddings)
        else:
            raise ValueError(f"Unsupported VECTOR_STORE_TYPE: {VECTOR_STORE_TYPE}")

        retriever = db.as_retriever(search_type="mmr", search_kwargs={"k": 10})

        prompt = PromptTemplate(
            input_variables=["context", "question"],
            template="""
You are an expert Old Icelandic dictionary interpreter.

Given the context from a scanned dictionary, extract the most likely definition or meaning of the term asked in the question.

Always try to interpret even partial or compound forms (like -elleftr or vera-elleftr), and explain their meaning in modern English if possible.

If absolutely no relevant context is found, only then say: "Not found in dictionary."

Context:
{context}

Question:
{question}
            """
        )

        llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.0, openai_api_key=OPENAI_API_KEY)

        qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=retriever,
            return_source_documents=True,
            chain_type_kwargs={"prompt": prompt}
        )

        augmented_question = _augment_question(question)
        result = qa.invoke({"query": augmented_question})

        answer = result["result"]
        sources = [
            f"{doc.metadata.get('source','unknown')} (Page {doc.metadata.get('page','N/A')})"
            for doc in result["source_documents"]
        ]

        # Log to MLflow
        with mlflow.start_run(run_name=f"query-{datetime.now().strftime('%Y%m%d-%H%M%S')}"):
            mlflow.log_param("question", question)
            mlflow.log_param("model", "gpt-3.5-turbo")
            mlflow.log_metric("response_length", len(answer))

            with open("mlflow_answer.txt", "w", encoding="utf-8") as f:
                f.write(answer)
            mlflow.log_artifact("mlflow_answer.txt")

            with open("mlflow_sources.txt", "w", encoding="utf-8") as f:
                f.write("\n".join(sources))
            mlflow.log_artifact("mlflow_sources.txt")

            os.remove("mlflow_answer.txt")
            os.remove("mlflow_sources.txt")

        return {
            "answer": answer,
            "source": "rag"
        }

    except AuthenticationError:
        raise ValueError("Invalid or missing OpenAI API key. Please check your .env file.")

    except Exception as e:
        print("[ERROR] in query_pdf():", str(e))
        raise RuntimeError(f"❌ Internal server error: {str(e)}")
