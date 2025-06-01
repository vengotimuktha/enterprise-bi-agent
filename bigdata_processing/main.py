# fastapi_app/main.py
from fastapi import FastAPI
from fastapi_app.models import QueryRequest, QueryResponse
from fastapi_app.rag_service import run_qa

app = FastAPI(title="Enterprise RAG API")

@app.post("/query", response_model=QueryResponse)
def query_docs(request: QueryRequest):
    answer, sources = run_qa(request.question)
    return QueryResponse(answer=answer, sources=sources)
