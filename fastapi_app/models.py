from pydantic import BaseModel
from typing import List

class QueryRequest(BaseModel):
    question: str
    index_path: str

class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
