import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
VECTOR_STORE_TYPE = os.getenv("VECTOR_STORE_TYPE", "faiss")
