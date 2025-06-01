import os
import shutil
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

from fastapi_app.config import OPENAI_API_KEY
from fastapi_app.upload_handler import process_and_store_pdf

app = FastAPI(title="Enterprise RAG API")

# Serve frontend files
app.mount(
    "/static",
    StaticFiles(directory=os.path.join(os.path.dirname(__file__), "frontend")),
    name="static"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"]
)

# Home route
@app.get("/", response_class=HTMLResponse)
async def root():
    return FileResponse(os.path.join(os.path.dirname(__file__), "frontend", "index.html"))

# Upload PDF route
@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files accepted.")

    try:
        os.makedirs("data", exist_ok=True)
        save_path = os.path.join("data", file.filename)

        with open(save_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        index_path = process_and_store_pdf(save_path)

        return {
            "status": "success",
            "message": "PDF processed and index created",
            "filename": file.filename,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "index_path": index_path
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Upload failed: {str(e)}")

# Query PDF route
@app.post("/query")
async def query_pdf(question: str = Form(...), index_path: str = Form(...)):
    if not os.path.isdir(index_path):
        raise HTTPException(status_code=400, detail="Invalid or missing index path.")

    try:
        # Load vector store and retriever
        embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)
        db = FAISS.load_local(
            index_path,
            embeddings=embeddings,
            allow_dangerous_deserialization=True
        )
        retriever = db.as_retriever(search_kwargs={"k": 5})

        # Retrieve documents
        docs = retriever.get_relevant_documents(question)
        if not docs:
            return JSONResponse(content={"answer": "❌ No relevant data found for your question."})

        # Log debug chunks
        for i, doc in enumerate(docs):
            print(f"\n=== Match {i} ===\n{doc.page_content[:300]}")

        # Run LLM QA chain
        llm = ChatOpenAI(openai_api_key=OPENAI_API_KEY, temperature=0.3, model_name="gpt-3.5-turbo")
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=retriever)
        result = qa_chain.run(question)

        return JSONResponse(content={"answer": result})

    except Exception as e:
        print("Error in /query:", e)
        return JSONResponse(status_code=500, content={"answer": f"❌ Internal server error: {str(e)}"})
