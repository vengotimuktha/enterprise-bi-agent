from reportlab.pdfgen import canvas
import os
import pytest
from fastapi.testclient import TestClient
from fastapi_app.main import app

client = TestClient(app)

def test_upload_pdf():
    test_pdf_path = "data/test_dummy.pdf"

    # Generate a valid test PDF
    os.makedirs("data", exist_ok=True)
    c = canvas.Canvas(test_pdf_path)
    c.drawString(100, 750, "This is a test PDF for unit testing.")
    c.save()

    # Upload the valid PDF
    with open(test_pdf_path, "rb") as pdf_file:
        response = client.post(
            "/upload-pdf",
            files={"file": ("test_dummy.pdf", pdf_file, "application/pdf")}
        )

    assert response.status_code == 200
    json_data = response.json()
    assert json_data["status"] == "success"
    assert json_data["filename"] == "test_dummy.pdf"

def test_query_rag():
    # Skip if no FAISS index is available yet
    if not os.path.exists("spark_vectorstores"):
        pytest.skip("No FAISS index found. Upload a PDF first.")

    response = client.post(
        "/query",
        json={"question": "Define the word elleftr"}
    )
    assert response.status_code == 200
    data = response.json()
    assert "answer" in data
    assert "sources" in data
