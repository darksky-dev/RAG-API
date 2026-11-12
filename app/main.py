from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import shutil
import os
from app.rag_engine import process_and_store_pdf, get_chat_response

from dotenv import load_dotenv
load_dotenv()

app = FastAPI(title="RAG API Portfolio", version="1.0")

class ChatRequest(BaseModel):
    question: str
    chat_history: list = []

@app.post("/upload/")
async def upload_document(file: UploadFile = File(...)):
    """Recibe un PDF, lo guarda temporalmente y procesa sus embeddings."""
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
        
    os.makedirs("uploads", exist_ok=True)
    file_path = f"uploads/{file.filename}"
    
    with open(file_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        chunks_saved = process_and_store_pdf(file_path)
        return {"message": f"Successfully processed {file.filename}", "chunks": chunks_saved}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        # Limpieza del archivo temporal
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat/")
async def chat_with_document(request: ChatRequest):
    """Permite hacer preguntas sobre los documentos indexados."""
    try:
        answer = get_chat_response(request.question, request.chat_history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))