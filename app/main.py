from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr
import shutil
import os

load_dotenv()

from app.rag_engine import process_and_store_pdf, get_chat_response

app = FastAPI(title="RAG API Portfolio", version="1.0")

# Access token to protect OpenAI API balance against bots
DEMO_TOKEN = os.getenv("DEMO_ACCESS_TOKEN", "PORTFOLIO2026")

class ChatRequest(BaseModel):
    question: str
    chat_history: list = []

# --- 1. REST ENDPOINTS (Available at /docs) ---

@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...), 
    x_access_token: str = Header(..., description="Use PORTFOLIO2026 for demo")
):
    if x_access_token != DEMO_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Access Token")
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
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat/")
async def chat_with_document(
    request: ChatRequest, 
    x_access_token: str = Header(..., description="Use PORTFOLIO2026 for demo")
):
    if x_access_token != DEMO_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Access Token")
    try:
        answer = get_chat_response(request.question, request.chat_history)
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 2. INTERACTIVE VISUAL INTERFACE (GRADIO) ---

def gradio_upload(pdf_file, token):
    if token.strip() != DEMO_TOKEN:
        return "❌ Invalid Access Token. Enter: PORTFOLIO2026"
    if pdf_file is None:
        return "⚠️ Please upload a PDF file first."
    try:
        chunks = process_and_store_pdf(pdf_file)
        return f"✅ PDF successfully indexed ({chunks} chunks stored in pgvector)."
    except Exception as e:
        return f"❌ Processing error: {str(e)}"

def gradio_chat(user_message, ui_history, lc_history, token):
    if token.strip() != DEMO_TOKEN:
        ui_history.append([user_message, "❌ Invalid Access Token. Enter PORTFOLIO2026 above."])
        return "", ui_history, lc_history
    if not user_message.strip():
        return "", ui_history, lc_history

    try:
        answer = get_chat_response(user_message, lc_history)
        lc_history.append([user_message, answer])
        ui_history.append([user_message, answer])
    except Exception as e:
        ui_history.append([user_message, f"❌ Error: {str(e)}"])

    return "", ui_history, lc_history

with gr.Blocks(title="Conversational RAG API Demo") as demo:
    gr.Markdown(
        """
        # 📄 Conversational RAG API — Live Portfolio Demo
        Upload any PDF to vectorize it in **PostgreSQL (`pgvector`)** and ask questions about its content using **LangChain (LCEL)** and **OpenAI (`gpt-4o-mini`)**.
        * **Recruiter Demo Access Token:** `PORTFOLIO2026` *(Enter it below to unlock the demo)*
        """
    )
    
    token_input = gr.Textbox(
        label="🔑 Access Token", 
        placeholder="Type here: PORTFOLIO2026",
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="1. Upload a PDF document", file_types=[".pdf"], type="filepath")
            upload_btn = gr.Button("Upload", variant="primary")
            upload_status = gr.Textbox(label="State", interactive=False)
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="2. Chat with the document", height=360)
            msg_input = gr.Textbox(
                label="Your question", 
                placeholder="Type your question about the PDF and press Enter..."
            )
            clear_btn = gr.Button("Clear conversation history")

    lc_history_state = gr.State([])

    upload_btn.click(
        fn=gradio_upload,
        inputs=[pdf_input, token_input],
        outputs=[upload_status]
    )

    msg_input.submit(
        fn=gradio_chat,
        inputs=[msg_input, chatbot, lc_history_state, token_input],
        outputs=[msg_input, chatbot, lc_history_state]
    )

    clear_btn.click(
        fn=lambda: ([], []),
        inputs=[],
        outputs=[chatbot, lc_history_state]
    )

# Mount Gradio at the root path without breaking /docs
app = gr.mount_gradio_app(app, demo, path="/")