from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr
import shutil
import os

load_dotenv()

from app.rag_engine import process_and_store_pdf, get_chat_response

app = FastAPI(title="RAG API Portfolio", version="1.0")

# Código de acceso para proteger tu saldo de OpenAI contra bots
DEMO_TOKEN = os.getenv("DEMO_ACCESS_TOKEN", "PORTFOLIO2026")

class ChatRequest(BaseModel):
    question: str
    chat_history: list = []

# --- 1. ENDPOINTS REST (Disponibles en /docs) ---

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


# --- 2. INTERFAZ VISUAL INTERACTIVA (GRADIO) ---

def gradio_upload(pdf_file, token):
    if token.strip() != DEMO_TOKEN:
        return "❌ Access Token inválido. Ingresa: PORTFOLIO2026"
    if pdf_file is None:
        return "⚠️ Por favor sube un archivo PDF primero."
    try:
        chunks = process_and_store_pdf(pdf_file)
        return f"✅ PDF indexado con éxito ({chunks} fragmentos guardados en pgvector)."
    except Exception as e:
        return f"❌ Error al procesar: {str(e)}"

def gradio_chat(user_message, ui_history, lc_history, token):
    if token.strip() != DEMO_TOKEN:
        ui_history.append([user_message, "❌ Access Token inválido. Ingresa: PORTFOLIO2026 arriba."])
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
        Sube cualquier PDF para vectorizarlo en **PostgreSQL (`pgvector`)** y haz preguntas sobre su contenido usando **LangChain (LCEL)** y **OpenAI (`gpt-4o-mini`)**.
        * **Recruiter Demo Access Token:** `PORTFOLIO2026` *(Ingrésalo abajo para desbloquear la demo)*
        * **Swagger REST API Docs:** Disponible en [`/docs`](/docs)
        """
    )
    
    token_input = gr.Textbox(
        label="🔑 Access Token (Requerido)", 
        placeholder="Escribe aquí: PORTFOLIO2026",
    )
    
    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="1. Sube un documento PDF", file_types=[".pdf"], type="filepath")
            upload_btn = gr.Button("Indexar PDF en Base de Datos Vectorial", variant="primary")
            upload_status = gr.Textbox(label="Estado", interactive=False)
        
        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="2. Conversa con el documento", height=360)
            msg_input = gr.Textbox(
                label="Tu pregunta", 
                placeholder="Escribe tu pregunta sobre el PDF y presiona Enter..."
            )
            clear_btn = gr.Button("Limpiar historial de conversación")

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

# Montamos Gradio en la raíz sin romper /docs
app = gr.mount_gradio_app(app, demo, path="/")