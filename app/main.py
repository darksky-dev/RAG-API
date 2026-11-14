from fastapi import FastAPI, UploadFile, File, HTTPException, Header
from pydantic import BaseModel
from dotenv import load_dotenv
import gradio as gr
import shutil
import os

load_dotenv()

from app.rag_engine import (
    process_and_store_pdf,
    get_chat_response,
    get_user_documents
)

app = FastAPI(title="RAG API Portfolio", version="1.0")

# Access token to protect OpenAI API balance against bots
DEMO_TOKEN = os.getenv("DEMO_ACCESS_TOKEN", "PORTFOLIO2026")

class ChatRequest(BaseModel):
    question: str
    filename: str
    chat_history: list = []

# --- 1. REST ENDPOINTS (Available at /docs) ---

@app.get("/documents/")
async def list_documents(
    x_access_token: str = Header(..., description="Use PORTFOLIO2026 for demo"),
    x_user_id: str = Header(..., description="Unique user or session identifier")
):
    if x_access_token != DEMO_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Access Token")
    return {"user_id": x_user_id, "documents": get_user_documents(x_user_id)}

@app.post("/upload/")
async def upload_document(
    file: UploadFile = File(...),
    x_access_token: str = Header(..., description="Use PORTFOLIO2026 for demo"),
    x_user_id: str = Header(..., description="Unique user or session identifier")
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
        chunks_saved = process_and_store_pdf(file_path, user_id=x_user_id, filename=file.filename)
        return {
            "message": f"Successfully processed {file.filename}",
            "filename": file.filename,
            "chunks": chunks_saved
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
    finally:
        if os.path.exists(file_path):
            os.remove(file_path)

@app.post("/chat/")
async def chat_with_document(
    request: ChatRequest,
    x_access_token: str = Header(..., description="Use PORTFOLIO2026 for demo"),
    x_user_id: str = Header(..., description="Unique user or session identifier")
):
    if x_access_token != DEMO_TOKEN:
        raise HTTPException(status_code=403, detail="Invalid Access Token")
    try:
        answer = get_chat_response(
            question=request.question,
            user_id=x_user_id,
            filename=request.filename,
            chat_history=request.chat_history
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# --- 2. INTERACTIVE VISUAL INTERFACE (GRADIO) ---

def init_user_session(user_id):
    """Loads the user's previously uploaded documents when opening the page."""
    docs = get_user_documents(user_id)
    selected = docs[0] if docs else None
    return user_id, gr.update(choices=docs, value=selected)

def gradio_upload(pdf_file, token, user_id):
    if token.strip() != DEMO_TOKEN:
        return "❌ Invalid Access Token. Enter: PORTFOLIO2026", gr.update()
    if not user_id:
        return "❌ Session ID not initialized. Please refresh the page.", gr.update()
    if pdf_file is None:
        return "⚠️ Please upload a PDF file first.", gr.update()

    try:
        filename = os.path.basename(pdf_file)
        chunks = process_and_store_pdf(pdf_file, user_id=user_id, filename=filename)
        docs = get_user_documents(user_id)
        return (
            f"✅ '{filename}' indexed ({chunks} chunks stored in your private workspace).",
            gr.update(choices=docs, value=filename)
        )
    except Exception as e:
        return f"❌ Processing error: {str(e)}", gr.update()

def gradio_chat(user_message, ui_history, lc_history, token, user_id, selected_doc):
    if token.strip() != DEMO_TOKEN:
        ui_history.append({"role": "user", "content": user_message})
        ui_history.append({"role": "assistant", "content": "❌ Invalid Access Token. Enter PORTFOLIO2026 above."})
        return "", ui_history, lc_history
    if not selected_doc:
        ui_history.append({"role": "user", "content": user_message})
        ui_history.append({"role": "assistant", "content": "⚠️ Please upload or select a PDF document first."})
        return "", ui_history, lc_history
    if not user_message.strip():
        return "", ui_history, lc_history

    try:
        answer = get_chat_response(
            question=user_message,
            user_id=user_id,
            filename=selected_doc,
            chat_history=lc_history
        )
        lc_history.append([user_message, answer])
        ui_history.append({"role": "user", "content": user_message})
        ui_history.append({"role": "assistant", "content": answer})
    except Exception as e:
        ui_history.append({"role": "user", "content": user_message})
        ui_history.append({"role": "assistant", "content": f"❌ Error: {str(e)}"})

    return "", ui_history, lc_history

with gr.Blocks(title="Conversational RAG API Demo") as demo:
    gr.Markdown(
        """
        # 📄 Conversational RAG API — Live Portfolio Demo
        Upload any PDF to vectorize it in **PostgreSQL (`pgvector`)** and ask questions about its content using **LangChain (LCEL)** and **OpenAI (`gpt-4o-mini`)**.
        * **Recruiter Demo Access Token:** `PORTFOLIO2026` *(Enter it below to unlock the demo)*
        * **Multi-Tenant Isolation:** Your uploaded documents are linked strictly to your private browser session and isolated from other users.
        """
    )

    # Hidden state that holds the persistent browser UUID
    user_id_state = gr.Textbox(visible=False)

    token_input = gr.Textbox(
        label="🔑 Access Token",
        placeholder="Type here: PORTFOLIO2026",
    )

    with gr.Row():
        with gr.Column(scale=1):
            pdf_input = gr.File(label="1. Upload a new PDF document", file_types=[".pdf"], type="filepath")
            upload_btn = gr.Button("Upload & Index", variant="primary")
            upload_status = gr.Textbox(label="State", interactive=False)
            
            doc_dropdown = gr.Dropdown(
                label="2. Select an active PDF from your workspace",
                choices=[],
                interactive=True
            )

        with gr.Column(scale=2):
            chatbot = gr.Chatbot(label="3. Chat with the selected document", height=360)
            msg_input = gr.Textbox(
                label="Your question",
                placeholder="Type your question about the selected PDF and press Enter..."
            )
            clear_btn = gr.Button("Clear conversation history")

    lc_history_state = gr.State([])

    # On page load, retrieve or generate a persistent user_id in browser localStorage and load their PDFs
    demo.load(
        fn=init_user_session,
        inputs=[user_id_state],
        outputs=[user_id_state, doc_dropdown],
        js="""(current_id) => {
            let uid = localStorage.getItem('rag_portfolio_uid');
            if (!uid) {
                uid = 'user-' + Math.random().toString(36).substring(2, 10) + Date.now().toString(36);
                localStorage.setItem('rag_portfolio_uid', uid);
            }
            return uid;
        }"""
    )

    upload_btn.click(
        fn=gradio_upload,
        inputs=[pdf_input, token_input, user_id_state],
        outputs=[upload_status, doc_dropdown]
    )

    # Clear chat history automatically when switching between documents
    doc_dropdown.change(
        fn=lambda: ([], []),
        inputs=[],
        outputs=[chatbot, lc_history_state]
    )

    msg_input.submit(
        fn=gradio_chat,
        inputs=[msg_input, chatbot, lc_history_state, token_input, user_id_state, doc_dropdown],
        outputs=[msg_input, chatbot, lc_history_state]
    )

    clear_btn.click(
        fn=lambda: ([], []),
        inputs=[],
        outputs=[chatbot, lc_history_state]
    )

# Mount Gradio at the root path without breaking /docs
app = gr.mount_gradio_app(app, demo, path="/")