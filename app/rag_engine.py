from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from app.database import get_vector_store

def process_and_store_pdf(file_path: str):
    """Extrae el texto del PDF, lo divide en chunks y lo guarda en pgvector."""
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    vectorstore = get_vector_store()
    vectorstore.add_documents(splits)
    return len(splits)

def get_chat_response(question: str, chat_history: list = None):
    """Recupera contexto de la base de datos y genera una respuesta con el LLM."""
    if chat_history is None:
        chat_history = []
        
    vectorstore = get_vector_store()
    llm = ChatOpenAI(temperature=0, model="gpt-3.5-turbo")
    
    # Creamos la cadena conversacional con memoria
    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(search_kwargs={"k": 3}),
        return_source_documents=True
    )
    
    result = qa_chain({"question": question, "chat_history": chat_history})
    return result["answer"]