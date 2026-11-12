from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
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
    """Recupera contexto vectorial y genera una respuesta usando LCEL moderno."""
    if chat_history is None:
        chat_history = []
        
    # 1. Formatear la memoria al estándar moderno de mensajes de LangChain
    formatted_history = []
    for human, ai in chat_history:
        formatted_history.extend([HumanMessage(content=human), AIMessage(content=ai)])
        
    vectorstore = get_vector_store()
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")
    
    # 2. Recuperar el contexto de la base de datos de forma explícita
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    
    # 3. Crear la plantilla del Prompt
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a helpful assistant. Answer the user's questions based strictly on the following context:\n\n{context}"),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}")
    ])
    
    # 4. Construir y ejecutar la cadena usando LCEL puro (sin langchain.chains)
    rag_chain = prompt | llm | StrOutputParser()
    
    return rag_chain.invoke({
        "context": context,
        "chat_history": formatted_history,
        "question": question
    })