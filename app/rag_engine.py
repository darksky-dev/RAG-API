import os
from sqlalchemy import create_engine, text
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.output_parsers import StrOutputParser
from app.database import get_vector_store, CONNECTION_STRING, COLLECTION_NAME

engine = create_engine(CONNECTION_STRING)

def get_user_documents(user_id: str) -> list[str]:
    """Returns a list of unique PDF filenames uploaded by a specific user_id."""
    if not user_id:
        return []
        
    query = text("""
        SELECT DISTINCT e.cmetadata->>'filename' AS filename
        FROM langchain_pg_embedding e
        JOIN langchain_pg_collection c ON e.collection_id = c.uuid
        WHERE c.name = :collection_name
          AND e.cmetadata->>'user_id' = :user_id
          AND e.cmetadata->>'filename' IS NOT NULL
        ORDER BY filename ASC
    """)
    try:
        with engine.connect() as conn:
            result = conn.execute(query, {"collection_name": COLLECTION_NAME, "user_id": user_id})
            return [row[0] for row in result if row[0]]
    except Exception:
        # Returns empty list if the tables have not been created yet
        return []

def delete_existing_document(user_id: str, filename: str):
    """Deletes previous chunks if the same user re-uploads a file with the same name."""
    query = text("""
        DELETE FROM langchain_pg_embedding
        WHERE collection_id = (
            SELECT uuid FROM langchain_pg_collection WHERE name = :collection_name LIMIT 1
        )
        AND cmetadata->>'user_id' = :user_id
        AND cmetadata->>'filename' = :filename
    """)
    try:
        with engine.begin() as conn:
            conn.execute(query, {
                "collection_name": COLLECTION_NAME,
                "user_id": user_id,
                "filename": filename
            })
    except Exception:
        pass

def process_and_store_pdf(file_path: str, user_id: str, filename: str = None):
    """Extracts text from PDF, tags chunks with user_id & filename, and stores them in pgvector."""
    if filename is None:
        filename = os.path.basename(file_path)

    loader = PyPDFLoader(file_path)
    documents = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)

    # Tag every chunk with user_id and filename for strict multi-tenant isolation
    for split in splits:
        split.metadata["user_id"] = user_id
        split.metadata["filename"] = filename

    vectorstore = get_vector_store()
    
    # Prevent duplicate vectors if the user uploads the same document again
    delete_existing_document(user_id, filename)
    
    vectorstore.add_documents(splits)
    return len(splits)

def get_chat_response(question: str, user_id: str, filename: str, chat_history: list = None):
    """Retrieves context filtered by user_id and filename, and generates an answer using LCEL."""
    if chat_history is None:
        chat_history = []

    formatted_history = []
    for human, ai in chat_history:
        formatted_history.extend([HumanMessage(content=human), AIMessage(content=ai)])

    vectorstore = get_vector_store()
    
    # Strict metadata filter: isolates context by both user_id and active filename
    retriever = vectorstore.as_retriever(
        search_kwargs={
            "k": 3,
            "filter": {
                "user_id": user_id,
                "filename": filename
            }
        }
    )
    llm = ChatOpenAI(temperature=0, model="gpt-4o-mini")

    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a highly precise and strict AI data analyst. Your sole purpose is to answer the user's questions based EXCLUSIVELY on the provided document context.

        Follow these rules strictly:
        1. Analyze the provided context carefully before answering.
        2. Answer the question using ONLY the information explicitly stated in the context.
        3. Do NOT use any outside knowledge, make assumptions, or attempt to guess the answer.
        4. If the context does not explicitly contain the information needed to answer the user's question, you MUST reply exactly with: "The provided document does not contain the answer to this question."

        Context:
            {context}"""),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{question}")
    ])

    rag_chain = prompt | llm | StrOutputParser()

    return rag_chain.invoke({
        "context": context,
        "chat_history": formatted_history,
        "question": question
    })