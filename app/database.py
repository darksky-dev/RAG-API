import os
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

# Cadena de conexión a nuestro contenedor Docker
CONNECTION_STRING = "postgresql+psycopg://admin:password@localhost:5432/rag_db"
COLLECTION_NAME = "portfolio_documents"

def get_vector_store():
    # Usaremos OpenAI para los embeddings por defecto, pero se puede cambiar a modelos locales
    embeddings = OpenAIEmbeddings()
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
    return vectorstore