import os
from dotenv import load_dotenv
from langchain_postgres.vectorstores import PGVector
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Usa DATABASE_URL en producción (Neon.tech) o el contenedor Docker en local
raw_db_url = os.getenv(
    "DATABASE_URL", 
    "postgresql+psycopg://admin:password@localhost:5432/rag_db"
)

# Neon entrega URLs con 'postgresql://', lo adaptamos al driver psycopg v3
if raw_db_url.startswith("postgres://"):
    CONNECTION_STRING = raw_db_url.replace("postgres://", "postgresql+psycopg://", 1)
elif raw_db_url.startswith("postgresql://"):
    CONNECTION_STRING = raw_db_url.replace("postgresql://", "postgresql+psycopg://", 1)
else:
    CONNECTION_STRING = raw_db_url

COLLECTION_NAME = "portfolio_documents"

def get_vector_store():
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING,
        use_jsonb=True,
    )
    return vectorstore