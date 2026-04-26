from langchain_core.documents import Document
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from config import settings

emb = HuggingFaceEmbeddings(model_name="intfloat/e5-large-v2")

def open_db() -> Chroma:
    try:
        vectordb = Chroma(
            collection_name="chroma_docs",
            embedding_function=emb,
            persist_directory=settings.index_dir,
        )
        try:
            vectordb._collection.count()
        except Exception as e:
            print(f"Database validation failed: {e}")
        return vectordb
    except Exception as e:
        print(f"Error opening database: {e}")
        raise