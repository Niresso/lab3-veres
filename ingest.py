import os
import glob
import pickle

from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import Field
from config import settings
from llama_index.core import SimpleDirectoryReader
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document
from langchain_community.retrievers import BM25Retriever
from chroma_db import open_db


"""
Knowledge ingestion pipeline.

Loads documents from data/ directory, splits into chunks,
generates embeddings, and saves the index to disk.

Usage: python ingest.py
"""


def ingest():
    # TODO:
    # 1. Load documents from config.data_dir (PDF, TXT, MD)
    # 2. Split into chunks using TextSplitter
    # 3. Generate embeddings
    # 4. Build vector store (FAISS, Qdrant, Chroma, etc.)
    # 5. Save index to config.index_dir
    # 6. Save chunks for BM25 retriever (pickle or JSON)
    data = load_documents()

    vectordb = open_db()
    docs = []
    for doc in data:
        print(doc['text'][:100])
        chunks = text_splitters(doc['text'])
        print(f"chunks: {len(chunks)}")
        for text in chunks:
            docs.append(chunks_to_docs(text, doc))

        print(f"chunks docs: {len(docs)}")

    vectordb.add_documents(docs)
    bm25_retriever = BM25Retriever.from_documents(docs)
    bm25_retriever.k = 5

    os.makedirs(settings.index_dir, exist_ok=True)
    bm25_path = os.path.join(settings.index_dir, "bm25_retriever.pkl")
    with open(bm25_path, "wb") as f:
        pickle.dump(bm25_retriever, f)
    print(f"BM25 retriever saved to {bm25_path}")


def extract_text_from_pdf(file_path):
    loader = PyPDFLoader(file_path)
    langchain_docs = loader.load()
    text = ""
    for pdf_doc in langchain_docs:
        text += pdf_doc.page_content + "\n"

    return text

def load_documents():
    documents = SimpleDirectoryReader(
        input_dir=f"./{settings.data_dir}",
        recursive=True,
        filename_as_id=True
    ).load_data()

    data = []

    for doc in documents:
        text = ''
        if doc.metadata.get("file_type") == "application/pdf":
            print(doc.metadata['file_path'])
            text = extract_text_from_pdf(doc.metadata['file_path'])
        else:
            text = doc.text

        data.append({
            "text": text,
            "metadata": doc.metadata,
            "file_type": doc.metadata.get("file_type"),
            "file_name": doc.metadata["file_name"],
        })

    return data

def text_splitters(text):
    recursive_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    recursive_chunks = recursive_splitter.create_documents([text])

    print(f"📊 Recursive splitter: {len(recursive_chunks)} chunks \n")
    chunks = []
    for i, chunk in enumerate(recursive_chunks):
        chunks.append(chunk.page_content.strip())

    return chunks

def chunks_to_docs(text: str, doc) -> Document:
    return Document(page_content=text, metadata={
        "file_type": doc["file_type"],
        "file_name": doc["file_name"],
    })




if __name__ == "__main__":
    ingest()
