import os
import pickle
import logging

import trafilatura
from ddgs import DDGS
from langchain_core.tools import tool
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import settings
from chroma_db import open_db

logging.basicConfig(level=logging.INFO)

os.makedirs(settings.path_save_file, exist_ok=True)


@tool
def web_search(query: str) -> list[dict]:
    """
    Search the internet using DuckDuckGo.

    Use when you need up-to-date information on a topic.

    Args:
        query: search query string

    Returns:
        List of up to 5 results, each containing: title, href, body
    """
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        return results or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
def read_url(url: str) -> str:
    """
    Fetch and extract the text content of a webpage.

    Use after web_search to read the full content of a promising link.

    Args:
        url: page URL

    Returns:
        Extracted page text (up to max_url_content_length characters)
    """
    try:
        downloaded = trafilatura.fetch_url(url)
        if not downloaded:
            return f"Error: could not fetch URL: {url}"
        text = trafilatura.extract(downloaded)
        if not text:
            return f"Error: could not extract text from URL: {url}"
        return text[:settings.max_url_content_length]
    except Exception as e:
        return f"Error reading URL {url}: {e}"


@tool
def knowledge_search(query: str) -> list[dict]:
    """
    Search the local knowledge base (ingested PDF/TXT documents).

    Uses hybrid search (BM25 + vector) with cross-encoder reranking.

    Args:
        query: search query string

    Returns:
        List of the most relevant document chunks, each containing: content, metadata
    """
    try:
        reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
        vectordb = open_db()
        vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

        bm25_path = os.path.join(settings.index_dir, "bm25_retriever.pkl")
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)

        ensemble_retriever = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7],
        )
        reranking_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble_retriever,
        )
        results = reranking_retriever.invoke(query)
        return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]
    except Exception as e:
        return [{"error": str(e)}]


@tool
def save_report(filename: str, content: str) -> str:
    """
    Save the final research report to a file. Requires user approval (HITL).

    Args:
        filename: file name, e.g. 'report.md'
        content: full report text in markdown

    Returns:
        Success message or error description
    """
    try:
        path = os.path.join(settings.path_save_file, filename)
        content_clean = content.encode("utf-8", errors="replace").decode("utf-8")
        with open(path, "w", encoding="utf-8") as f:
            f.write(content_clean)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error writing report: {e}"
