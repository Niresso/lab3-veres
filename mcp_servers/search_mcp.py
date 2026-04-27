"""
SearchMCP — port 8901
Tools: web_search, read_url, knowledge_search
Resources: knowledge-base-stats
"""
import os
import pickle
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import trafilatura
from ddgs import DDGS
from fastmcp import FastMCP
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder

from config import settings
from chroma_db import open_db

mcp = FastMCP(name="SearchMCP")


@mcp.tool
def web_search(query: str) -> list[dict]:
    """
    Search the internet using DuckDuckGo.

    Args:
        query: search query string

    Returns:
        List of up to 5 results: title, href, body
    """
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        return results or []
    except Exception as e:
        return [{"error": str(e)}]


@mcp.tool
def read_url(url: str) -> str:
    """
    Fetch and extract the text content of a webpage.

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


@mcp.tool
def knowledge_search(query: str) -> list[dict]:
    """
    Search the local knowledge base using hybrid BM25 + vector search with reranking.

    Args:
        query: search query string

    Returns:
        List of relevant document chunks: content, metadata
    """
    try:
        reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
        compressor = CrossEncoderReranker(model=reranker_model, top_n=3)
        vectordb = open_db()
        vector_retriever = vectordb.as_retriever(search_kwargs={"k": 10})

        bm25_path = os.path.join(settings.index_dir, "bm25_retriever.pkl")
        with open(bm25_path, "rb") as f:
            bm25_retriever = pickle.load(f)

        ensemble = EnsembleRetriever(
            retrievers=[bm25_retriever, vector_retriever],
            weights=[0.3, 0.7],
        )
        retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=ensemble,
        )
        docs = retriever.invoke(query)
        return [{"content": d.page_content, "metadata": d.metadata} for d in docs]
    except Exception as e:
        return [{"error": str(e)}]


@mcp.resource("resource://knowledge-base-stats")
def knowledge_base_stats() -> dict:
    """Returns document count and last update time of the knowledge base index."""
    try:
        bm25_path = os.path.join(settings.index_dir, "bm25_retriever.pkl")
        mtime = os.path.getmtime(bm25_path)
        import datetime
        last_updated = datetime.datetime.fromtimestamp(mtime).isoformat()
        with open(bm25_path, "rb") as f:
            bm25 = pickle.load(f)
        doc_count = len(bm25.docs) if hasattr(bm25, "docs") else "unknown"
        return {"document_chunks": doc_count, "last_updated": last_updated}
    except Exception as e:
        return {"error": str(e)}


if __name__ == "__main__":
    mcp.run(
        transport="streamable-http",
        host="0.0.0.0",
        port=settings.search_mcp_port,
        uvicorn_config={"ws": "none"},
    )
