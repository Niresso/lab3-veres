import os
import pickle
import logging
import trafilatura

from ddgs import DDGS
from langchain_core.tools import tool
from config import settings

logging.basicConfig(level=logging.INFO)

os.makedirs(settings.path_save_file, exist_ok=True)
from langchain_classic.retrievers import EnsembleRetriever
from langchain_classic.retrievers import ContextualCompressionRetriever
from langchain_classic.retrievers.document_compressors import CrossEncoderReranker
from langchain_community.cross_encoders import HuggingFaceCrossEncoder
from chroma_db import open_db


@tool
def web_search(query: str) -> list[dict]:
    """
    Шукає інформацію в інтернеті через DuckDuckGo.

    Використовуй, коли потрібно знайти актуальну інформацію на певну тему.

    Args:
        query: пошуковий запит

    Returns:
        Список до 5 результатів, кожен містить:
          - title (str): заголовок сторінки
          - href (str): URL сторінки
          - body (str): короткий опис/уривок
    """
    try:
        results = DDGS().text(query, max_results=settings.max_search_results)
        return results or []
    except Exception as e:
        return [{"error": str(e)}]


@tool
def read_url(url: str) -> str:
    """
    Завантажує та витягує текстовий вміст веб-сторінки за URL.

    Використовуй, коли потрібно отримати повний текст конкретної сторінки
    (наприклад, після web_search для детального читання статті).

    Args:
        url: адреса сторінки

    Returns:
        Текст сторінки (рядок, до max_url_content_length символів)
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
def write_report(filename: str, content: str) -> str:
    """
    Зберігає текстовий звіт у файл.

    Використовуй, коли користувач явно просить зберегти результати або звіт.

    Args:
        filename: назва файлу (наприклад 'report.md')
        content: текст для збереження

    Returns:
        Повідомлення про успіх або помилку
    """
    try:
        path = os.path.join(settings.path_save_file, filename)
        content = content.encode('utf-8', errors='replace').decode('utf-8')
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error writing report: {e}"

@tool
def knowledge_search(query: str) -> list[dict]:
    """
    Шукає інформацію у локальній базі знань (PDF, TXT документи).

    Використовуй, коли потрібно відповісти на запитання про завантажені документи.
    Використовує гібридний пошук (BM25 + векторний) з реранкінгом для кращої точності.

    Args:
        query: пошуковий запит

    Returns:
        Список найбільш релевантних фрагментів документів, кожен містить:
          - content (str): текст фрагменту
          - metadata (dict): метадані (назва файлу, тип)
    """
    reranker_model = HuggingFaceCrossEncoder(model_name="BAAI/bge-reranker-base")
    compressor = CrossEncoderReranker(
        model=reranker_model,
        top_n=3  # keep only 3 most relevant
    )
    vectordb = open_db()
    vector_retriever = vectordb.as_retriever(search_kwargs={"k": 100})


    bm25_path = os.path.join(settings.index_dir, "bm25_retriever.pkl")
    with open(bm25_path, "rb") as f:
        bm25_retriever = pickle.load(f)

    ensemble_retriever = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[0.3, 0.7]  # 30% BM25 + 70% Vector
    )

    reranking_retriever = ContextualCompressionRetriever(
        base_compressor=compressor,
        base_retriever=ensemble_retriever
    )

    results = reranking_retriever.invoke(query)
    for i, doc in enumerate(results):
        print(f"Result {i+1}:")
        print(f"  {doc.page_content[:150]}...")
        print()
    return [{"content": doc.page_content, "metadata": doc.metadata} for doc in results]