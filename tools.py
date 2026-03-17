import os
import logging
import trafilatura

from ddgs import DDGS
from langchain_core.tools import tool
from config import settings

logging.basicConfig(level=logging.INFO)

os.makedirs(settings.path_save_file, exist_ok=True)


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
        with open(path, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"Report saved to {path}"
    except Exception as e:
        return f"Error writing report: {e}"
