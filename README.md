# lab3-veres

Мультиагентна дослідницька система на основі LangChain / LangGraph / MCP / ACP.

Архітектура: **Supervisor** координує три суб-агенти за патерном Plan → Research → Critique.
Critic може повернути Researcher на доопрацювання (до 2 раундів).
Збереження звіту потребує підтвердження користувача (HITL).

```
User → Supervisor → ACP → Planner / Researcher / Critic → MCP → SearchMCP / ReportMCP
```

## Встановлення

```bash
python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Налаштування

Скопіювати `.env.examle` в `.env`:

```
API_KEY=ваш_openai_ключ
MODEL_NAME=gpt-4o-mini
PATH_SAVE_FILE=output
DATA_DIR=data
INDEX_DIR=index
```

## Запуск

```bash
# 1. Побудувати індекс знань (один раз або при зміні документів у data/)
python ingest.py

# 2. Запустити MCP та ACP сервери (кожен в окремому терміналі)
python mcp_servers/search_mcp.py   # порт 8901 — web_search, read_url, knowledge_search
python mcp_servers/report_mcp.py   # порт 8902 — save_report
python acp_server.py               # порт 8903 — planner, researcher, critic

# 3. Запустити Supervisor REPL
python main.py
```

## HITL — підтвердження збереження звіту

Коли Supervisor готовий зберегти звіт, система зупиняється і питає:

```
Options: [approve] / [edit <your feedback>] / [reject]
Decision: approve
```

- `approve` — зберегти звіт як є
- `edit додай висновки` — Supervisor переробляє і питає знову
- `reject` — скасувати збереження

pip install --upgrade uvicorn websockets