# lab3-veres — Multi-Agent AI Systems

Два незалежні мультиагентні пайплайни, побудовані на LangChain / LangGraph:

| Система | Патерн | Запуск |
|---|---|---|
| **Research System** (homework-8 / homework-10) | Supervisor → Planner → Researcher → Critic | `python main.py` |
| **Dev Team System** (diploma) | BA → Developer ↔ QA (Evaluator-Optimizer) | `python dev_main.py` |

---

## Research System (homework-8 + homework-10)

### Архітектура

```
User
 └─► Supervisor
       ├─► plan()      → Planner Agent  → ResearchPlan (JSON)
       ├─► research()  → Research Agent → Markdown findings
       ├─► critique()  → Critic Agent   → CritiqueResult (JSON)
       │     └─ verdict=REVISE → research() знову (≤2 раунди)
       └─► save_report() ← HITL gate (approve / edit / reject)
```

**Агенти та інструменти:**

| Агент | Інструменти | Structured Output |
|---|---|---|
| Planner | `web_search`, `knowledge_search` | `ResearchPlan` |
| Researcher | `web_search`, `read_url`, `knowledge_search` | — (Markdown) |
| Critic | `web_search`, `read_url`, `knowledge_search` | `CritiqueResult` |

### Встановлення та запуск

```bash
python -m venv .venv
source .venv/bin/activate       # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# Налаштування (.env)
API_KEY=ваш_openai_ключ
MODEL_NAME=gpt-4o-mini
PATH_SAVE_FILE=output
DATA_DIR=data
INDEX_DIR=index

# 1. Побудувати RAG-індекс (покласти PDF/TXT у ./data/)
python ingest.py

# 2. Запустити
python main.py
```

### HITL — підтвердження збереження звіту

```
Options: [approve] / [edit <your feedback>] / [reject]
Decision: approve
```

- `approve` — зберегти звіт як є
- `edit додай висновки` — Supervisor переробляє і питає знову
- `reject` — скасувати збереження

---

## Dev Team System (diploma)

Симулює AI-команду розробки: Business Analyst аналізує user story, Developer пише код, QA перевіряє його в автоматичному циклі.

### Архітектура

```
User story
    │
    ▼
BA Agent ──(web_search + knowledge_search)──► SpecOutput (JSON)
    │
    ▼
HITL gate  ◄─── "approve" / feedback (повертає до BA)
    │ approve
    ▼
Developer Agent ──(web_search + python_repl + write_file)──► CodeOutput + файли на диску
    │
    ▼
QA Agent ──(python_repl + read_file)──► ReviewOutput (JSON)
    │
    ├─ verdict=REVISION_NEEDED (iteration < 5) ──► Developer (з issues + suggestions)
    │
    └─ verdict=APPROVED  ──► END
```

**Патерн:** Evaluator-Optimizer (Anthropic) — генерація→оцінка з циклом до 5 ітерацій.

### Structured Output контракти

```python
# Business Analyst → Developer
SpecOutput(
    title: str,
    requirements: list[str],
    acceptance_criteria: list[str],   # бінарні, тестовані
    tech_stack: list[str],
    estimated_complexity: "simple" | "medium" | "complex",
    coding_standards: list[str],
)

# Developer → QA
CodeOutput(
    description: str,
    files_created: list[str],         # реальні файли у workspace
    source_code: str,                 # головний модуль
    test_results: str,                # вивід python_repl
)

# QA → Developer | END
ReviewOutput(
    verdict: "APPROVED" | "REVISION_NEEDED",
    score: float,                     # 0.0–1.0
    issues: list[str],               # конкретні проблеми (file:line)
    suggestions: list[str],          # actionable виправлення
    acceptance_criteria_met: list[str],
    acceptance_criteria_failed: list[str],
)
```

### Інструменти

| Інструмент | Агенти | Призначення |
|---|---|---|
| `web_search` (DuckDuckGo) | BA, Developer | Документація, бібліотеки, патерни |
| `knowledge_search` (RAG) | BA | Внутрішні стандарти кодування, ADR |
| `python_repl` | Developer, QA | Виконання та тестування коду (sandbox) |
| `write_file` | Developer | Створення файлів проєкту у workspace |
| `read_file` | Developer, QA | Читання файлів з workspace |
| `list_files` | Developer, QA | Перегляд структури workspace |

### Sandboxed Python REPL

- **AST-перевірка** перед виконанням блокує заборонені модулі
- **Заборонені:** `os`, `subprocess`, `sys`, `shutil`, `socket`, `threading`, `multiprocessing`, `ctypes`, `importlib`, `pickle`
- **Timeout:** 10 секунд (захист від нескінченних циклів)
- **Ліміт виводу:** 4000 символів

### Встановлення та запуск

```bash
# Додаткові змінні у .env (або окремий .env):
MODEL_NAME=gpt-4o-mini
API_KEY=ваш_openai_ключ
DEV_WORKSPACE_DIR=dev_workspace   # де Developer створює файли

# Запуск
python dev_main.py
```

### Приклад сесії

```
User story: As a user, I want a function that validates email addresses

[ba] → web_search(email validation python regex...)
[ba] → knowledge_search(coding standards...)
[ba] ← SpecOutput: Email Validator | complexity: simple

==============================
  SPECIFICATION: Email Validator
==============================
  Complexity: simple

Requirements:
  • validate_email(email: str) -> bool
  • Returns True for valid RFC-style addresses
  • Returns False for empty string, missing @, missing domain
  • Handles None without raising exceptions

Acceptance Criteria:
  ✓ validate_email('user@example.com') == True
  ✓ validate_email('') == False
  ✓ validate_email(None) returns False (no exception)
==============================

Review the specification. Type 'approve' or describe what needs to change.
Decision: approve

[developer] → write_file(src/email_validator.py ...)
[developer] → python_repl(# test validate_email...)
[qa] → read_file(src/email_validator.py)
[qa] → python_repl(validate_email(None) ...)

✅  QA Verdict: APPROVED  (score: 0.92)
Passed criteria:
  ✓ validate_email('user@example.com') == True
  ✓ validate_email('') == False
  ✓ validate_email(None) returns False (no exception)

Files created: src/email_validator.py, tests/test_email_validator.py
```

---

## Тести (LLM-as-a-Judge)

Обидві системи покриті автоматизованими тестами на базі **DeepEval**.

### Запуск

```bash
# Всі тести
deepeval test run tests/

# Тільки Dev Team тести
deepeval test run tests/test_ba_dev.py tests/test_developer_dev.py tests/test_qa_dev.py tests/test_e2e_dev.py

# Тільки Research System тести
deepeval test run tests/test_planner.py tests/test_researcher.py tests/test_critic.py tests/test_e2e.py

# Тести інструментів (без API key)
pytest tests/test_e2e_dev.py::test_e2e_repl_sandboxing tests/test_e2e_dev.py::test_e2e_repl_timeout tests/test_e2e_dev.py::test_e2e_workspace_tools -v
```

### Покриття тестами

#### Research System (homework-10)

| Файл | Що тестується | Метрики DeepEval |
|---|---|---|
| `test_planner.py` | Якість плану, виклик search-інструментів, поля SpecOutput | `GEval` Plan Quality |
| `test_researcher.py` | Groundedness відповіді, повнота, Markdown-формат | `GEval` Groundedness, Completeness |
| `test_critic.py` | Якість critique, консистентність verdict↔revision_requests | `GEval` Critique Quality |
| `test_tools.py` | Tool correctness для кожного агента | `ToolCorrectnessMetric` |
| `test_e2e.py` | Повний pipeline на golden dataset (15 прикладів) | `AnswerRelevancyMetric`, `GEval` Correctness |

#### Dev Team System (diploma)

| Файл | Що тестується | Метрики DeepEval |
|---|---|---|
| `test_ba_dev.py` | Специфікація повна, тестована, BA викликає search | `GEval` Spec Completeness/Accuracy |
| `test_developer_dev.py` | Код покриває requirements, якість коду, файли створені | `GEval` Code Coverage/Quality |
| `test_qa_dev.py` | QA знаходить реальні проблеми у **навмисно поганому коді** | `GEval` Issue Detection/Verdict Consistency |
| `test_e2e_dev.py` | Повний pipeline + REPL sandbox + timeout + workspace | `GEval` E2E Quality/Correctness |

### Golden Datasets

- `tests/golden_dataset.json` — 15 прикладів для Research System (happy path / edge case / failure)
- `tests/golden_dataset_dev.json` — 10 прикладів для Dev Team (happy path / edge case / failure)

---

## Структура проєкту

```
lab3-veres/
├── agents/
│   ├── planner.py        # Research System: Planner
│   ├── research.py       # Research System: Researcher
│   ├── critic.py         # Research System: Critic
│   ├── ba.py             # Dev Team: Business Analyst
│   ├── developer.py      # Dev Team: Developer
│   └── qa.py             # Dev Team: QA Engineer
├── tests/
│   ├── golden_dataset.json        # Research System golden examples
│   ├── golden_dataset_dev.json    # Dev Team golden examples
│   ├── conftest.py                # Research System test helpers
│   ├── conftest_dev.py            # Dev Team test helpers
│   ├── test_planner.py
│   ├── test_researcher.py
│   ├── test_critic.py
│   ├── test_tools.py
│   ├── test_e2e.py
│   ├── test_ba_dev.py
│   ├── test_developer_dev.py
│   ├── test_qa_dev.py
│   └── test_e2e_dev.py
├── dev_workspace/         # Файли, створені Developer (генерується автоматично)
├── output/                # Звіти Research System
├── data/                  # Документи для RAG-індексу
├── index/                 # BM25 + Chroma індекс
├── schemas.py             # ResearchPlan, CritiqueResult
├── dev_schemas.py         # SpecOutput, CodeOutput, ReviewOutput
├── config.py              # Налаштування + промпти Research System
├── dev_config.py          # Налаштування + промпти Dev Team
├── tools.py               # web_search, read_url, knowledge_search, save_report
├── tools_repl.py          # Sandboxed Python REPL
├── tools_fs.py            # write_file, read_file, list_files
├── supervisor.py          # Research System LangGraph graph
├── dev_graph.py           # Dev Team LangGraph StateGraph
├── main.py                # Research System entry point
├── dev_main.py            # Dev Team entry point
├── ingest.py              # RAG ingestion pipeline
├── chroma_db.py           # ChromaDB helpers
└── requirements.txt
```

## Змінні середовища (.env)

```env
API_KEY=ваш_openai_ключ
MODEL_NAME=gpt-4o-mini

# Research System
PATH_SAVE_FILE=output
DATA_DIR=data
INDEX_DIR=index

# Dev Team
DEV_WORKSPACE_DIR=dev_workspace

# Моніторинг (опційно)
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=dev-agent-system
LANGCHAIN_API_KEY=ваш_langsmith_ключ
```
