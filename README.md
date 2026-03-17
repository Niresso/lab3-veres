# lab3-veres

Агент для веб-пошуку та аналізу інформації на основі LangChain.

## Встановлення

```bash
# Створити віртуальне середовище
python3 -m venv venv

# Активувати середовище
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows

# Встановити залежності
pip install -r requirements.txt
```

## Налаштування

Скопіювати `.env.examle` в `.env` та заповнити змінні:

```
API_KEY=ваш_api_ключ
MODEL_NAME=назва_моделі  # наприклад: gpt-4o
```

## Запуск

```bash
python main.py
```

## Якщо потрібно зберегти файл то треба написати наприклад
You: збережи це 
