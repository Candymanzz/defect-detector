# Backend — API детектора дефектов (Patchcore)

Бэкенд реализован как **отдельное приложение** в папке **`../server`**.

Здесь только описание; исходный код и запуск — в [../server](../server).

## Запуск

```bash
cd ../server
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

API: http://localhost:8000, документация: http://localhost:8000/docs.

## Назначение папки backend/

Символическое разделение: фронтенд (`client/`) и бэкенд (`server/`) — два приложения. Папка `backend/` указывает на бэкенд-приложение и содержит только этот README; весь код в `server/`.
