# Defect Detector — детектор дефектов этикетки на ведре

Два независимых приложения: **фронтенд** (UI, сравнение по MSE, вызов Patchcore) и **бэкенд** (API Patchcore/Anomalib).

## Структура

| Приложение | Папка    | Стек              | Назначение |
|------------|----------|-------------------|------------|
| **Фронтенд** | `client/` | React, Vite, TS   | Загрузка эталона/теста, контур ROI, сравнение по MSE, вызов API Patchcore, совмещённый вывод |
| **Бэкенд**   | `server/` | FastAPI, Anomalib, Patchcore | Обучение Patchcore, анализ изображения, калибровка порога |

Фронтенд и бэкенд запускаются отдельно и общаются по HTTP (CORS разрешён).

## Быстрый старт

### 1. Бэкенд (API)

```bash
cd server
python -m venv .venv
.\.venv\Scripts\Activate.ps1   # Windows PowerShell
pip install -r requirements.txt
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Подробнее: [server/README.md](server/README.md).

### 2. Фронтенд

```bash
cd client
npm install
npm run dev
```

Откройте указанный в терминале URL (обычно http://localhost:5173).  
URL бэкенда по умолчанию: `http://localhost:8000`. Его можно сменить в UI или задать при сборке через `VITE_API_URL` (см. `client/.env.example`).

### 3. Первый запуск бэкенда

Перед анализом нужно обучить модель и (по желанию) откалибровать порог:

- **Обучение:** `POST http://localhost:8000/api/patchcore/train` (или кнопка/скрипт из server).
- **Калибровка:** `GET http://localhost:8000/api/patchcore/calibrate` или скрипт `server/calibrate_dump_scores.py`.

Данные для обучения: `server/dataset/normal/` (и при необходимости `server/dataset/abnormal/`).

## Переменные окружения

- **Фронтенд (сборка):** `VITE_API_URL` — базовый URL API (по умолчанию `http://localhost:8000`). Пример: `client/.env.example`.
- **Бэкенд:** при необходимости задаётся в `server` (см. server/README.md).

## Сборка фронтенда для продакшена

```bash
cd client
# При необходимости задайте URL бэкенда:
# echo "VITE_API_URL=https://api.example.com" > .env
npm run build
```

Статика в `client/dist/`. Раздавайте любым HTTP-сервером (Nginx, static hosting); запросы к API уходят на `VITE_API_URL`.

## Добавление своего сервиса между клиентом и бэкендом

Если нужно вставить прокси, BFF, шлюз или другой слой между фронтендом и API (логирование, авторизация, свой домен): контракт API, форматы запросов/ответов и примеры описаны в **[docs/Integration.md](docs/Integration.md)**.
