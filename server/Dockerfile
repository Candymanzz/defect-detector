# Backend Defect Detector (FastAPI + Anomalib/Patchcore)
# Сборка: docker build -t defect-detector-backend ./server
# Запуск: docker run -p 8000:8000 -v $(pwd)/server/dataset:/app/dataset -v defect-results:/app/results defect-detector-backend

FROM python:3.11-slim

WORKDIR /app

# Системные зависимости (для компиляции пакетов при необходимости)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Зависимости Python
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Код приложения
COPY main.py .
COPY calibrate_dump_scores.py .

# Каталоги для датасета и результатов (монтируются как volume)
RUN mkdir -p /app/dataset/normal /app/dataset/abnormal /app/results

ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# По умолчанию uvicorn без --reload (для продакшена в контейнере)
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
