# Frontend Defect Detector (React + Vite)
# Сборка с URL бэкенда: docker build --build-arg VITE_API_URL=http://localhost:8000 -t defect-detector-frontend ./client
# Запуск: docker run -p 80:80 defect-detector-frontend

# Этап 1: сборка
FROM node:20-alpine AS builder

WORKDIR /app

COPY package.json package-lock.json* ./
RUN npm ci

COPY . .

# URL API бэкенда при сборке (для fetch с фронта). Пусто = относительные пути.
ARG VITE_API_URL=
ENV VITE_API_URL=$VITE_API_URL

RUN npm run build

# Этап 2: раздача статики через nginx
FROM nginx:alpine

# Удаляем дефолтную конфигурацию и подключаем свою
RUN rm -rf /usr/share/nginx/html/*
COPY --from=builder /app/dist /usr/share/nginx/html

# Проксирование /api на бэкенд при необходимости (если фронт и бэк за одним хостом)
# Раскомментируйте и настройте при использовании одного домена:
# COPY nginx.conf /etc/nginx/conf.d/default.conf

EXPOSE 80

CMD ["nginx", "-g", "daemon off;"]
