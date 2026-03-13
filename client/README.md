# Frontend — детектор дефектов (UI)

React + Vite + TypeScript. Эталон → контур зоны → тестовое фото → сравнение по MSE и/или через Patchcore API.

## Запуск

```bash
npm install
npm run dev
```

Откройте URL из терминала (обычно http://localhost:5173).

## Переменные окружения

При сборке (`npm run build`) используется:

- **VITE_API_URL** — базовый URL бэкенда (по умолчанию `http://localhost:8000`).

Скопируйте `.env.example` в `.env` и при необходимости измените:

```bash
cp .env.example .env
```

## Сборка

```bash
npm run build
```

Результат в `dist/`. Для продакшена раздавайте статику и настройте прокси к бэкенду или укажите `VITE_API_URL` на публичный URL API.

## Структура src/

- **api/** — клиент API бэкенда (Patchcore analyze).
- **components/** — шаги и блоки результатов (ReferenceStep, TestStep, PatchcoreStep, ResultBlocks и др.).
- **lib/** — алгоритм сравнения по MSE (compare.ts).
- **types/** — общие типы (Point, Rect, CompareResult).
- **config.ts** — конфиг (в т.ч. API URL из env).

Подробнее про взаимодействие с бэкендом и добавление промежуточного сервиса: [../docs/Integration.md](../docs/Integration.md).
