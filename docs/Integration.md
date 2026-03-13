# Взаимодействие с сервисами Defect Detector

Документ для тех, кто хочет **добавить свой сервис между клиентом и бэкендом** (прокси, BFF, шлюз, логирование, авторизация и т.п.).

---

## 1. Архитектура

```
┌─────────────┐         ┌─────────────────────┐         ┌─────────────┐
│   Клиент    │  HTTP   │  Ваш сервис (опц.)   │  HTTP   │   Бэкенд    │
│  (client/)  │ ──────► │  прокси / BFF / ...  │ ──────► │  (server/)  │
│  React+Vite │         │                     │         │  FastAPI     │
└─────────────┘         └─────────────────────┘         └─────────────┘
       │                              │
       │  VITE_API_URL                │  upstream URL к server
       └──────────────────────────────┘
```

- **Клиент** ходит по одному базовому URL (переменная `VITE_API_URL` при сборке или поле «URL бэкенда» в UI). Все запросы уходят на этот URL.
- **Ваш сервис** принимает те же запросы и либо проксирует их в `server/`, либо обрабатывает сам и при необходимости вызывает бэкенд.
- **Бэкенд** — FastAPI в `server/`, порт 8000 по умолчанию.

---

## 2. Контракт бэкенда (server)

Бэкенд предоставляет следующие эндпоинты.

### 2.1. Проверка живости

| Метод | Путь      | Назначение        | Ответ (200)              |
|-------|------------|-------------------|---------------------------|
| GET   | `/`        | Имя и статус      | `{"service":"...", "status":"ok"}` |
| GET   | `/health`  | Health check      | `{"status":"ok"}`         |

Без тела запроса. Используйте для проверки доступности перед проксированием или для оркестрации (K8s, Docker).

### 2.2. Анализ изображения (Patchcore)

| Метод | Путь                      | Назначение                    |
|-------|----------------------------|-------------------------------|
| POST  | `/api/anomalib/analyze`    | Анализ тестового фото на брак |

**Content-Type:** `multipart/form-data`.

**Поля формы:**

| Имя        | Тип    | Обязательное | Описание |
|------------|--------|--------------|----------|
| `reference`| файл   | нет          | Эталонное изображение (для совместимости; Patchcore его не использует) |
| `test`     | файл   | да           | Тестовое изображение для проверки на дефекты |
| `threshold`| строка | нет          | Число 0–1, по умолчанию 0.5 (порог для UI)   |

**Ответ 200 (JSON):**

```json
{
  "defect": false,
  "score": 0.234,
  "threshold": 0.5,
  "heatmap_base64": "iVBORw0KGgo...",
  "message": null,
  "raw_score": 8.05
}
```

| Поле            | Тип    | Описание |
|-----------------|--------|----------|
| `defect`        | bool   | Брак (true) или норма (false) по порогу |
| `score`         | float  | Нормализованный скор 0–1 (Min-Max по калибровке) |
| `threshold`     | float  | Порог, переданный в запросе |
| `heatmap_base64`| str \| null | PNG карты аномалий в base64 или null |
| `message`       | str \| null | Сообщение об ошибке или null |
| `raw_score`     | float \| null | Сырой скор до нормализации |

**Ошибки:** при 4xx/5xx тело — JSON `{"detail": "строка или объект"}`. Клиент читает `detail` для отображения пользователю.

### 2.3. Обучение модели

| Метод | Путь                    | Назначение           |
|-------|--------------------------|----------------------|
| POST  | `/api/patchcore/train`   | Запуск обучения Patchcore |

Тело не требуется. Ответ 200 — JSON в формате `PatchcoreResult` (например, `defect: false`, `message` с путём к модели).

### 2.4. Калибровка порога

| Метод | Путь                         | Назначение |
|-------|-------------------------------|------------|
| GET   | `/api/patchcore/calibrate`    | Прогон по normal/abnormal, расчёт границ Min-Max и предложение порогов |

**Query-параметры:** `limit_per_class` (int, необязательно) — лимит картинок на класс для калибровки.

**Ответ 200 (JSON):** объект с полями `normal_count`, `abnormal_count`, `normal_stats`, `abnormal_stats`, `suggested_t_low`, `suggested_t_high`, `samples` (массив с `path`, `kind`, `score`, `raw_score`).

---

## 3. Контракт клиента (client)

Клиент делает запросы только к **одному базовому URL** (из `config.apiBaseUrl` или из поля «URL бэкенда»).

- Для анализа используется **только** `POST /api/anomalib/analyze` с `multipart/form-data`: поля `reference`, `test`, `threshold`.
- Ожидаемый ответ — JSON с полями `defect`, `score`, `threshold`, `raw_score`, `heatmap_base64`, `message` (как в п. 2.2).
- При ошибке клиент обрабатывает любой 4xx/5xx и по возможности парсит `detail` из JSON тела.

Остальные эндпоинты (`/`, `/health`, train, calibrate) клиент по умолчанию не вызывает; их может вызывать ваш промежуточный сервис или админка.

---

## 4. Как вставить свой сервис между клиентом и бэкендом

### 4.1. Вариант A: прозрачный прокси

Клиент указывает в `VITE_API_URL` (или в UI) **адрес вашего сервиса**. Ваш сервис принимает те же пути и методы и проксирует запросы на бэкенд.

Что нужно сделать:

1. Принимать запросы на те же пути, что и бэкенд (как минимум `POST /api/anomalib/analyze`), с теми же заголовками и телом.
2. Проксировать запрос на бэкенд (например, на `http://localhost:8000`), сохраняя:
   - метод и путь;
   - `Content-Type: multipart/form-data` и тело формы для `/api/anomalib/analyze`;
   - при необходимости заголовки (например, `X-Request-Id`).
3. Возвращать клиенту ответ бэкенда (статус, заголовки, тело) без изменений.
4. Настроить CORS для источника клиента (или разрешить нужные `Origin`), если клиент и ваш сервис на разных доменах/портах.

Бэкенд уже отдаёт `Access-Control-Allow-Origin: *`. Если клиент ходит на ваш сервис, CORS должен отдавать ваш сервис.

### 4.2. Вариант B: BFF (Backend for Frontend)

Ваш сервис — единственная точка входа для клиента. Он может:

- принимать запросы по **своим** путям (например, `POST /api/analyze`);
- внутри вызывать бэкенд `POST http://backend:8000/api/anomalib/analyze` с переданными файлами и порогом;
- при необходимости дополнять ответ (логирование, метрики, доп. поля);
- возвращать клиенту ответ в том же формате, что ожидает клиент (поля из п. 2.2).

Тогда в клиенте нужно поменять URL вызова: либо задать `VITE_API_URL` на ваш BFF, либо в коде заменить путь на путь BFF (например, `POST /api/analyze` вместо `POST /api/anomalib/analyze`), сохранив формат тела и ответа.

### 4.3. Вариант C: шлюз с авторизацией/ограничениями

Перед проксированием на бэкенд ваш сервис может:

- проверять заголовок (например, `Authorization`) или cookie;
- ограничивать размер тела или тип файлов;
- логировать запросы и ответы.

Важно: для `POST /api/anomalib/analyze` не менять структуру `multipart/form-data` (поля `reference`, `test`, `threshold`) и формат JSON-ответа, иначе текущий клиент перестанет корректно отображать результат.

---

## 5. CORS, заголовки, ошибки

- **Бэкенд:** включён `CORSMiddleware` с `allow_origins=["*"]`, `allow_credentials=True`, `allow_methods=["*"]`, `allow_headers=["*"]`. Для продакшена при необходимости сузьте `allow_origins` до домена клиента.
- **Ваш сервис:** если клиент с другого origin (другой порт/домен), ответы вашего сервиса должны содержать заголовки CORS (например, `Access-Control-Allow-Origin`), иначе браузер заблокирует ответ.
- **Ошибки бэкенда:** при 4xx/5xx FastAPI возвращает JSON `{"detail": "..."}`. Клиент при ошибке показывает пользователю строку из `detail` или текст ответа.

---

## 6. Примеры

### 6.1. Nginx как обратный прокси

Клиент и бэкенд на одном домене; Nginx раздаёт статику клиента и проксирует `/api/` на бэкенд:

```nginx
server {
  listen 80;
  server_name example.com;
  root /var/www/client/dist;
  index index.html;
  location / {
    try_files $uri $uri/ /index.html;
  }
  location /api/ {
    proxy_pass http://127.0.0.1:8000;
    proxy_http_version 1.1;
    proxy_set_header Host $host;
    proxy_set_header X-Real-IP $remote_addr;
    client_max_body_size 50M;
  }
}
```

Клиент собирается с `VITE_API_URL=https://example.com` (или пусто, тогда запросы на тот же origin).

### 6.2. Небольшой BFF на Node (Express)

Принимает тот же multipart и проксирует на бэкенд:

```javascript
const express = require('express');
const multer = require('multer');
const FormData = require('form-data');
const fetch = require('node-fetch');

const app = express();
const upload = multer({ storage: multer.memoryStorage() });
const BACKEND = process.env.BACKEND_URL || 'http://localhost:8000';

app.post('/api/analyze', upload.fields([{ name: 'reference' }, { name: 'test' }]), async (req, res) => {
  const form = new FormData();
  if (req.files.reference) form.append('reference', req.files.reference[0].buffer, { filename: 'reference.png' });
  form.append('test', req.files.test[0].buffer, { filename: 'test.png' });
  form.append('threshold', req.body.threshold || '0.5');
  const r = await fetch(`${BACKEND}/api/anomalib/analyze`, { method: 'POST', body: form, headers: form.getHeaders() });
  const data = await r.json().catch(() => ({}));
  res.status(r.status).json(data);
});
```

Клиент в этом случае должен слать запросы на `https://your-bff/api/analyze` с теми же полями формы; ответ — в формате бэкенда (п. 2.2).

### 6.3. Прокси в Vite (разработка)

Чтобы в dev не настраивать CORS и ходить на один origin, можно проксировать через Vite:

```ts
// vite.config.ts
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      '/api': { target: 'http://localhost:8000', changeOrigin: true },
    },
  },
});
```

Клиент тогда использует относительный путь (например, `config.apiBaseUrl = ''` или `'/api'` в dev), и запросы к `/api/...` уйдут на бэкенд через Vite.

---

## 7. Краткая сводка для интеграции

| Кто       | Действие |
|-----------|----------|
| **Клиент** | Шлёт все запросы на один базовый URL (`VITE_API_URL` или поле в UI). Главный вызов: `POST /api/anomalib/analyze` (multipart: `reference`, `test`, `threshold`). Ожидает JSON с полями `defect`, `score`, `threshold`, `heatmap_base64`, `message`, `raw_score`. |
| **Ваш сервис** | Принимает запросы с клиента; при необходимости проверяет авторизацию/лимиты; проксирует на бэкенд или реализует тот же контракт, вызывая бэкенд внутри. Важно сохранить формат запроса и ответа для `/api/anomalib/analyze`, чтобы клиент работал без правок. |
| **Бэкенд** | Слушает на своём порту (8000), отдаёт описанные эндпоинты и форматы. CORS уже включён. |

Если нужно расширить API (новые поля, эндпоинты), опишите изменения в контракте в этом документе или в `server/README.md`, чтобы и клиент, и промежуточный сервис могли согласованно обновиться.
