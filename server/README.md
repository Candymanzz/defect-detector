# Бэкенд Anomalib (Intel) для детектора дефектов

Сервис принимает эталонное и тестовое изображения и возвращает оценку аномалии (брак/норма) с помощью модели **Padim** из библиотеки [Anomalib](https://github.com/openvinotoolkit/anomalib).

## Требования

- Python 3.10+ (установите с [python.org](https://www.python.org/downloads/); не используйте заглушку из Microsoft Store, если Python не установлен через неё)
- Рекомендуется виртуальное окружение (venv или conda)

## Установка

```bash
cd server
python -m venv .venv
```

**Вариант 1 — с активацией окружения**

В PowerShell (Windows):
```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Если скрипты запрещены политикой, выполните один раз:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

В cmd (Windows):
```cmd
.venv\Scripts\activate.bat
pip install -r requirements.txt
```

В Linux/macOS:
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

**Вариант 2 — без активации (PowerShell/cmd)**

Укажите полный путь к `python.exe` из venv (после `python -m venv .venv`):

```powershell
# Пример, если Python установлен в C:\Python310:
# C:\Python310\python.exe -m venv .venv

.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Если команда `.\\.venv\Scripts\python.exe` не найдена, сначала создайте окружение из папки `server` командой (используйте путь к реальному Python, не к заглушке из Windows Apps):

```powershell
& "C:\путь\к\python.exe" -m venv .venv
```

При первом запуске Anomalib может подтянуть веса моделей (например, ResNet для Padim).

## Запуск

**Если окружение активировано** (см. выше):
```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

**Без активации (PowerShell/cmd из папки `server`):**
```powershell
.\.venv\Scripts\python.exe -m uvicorn main:app --reload --host 0.0.0.0 --port 8000
```

Или просто:
```bash
python main.py
```

Сервер будет доступен по адресу `http://localhost:8000`. Документация API: `http://localhost:8000/docs`.

## API

- **GET /** — проверка работы сервиса  
- **GET /health** — health check  
- **POST /api/anomalib/analyze** — анализ дефекта  
  - Тело: `multipart/form-data`  
  - Поля: `reference` (файл изображения), `test` (файл изображения), `threshold` (число, по умолчанию 0.5)  
  - Ответ: `{ "defect": bool, "score": float, "threshold": float, "heatmap_base64": str | null, "message": str | null }`

В клиенте на шаге «Тестовое фото» включён блок **Режим Anomalib**: укажите URL бэкенда (например `http://localhost:8000`) и нажмите «Сравнить через Anomalib».
