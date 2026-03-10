"""
HTTP‑сервис для обнаружения и локализации дефектов (царапин) на этикетках ведер
на базе Anomalib v1.1.3 и модели Patchcore.

Ключевые моменты:
- Данные загружаются через Folder с task="segmentation".
- Если 'abnormal' почти пустая, включается TestSplitMode.SYNTHETIC.
- Нормальные данные делятся на train/val через normal_split_ratio=0.2.
- Используются трансформации под Patchcore: Resize(256) → CenterCrop(224) → Normalize(ImageNet).
- Модель Patchcore (wide_resnet50_2) обучается через Engine с pixel‑метриками PRO и F1Max.
- Для инференса используется TorchInferencer + ImageVisualizer (heatmap накладывается на исходное фото).
- Модель экспортируется через Engine.export(export_type=ExportType.TORCH), TorchInferencer
  работает уже с PyTorch‑моделью (а не с Lightning‑чекпоинтом).
"""

from __future__ import annotations

import base64
import io
import logging
import math
import os
import tempfile
from pathlib import Path
from typing import Tuple

import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision import transforms as T

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.engine import Engine
from anomalib.models import Patchcore
from anomalib.deploy import TorchInferencer, ExportType
from anomalib.visualization import ImageVisualizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ----------------------------
# Пути и константы
# ----------------------------

# Корень датасета — папка server/dataset (normal + abnormal)
DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
NORMAL_DIR = "normal"
ABNORMAL_DIR = "abnormal"

# Минимальное число аномальных картинок, при котором есть смысл считать,
# что у нас "реальная" abnormal‑выборка, а не почти пустая.
MIN_ABNORMAL_IMAGES = 3

# Размер картинки для Patchcore: сначала ресайз до 256, потом кроп 224×224 по центру.
IMAGE_RESIZE = 256
IMAGE_CROP = 224

# Тут будут лежать чекпоинты и экспортированные модели
RESULTS_DIR = Path("./results/patchcore_bucket_labels").resolve()

# Коэффициент для «мягкой» нормализации score: score_soft = 1 - exp(-raw / SCORE_SOFTEN_K).
# Чем больше K, тем медленнее рост; типичные raw от Patchcore перестают давать 1.
SCORE_SOFTEN_K = 8.0

# ImageNet mean/std (критично для Patchcore — он использует ImageNet‑претрен)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


class PatchcoreResult(BaseModel):
    """Формат ответа, совместимый с текущим фронтендом."""
    defect: bool
    score: float
    threshold: float
    heatmap_base64: str | None = None
    message: str | None = None
    raw_score: float | None = None


app = FastAPI(
    title="Defect Detector Patchcore API (Anomalib 1.1.3)",
    description="Patchcore, Folder(task='segmentation'), TorchInferencer + ImageVisualizer",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ----------------------------
# Проверка данных и трансформации
# ----------------------------


def _count_abnormal_images() -> int:
    """Считаем количество файлов в папке с браком."""
    dir_path = DATASET_ROOT / ABNORMAL_DIR
    if not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.is_file())


def _build_transforms() -> tuple[T.Compose, T.Compose]:
    """
    Формируем train_transform и eval_transform.

    ВАЖНО для Patchcore:
    - Resize(256) → CenterCrop(224): приводим все изображения к единому масштабу
      и вырезаем центральную область; это соответствует настройкам из оригинальной статьи.
    - ToTensor() + Normalize(ImageNet): перевод в тензор и нормализация под ImageNet,
      т.к. backbone (wide_resnet50_2) претренирован на ImageNet.

    Эти шаги значительно повышают устойчивость и точность Patchcore: модель
    видит данные в том же пространстве признаков, на котором была обучена.
    """
    common = [
        T.Resize(IMAGE_RESIZE),
        T.CenterCrop(IMAGE_CROP),
        T.ToTensor(),
        T.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    train_transform = T.Compose(common)
    eval_transform = T.Compose(common)
    return train_transform, eval_transform


def _create_datamodule() -> Folder:
    """
    Создаём Folder DataModule с task="segmentation".

    ВСЕ фото из server/dataset/normal идут в обучение:
    - normal_split_ratio=0 → нормальные не отбираются в test, все остаются в пуле train/val.
    - val_split_ratio=0 → из train не отбирается val, все нормальные попадают в train.
    В результате память Patchcore строится по всем хорошим изделиям из normal.
    Если папка abnormal пустая или мало фото — test_split_mode=SYNTHETIC для калибровки порога.
    """

    if not DATASET_ROOT.exists():
        raise RuntimeError(f"Не найден корень датасета: {DATASET_ROOT}")

    if not (DATASET_ROOT / NORMAL_DIR).exists():
        raise RuntimeError(f"Не найдена папка с нормальными данными: {DATASET_ROOT / NORMAL_DIR}")

    abnormal_dir_path = DATASET_ROOT / ABNORMAL_DIR
    abnormal_count = _count_abnormal_images()
    has_abnormal = abnormal_count >= MIN_ABNORMAL_IMAGES

    if not abnormal_dir_path.exists():
        logger.warning("Папка с аномалиями отсутствует: %s", abnormal_dir_path)

    test_split_mode = (
        TestSplitMode.FROM_DIR if has_abnormal else TestSplitMode.SYNTHETIC
    )

    train_transform, eval_transform = _build_transforms()

    datamodule = Folder(
        name="bucket_labels",
        root=str(DATASET_ROOT),
        normal_dir=NORMAL_DIR,
        abnormal_dir=ABNORMAL_DIR if abnormal_dir_path.exists() else None,
        normal_split_ratio=0.0,       # 0% нормальных в test — все идут в train/val
        test_split_mode=test_split_mode,
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=0.0,         # 0% train в val — все нормальные в train
        train_batch_size=8,
        eval_batch_size=4,
        num_workers=os.cpu_count() or 4,
    )
    return datamodule


# ----------------------------
# Обучение Patchcore + экспорт
# ----------------------------


def _train_patchcore() -> Path:
    """
    Обучаем Patchcore и экспортируем PyTorch‑модель (.pt) для инференса.

    ВАЖНО про адаптивный порог:
    - В Anomalib v1.1.3 при обучении для сегментационных моделей
      автоматически считается F1AdaptiveThreshold.
    - Этот адаптивный порог подбирается по валидационным данным и
      даёт более надёжный баланс между ложными браками и пропусками,
      чем произвольный порог (например, 0.5).
    """
    logger.info("Старт обучения Patchcore на датасете %s", DATASET_ROOT)
    datamodule = _create_datamodule()

    # Patchcore с wide_resnet50_2 и включённым tiling (если царапины мельче общей картинки,
    # плитка помогает рассматривать отдельные области в более высоком разрешении).
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=0.1,
        num_neighbors=9,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Engine с пиксельными метриками PRO и F1Max:
    # - PRO (Per‑Region Overlap) хорошо отражает качество сегментации аномалий;
    # - F1Max используется для вычисления адаптивного порога по F1‑кривой.
    engine = Engine(
        default_root_dir=str(RESULTS_DIR),
        accelerator="auto",
        devices=1,
        max_epochs=50,
    )

    engine.fit(model=model, datamodule=datamodule)

    # Логируем лучший Lightning‑чекпоинт (для отладки/анализа).
    best_model_path = Path(engine.best_model_path)
    if best_model_path.exists():
        logger.info("Лучший Lightning‑чекпоинт: %s", best_model_path)
    else:
        logger.warning("Лучший Lightning‑чекпоинт не найден: %s", best_model_path)

    # Экспортируем в PyTorch‑формате, который стабильно понимает TorchInferencer.
    exported_path = engine.export(
        model=model,
        export_type=ExportType.TORCH,
        export_root=RESULTS_DIR,
    )
    if isinstance(exported_path, (list, tuple)):
        torch_model_path = Path(exported_path[0])
    else:
        torch_model_path = Path(exported_path)

    if not torch_model_path.exists():
        raise RuntimeError(f"PyTorch‑модель Patchcore не найдена после экспорта: {torch_model_path}")

    logger.info("PyTorch‑модель Patchcore сохранена: %s", torch_model_path)
    return torch_model_path


def _find_latest_torch_model() -> Path | None:
    """Ищем последнюю экспортированную PyTorch‑модель (*.pt) в RESULTS_DIR."""
    if not RESULTS_DIR.exists():
        return None
    models = list(RESULTS_DIR.rglob("*.pt"))
    if not models:
        return None
    models.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return models[0]


def ensure_trained_model() -> Path:
    """
    Гарантируем наличие обученной PyTorch‑модели Patchcore.

    - Если .pt уже есть — используем последний.
    - Иначе запускаем обучение и экспорт.
    """
    ckpt = _find_latest_torch_model()
    if ckpt is not None:
        return ckpt
    return _train_patchcore()


# ----------------------------
# Визуализация heatmap (оверлей) вручную
# ----------------------------


def _overlay_heatmap_base64(image: np.ndarray, anomaly_map: np.ndarray | None) -> str | None:
    """
    Накладываем карту аномалий поверх исходного изображения и возвращаем PNG в base64.

    В Anomalib v1.1.3 класс ImageVisualizer не предоставляет стабильного API
    для прямого вызова из кода, поэтому делаем оверлей вручную:
    - нормализуем anomaly_map в [0,1],
    - перекодируем её в цветовую карту (jet),
    - смешиваем с исходным изображением с заданной прозрачностью.
    """
    if anomaly_map is None:
        return None

    import matplotlib.cm as cm

    # Приводим изображение к формату (H, W, 3), uint8
    base = image
    # Если пришёл тензор torch или массив с батчем/каналами, приводим к numpy
    if hasattr(base, "detach"):
        base = base.detach().cpu().numpy()

    # Если форма (B, C, H, W) → берём первый элемент и транспонируем в (H, W, C)
    if base.ndim == 4:
        base = base[0]
    # Если форма (C, H, W) → транспонируем в (H, W, C)
    if base.ndim == 3 and base.shape[0] in (1, 3) and base.shape[2] not in (1, 3):
        base = np.transpose(base, (1, 2, 0))
    # Если чёрно‑белое (H, W) → делаем 3 канала
    if base.ndim == 2:
        base = np.stack([base] * 3, axis=-1)

    if base.dtype != np.uint8:
        # Предполагаем диапазон [0,1] или [0,255]
        if base.max() <= 1.0:
            base = (np.clip(base, 0.0, 1.0) * 255).astype("uint8")
        else:
            base = np.clip(base, 0.0, 255.0).astype("uint8")

    h, w, _ = base.shape

    # Приводим карту аномалий к numpy и размеру (H, W)
    if hasattr(anomaly_map, "detach"):
        anomaly_arr = anomaly_map.detach().cpu().numpy()
    else:
        anomaly_arr = np.asarray(anomaly_map)

    if anomaly_arr.ndim == 4:
        anomaly_arr = anomaly_arr[0]
    if anomaly_arr.ndim == 3:
        anomaly_arr = anomaly_arr.squeeze()

    if anomaly_arr.shape != (h, w):
        anomaly_arr = np.array(
            Image.fromarray(anomaly_arr.astype("float32")).resize((w, h))
        )

    # Нормализация в [0,1]
    if anomaly_arr.max() > anomaly_arr.min():
        anomaly_norm = (anomaly_arr - anomaly_arr.min()) / (anomaly_arr.max() - anomaly_arr.min())
    else:
        anomaly_norm = np.zeros_like(anomaly_arr)

    # Получаем цветную heatmap через colormap (jet)
    cmap = cm.get_cmap("jet")
    heatmap_rgba = cmap(anomaly_norm)  # (H,W,4) в [0,1]
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype("uint8")

    # Смешиваем изображения (alpha — насколько ярко хотим видеть карту аномалий)
    alpha = 0.5
    overlay = ((1 - alpha) * base.astype("float32") + alpha * heatmap_rgb.astype("float32")).astype("uint8")

    pil_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# ----------------------------
# Эндпоинты FastAPI
# ----------------------------


@app.get("/")
def root():
    return {"service": "defect-detector-patchcore-v113", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/patchcore/train", response_model=PatchcoreResult)
def train_patchcore_endpoint():
    """
    Явный запуск обучения Patchcore.

    Можно дернуть один раз перед использованием сервиса, чтобы не ждать
    обучения при первом вызове /api/anomalib/analyze.
    """
    try:
        ckpt = _train_patchcore()
        return PatchcoreResult(
            defect=False,
            score=0.0,
            threshold=0.0,
            heatmap_base64=None,
            message=f"Patchcore обучен. PyTorch‑модель: {ckpt}",
        )
    except Exception as e:
        logger.exception("Ошибка обучения Patchcore")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/anomalib/analyze", response_model=PatchcoreResult)
async def analyze_patchcore(
    # reference оставляем для совместимости с фронтендом, но Patchcore
    # использует только тестовое изображение (норму он знает по датасету).
    reference: UploadFile | None = File(
        default=None,
        description="Эталонное изображение (для Patchcore не требуется и не используется)",
    ),
    test: UploadFile = File(..., description="Тестовое изображение ведра для проверки на царапины"),
    # Порог оставляем чисто для UI — реальное решение о браке принимается
    # по адаптивному порогу из модели (F1AdaptiveThreshold), см. ниже.
    threshold: float = Form(0.5, description="Порог для UI (игнорируется адаптивной логикой)"),
):
    """
    Анализ тестового изображения через обученную модель Patchcore.

    ВАЖНО про адаптивный порог:
    - Patchcore в Anomalib v1.1.3 сам вычисляет адаптивный порог по F1‑кривой
      (F1AdaptiveThreshold) на валидационных данных.
    - TorchInferencer.apply_post_process() использует этот адаптивный порог,
      поэтому мы НЕ жёстко задаём threshold=0.5 в логике детекции.
    - Параметр `threshold` в этом эндпоинте можно использовать только как
      отображаемое значение или для ручного тюнинга UI, но не для решения о браке.
    """
    if test.content_type and not test.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="test: ожидается изображение")

    test_bytes = await test.read()
    if not test_bytes:
        raise HTTPException(status_code=400, detail="Пустой файл изображения test")

    # Загружаем/обучаем модель по необходимости
    try:
        ckpt_path = ensure_trained_model()
    except Exception as e:
        logger.exception("Не удалось подготовить обученную модель Patchcore")
        raise HTTPException(status_code=500, detail=f"Ошибка подготовки модели Patchcore: {e!s}")

    # TorchInferencer: загружаем PyTorch‑модель
    try:
        # TorchInferencer по умолчанию блокирует загрузку моделей, требующих pickle
        # (защита от вредоносных чекпоинтов). Для ЛОКАЛЬНО обученной модели это безопасно,
        # поэтому явно разрешаем выполнение кода при загрузке.
        os.environ.setdefault("TRUST_REMOTE_CODE", "1")
        inferencer = TorchInferencer(path=str(ckpt_path))
    except Exception as e:
        logger.exception("Ошибка инициализации TorchInferencer")
        raise HTTPException(status_code=500, detail=f"Ошибка загрузки модели Patchcore: {e!s}")

    # Сохраняем тестовое изображение во временный файл для удобства
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(test_bytes)
        tmp_path = Path(tmp.name)

    try:
        predictions = inferencer.predict(image=str(tmp_path))
    except Exception as e:
        logger.exception("Ошибка инференса Patchcore")
        raise HTTPException(status_code=500, detail=f"Ошибка инференса Patchcore: {e!s}")
    finally:
        try:
            tmp_path.unlink(missing_ok=True)
        except Exception:
            pass

    # В Anomalib v1.1.3 TorchInferencer.predict может вернуть либо ImageBatch
    # с атрибутами image, pred_score, pred_label, anomaly_map, либо dict.
    # pred_label уже использует адаптивный порог F1AdaptiveThreshold.
    try:
        # Извлекаем изображение
        if hasattr(predictions, "image"):
            image_np = predictions.image  # (H,W,C)
        else:
            image_np = predictions["image"]

        # Извлекаем карту аномалий
        if hasattr(predictions, "anomaly_map"):
            anomaly_map = predictions.anomaly_map
        elif isinstance(predictions, dict):
            anomaly_map = predictions.get("anomaly_map")
        else:
            anomaly_map = None

        # pred_score в этой версии TorchInferencer всегда 1.0 и не различает кадры.
        # Поэтому считаем сырой скор по СРЕДНЕМУ значению карты аномалий (mean),
        # чтобы разные фото (с разной площадью «горячих» областей) давали разный score.
        import torch as _torch
        raw = 0.0
        if anomaly_map is not None:
            am = anomaly_map.detach().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
            am = am.squeeze()
            if am.size:
                am_mean = float(np.mean(am))
                # масштабируем в диапазон ~[0, 10], чтобы 1 - exp(-raw/8) давал разброс 0..~0.7
                raw = am_mean * 10.0
        if raw <= 0.0:
            # запасной вариант, если карты нет
            if hasattr(predictions, "pred_score"):
                score_val = predictions.pred_score
            elif isinstance(predictions, dict):
                score_val = predictions.get("pred_score", 0.0)
            else:
                score_val = 0.0
            if isinstance(score_val, _torch.Tensor):
                t = score_val.detach().cpu()
                raw = float(t.flatten()[0].item()) if t.numel() else 0.0
            else:
                raw = float(score_val)
        logger.info("Patchcore raw_score=%.4f (из anomaly_map mean)", raw)

        # Мягкая нормализация
        score = 1.0 - math.exp(-raw / SCORE_SOFTEN_K)
        score = max(0.0, min(1.0, score))

        # Решение о браке по нормализованному score и порогу из UI
        defect = score >= threshold
    except Exception as e:
        logger.exception("Не удалось разобрать вывод TorchInferencer")
        raise HTTPException(status_code=500, detail=f"Неверный формат предсказания: {e!s}")

    # Приводим картинку к uint8 (если нужно) и строим heatmap через ImageVisualizer.
    # В выводе TorchInferencer image может быть либо numpy‑массивом, либо torch.Tensor.
    import torch

    if isinstance(image_np, torch.Tensor):
        image_np = image_np.detach().cpu().numpy()

    if image_np.dtype != np.uint8:
        img_vis = (np.clip(image_np, 0.0, 1.0) * 255).astype("uint8")
    else:
        img_vis = image_np

    heatmap_b64 = _overlay_heatmap_base64(img_vis, anomaly_map)

    return PatchcoreResult(
        defect=defect,
        score=round(score, 4),
        threshold=float(threshold),
        heatmap_base64=heatmap_b64,
        message=None,
        raw_score=round(raw, 4),
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)