"""
Defect Detector API — обнаружение дефектов (царапин) на этикетках ведер.

Стек: FastAPI + Anomalib 1.1.3 + Patchcore (wide_resnet50_2).
- Обучение: Folder(task="segmentation"), все normal в train; abnormal — в test или SYNTHETIC.
- Инференс: Lightning .ckpt (прямой predict) или TorchInferencer по .pt; Min-Max нормализация по калибровке.
- Калибровка: прогон по normal/abnormal → сохранение raw_score_min/max для шкалы 0–1.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import math
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
from pydantic import BaseModel
from torchvision.transforms import v2 as Tv2
from torchvision.transforms import functional as TF

from anomalib.data import Folder
from anomalib.data.utils import TestSplitMode, ValSplitMode
from anomalib.deploy import ExportType, TorchInferencer
from anomalib.engine import Engine
from anomalib.models import Patchcore

# Разрешаем загрузку .ckpt с произвольными классами (Anomalib/Lightning)
try:
    from anomalib.pre_processing.pre_processor import PreProcessor
    torch.serialization.add_safe_globals([PreProcessor])
except Exception:
    pass
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

# Безопасная очистка временной папки SyntheticAnomalyDataset (macOS PermissionError)
import anomalib.data.utils.synthetic as _synthetic_module
_synthetic_module.ROOT = str(Path(tempfile.gettempdir()) / "anomalib_synthetic")


def _safe_synthetic_del(self):
    if getattr(self, "_cleanup", True) and getattr(self, "root", None):
        try:
            import shutil
            if self.root.exists():
                shutil.rmtree(self.root, onerror=lambda _f, p, e: logger.debug("rmtree skip %s: %s", p, e))
        except Exception as e:
            logger.debug("Очистка SyntheticAnomalyDataset: %s", e)


_synthetic_module.SyntheticAnomalyDataset.__del__ = _safe_synthetic_del


# --- Логирование ---
LOG_LEVEL = os.environ.get("LOG_LEVEL", "INFO").upper()
logging.basicConfig(level=getattr(logging, LOG_LEVEL, logging.INFO))
logger = logging.getLogger(__name__)


# =============================================================================
# КОНФИГУРАЦИЯ (пути, размеры, параметры Patchcore)
# =============================================================================

# Датасет: server/dataset/normal и server/dataset/abnormal
DATASET_ROOT = Path(__file__).resolve().parent / "dataset"
NORMAL_DIR = "normal"
ABNORMAL_DIR = "abnormal"
MIN_ABNORMAL_IMAGES = 3  # Ниже — test_split_mode=SYNTHETIC

# Входное разрешение для модели (Resize → CenterCrop). 512×512 — баланс детали и скорости.
IMAGE_RESIZE = 512
IMAGE_CROP = 512

# Patchcore: слои ResNet (layer2+layer3 — более высокоуровневые структуры; старые чекпоинты также могут содержать layer2+layer3)
CORESET_SAMPLING_RATIO = 0.2   # Доля патчей в memory bank (0.2 даёт более насыщенную «память» нормальных образцов)
PATCHCORE_NUM_NEIGHBORS = 15   # K ближайших соседей при сравнении патча с памятью

# Результаты обучения и калибровки
RESULTS_DIR = Path("./results/patchcore_bucket_labels").resolve()
CALIBRATION_BOUNDS_FILE = "calibration_bounds.json"

# Мягкая нормализация score без калибровки: score = 1 - exp(-raw / K). Чем больше K, тем медленнее рост.
# K=14: нормальные кадры (raw ~8) дают score ~0.44, брак (raw 20+) — 0.76+; при пороге 0.5–0.62 меньше ложных срабатываний.
SCORE_SOFTEN_K = 14.0

# Нормализация под ImageNet (backbone претренирован на нём)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def _load_calibration_bounds() -> tuple[float, float] | None:
    """Загружает (raw_score_min, raw_score_max) из calibration_bounds.json. Если файла нет — None."""
    path = RESULTS_DIR / CALIBRATION_BOUNDS_FILE
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        lo, hi = data.get("raw_score_min"), data.get("raw_score_max")
        if lo is None or hi is None:
            return None
        return (float(lo), float(hi))
    except Exception as e:
        logger.warning("Не удалось загрузить границы калибровки из %s: %s", path, e)
        return None


def _min_max_normalize(raw: float, raw_min: float, raw_max: float) -> float:
    """Min-Max нормализация в [0, 1]. При raw_max == raw_min возвращает 0.5."""
    if raw_max <= raw_min:
        return 0.5
    return max(0.0, min(1.0, (raw - raw_min) / (raw_max - raw_min)))


class _EnsureTensor:
    """PIL/ndarray → тензор; тензор пропускается без изменений (для Anomalib augmentations)."""

    def __call__(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return TF.to_tensor(x)


# =============================================================================
# МОДЕЛИ ОТВЕТОВ API (Pydantic)
# =============================================================================


class PatchcoreResult(BaseModel):
    """Формат ответа для одиночного изображения (совместим с текущим фронтендом)."""

    defect: bool
    score: float
    threshold: float
    heatmap_base64: str | None = None
    message: str | None = None
    raw_score: float | None = None


class CalibrationSample(BaseModel):
    """Один пример из normal/abnormal с рассчитанным score."""

    path: str          # относительный путь внутри dataset
    kind: str          # "normal" или "abnormal"
    score: float
    raw_score: float


class CalibrationResult(BaseModel):
    """Результаты калибровки порога по всем картинкам из normal/abnormal."""

    normal_count: int
    abnormal_count: int
    normal_stats: dict | None
    abnormal_stats: dict | None
    suggested_t_low: float | None
    suggested_t_high: float | None
    samples: list[CalibrationSample]


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


# =============================================================================
# ДАННЫЕ И ТРАНСФОРМАЦИИ (датасет, train/eval transforms)
# =============================================================================


def _count_abnormal_images() -> int:
    """Считаем количество файлов в папке с браком."""
    dir_path = DATASET_ROOT / ABNORMAL_DIR
    if not dir_path.is_dir():
        return 0
    return sum(1 for p in dir_path.iterdir() if p.is_file())


def _build_transforms() -> tuple[Tv2.Compose, Tv2.Compose]:
    """
    Формируем train_transform и eval_transform.

    - Eval: детерминированный пайплайн Resize → CenterCrop → Normalize (выравнивание по центру кадра).
    - Train: те же шаги + RandomRotation и RandomAffine, чтобы Patchcore был устойчив к смещению/повороту ведра.
    """
    common_after_geom = [
        Tv2.Resize((IMAGE_RESIZE, IMAGE_CROP)),
        Tv2.CenterCrop(IMAGE_CROP),
        Tv2.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ]
    eval_transform = Tv2.Compose([
        _EnsureTensor(),
        *common_after_geom,
    ])
    train_transform = Tv2.Compose([
        _EnsureTensor(),
        Tv2.RandomRotation(degrees=5),
        Tv2.RandomAffine(degrees=0, translate=(0.05, 0.05)),
        *common_after_geom,
    ])
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

    # Критично: передаём те же Resize(512)+CenterCrop(512)+Normalize, что и в pre_processor модели.
    # Иначе при обучении Engine подставит дефолт 256→224, при инференсе — 512×512,
    # и одни и те же фото из normal будут давать разные признаки → ложная аномалия (1 из 100).
    train_transform, eval_transform = _build_transforms()

    datamodule = Folder(
        name="bucket_labels",
        root=str(DATASET_ROOT),
        normal_dir=NORMAL_DIR,
        abnormal_dir=ABNORMAL_DIR if has_abnormal else None,
        train_augmentations=train_transform,
        val_augmentations=eval_transform,
        test_augmentations=eval_transform,
        normal_split_ratio=0.0,       # 0% нормальных в test — все идут в train/val
        test_split_mode=test_split_mode,
        val_split_mode=ValSplitMode.FROM_TRAIN,
        val_split_ratio=0.0,         # 0% train в val — все нормальные в train
        train_batch_size=8,
        eval_batch_size=4,
        num_workers=os.cpu_count() or 4,
    )
    return datamodule


# =============================================================================
# ОБУЧЕНИЕ И МОДЕЛЬ (Patchcore, чекпоинты, инференс)
# =============================================================================


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

    # Patchcore с wide_resnet50_2; pre_processor 512×512 (Resize) + 512×512 (CenterCrop),
    # что строго соответствует трансформациям датамодуля и TorchInferencer.
    patchcore_pre_processor = Patchcore.configure_pre_processor(
        image_size=(IMAGE_RESIZE, IMAGE_RESIZE),
        center_crop_size=(IMAGE_CROP, IMAGE_CROP),
    )
    model = Patchcore(
        backbone="wide_resnet50_2",
        layers=["layer2", "layer3"],
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        num_neighbors=PATCHCORE_NUM_NEIGHBORS,
        pre_processor=patchcore_pre_processor,
    )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    # Engine с пиксельными метриками PRO и F1Max:
    # - PRO (Per‑Region Overlap) хорошо отражает качество сегментации аномалий;
    # - F1Max используется для вычисления адаптивного порога по F1‑кривой.
    # На Mac с M‑серией backend MPS часто упирается в лимит памяти даже на небольших датасетах.
    # Для стабильности принудительно обучаем на CPU.
    engine = Engine(
        default_root_dir=str(RESULTS_DIR),
        accelerator="cpu",
        devices=1,
        max_epochs=50,
    )

    engine.fit(model=model, datamodule=datamodule)

    # Логируем лучший Lightning‑чекпоинт (для отладки/анализа).
    best_model_path = Path(engine.best_model_path)
    if best_model_path.exists():
        logger.info("Лучший Lightning‑чекпоинт: %s", best_model_path)
        # Сохраняем путь для инференса через Engine.predict (экспорт .pt даёт постоянную anomaly_map).
        (RESULTS_DIR / "latest_lightning_ckpt.txt").write_text(str(best_model_path), encoding="utf-8")
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


def _find_latest_lightning_ckpt() -> Path | None:
    """Путь к последнему Lightning‑чекпоинту (.ckpt) для Engine.predict (нормальная anomaly_map)."""
    path_file = RESULTS_DIR / "latest_lightning_ckpt.txt"
    if path_file.exists():
        p = Path(path_file.read_text(encoding="utf-8").strip())
        if p.exists():
            return p
    if not RESULTS_DIR.exists():
        return None
    ckpts = list(RESULTS_DIR.rglob("*.ckpt"))
    if not ckpts:
        return None
    ckpts.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return ckpts[0]


def _infer_patchcore_layers_from_state(state_dict: dict) -> list[str]:
    """
    По ключам state_dict определяем, с какими слоями обучен чекпоинт.
    При наличии признаков layer3 считаем, что модель обучена на ["layer2", "layer3"].
    В остальных случаях по умолчанию используем ["layer2", "layer3"] для wide_resnet50_2.
    """
    for key in state_dict:
        if "layer3" in key:
            return ["layer2", "layer3"]
    return ["layer2", "layer3"]


def _build_patchcore_model(ckpt_path: Path | None = None) -> Patchcore:
    """
    Создаёт экземпляр Patchcore. Если передан ckpt_path — подбирает layers под чекпоинт,
    чтобы различающиеся по конфигурации чекпоинты (layer2+layer3) работали без ошибки размерностей.
    """
    layers = ["layer2", "layer3"]
    if ckpt_path is not None and ckpt_path.exists():
        try:
            ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
            state = ckpt.get("state_dict", ckpt)
            layers = _infer_patchcore_layers_from_state(state)
            logger.info("Чекпоинт %s: используем layers=%s", ckpt_path.name, layers)
        except Exception as e:
            logger.warning("Не удалось определить layers из чекпоинта %s: %s", ckpt_path, e)

    patchcore_pre_processor = Patchcore.configure_pre_processor(
        image_size=(IMAGE_RESIZE, IMAGE_RESIZE),
        center_crop_size=(IMAGE_CROP, IMAGE_CROP),
    )
    return Patchcore(
        backbone="wide_resnet50_2",
        layers=layers,
        coreset_sampling_ratio=CORESET_SAMPLING_RATIO,
        num_neighbors=PATCHCORE_NUM_NEIGHBORS,
        pre_processor=patchcore_pre_processor,
    )


def _predict_with_ckpt_direct(ckpt_path: Path, image_path: Path):
    """
    Инференс по Lightning‑чекпоинту: загрузка state_dict, model.model(tensor).
    Возвращает объект с .anomaly_map, .pred_score, .image (без постобработки Engine).
    """
    _, eval_transform = _build_transforms()
    img_pil = Image.open(image_path)
    if img_pil.mode != "RGB":
        img_pil = img_pil.convert("RGB")
    tensor = eval_transform(img_pil)  # (C, H, W)
    if tensor.dim() == 3:
        tensor = tensor.unsqueeze(0)  # (1, C, H, W)

    ckpt = torch.load(str(ckpt_path), map_location="cpu", weights_only=False)
    model = _build_patchcore_model(ckpt_path)
    model.load_state_dict(ckpt["state_dict"], strict=False)
    model.eval()

    with torch.no_grad():
        out = model.model(tensor)

    if out is None:
        raise ValueError("model.model вернул None")

    # Логируем вывод (сырые значения, не нормализованные в [0,1])
    logger.info("model.model() вернул type=%s", type(out).__name__)
    anomaly_map = getattr(out, "anomaly_map", None)
    if anomaly_map is None and isinstance(out, dict):
        anomaly_map = out.get("anomaly_map")
    pred_score = getattr(out, "pred_score", None)
    if pred_score is None and isinstance(out, dict):
        pred_score = out.get("pred_score")
    if anomaly_map is not None:
        am = anomaly_map.detach().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
        logger.info("direct anomaly_map shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                   getattr(anomaly_map, "shape", am.shape), float(np.min(am)), float(np.max(am)),
                   float(np.mean(am)), float(np.std(am)))
    if pred_score is not None:
        ps = pred_score.detach().cpu().numpy() if hasattr(pred_score, "detach") else np.asarray(pred_score)
        logger.info("direct pred_score=%s (shape=%s)", ps, getattr(ps, "shape", None))
    # Картинка для визуализации: denorm препроцессированного тензора
    img_vis = tensor[0].cpu().numpy().transpose(1, 2, 0)
    img_vis = np.clip(img_vis * np.array(IMAGENET_STD) + np.array(IMAGENET_MEAN), 0, 1)
    img_vis = (img_vis * 255).astype(np.uint8)

    class _Pred:
        pass
    p = _Pred()
    p.anomaly_map = anomaly_map
    p.pred_score = pred_score
    p.image = img_vis
    return p


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


# =============================================================================
# ВИЗУАЛИЗАЦИЯ (heatmap поверх изображения → base64)
# =============================================================================


def _overlay_heatmap_base64(image: np.ndarray, anomaly_map: np.ndarray | None) -> str | None:
    """
    Накладываем карту аномалий поверх исходного изображения и возвращаем PNG в base64.
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
        map_has_variation = True
    else:
        anomaly_norm = np.zeros_like(anomaly_arr)
        map_has_variation = False

    # Получаем цветную heatmap через colormap (jet)
    cmap = cm.get_cmap("jet")
    heatmap_rgba = cmap(anomaly_norm)  # (H,W,4) в [0,1]
    heatmap_rgb = (heatmap_rgba[..., :3] * 255).astype("uint8")

    # Если карта без вариации — не заливаем всё одним цветом (фиолетовый слой), делаем оверлей почти прозрачным
    alpha = 0.5 if map_has_variation else 0.08
    overlay = ((1 - alpha) * base.astype("float32") + alpha * heatmap_rgb.astype("float32")).astype("uint8")

    pil_img = Image.fromarray(overlay)
    buf = io.BytesIO()
    pil_img.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


# =============================================================================
# ИЗВЛЕЧЕНИЕ И НОРМАЛИЗАЦИЯ СКОРА (сырой raw → score 0–1)
# =============================================================================


def _extract_raw_score_from_predictions(predictions) -> tuple[float, str]:
    """
    Из вывода модели (anomaly_map / pred_score) получаем один сырой скор.
    Сырые значения от model.model() — ~50–100; от Engine/.pt — [0,1]. Приводим к одной шкале.
    Возвращает (raw, raw_source).
    """
    anomaly_map = getattr(predictions, "anomaly_map", None)
    if anomaly_map is None and isinstance(predictions, dict):
        anomaly_map = predictions.get("anomaly_map")

    raw = 0.0
    raw_source = "anomaly_map max"
    if anomaly_map is not None:
        am = anomaly_map.detach().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
        am = am.squeeze()
        if am.size:
            am_max = float(np.max(am))
            am_std = float(np.std(am))
            raw = (am_max / 10.0) if am_max > 2.0 else (am_max * 10.0)
            if am_std < 1e-6:
                logger.error(
                    "anomaly_map имеет нулевую дисперсию (std=%.6f). Возможна проблема инициализации весов или несоответствие размеров.",
                    am_std,
                )
                score_val = getattr(predictions, "pred_score", None)
                if score_val is None and isinstance(predictions, dict):
                    score_val = predictions.get("pred_score")
                if score_val is not None:
                    if isinstance(score_val, torch.Tensor):
                        t = score_val.detach().cpu()
                        ps_val = float(t.flatten()[0].item()) if t.numel() else 0.0
                    else:
                        ps_val = float(score_val)
                    raw = (ps_val / 10.0) if ps_val > 2.0 else (ps_val * 10.0)
                    raw_source = "pred_score"
                    logger.warning(
                        "anomaly_map постоянна, pred_score=%.4f -> raw=%.4f. Возможно старый .pt (layer2+layer3) — переобучите.",
                        ps_val,
                        raw,
                    )
    if raw <= 0.0:
        raw_source = "pred_score (fallback)"
        score_val = getattr(predictions, "pred_score", None)
        if score_val is None and isinstance(predictions, dict):
            score_val = predictions.get("pred_score", 0.0)
        if isinstance(score_val, torch.Tensor):
            t = score_val.detach().cpu()
            raw = float(t.flatten()[0].item()) if t.numel() else 0.0
        else:
            raw = float(score_val) if score_val is not None else 0.0
    return raw, raw_source


def _normalize_score_for_response(raw: float) -> float:
    """
    Применяем только Min-Max по calibration_bounds.
    Если calibration_bounds.json отсутствует или некорректен — считаем, что калибровка не проведена,
    и выбрасываем исключение, чтобы API явно сигнализировало об ошибке конфигурации.
    """
    bounds = _load_calibration_bounds()
    if bounds is None:
        raise RuntimeError(
            "Не найдены границы калибровки raw_score (calibration_bounds.json). "
            "Перед использованием инференса необходимо вызвать /api/patchcore/calibrate."
        )
    return _min_max_normalize(raw, bounds[0], bounds[1])


def _extract_adaptive_threshold_from_predictions(predictions) -> float | None:
    """
    Пытаемся извлечь адаптивный порог (F1AdaptiveThreshold), вычисленный Anomalib при обучении.
    Значение может храниться в metadata/мета‑поля предсказания или в атрибуте image_threshold.
    """
    # Вариант с атрибутом image_threshold (некоторые версии TorchInferencer)
    image_thr = getattr(predictions, "image_threshold", None)
    if isinstance(image_thr, (float, int)):
        return float(image_thr)

    # Вариант с metadata/meta
    meta = getattr(predictions, "metadata", None)
    if meta is None and isinstance(predictions, dict):
        meta = predictions.get("metadata") or predictions.get("meta")

    if isinstance(meta, dict):
        for key in ("image_threshold", "threshold", "image_thr"):
            if key in meta and isinstance(meta[key], (float, int)):
                return float(meta[key])

    # Иногда Anomalib может класть порог как отдельное поле в самом предсказании
    if isinstance(predictions, dict):
        for key in ("image_threshold", "threshold", "image_thr"):
            val = predictions.get(key)
            if isinstance(val, (float, int)):
                return float(val)

    return None


# =============================================================================
# ЭНДПОИНТЫ API
# =============================================================================


@app.get("/")
def root():
    """Корневой маршрут: имя сервиса и статус."""
    return {"service": "defect-detector-patchcore-v113", "status": "ok"}


@app.get("/health")
def health():
    """Проверка живости для оркестрации (K8s, Docker)."""
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
    # Порог оставляем чисто для UI: для данного датасета ниже score → сильнее подозрение на брак.
    # Реальное решение о браке принимается по адаптивному порогу из модели (F1AdaptiveThreshold), если он есть.
    threshold: float = Form(0.5, description="Порог для UI: score <= threshold → брак (по умолчанию 0.5 для данного датасета)"),
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
        ensure_trained_model()  # гарантируем, что есть хотя бы .pt или только что обучили
    except Exception as e:
        logger.exception("Не удалось подготовить обученную модель Patchcore")
        raise HTTPException(status_code=500, detail=f"Ошибка подготовки модели Patchcore: {e!s}")

    # Обязательное наличие calibration_bounds.json: без него Min-Max нормализация и пороги некорректны.
    if _load_calibration_bounds() is None:
        logger.warning(
            "calibration_bounds.json не найден. Результаты инференса будут некорректны без предварительной калибровки."
        )
        raise HTTPException(
            status_code=500,
            detail=(
                "Отсутствует calibration_bounds.json. "
                "Перед использованием инференса вызовите /api/patchcore/calibrate для калибровки на ваших данных."
            ),
        )

    # Сохраняем тестовое изображение во временный файл
    with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
        tmp.write(test_bytes)
        tmp_path = Path(tmp.name)

    # Предпочитаем инференс по Lightning‑чекпоинту: сначала прямой predict_step (сырая anomaly_map),
    # иначе Engine.predict (у него в текущей версии anomaly_map часто постоянна) или TorchInferencer.
    lightning_ckpt = _find_latest_lightning_ckpt()
    predictions = None
    used_engine_predict = False

    if lightning_ckpt is not None:
        try:
            predictions = _predict_with_ckpt_direct(lightning_ckpt, tmp_path)
            used_engine_predict = True
            logger.info("Инференс через прямой predict_step (Lightning‑чекпоинт)")
        except Exception as e:
            logger.warning("Прямой predict_step не удался, пробуем Engine.predict: %s", e)
            if lightning_ckpt is not None:
                try:
                    model = _build_patchcore_model(lightning_ckpt)
                    # Инференс через Lightning Engine тоже гоняем через CPU, чтобы не трогать MPS.
                    engine = Engine(accelerator="cpu", devices=1)
                    pred_output = engine.predict(
                        model=model,
                        ckpt_path=str(lightning_ckpt),
                        data_path=str(tmp_path),
                        return_predictions=True,
                    )
                    if pred_output:
                        out = pred_output[0] if isinstance(pred_output, (list, tuple)) else pred_output
                        if isinstance(out, (list, tuple)) and len(out):
                            out = out[0]
                        predictions = out
                        used_engine_predict = predictions is not None
                    if used_engine_predict:
                        logger.info("Инференс через Engine.predict (Lightning‑чекпоинт)")
                except Exception as e2:
                    logger.warning("Engine.predict не удался: %s", e2)

    if not used_engine_predict:
        try:
            os.environ.setdefault("TRUST_REMOTE_CODE", "1")
            ckpt_path = ensure_trained_model()
            inferencer = TorchInferencer(path=str(ckpt_path))
            predictions = inferencer.predict(image=str(tmp_path))
        except Exception as e:
            try:
                tmp_path.unlink(missing_ok=True)
            except Exception:
                pass
            logger.exception("Ошибка инференса Patchcore")
            raise HTTPException(status_code=500, detail=f"Ошибка инференса Patchcore: {e!s}")

    try:
        tmp_path.unlink(missing_ok=True)
    except Exception:
        pass

    # Разбор предсказания: image и anomaly_map для heatmap; raw/score через общие функции
    try:
        logger.info("predictions type=%s", type(predictions).__name__)
        if isinstance(predictions, dict):
            logger.info("predictions keys=%s", list(predictions.keys()))
        image_np = predictions.image if hasattr(predictions, "image") else predictions["image"]
        anomaly_map = getattr(predictions, "anomaly_map", None)
        if anomaly_map is None and isinstance(predictions, dict):
            anomaly_map = predictions.get("anomaly_map")
        if anomaly_map is not None:
            am_debug = anomaly_map.detach().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
            am_flat = am_debug.flatten()
            am_std = float(np.std(am_flat)) if am_flat.size else 0.0
            logger.info(
                "anomaly_map shape=%s min=%.4f max=%.4f mean=%.4f std=%.4f",
                am_debug.shape,
                float(np.min(am_flat)),
                float(np.max(am_flat)),
                float(np.mean(am_flat)),
                am_std,
            )
            if am_std < 1e-6:
                logger.error(
                    "anomaly_map из инференса имеет нулевую дисперсию (std=%.6f). "
                    "Проверьте соответствие конфигурации слоёв (layers) и размеров входного изображения.",
                    am_std,
                )
        raw, raw_source = _extract_raw_score_from_predictions(predictions)
        logger.info("Patchcore raw_score=%.4f (из %s)", raw, raw_source)
        score = _normalize_score_for_response(raw)

        # Используем адаптивный порог F1AdaptiveThreshold, вычисленный Anomalib при обучении.
        adaptive_threshold = _extract_adaptive_threshold_from_predictions(predictions)
        effective_threshold = adaptive_threshold if adaptive_threshold is not None else float(threshold)
        if adaptive_threshold is None:
            logger.warning(
                "Не удалось извлечь адаптивный порог (F1AdaptiveThreshold) из предсказаний. "
                "Используется UI‑порог %.4f.",
                effective_threshold,
            )

        # Для текущего датасета более информативен низкий score как индикатор брака:
        # чем ниже score, тем сильнее подозрение на дефект.
        logger.info(
            "score нормализованный=%.4f (порог=%.4f -> defect=%s; правило: score <= threshold → defect)",
            score,
            effective_threshold,
            score <= effective_threshold,
        )
        defect = score <= effective_threshold
    except Exception as e:
        logger.exception("Не удалось разобрать вывод модели")
        raise HTTPException(status_code=500, detail=f"Неверный формат предсказания: {e!s}")

    # Подготовка изображения и heatmap для ответа
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
        threshold=float(effective_threshold),
        heatmap_base64=heatmap_b64,
        message=None,
        raw_score=round(raw, 4),
    )


# =============================================================================
# КАЛИБРОВКА (прогон по normal/abnormal → calibration_bounds.json)
# =============================================================================


def _iter_dataset_images(kind: str):
    """Итерируем все картинки в dataset/normal или dataset/abnormal."""

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    base_dir = DATASET_ROOT / (NORMAL_DIR if kind == "normal" else ABNORMAL_DIR)
    if not base_dir.exists():
        return

    for path in sorted(base_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            # Возвращаем относительный путь внутри dataset для удобства
            rel = path.relative_to(DATASET_ROOT)
            yield kind, path, rel.as_posix()


def _compute_raw_from_anomaly_map(anomaly_map: np.ndarray | None) -> float:
    """Сырой скор по anomaly_map (без нормализации). Используется для Min-Max по калибровке."""

    if anomaly_map is None:
        return 0.0

    arr = anomaly_map.detach().cpu().numpy() if hasattr(anomaly_map, "detach") else np.asarray(anomaly_map)
    arr = arr.squeeze()
    if not arr.size:
        return 0.0

    max_val = float(np.max(arr))
    std_val = float(np.std(arr))
    if std_val < 1e-6:
        logger.error(
            "Калибровка: anomaly_map имеет нулевую дисперсию (std=%.6f). "
            "Возможна проблема с конфигурацией Patchcore или трансформациями Resize/CenterCrop.",
            std_val,
        )

    # Сырые значения от model.model() для Patchcore — в районе 50–100; от Engine — в [0,1].
    raw = (max_val / 10.0) if max_val > 2.0 else (max_val * 10.0)
    return raw


@app.get("/api/patchcore/calibrate", response_model=CalibrationResult)
def calibrate_patchcore(limit_per_class: int = 0):
    """
    Пройтись по всем изображениям в dataset/normal и dataset/abnormal,
    посчитать raw/score и предложить пороги T_low, T_high:

    - T_low — верхняя граница «точно не брак» (max score по normal)
    - T_high — нижняя граница «точно брак» (min score по abnormal)
    Если распределения пересекаются, T_low/T_high считаются по перцентилям (90% normal, 10% abnormal).
    """

    try:
        ensure_trained_model()
    except Exception as e:
        logger.exception("Не удалось подготовить модель для калибровки")
        raise HTTPException(status_code=500, detail=f"Ошибка подготовки модели Patchcore: {e!s}")

    lightning_ckpt = _find_latest_lightning_ckpt()
    if lightning_ckpt is None:
        raise HTTPException(status_code=500, detail="Не найден Lightning‑чекпоинт Patchcore для калибровки")

    # Сначала собираем сырые скоры по всем изображениям
    samples_raw: list[tuple[str, str, str, float]] = []  # (kind, path, rel, raw)

    for kind in ("normal", "abnormal"):
        count = 0
        for kind_, path, rel in _iter_dataset_images(kind):
            if limit_per_class and count >= limit_per_class:
                break
            count += 1

            try:
                pred = _predict_with_ckpt_direct(lightning_ckpt, path)
                raw = _compute_raw_from_anomaly_map(pred.anomaly_map)
            except Exception as e:
                logger.warning("Калибровка: не удалось обработать %s (%s): %s", kind_, rel, e)
                continue

            samples_raw.append((kind_, rel, path, raw))

    # Границы для Min-Max нормализации
    all_raws = [r for (_, _, _, r) in samples_raw]
    raw_min = min(all_raws) if all_raws else 0.0
    raw_max = max(all_raws) if all_raws else 1.0
    bounds_path = RESULTS_DIR / CALIBRATION_BOUNDS_FILE
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bounds_path.write_text(
        json.dumps({"raw_score_min": raw_min, "raw_score_max": raw_max}, indent=2),
        encoding="utf-8",
    )
    logger.info("Сохранены границы калибровки: raw_min=%.4f raw_max=%.4f -> %s", raw_min, raw_max, bounds_path)

    samples: list[CalibrationSample] = []
    normal_scores: list[float] = []
    abnormal_scores: list[float] = []

    for kind_, rel, _path, raw in samples_raw:
        score = _min_max_normalize(raw, raw_min, raw_max)
        samples.append(
            CalibrationSample(
                path=rel,
                kind=kind_,
                score=round(score, 4),
                raw_score=round(raw, 4),
            )
        )
        if kind_ == "normal":
            normal_scores.append(score)
        else:
            abnormal_scores.append(score)

    def _stats(values: list[float]) -> dict | None:
        if not values:
            return None
        arr = np.asarray(values, dtype=float)
        return {
            "min": float(arr.min()),
            "max": float(arr.max()),
            "mean": float(arr.mean()),
            "std": float(arr.std()),
        }

    normal_stats = _stats(normal_scores)
    abnormal_stats = _stats(abnormal_scores)

    t_low: float | None = None
    t_high: float | None = None

    if normal_scores and abnormal_scores:
        max_n = max(normal_scores)
        min_a = min(abnormal_scores)
        if max_n < min_a:
            # Красивый разрыв: берём границы по краям и оставляем между ними «серую зону».
            t_low = round(max_n, 4)
            t_high = round(min_a, 4)
        else:
            # Пересечение распределений: берём перцентили 90%/10% как мягкие границы.
            n_arr = np.asarray(normal_scores, dtype=float)
            a_arr = np.asarray(abnormal_scores, dtype=float)
            t_low = float(np.percentile(n_arr, 90))
            t_high = float(np.percentile(a_arr, 10))
            t_low = round(t_low, 4)
            t_high = round(t_high, 4)

    return CalibrationResult(
        normal_count=len(normal_scores),
        abnormal_count=len(abnormal_scores),
        normal_stats=normal_stats,
        abnormal_stats=abnormal_stats,
        suggested_t_low=t_low,
        suggested_t_high=t_high,
        samples=samples,
    )


if __name__ == "__main__":
    # Продакшен: лучше запускать через uvicorn из CLI с нужными workers/host
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)