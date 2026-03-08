"""
Сервис анализа изображений через Anomalib (Intel).
Принимает эталонное и тестовое изображения, обучает Padim на эталоне и предсказывает аномалию на тесте.
"""

from __future__ import annotations

import base64
import io
import logging
import tempfile
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Defect Detector Anomalib API",
    description="Анализ дефектов этикетки с помощью Anomalib (Padim)",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class AnomalibResult(BaseModel):
    """Результат анализа Anomalib."""
    defect: bool
    score: float
    threshold: float
    heatmap_base64: str | None = None  # PNG heatmap для визуализации
    message: str | None = None


def run_anomalib_analysis(reference_bytes: bytes, test_bytes: bytes, score_threshold: float = 0.5) -> AnomalibResult:
    """
    Обучает Padim на эталонном изображении и предсказывает аномалию на тестовом.
    """
    try:
        from anomalib.data import Folder
        from anomalib.engine import Engine
        from anomalib.models import Padim
        from anomalib.deploy import TorchInferencer
        from PIL import Image
    except ImportError as e:
        raise HTTPException(status_code=500, detail=f"Anomalib не установлен: {e}") from e

    with tempfile.TemporaryDirectory(prefix="anomalib_defect_") as tmpdir:
        root = Path(tmpdir)
        normal_dir = root / "normal"
        normal_dir.mkdir(parents=True, exist_ok=True)

        try:
            # Сохраняем эталон в normal/
            ref_path = normal_dir / "reference.png"
            _save_image(reference_bytes, ref_path)

            # Сохраняем тестовое изображение в корень
            test_path = root / "test.png"
            _save_image(test_bytes, test_path)

            # Один эталон — мало для стабильного split; делаем копии (нужно несколько для train/val)
            for i in range(2, 5):
                copy_path = normal_dir / f"reference_{i}.png"
                if ref_path.exists():
                    Image.open(ref_path).save(copy_path)
        except Exception as e:
            logger.exception("Ошибка сохранения изображений")
            return AnomalibResult(
                defect=True,
                score=1.0,
                threshold=score_threshold,
                heatmap_base64=None,
                message=f"Ошибка загрузки изображений: {e!s}",
            )

        try:
            datamodule = Folder(
                name="defect_session",
                root=str(root),
                normal_dir="normal",
                abnormal_dir=None,
                normal_split_ratio=0.2,
                train_batch_size=1,
                eval_batch_size=1,
                num_workers=0,
            )
            datamodule.setup()

            model = Padim()
            engine = Engine(
                default_root_dir=str(root / "results"),
                accelerator="auto",
                devices=1,
                limit_val_batches=0,  # без валидации при одном классе (норма)
            )

            # Обучение на эталоне (Padim — one-class, только норма)
            engine.fit(model, datamodule=datamodule)

            # Сохраняем чекпоинт и предсказываем через Inferencer (избегаем engine.predict — там "expected 2, got 0")
            ckpt_path = root / "model.ckpt"
            engine.trainer.save_checkpoint(str(ckpt_path))

            inferencer = TorchInferencer(path=str(ckpt_path), device="auto")
            batch_result = inferencer.predict(image=str(test_path))

            # ImageBatch для одного изображения: pred_score и anomaly_map — тензоры с batch_dim=1
            if hasattr(batch_result, "pred_score"):
                ps = batch_result.pred_score
                score = float(ps.item() if ps.numel() == 1 else ps[0].item())
            else:
                score = 0.0
            anomaly_map = getattr(batch_result, "anomaly_map", None)

            # Упаковываем в список из одного элемента для единообразной обработки ниже
            predictions = [type("Pred", (), {"pred_score": score, "anomaly_map": anomaly_map})()]
        except Exception as e:
            return AnomalibResult(
                defect=True,
                score=1.0,
                threshold=score_threshold,
                heatmap_base64=None,
                message=f"Ошибка Anomalib: {e!s}",
            )

        if not predictions or len(predictions) == 0:
            return AnomalibResult(
                defect=False,
                score=0.0,
                threshold=score_threshold,
                heatmap_base64=None,
                message="Нет предсказаний",
            )

        pred = predictions[0]
        if hasattr(pred, "pred_score"):
            score = float(pred.pred_score)
            anomaly_map = getattr(pred, "anomaly_map", None)
        elif isinstance(pred, dict):
            score = float(pred.get("pred_score", 0.0))
            anomaly_map = pred.get("anomaly_map")
        else:
            score = 0.0
            anomaly_map = None

        heatmap_b64 = None
        if anomaly_map is not None:
            try:
                import numpy as np
                from PIL import Image as PILImage
                if hasattr(anomaly_map, "cpu"):
                    arr = anomaly_map.cpu().numpy()
                else:
                    arr = np.asarray(anomaly_map)
                if arr.ndim == 3:
                    arr = arr.squeeze()
                # Масштаб 0–255 для визуализации
                if arr.max() > arr.min():
                    arr = (arr - arr.min()) / (arr.max() - arr.min()) * 255
                else:
                    arr = np.zeros_like(arr)
                img = PILImage.fromarray(arr.astype("uint8")).resize((256, 256))
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                heatmap_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
            except Exception:
                pass

        return AnomalibResult(
            defect=score >= score_threshold,
            score=score,
            threshold=score_threshold,
            heatmap_base64=heatmap_b64,
        )


def _save_image(data: bytes, path: Path) -> None:
    """Сохраняет байты как изображение, конвертируя в RGB при необходимости."""
    from PIL import Image
    img = Image.open(io.BytesIO(data))
    if img.mode not in ("RGB", "L"):
        img = img.convert("RGB")
    img.save(path)


@app.get("/")
def root():
    return {"service": "defect-detector-anomalib", "status": "ok"}


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/api/anomalib/analyze", response_model=AnomalibResult)
async def analyze(
    reference: UploadFile = File(..., description="Эталонное изображение (без дефекта)"),
    test: UploadFile = File(..., description="Тестовое изображение для проверки"),
    threshold: float = Form(0.5, description="Порог по score: выше = брак"),
):
    """Анализ: обучение на эталоне (Padim) и предсказание аномалии на тесте."""
    if reference.content_type and not reference.content_type.startswith("image/"):
        raise HTTPException(400, "reference: ожидается изображение")
    if test.content_type and not test.content_type.startswith("image/"):
        raise HTTPException(400, "test: ожидается изображение")

    ref_bytes = await reference.read()
    test_bytes = await test.read()
    if not ref_bytes or not test_bytes:
        raise HTTPException(400, "Пустой файл изображения")

    try:
        return run_anomalib_analysis(ref_bytes, test_bytes, score_threshold=threshold)
    except HTTPException:
        raise
    except Exception as e:
        logger.exception("Ошибка при анализе Anomalib")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
