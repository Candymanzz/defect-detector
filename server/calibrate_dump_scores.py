import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from main import (
    DATASET_ROOT,
    NORMAL_DIR,
    ABNORMAL_DIR,
    RESULTS_DIR,
    CALIBRATION_BOUNDS_FILE,
    _build_patchcore_model,
    _build_transforms,
    _find_latest_lightning_ckpt,
    ensure_trained_model,
    _compute_raw_from_anomaly_map,
    _min_max_normalize,
)


def iter_images(kind: str):
    """Итерируем все изображения в dataset/normal или dataset/abnormal."""

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
    base_dir = DATASET_ROOT / (NORMAL_DIR if kind == "normal" else ABNORMAL_DIR)
    if not base_dir.exists():
        return

    for path in sorted(base_dir.rglob("*")):
        if path.is_file() and path.suffix.lower() in exts:
            rel = path.relative_to(DATASET_ROOT)
            yield kind, path, rel.as_posix()


def main():
    """Последовательно прогоняет все normal/abnormal и пишет result JSON на диск."""

    ensure_trained_model()
    ckpt = _find_latest_lightning_ckpt()
    if ckpt is None:
        raise RuntimeError("Не найден Lightning‑чекпоинт Patchcore для калибровки.")

    # Считаем, сколько файлов обработаем
    file_list = []
    for k in ("normal", "abnormal"):
        file_list.extend(list(iter_images(k)))
    total = len(file_list)
    print(f"Найдено {total} изображений. Загрузка модели...", flush=True)

    # Готовим модель с теми же слоями, что и чекпоинт (layer2+layer3 или layer1+layer2).
    model = _build_patchcore_model(ckpt)
    state = torch.load(str(ckpt), map_location="cpu", weights_only=False)
    model.load_state_dict(state["state_dict"], strict=False)
    model.eval()

    _, eval_transform = _build_transforms()
    print("Модель загружена. Обработка изображений...", flush=True)

    results = []
    done = 0

    for kind in ("normal", "abnormal"):
        for kind_, path, rel in iter_images(kind):
            done += 1
            try:
                img = Image.open(path)
                if img.mode != "RGB":
                    img = img.convert("RGB")
                tensor = eval_transform(img)  # (C,H,W)
                if tensor.dim() == 3:
                    tensor = tensor.unsqueeze(0)

                with torch.no_grad():
                    out = model.model(tensor)

                anomaly_map = getattr(out, "anomaly_map", None)
                if anomaly_map is None and isinstance(out, dict):
                    anomaly_map = out.get("anomaly_map")

                raw = _compute_raw_from_anomaly_map(anomaly_map)
                print(f"  [{done}/{total}] {rel} raw_score={raw:.4f}", flush=True)
            except Exception as e:  # noqa: BLE001
                print(f"  [{done}/{total}] {rel} ОШИБКА: {e}", flush=True)
                results.append(
                    {
                        "path": rel,
                        "kind": kind_,
                        "error": str(e),
                    }
                )
                continue

            results.append(
                {
                    "path": rel,
                    "kind": kind_,
                    "raw_score": raw,
                }
            )

    # Min-Max границы и нормализованный score
    raw_scores = [r["raw_score"] for r in results if "raw_score" in r]
    raw_min = min(raw_scores) if raw_scores else 0.0
    raw_max = max(raw_scores) if raw_scores else 1.0
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    bounds_path = RESULTS_DIR / CALIBRATION_BOUNDS_FILE
    with bounds_path.open("w", encoding="utf-8") as f:
        json.dump({"raw_score_min": raw_min, "raw_score_max": raw_max}, f, indent=2)
    print(f"Границы калибровки: raw_min={raw_min:.4f} raw_max={raw_max:.4f} -> {bounds_path}")

    for r in results:
        if "raw_score" in r:
            r["score"] = _min_max_normalize(r["raw_score"], raw_min, raw_max)

    out_path = RESULTS_DIR / "calibration_scores.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Сохранено {len(results)} записей в {out_path}")


if __name__ == "__main__":
    main()

