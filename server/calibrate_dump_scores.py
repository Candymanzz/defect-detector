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
    _build_patchcore_model,
    _build_transforms,
    _find_latest_lightning_ckpt,
    ensure_trained_model,
    _compute_raw_and_score_from_anomaly_map,
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

    # Готовим одну модель и один набор трансформов для всех изображений.
    model = _build_patchcore_model()
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

                raw, score = _compute_raw_and_score_from_anomaly_map(anomaly_map)
                print(f"  [{done}/{total}] {rel} score={score:.4f}", flush=True)
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
                    "score": score,
                }
            )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULTS_DIR / "calibration_scores.json"
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    print(f"Сохранено {len(results)} записей в {out_path}")


if __name__ == "__main__":
    main()

