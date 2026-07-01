"""
Görev 2 değerlendirme — 3D Öklid ortalama hata (Denklem 2).

Kullanım:
    python eval_gorev2.py <tahmin_csv> <ground_truth_csv>

CSV formatı: frame_id, translation_x, translation_y, translation_z
"""

import sys
import csv
import math
from pathlib import Path


def load_csv(path: str) -> dict[str, tuple[float, float, float]]:
    data = {}
    with open(path) as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = row["frame_id"]
            data[fid] = (
                float(row["translation_x"]),
                float(row["translation_y"]),
                float(row["translation_z"]),
            )
    return data


def euclidean_3d(p1: tuple, p2: tuple) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(p1, p2)))


def evaluate(pred_path: str, gt_path: str) -> float:
    """Ortalama 3D Öklid hatasını (metre) döner."""
    preds = load_csv(pred_path)
    gts = load_csv(gt_path)

    common = set(preds) & set(gts)
    if not common:
        print("Eşleşen kare yok!")
        return float("inf")

    errors = [euclidean_3d(preds[fid], gts[fid]) for fid in common]
    mean_err = sum(errors) / len(errors)
    print(f"Kare sayısı : {len(common)}")
    print(f"Ortalama hata: {mean_err:.4f} m")
    print(f"Maks hata    : {max(errors):.4f} m")
    print(f"Min hata     : {min(errors):.4f} m")
    return mean_err


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Kullanım: python eval_gorev2.py <tahmin.csv> <ground_truth.csv>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
