"""
Görev 1 değerlendirme — mAP@0.5 (IoU ≥ 0.5).

Kullanım:
    python eval_map.py <tahmin_json> <ground_truth_json>

JSON formatı (liste):
[
  {
    "frame_id": "frame_0001",
    "objects": [
      {"cls": "0", "top_left_x": 100, "top_left_y": 200,
       "bottom_right_x": 300, "bottom_right_y": 400,
       "score": 0.9, "motion_status": "1", "landing_status": "-1"}
    ]
  }, ...
]
"""

import sys
import json
from collections import defaultdict


def iou(b1: dict, b2: dict) -> float:
    x1 = max(b1["top_left_x"], b2["top_left_x"])
    y1 = max(b1["top_left_y"], b2["top_left_y"])
    x2 = min(b1["bottom_right_x"], b2["bottom_right_x"])
    y2 = min(b1["bottom_right_y"], b2["bottom_right_y"])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1["bottom_right_x"] - b1["top_left_x"]) * (b1["bottom_right_y"] - b1["top_left_y"])
    a2 = (b2["bottom_right_x"] - b2["top_left_x"]) * (b2["bottom_right_y"] - b2["top_left_y"])
    union = a1 + a2 - inter
    return inter / union if union > 0 else 0.0


def compute_ap(precisions: list[float], recalls: list[float]) -> float:
    """11-nokta interpolasyon ile AP hesaplar."""
    ap = 0.0
    for thr in [i / 10 for i in range(11)]:
        p_max = max((p for p, r in zip(precisions, recalls) if r >= thr), default=0.0)
        ap += p_max / 11
    return ap


def evaluate(pred_path: str, gt_path: str, iou_thr: float = 0.5) -> float:
    with open(pred_path) as f:
        preds_list = json.load(f)
    with open(gt_path) as f:
        gts_list = json.load(f)

    preds_by_frame = {r["frame_id"]: r["objects"] for r in preds_list}
    gts_by_frame   = {r["frame_id"]: r["objects"] for r in gts_list}

    # sınıf bazlı TP/FP/FN
    class_tp  = defaultdict(list)
    class_fp  = defaultdict(list)
    class_gt  = defaultdict(int)
    class_sc  = defaultdict(list)

    for frame_id, gt_objs in gts_by_frame.items():
        pred_objs = preds_by_frame.get(frame_id, [])
        matched = set()

        # sınıf sayısı
        for g in gt_objs:
            class_gt[g["cls"]] += 1

        # score'a göre sırala
        pred_sorted = sorted(pred_objs, key=lambda x: x.get("score", 1.0), reverse=True)

        for p in pred_sorted:
            cls = p["cls"]
            best_iou, best_idx = 0.0, -1
            for i, g in enumerate(gt_objs):
                if g["cls"] != cls or i in matched:
                    continue
                v = iou(p, g)
                if v > best_iou:
                    best_iou, best_idx = v, i

            class_sc[cls].append(p.get("score", 1.0))
            if best_iou >= iou_thr:
                class_tp[cls].append(1)
                class_fp[cls].append(0)
                matched.add(best_idx)
            else:
                class_tp[cls].append(0)
                class_fp[cls].append(1)

    all_classes = set(class_gt) | set(class_tp)
    aps = {}
    for cls in sorted(all_classes):
        tp_arr = class_tp[cls]
        fp_arr = class_fp[cls]
        n_gt   = class_gt[cls]
        if not tp_arr:
            aps[cls] = 0.0
            continue

        # kümülatif
        cum_tp = []
        cum_fp = []
        s = 0
        for v in tp_arr:
            s += v
            cum_tp.append(s)
        s = 0
        for v in fp_arr:
            s += v
            cum_fp.append(s)

        precisions = [tp / (tp + fp) if (tp + fp) > 0 else 0 for tp, fp in zip(cum_tp, cum_fp)]
        recalls    = [tp / n_gt if n_gt > 0 else 0 for tp in cum_tp]
        aps[cls]   = compute_ap(precisions, recalls)
        print(f"  Sınıf {cls}: AP={aps[cls]:.4f}  GT={n_gt}")

    mAP = sum(aps.values()) / len(aps) if aps else 0.0
    print(f"mAP@{iou_thr}: {mAP:.4f}")
    return mAP


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Kullanım: python eval_map.py <tahmin.json> <ground_truth.json>")
        sys.exit(1)
    evaluate(sys.argv[1], sys.argv[2])
