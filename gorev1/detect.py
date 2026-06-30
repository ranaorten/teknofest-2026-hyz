"""
Görev 1: Nesne Tespiti
- Sınıflar: 0=taşıt, 1=insan, 2=UAP, 3=UAİ
- Taşıt için motion_status, UAP/UAİ için landing_status gönderilir.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
from server_io import DetectedObject


CLASS_NAMES = {0: "tasit", 1: "insan", 2: "UAP", 3: "UAI"}


def detect(frame: np.ndarray, model=None) -> list[DetectedObject]:
    """
    Bir kare üzerinde nesne tespiti yapar.
    model=None iken boş liste döner (iskelet).
    """
    if model is None:
        return []

    results = model(frame)
    detections = []
    for box in results[0].boxes:
        cls = int(box.cls[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        obj = DetectedObject(
            cls=str(cls),
            top_left_x=x1,
            top_left_y=y1,
            bottom_right_x=x2,
            bottom_right_y=y2,
            motion_status="1" if cls == 0 else "-1",
            landing_status="1" if cls in (2, 3) else "-1",
        )
        detections.append(obj)
    return detections


def load_model(weights_path: str = "weights/best.pt"):
    from ultralytics import YOLO
    return YOLO(weights_path)


if __name__ == "__main__":
    img_path = sys.argv[1] if len(sys.argv) > 1 else None
    if img_path:
        frame = cv2.imread(img_path)
        results = detect(frame)
        print(f"{len(results)} nesne tespit edildi.")
    else:
        print("Kullanım: python detect.py <görüntü_yolu>")
