"""
Video / kare listesi yükleyici.
- Gerçek modda sunucudan FrameInfo listesi alır.
- Mock modda yerel video dosyasını 7.5 fps mantığıyla iter.
- İlk 450 kare GPS sağlıklı (health=1), sonrası değişken.
"""

import cv2
import numpy as np
from dataclasses import dataclass
from typing import Generator
from pathlib import Path


HEALTHY_FRAME_COUNT = 450   # ilk 1 dakika = 450 kare kesinlikle sağlıklı


@dataclass
class Frame:
    index: int
    image: np.ndarray
    frame_id: str
    translation_x: float
    translation_y: float
    translation_z: float
    gps_health_status: int   # 1=sağlıklı, 0=sağlıksız


def iter_video(video_path: str, start_health_drop: int | None = None) -> Generator[Frame, None, None]:
    """
    Yerel video dosyasını Frame nesneleri olarak iter.
    start_health_drop: bu kareden itibaren GPS sağlıksız (None → hep sağlıklı).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        health = 1
        if idx >= HEALTHY_FRAME_COUNT:
            if start_health_drop is not None and idx >= start_health_drop:
                health = 0

        yield Frame(
            index=idx,
            image=frame,
            frame_id=f"frame_{idx:04d}",
            translation_x=0.0,
            translation_y=0.0,
            translation_z=50.0,
            gps_health_status=health,
        )
        idx += 1

    cap.release()


def iter_server_frames(frame_list: list) -> Generator[Frame, None, None]:
    """
    Sunucudan alınan FrameInfo listesini Frame nesnelerine çevirir.
    Görüntüyü image_url'den indirir.
    """
    import requests
    import numpy as np

    for idx, info in enumerate(frame_list):
        resp = requests.get(info.image_url, timeout=10)
        arr = np.frombuffer(resp.content, np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        yield Frame(
            index=idx,
            image=image,
            frame_id=info.url,
            translation_x=info.translation_x,
            translation_y=info.translation_y,
            translation_z=info.translation_z,
            gps_health_status=info.gps_health_status,
        )


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "data/THYZ_2026_Ornek_Veri_1.MP4"
    for f in iter_video(path):
        h, w = f.image.shape[:2]
        print(f"[{f.index:04d}] {w}x{h}  GPS={f.gps_health_status}  "
              f"X={f.translation_x:.2f} Y={f.translation_y:.2f} Z={f.translation_z:.2f}")
