"""
Mock server I/O — şartname Bölüm 8 ve Tablo 6'ya göre.
Gerçek yarışmada SERVER_URL yarışma günü paylaşılır (ör: http://127.0.0.25:5000).
"""

import requests
import json
from dataclasses import dataclass, field
from typing import Optional

SERVER_URL = "http://127.0.0.1:5000"


@dataclass
class FrameInfo:
    url: str
    image_url: str
    video_name: str
    session: str
    translation_x: float
    translation_y: float
    translation_z: float
    gps_health_status: int  # 1 = sağlıklı, 0 = sağlıksız


@dataclass
class DetectedObject:
    cls: str               # "0"=taşıt "1"=insan "2"=UAP "3"=UAİ
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int
    landing_status: str = "-1"   # "-1","0","1"  (sadece UAP/UAİ)
    motion_status: str = "-1"    # "-1","0","1"  (sadece taşıt)


@dataclass
class DetectedTranslation:
    translation_x: float
    translation_y: float
    translation_z: float


@dataclass
class DetectedUndefinedObject:
    object_id: str
    top_left_x: int
    top_left_y: int
    bottom_right_x: int
    bottom_right_y: int


def get_frame_list(session_url: str) -> list[FrameInfo]:
    """Sunucudan oturumdaki tüm kare listesini alır."""
    try:
        resp = requests.get(session_url, timeout=10)
        resp.raise_for_status()
        return [FrameInfo(**item) for item in resp.json()]
    except Exception as e:
        print(f"[MOCK] get_frame_list hatası: {e}")
        return _mock_frame_list()


def get_frame_image(image_url: str) -> bytes:
    """Bir karenin görüntü verisini indirir."""
    try:
        resp = requests.get(image_url, timeout=10)
        resp.raise_for_status()
        return resp.content
    except Exception as e:
        print(f"[MOCK] get_frame_image hatası: {e}")
        return b""


def send_result(
    prediction_id: str,
    user_url: str,
    frame_url: str,
    detected_objects: list[DetectedObject],
    detected_translations: list[DetectedTranslation],
    detected_undefined_objects: list[DetectedUndefinedObject],
) -> bool:
    """Tahmin sonuçlarını sunucuya gönderir (Şekil 17 JSON formatı)."""
    payload = {
        "id": prediction_id,
        "user": user_url,
        "frame": frame_url,
        "detected_objects": [vars(o) for o in detected_objects],
        "detected_translations": [vars(t) for t in detected_translations],
        "detected_undefined_objects": [vars(u) for u in detected_undefined_objects],
    }
    try:
        resp = requests.post(
            f"{SERVER_URL}/api/submit/",
            json=payload,
            timeout=10,
        )
        resp.raise_for_status()
        return True
    except Exception as e:
        print(f"[MOCK] send_result hatası: {e}")
        print("[MOCK] Gönderilecek payload:", json.dumps(payload, indent=2, ensure_ascii=False))
        return False


# ---------------------------------------------------------------------------
# Mock yardımcılar — sunucu yokken lokal test için
# ---------------------------------------------------------------------------

def _mock_frame_list(n: int = 5) -> list[FrameInfo]:
    return [
        FrameInfo(
            url=f"http://mock/frames/{i}/",
            image_url=f"http://mock/images/frame_{i:04d}.jpg",
            video_name="THYZ_2026_Ornek_Veri_1",
            session="http://mock/sessions/1/",
            translation_x=float(i) * 0.5,
            translation_y=float(i) * 0.3,
            translation_z=50.0,
            gps_health_status=1,
        )
        for i in range(n)
    ]


if __name__ == "__main__":
    frames = _mock_frame_list(3)
    for f in frames:
        print(f"Kare: {f.url}  GPS sağlık: {f.gps_health_status}")

    ok = send_result(
        prediction_id="test-001",
        user_url="http://mock/users/1/",
        frame_url=frames[0].url,
        detected_objects=[
            DetectedObject(cls="0", top_left_x=100, top_left_y=200,
                           bottom_right_x=250, bottom_right_y=350,
                           motion_status="1")
        ],
        detected_translations=[
            DetectedTranslation(translation_x=0.5, translation_y=0.3, translation_z=50.0)
        ],
        detected_undefined_objects=[],
    )
    print("Gönderim:", "OK" if ok else "MOCK (sunucu yok)")
