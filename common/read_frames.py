"""
Video dosyasını kare kare okur ve mock sunucu I/O ile eşleştirir.
Kullanım: python common/read_frames.py data/THYZ_2026_Ornek_Veri_1.MP4
"""

import sys
import cv2
from pathlib import Path
from server_io import _mock_frame_list, send_result, DetectedObject, DetectedTranslation


def read_video_frames(video_path: str):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Video açılamadı: {video_path}")
        return

    frame_idx = 0
    mock_frames = _mock_frame_list(n=9999)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        info = mock_frames[frame_idx] if frame_idx < len(mock_frames) else None
        h, w = frame.shape[:2]
        print(f"Kare {frame_idx:04d} | {w}x{h} | GPS: {info.gps_health_status if info else '?'} | "
              f"X={info.translation_x:.2f} Y={info.translation_y:.2f} Z={info.translation_z:.2f}" if info
              else f"Kare {frame_idx:04d} | {w}x{h}")

        frame_idx += 1

    cap.release()
    print(f"Toplam {frame_idx} kare okundu.")


if __name__ == "__main__":
    path = sys.argv[1] if len(sys.argv) > 1 else "data/THYZ_2026_Ornek_Veri_1.MP4"
    read_video_frames(path)
