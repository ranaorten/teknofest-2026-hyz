"""
Görev 2: Pozisyon Kestirimi
GPS sağlık=1 → sunucudan gelen değeri kullan veya kendi algoritmanı gönder.
GPS sağlık=0 → visual odometry ile kestir (visual_odometry.py).
"""

import sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
from server_io import FrameInfo, DetectedTranslation


def estimate_position(
    frame,
    prev_frame,
    frame_info: FrameInfo,
    vo_state: dict,
    camera_matrix: np.ndarray,
) -> DetectedTranslation:
    """
    GPS sağlıklıysa sunucu değerini döner.
    Sağlıksızsa visual odometry ile kestiri yapar.
    """
    if frame_info.gps_health_status == 1:
        return DetectedTranslation(
            translation_x=frame_info.translation_x,
            translation_y=frame_info.translation_y,
            translation_z=frame_info.translation_z,
        )

    # GPS sağlıksız — visual odometry
    from visual_odometry import step as vo_step
    dx, dy, dz = vo_step(frame, prev_frame, vo_state, camera_matrix)
    return DetectedTranslation(
        translation_x=vo_state.get("x", 0.0) + dx,
        translation_y=vo_state.get("y", 0.0) + dy,
        translation_z=vo_state.get("z", 0.0) + dz,
    )


def build_camera_matrix(fx, fy, cx, cy) -> np.ndarray:
    return np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float64)


if __name__ == "__main__":
    import json
    cal = json.load(open(Path(__file__).parent.parent / "data/calibration.json"))
    cm = cal["camera_matrix"]
    K = build_camera_matrix(cm["fx"], cm["fy"], cm["cx"], cm["cy"])
    print("Kamera matrisi:\n", K)
