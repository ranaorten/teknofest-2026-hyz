"""
Kamera kalibrasyonu: K matrisi, distorsiyon katsayıları, undistort.
"""

import json
import cv2
import numpy as np
from pathlib import Path

DEFAULT_CAL = Path(__file__).parent.parent / "data/calibration.json"


def load_calibration(path: str | Path = DEFAULT_CAL) -> tuple[np.ndarray, np.ndarray]:
    """(K, dist_coeffs) döner. K: 3x3, dist: (k1,k2,p1,p2,k3)."""
    with open(path) as f:
        cal = json.load(f)
    cm = cal["camera_matrix"]
    dc = cal["dist_coeffs"]
    K = np.array(
        [[cm["fx"], 0, cm["cx"]],
         [0, cm["fy"], cm["cy"]],
         [0,        0,        1]],
        dtype=np.float64,
    )
    dist = np.array(
        [dc["k1"], dc["k2"], dc["p1"], dc["p2"], dc["k3"]],
        dtype=np.float64,
    )
    return K, dist


def undistort(frame: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Lens bozulmasını giderir."""
    h, w = frame.shape[:2]
    new_K, roi = cv2.getOptimalNewCameraMatrix(K, dist, (w, h), alpha=0)
    undist = cv2.undistort(frame, K, dist, None, new_K)
    x, y, rw, rh = roi
    return undist[y:y+rh, x:x+rw] if all([rw, rh]) else undist


if __name__ == "__main__":
    K, dist = load_calibration()
    print("K:\n", K)
    print("dist:", dist)
