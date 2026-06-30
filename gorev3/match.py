"""
Görev 3: Görüntü Eşleme (tanımsız nesne tespiti)
Oturum başında verilen referans nesne görüntülerini video karelerinde bulur.
"""

import sys
import cv2
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "common"))
from server_io import DetectedUndefinedObject


def load_reference_images(ref_dir: str) -> dict[str, np.ndarray]:
    """Referans nesne görüntülerini yükler. {object_id: görüntü}"""
    refs = {}
    for p in Path(ref_dir).glob("*.*"):
        img = cv2.imread(str(p))
        if img is not None:
            refs[p.stem] = img
    return refs


def match_frame(
    frame: np.ndarray,
    references: dict[str, np.ndarray],
) -> list[DetectedUndefinedObject]:
    """
    SIFT + BFMatcher ile referans nesneleri frame içinde arar.
    Eşleşme bulunamazsa boş liste döner.
    """
    sift = cv2.SIFT_create()
    kp_frame, des_frame = sift.detectAndCompute(frame, None)
    if des_frame is None:
        return []

    bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    results = []

    for obj_id, ref_img in references.items():
        kp_ref, des_ref = sift.detectAndCompute(ref_img, None)
        if des_ref is None:
            continue

        matches = bf.knnMatch(des_ref, des_frame, k=2)
        good = [m for m, n in matches if m.distance < 0.75 * n.distance]

        if len(good) < 8:
            continue

        src_pts = np.float32([kp_ref[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp_frame[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
        M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)

        if M is None:
            continue

        h, w = ref_img.shape[:2]
        corners = np.float32([[0, 0], [w, 0], [w, h], [0, h]]).reshape(-1, 1, 2)
        projected = cv2.perspectiveTransform(corners, M).reshape(-1, 2)

        x1, y1 = projected.min(axis=0).astype(int)
        x2, y2 = projected.max(axis=0).astype(int)

        results.append(DetectedUndefinedObject(
            object_id=obj_id,
            top_left_x=int(x1),
            top_left_y=int(y1),
            bottom_right_x=int(x2),
            bottom_right_y=int(y2),
        ))

    return results


if __name__ == "__main__":
    ref_dir = sys.argv[1] if len(sys.argv) > 1 else "data/references"
    img_path = sys.argv[2] if len(sys.argv) > 2 else None

    refs = load_reference_images(ref_dir)
    print(f"{len(refs)} referans nesne yüklendi.")

    if img_path:
        frame = cv2.imread(img_path)
        found = match_frame(frame, refs)
        print(f"{len(found)} eşleşme bulundu.")
        for f in found:
            print(f"  {f.object_id}: ({f.top_left_x},{f.top_left_y}) → ({f.bottom_right_x},{f.bottom_right_y})")
