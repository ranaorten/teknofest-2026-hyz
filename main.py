"""
Uçtan uca akış — tek komutla çalışır:
    python main.py --video data/THYZ_2026_Ornek_Veri_1.MP4
    python main.py --live --session <url>   (gerçek yarışma sunucusu)

Görev 1 ve Görev 3 şimdilik baseline (boş ama geçerli çıktı).
Görev 2 (pozisyon), ORB-SLAM3 tabanlı PositionEstimatorORB ile gerçek
kestirim yapar (bkz. gorev2/position_orbslam.py, DEVAM_NOTU.md).
"""

import sys
import json
import argparse
import time
import uuid
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent / "common"))

from common.video_loader import VideoLoader
from common.server_io import (
    FrameInfo, DetectedObject, DetectedTranslation, DetectedUndefinedObject,
    get_frame_list, get_frame_image, send_result,
)

from gorev2.orb_slam3_bridge import OrbSlam3Bridge
from gorev2.position_orbslam import PositionEstimatorORB

import numpy as np
import cv2


VOCAB_PATH = "/home/rana/ORB_SLAM3/Vocabulary/ORBvoc.txt"
SETTINGS_PATH = "/home/rana/ORB_SLAM3/my_camera.yaml"
EXECUTABLE_PATH = "/home/rana/ORB_SLAM3/Examples/Monocular/mono_stream"
FPS = 7.5


def baseline_detect(frame) -> list:
    """Görev 1 baseline: nesne tespit edilmedi."""
    return []


def baseline_match(frame) -> list:
    """Görev 3 baseline: eşleşme yok."""
    return []


def estimate_position(estimator: PositionEstimatorORB, frame, frame_info) -> DetectedTranslation:
    """
    gps_health_status'e gore karar:
      - saglikli (1): referans degeri gonderebiliriz (guvenli varsayilan).
      - sagliksiz (0): estimator'in kestirimini gondermek ZORUNLUYUZ.
    """
    gps = {
        "gps_health_status": frame_info.gps_health_status,
        "translation_x": frame_info.translation_x,
        "translation_y": frame_info.translation_y,
        "translation_z": frame_info.translation_z,
    }
    x, y, z = estimator.update(frame, gps)

    if frame_info.gps_health_status == 1:
        return DetectedTranslation(
            translation_x=frame_info.translation_x,
            translation_y=frame_info.translation_y,
            translation_z=frame_info.translation_z,
        )
    return DetectedTranslation(translation_x=x, translation_y=y, translation_z=z)


def run_offline(video_path: str, output_path: str, dry_run: bool = True):
    """Video dosyasindan okuyarak calisir (gercek sunucu yerine yerel test)."""
    import pandas as pd

    loader = VideoLoader(video_path, target_fps=FPS)

    gt_df = pd.read_csv("data/ground_truth.csv")
    gt_df["frame_idx"] = gt_df["frame_numbers"].str.replace("frame_", "").astype(int)
    gt_df = gt_df.sort_values("frame_idx").reset_index(drop=True)

    bridge = OrbSlam3Bridge(
        vocab_path=VOCAB_PATH, settings_path=SETTINGS_PATH,
        executable_path=EXECUTABLE_PATH, fps=FPS,
    )
    bridge.start()
    time.sleep(2)
    estimator = PositionEstimatorORB(bridge)

    results = []
    n_healthy = 450

    try:
        while True:
            result = loader.next_frame()
            if result is None:
                break
            idx, frame = result

            raw_idx = idx * loader.skip
            if raw_idx >= len(gt_df):
                break
            gt_row = gt_df.iloc[raw_idx]

            is_healthy = idx < n_healthy
            frame_info = FrameInfo(
                url=f"local/frame_{idx}", image_url="", video_name=Path(video_path).name,
                session="local",
                translation_x=float(gt_row["translation_x"]),
                translation_y=float(gt_row["translation_y"]),
                translation_z=float(gt_row["translation_z"]),
                gps_health_status=1 if is_healthy else 0,
            )

            objects = baseline_detect(frame)
            translation = estimate_position(estimator, frame, frame_info)
            undefined = baseline_match(frame)

            result_dict = {
                "frame_id": frame_info.url,
                "gps_health": frame_info.gps_health_status,
                "detected_objects": [vars(o) for o in objects],
                "detected_translations": [vars(translation)],
                "detected_undefined_objects": [vars(u) for u in undefined],
            }
            results.append(result_dict)

            if not dry_run:
                send_result(
                    prediction_id=str(uuid.uuid4()),
                    user_url="http://server/users/1/",
                    frame_url=frame_info.url,
                    detected_objects=objects,
                    detected_translations=[translation],
                    detected_undefined_objects=undefined,
                )

            if idx % 100 == 0:
                print(f"[{idx:04d}] GPS={frame_info.gps_health_status} "
                      f"pos=({translation.translation_x:.2f},"
                      f"{translation.translation_y:.2f},"
                      f"{translation.translation_z:.2f})")
    finally:
        bridge.stop()
        loader.release()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n✓ {len(results)} kare işlendi → {output_path}")


def run_live(session_url: str, output_path: str):
    """Gercek yarisma sunucusundan kare listesini alip isler."""
    frames = get_frame_list(session_url)
    print(f"{len(frames)} kare alindi (oturum: {session_url})")

    bridge = OrbSlam3Bridge(
        vocab_path=VOCAB_PATH, settings_path=SETTINGS_PATH,
        executable_path=EXECUTABLE_PATH, fps=FPS,
    )
    bridge.start()
    time.sleep(2)
    estimator = PositionEstimatorORB(bridge)

    results = []
    try:
        for i, frame_info in enumerate(frames):
            img_bytes = get_frame_image(frame_info.image_url)
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            frame = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
            if frame is None:
                print(f"[UYARI] Kare decode edilemedi: {frame_info.url}")
                continue

            objects = baseline_detect(frame)
            translation = estimate_position(estimator, frame, frame_info)
            undefined = baseline_match(frame)

            send_result(
                prediction_id=str(uuid.uuid4()),
                user_url="http://server/users/1/",
                frame_url=frame_info.url,
                detected_objects=objects,
                detected_translations=[translation],
                detected_undefined_objects=undefined,
            )

            results.append({
                "frame_id": frame_info.url,
                "gps_health": frame_info.gps_health_status,
                "detected_translations": [vars(translation)],
            })

            if i % 100 == 0:
                print(f"[{i:04d}] GPS={frame_info.gps_health_status} "
                      f"pos=({translation.translation_x:.2f},"
                      f"{translation.translation_y:.2f},"
                      f"{translation.translation_z:.2f})")
    finally:
        bridge.stop()

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n✓ {len(results)} kare işlendi → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", default="data/THYZ_2026_Ornek_Veri_1.MP4")
    parser.add_argument("--output", default="outputs/results.json")
    parser.add_argument("--live", action="store_true", help="Gerçek sunucuya bağlan")
    parser.add_argument("--session", default=None, help="--live ile: oturum URL'i")
    parser.add_argument("--send", action="store_true",
                         help="Offline modda da sunucuya gonder (varsayilan: sadece dosyaya yaz)")
    args = parser.parse_args()

    if args.live:
        if not args.session:
            print("HATA: --live ile --session <url> gerekli")
            sys.exit(1)
        run_live(args.session, args.output)
    else:
        run_offline(args.video, args.output, dry_run=not args.send)
