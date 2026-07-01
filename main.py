"""
Uçtan uca akış — tek komutla çalışır:
    python main.py --video data/THYZ_2026_Ornek_Veri_1.MP4

Dummy/baseline modda her görev boş ama geçerli çıktı üretir.
"""

import sys
import json
import argparse
import uuid
from pathlib import Path

# common modülleri
sys.path.insert(0, str(Path(__file__).parent / "common"))
from calibration import load_calibration, undistort
from video_loader import iter_video, HEALTHY_FRAME_COUNT
from server_io import (
    DetectedObject, DetectedTranslation, DetectedUndefinedObject, send_result,
)

# görev modülleri
sys.path.insert(0, str(Path(__file__).parent / "gorev1"))
sys.path.insert(0, str(Path(__file__).parent / "gorev2"))
sys.path.insert(0, str(Path(__file__).parent / "gorev3"))


# ── Baseline / dummy fonksiyonlar ──────────────────────────────────────────

def baseline_detect(frame) -> list[DetectedObject]:
    """Görev 1 baseline: nesne tespit edilmedi."""
    return []


def baseline_position(frame_info) -> DetectedTranslation:
    """Görev 2 baseline: sunucudan gelen değeri olduğu gibi döndür."""
    return DetectedTranslation(
        translation_x=frame_info.translation_x,
        translation_y=frame_info.translation_y,
        translation_z=frame_info.translation_z,
    )


def baseline_match(frame) -> list[DetectedUndefinedObject]:
    """Görev 3 baseline: eşleşme yok."""
    return []


# ── Ana döngü ──────────────────────────────────────────────────────────────

def run(video_path: str, output_path: str = "outputs/results.json", dry_run: bool = True):
    K, dist = load_calibration()
    results = []
    prev_frame = None

    for f in iter_video(video_path):
        frame_ud = undistort(f.image, K, dist)

        objects      = baseline_detect(frame_ud)
        translation  = baseline_position(f)
        undefined    = baseline_match(frame_ud)

        result = {
            "frame_id":    f.frame_id,
            "gps_health":  f.gps_health_status,
            "detected_objects":            [vars(o) for o in objects],
            "detected_translations":       [vars(translation)],
            "detected_undefined_objects":  [vars(u) for u in undefined],
        }
        results.append(result)

        if not dry_run:
            send_result(
                prediction_id=str(uuid.uuid4()),
                user_url="http://server/users/1/",
                frame_url=f.frame_id,
                detected_objects=objects,
                detected_translations=[translation],
                detected_undefined_objects=undefined,
            )

        if f.index % 50 == 0:
            print(f"[{f.index:04d}] GPS={f.gps_health_status} "
                  f"obj={len(objects)} undef={len(undefined)}")

        prev_frame = frame_ud

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as fp:
        json.dump(results, fp, ensure_ascii=False, indent=2)
    print(f"\n✓ {len(results)} kare işlendi → {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video",   default="data/THYZ_2026_Ornek_Veri_1.MP4")
    parser.add_argument("--output",  default="outputs/results.json")
    parser.add_argument("--live",    action="store_true", help="Gerçek sunucuya gönder")
    args = parser.parse_args()

    run(args.video, args.output, dry_run=not args.live)
