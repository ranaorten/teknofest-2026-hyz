"""
mono_stream bridge'ini tum video boyunca calistirip her decimated kare
icin (x, y, z_slam, median_depth) cikarir.
"""

import sys
sys.path.insert(0, ".")

import time
import numpy as np
import pandas as pd
import cv2

from gorev2.orb_slam3_bridge import OrbSlam3Bridge


def main():
    video_path = "/home/rana/Downloads/THYZ_2026_Ornek_Veri_1.MP4"
    vocab_path = "/home/rana/ORB_SLAM3/Vocabulary/ORBvoc.txt"
    settings_path = "/home/rana/ORB_SLAM3/my_camera.yaml"
    executable_path = "/home/rana/ORB_SLAM3/Examples/Monocular/mono_stream"
    out_csv = "orb_slam3_with_depth.csv"
    skip = 4

    bridge = OrbSlam3Bridge(
        vocab_path=vocab_path,
        settings_path=settings_path,
        executable_path=executable_path,
        fps=7.4925,
    )
    bridge.start()
    time.sleep(2)

    cap = cv2.VideoCapture(video_path)
    raw_frame_idx = 0
    decimated_idx = 0
    rows = []
    t0 = time.time()

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if raw_frame_idx % skip == 0:
            result = bridge.push_frame(frame)
            if result is not None:
                x, y, z, depth = result
            else:
                x = y = z = depth = None

            rows.append({
                "decimated_idx": decimated_idx,
                "raw_frame_idx": raw_frame_idx,
                "slam_x": x, "slam_y": y, "slam_z": z, "depth": depth,
            })

            if decimated_idx % 200 == 0:
                print(f"  {decimated_idx} kare islendi...")

            decimated_idx += 1

        raw_frame_idx += 1

    cap.release()
    bridge.stop()

    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)

    elapsed = time.time() - t0
    print(f"\nToplam {len(df)} satir yazildi: {out_csv}")
    print(f"Sure: {elapsed:.1f}s ({len(df)/elapsed:.2f} kare/s)")


if __name__ == "__main__":
    main()
