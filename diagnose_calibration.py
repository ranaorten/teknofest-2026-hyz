"""
Kalibrasyon sonrasi hatayi zaman ve eksen bazinda analiz eder.
"""

import numpy as np
import pandas as pd
import re
import sys

from compute_calibration import (
    load_keyframe_trajectory,
    load_ground_truth,
    match_by_timestamp,
)


def main():
    kf_path = sys.argv[1] if len(sys.argv) > 1 else "../ORB_SLAM3/KeyFrameTrajectory.txt"
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth.csv"
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 7.5

    slam_ts, slam_pos = load_keyframe_trajectory(kf_path)
    gt_df = load_ground_truth(gt_path, fps=fps)

    gt_timestamps = gt_df["timestamp"].values
    gt_positions = gt_df[["translation_x", "translation_y", "translation_z"]].values

    matched_slam = []
    matched_gt = []
    matched_ts = []
    for t, pos in zip(slam_ts, slam_pos):
        idx = np.argmin(np.abs(gt_timestamps - t))
        dt = abs(gt_timestamps[idx] - t)
        if dt <= (1.0 / fps / 2):
            matched_slam.append(pos)
            matched_gt.append(gt_positions[idx])
            matched_ts.append(t)

    matched_slam = np.array(matched_slam)
    matched_gt = np.array(matched_gt)
    matched_ts = np.array(matched_ts)

    calib = np.load("calib.npz")
    scale, R, t_vec = calib["scale"], calib["R"], calib["t"]

    pred = (scale * (R @ matched_slam.T).T) + t_vec
    diff = pred - matched_gt
    errors = np.linalg.norm(diff, axis=1)

    print("=== Eksen bazinda ortalama mutlak hata ===")
    print(f"X: {np.abs(diff[:,0]).mean():.4f} m")
    print(f"Y: {np.abs(diff[:,1]).mean():.4f} m")
    print(f"Z: {np.abs(diff[:,2]).mean():.4f} m")

    print("\n=== Zamana gore hata (ilk / orta / son ucte bir) ===")
    n = len(errors)
    thirds = [errors[:n//3], errors[n//3:2*n//3], errors[2*n//3:]]
    labels = ["Ilk 1/3 (0-20sn)", "Orta 1/3 (20-40sn)", "Son 1/3 (40-60sn)"]
    for label, chunk in zip(labels, thirds):
        print(f"{label}: ortalama={chunk.mean():.4f} m, max={chunk.max():.4f} m")

    print("\n=== En kotu 5 kare ===")
    worst_idx = np.argsort(errors)[-5:][::-1]
    for i in worst_idx:
        print(f"  t={matched_ts[i]:.2f}s  hata={errors[i]:.2f}m  "
              f"pred=({pred[i,0]:.2f},{pred[i,1]:.2f},{pred[i,2]:.2f})  "
              f"gt=({matched_gt[i,0]:.2f},{matched_gt[i,1]:.2f},{matched_gt[i,2]:.2f})")

    print("\n=== GT hareket araligi (referans icin) ===")
    print(f"X: {gt_positions[:,0].min():.2f} .. {gt_positions[:,0].max():.2f}")
    print(f"Y: {gt_positions[:,1].min():.2f} .. {gt_positions[:,1].max():.2f}")
    print(f"Z: {gt_positions[:,2].min():.2f} .. {gt_positions[:,2].max():.2f}")


if __name__ == "__main__":
    main()
