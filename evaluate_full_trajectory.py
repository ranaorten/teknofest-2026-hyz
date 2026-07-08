"""
Tam KeyFrameTrajectory.txt'yi (GPS'li + GPS'siz tum sure) alir,
SADECE ilk 450 karelik (GPS'li) bolgeden hesaplanmis calib.npz'yi uygular,
ve GPS'li / GPS'siz bolgeleri AYRI AYRI degerlendirir.
"""

import numpy as np
import pandas as pd
import re
import sys

from compute_calibration import load_keyframe_trajectory, load_ground_truth


def match_by_timestamp(slam_timestamps, slam_positions, gt_df, max_dt):
    gt_timestamps = gt_df["timestamp"].values
    gt_positions = gt_df[["translation_x", "translation_y", "translation_z"]].values

    matched_slam, matched_gt, matched_ts = [], [], []
    unmatched = 0
    for t, pos in zip(slam_timestamps, slam_positions):
        idx = np.argmin(np.abs(gt_timestamps - t))
        dt = abs(gt_timestamps[idx] - t)
        if dt <= max_dt:
            matched_slam.append(pos)
            matched_gt.append(gt_positions[idx])
            matched_ts.append(t)
        else:
            unmatched += 1
    return (np.array(matched_slam), np.array(matched_gt),
            np.array(matched_ts), unmatched)


def main():
    kf_path = sys.argv[1] if len(sys.argv) > 1 else "KeyFrameTrajectory.txt"
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth.csv"
    raw_fps = float(sys.argv[3]) if len(sys.argv) > 3 else 29.97
    gps_healthy_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

    slam_ts, slam_pos = load_keyframe_trajectory(kf_path)
    gt_df = load_ground_truth(gt_path, fps=raw_fps)

    decimated_fps = raw_fps / 4.0
    max_dt = 1.0 / decimated_fps / 2.0

    matched_slam, matched_gt, matched_ts, unmatched = match_by_timestamp(
        slam_ts, slam_pos, gt_df, max_dt
    )
    print(f"Toplam keyframe: {len(slam_ts)}, eslesen: {len(matched_slam)}, "
          f"eslesmeyen: {unmatched}")

    calib = np.load("calib.npz")
    scale, R, t_vec = calib["scale"], calib["R"], calib["t"]

    pred = (scale * (R @ matched_slam.T).T) + t_vec
    errors = np.linalg.norm(pred - matched_gt, axis=1)

    is_healthy = matched_ts <= gps_healthy_seconds
    is_denied = ~is_healthy

    print(f"\n=== GPS'li bolge (0 - {gps_healthy_seconds:.0f}s), "
          f"kalibrasyonun egitildigi bolge ===")
    if is_healthy.sum() > 0:
        h_err = errors[is_healthy]
        print(f"  {is_healthy.sum()} kare, ortalama hata: {h_err.mean():.4f} m, "
              f"max: {h_err.max():.4f} m")

    print(f"\n=== GPS'siz bolge ({gps_healthy_seconds:.0f}s - son), "
          f"YARISMA PUANLAMA SENARYOSU ===")
    if is_denied.sum() > 0:
        d_err = errors[is_denied]
        print(f"  {is_denied.sum()} kare, ortalama hata: {d_err.mean():.4f} m, "
              f"max: {d_err.max():.4f} m")
        print(f"  medyan: {np.median(d_err):.4f} m")

        denied_ts = matched_ts[is_denied]
        denied_err = d_err
        t_min, t_max = denied_ts.min(), denied_ts.max()
        n_bins = 5
        edges = np.linspace(t_min, t_max, n_bins + 1)
        print(f"\n  Zaman dilimlerine gore (GPS kesildikten sonra hata birikiyor mu?):")
        for i in range(n_bins):
            mask = (denied_ts >= edges[i]) & (denied_ts < edges[i + 1] + 1e-9)
            if mask.sum() > 0:
                print(f"    {edges[i]:6.1f}s - {edges[i+1]:6.1f}s: "
                      f"ortalama={denied_err[mask].mean():6.3f} m, "
                      f"max={denied_err[mask].max():6.3f} m, n={mask.sum()}")
    else:
        print("  UYARI: GPS'siz bolgede eslesen kare bulunamadi!")

    print(f"\n=== Genel (tum eslesenler) ===")
    print(f"  Ortalama: {errors.mean():.4f} m, Max: {errors.max():.4f} m, "
          f"Medyan: {np.median(errors):.4f} m")


if __name__ == "__main__":
    main()
