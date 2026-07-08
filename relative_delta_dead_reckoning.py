"""
Hibrit yaklasim: ORB-SLAM3'un ardisik keyframe'ler arasindaki RELATIF hareketini
GPS'li bolgede kalibre eder, GPS'siz bolgede bu kalibre edilmis delta'lari
kumulatif olarak entegre eder (dead-reckoning).
"""

import numpy as np
import pandas as pd
import re
import sys

from compute_calibration import load_keyframe_trajectory, load_ground_truth


def procrustes_scale_rotation(src_vecs, dst_vecs):
    """
    src_vecs, dst_vecs: (N,3) hareket vektorleri (delta/yon vektorleri).
    Ceviri YOK - sadece scale*R ile src'yi dst'ye en iyi oturtan donusumu bulur.
    Dondurur: scale (float), R (3,3)  ->  dst_esti = scale * R @ src
    """
    n, dim = src_vecs.shape
    cov = (dst_vecs.T @ src_vecs) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt

    var_src = (src_vecs ** 2).sum() / n
    scale = np.trace(np.diag(D) @ S) / var_src

    return scale, R


def main():
    kf_path = sys.argv[1] if len(sys.argv) > 1 else "KeyFrameTrajectory.txt"
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth.csv"
    raw_fps = float(sys.argv[3]) if len(sys.argv) > 3 else 29.97
    gps_healthy_seconds = float(sys.argv[4]) if len(sys.argv) > 4 else 60.0

    slam_ts, slam_pos = load_keyframe_trajectory(kf_path)
    gt_df = load_ground_truth(gt_path, fps=raw_fps)
    gt_timestamps = gt_df["timestamp"].values
    gt_positions = gt_df[["translation_x", "translation_y", "translation_z"]].values

    def gt_at(t):
        idx = np.argmin(np.abs(gt_timestamps - t))
        return gt_positions[idx]

    healthy_mask = slam_ts <= gps_healthy_seconds
    healthy_ts = slam_ts[healthy_mask]
    healthy_pos = slam_pos[healthy_mask]

    src_deltas = []
    dst_deltas = []
    for i in range(len(healthy_ts) - 1):
        d_slam = healthy_pos[i + 1] - healthy_pos[i]
        gt_i = gt_at(healthy_ts[i])
        gt_ip1 = gt_at(healthy_ts[i + 1])
        d_gt = gt_ip1 - gt_i
        src_deltas.append(d_slam)
        dst_deltas.append(d_gt)

    src_deltas = np.array(src_deltas)
    dst_deltas = np.array(dst_deltas)
    print(f"GPS'li bolgede {len(src_deltas)} ardisik delta cifti bulundu")

    scale, R = procrustes_scale_rotation(src_deltas, dst_deltas)
    print(f"\nRelatif kalibrasyon: scale={scale:.6f}")
    print(f"R=\n{R}")

    pred_deltas = (scale * (R @ src_deltas.T).T)
    delta_errors = np.linalg.norm(pred_deltas - dst_deltas, axis=1)
    print(f"\nDelta-basi ortalama hata (GPS'li bolge): {delta_errors.mean():.4f} m")

    denied_mask = slam_ts > gps_healthy_seconds
    denied_ts = slam_ts[denied_mask]
    denied_pos = slam_pos[denied_mask]

    last_healthy_idx = np.where(healthy_mask)[0][-1]
    current_pos_est = gt_at(slam_ts[last_healthy_idx]).copy()
    current_slam_pos = slam_pos[last_healthy_idx].copy()

    estimates = []
    gts = []
    ts_list = []

    prev_slam = current_slam_pos
    prev_est = current_pos_est

    for t, pos in zip(denied_ts, denied_pos):
        d_slam = pos - prev_slam
        d_est = scale * (R @ d_slam)
        new_est = prev_est + d_est

        estimates.append(new_est.copy())
        gts.append(gt_at(t))
        ts_list.append(t)

        prev_slam = pos
        prev_est = new_est

    estimates = np.array(estimates)
    gts = np.array(gts)
    ts_arr = np.array(ts_list)

    errors = np.linalg.norm(estimates - gts, axis=1)

    print(f"\n=== GPS'siz bolge (dead-reckoning ile) ===")
    print(f"  {len(errors)} kare, ortalama hata: {errors.mean():.4f} m, "
          f"max: {errors.max():.4f} m, medyan: {np.median(errors):.4f} m")

    n_bins = 5
    t_min, t_max = ts_arr.min(), ts_arr.max()
    edges = np.linspace(t_min, t_max, n_bins + 1)
    print(f"\n  Zaman dilimlerine gore:")
    for i in range(n_bins):
        mask = (ts_arr >= edges[i]) & (ts_arr < edges[i + 1] + 1e-9)
        if mask.sum() > 0:
            print(f"    {edges[i]:6.1f}s - {edges[i+1]:6.1f}s: "
                  f"ortalama={errors[mask].mean():6.3f} m, "
                  f"max={errors[mask].max():6.3f} m, n={mask.sum()}")

    diff = estimates - gts
    print(f"\n  Eksen bazinda ortalama mutlak hata:")
    print(f"    X: {np.abs(diff[:,0]).mean():.4f} m")
    print(f"    Y: {np.abs(diff[:,1]).mean():.4f} m")
    print(f"    Z: {np.abs(diff[:,2]).mean():.4f} m")


if __name__ == "__main__":
    main()
