"""
ORB-SLAM3 KeyFrameTrajectory.txt ile ground_truth.csv'yi eslestirip
Umeyama algoritmasi ile scale, R, t hesaplar.
"""

import numpy as np
import pandas as pd
import re


def load_keyframe_trajectory(path):
    data = np.loadtxt(path)
    timestamps = data[:, 0]
    positions = data[:, 1:4]
    return timestamps, positions


def load_ground_truth(path, fps=7.5):
    df = pd.read_csv(path)
    df["frame_idx"] = df["frame_numbers"].apply(
        lambda s: int(re.sub(r"[^0-9]", "", s))
    )
    df["timestamp"] = df["frame_idx"] / fps
    df = df.sort_values("frame_idx").reset_index(drop=True)
    return df


def match_by_timestamp(slam_timestamps, slam_positions, gt_df, max_dt=1.0 / 7.5 / 2):
    gt_timestamps = gt_df["timestamp"].values
    gt_positions = gt_df[["translation_x", "translation_y", "translation_z"]].values

    matched_slam = []
    matched_gt = []
    unmatched = 0

    for t, pos in zip(slam_timestamps, slam_positions):
        idx = np.argmin(np.abs(gt_timestamps - t))
        dt = abs(gt_timestamps[idx] - t)
        if dt <= max_dt:
            matched_slam.append(pos)
            matched_gt.append(gt_positions[idx])
        else:
            unmatched += 1

    print(f"Eslesen: {len(matched_slam)} / {len(slam_timestamps)}  (eslesmeyen: {unmatched})")
    return np.array(matched_slam), np.array(matched_gt)


def umeyama_alignment(src, dst):
    assert src.shape == dst.shape
    n, dim = src.shape

    src_mean = src.mean(axis=0)
    dst_mean = dst.mean(axis=0)

    src_centered = src - src_mean
    dst_centered = dst - dst_mean

    cov = (dst_centered.T @ src_centered) / n

    U, D, Vt = np.linalg.svd(cov)

    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1

    R = U @ S @ Vt

    var_src = (src_centered ** 2).sum() / n
    scale = np.trace(np.diag(D) @ S) / var_src

    t = dst_mean - scale * R @ src_mean

    return scale, R, t


def evaluate(src, dst, scale, R, t):
    pred = (scale * (R @ src.T).T) + t
    errors = np.linalg.norm(pred - dst, axis=1)
    return errors.mean(), errors.max(), errors


if __name__ == "__main__":
    import sys

    kf_path = sys.argv[1] if len(sys.argv) > 1 else "KeyFrameTrajectory.txt"
    gt_path = sys.argv[2] if len(sys.argv) > 2 else "data/ground_truth.csv"
    fps = float(sys.argv[3]) if len(sys.argv) > 3 else 7.5

    print(f"Okunuyor: {kf_path}")
    slam_ts, slam_pos = load_keyframe_trajectory(kf_path)
    print(f"  {len(slam_ts)} keyframe bulundu")

    print(f"Okunuyor: {gt_path}")
    gt_df = load_ground_truth(gt_path, fps=fps)
    print(f"  {len(gt_df)} GT satiri bulundu")

    matched_slam, matched_gt = match_by_timestamp(slam_ts, slam_pos, gt_df)

    if len(matched_slam) < 10:
        print("HATA: Cok az eslesme var, zaman eksenini kontrol et (fps degeri dogru mu?)")
        sys.exit(1)

    scale, R, t = umeyama_alignment(matched_slam, matched_gt)

    print("\n=== Kalibrasyon Sonuclari ===")
    print(f"scale = {scale:.6f}")
    print(f"R =\n{R}")
    print(f"t = {t}")

    mean_err, max_err, errors = evaluate(matched_slam, matched_gt, scale, R, t)
    print(f"\nOrtalama hata: {mean_err:.4f} m")
    print(f"Maksimum hata: {max_err:.4f} m")

    np.savez("calib.npz", scale=scale, R=R, t=t)
    print("\nKaydedildi: calib.npz")
