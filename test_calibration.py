"""Ucten uca test: PositionEstimator ile Denklem 2 skoru."""
import sys
sys.path.insert(0, ".")
import csv
import random
import numpy as np
from common.video_loader import VideoLoader
from gorev2.visual_odometry import VisualOdometry
from gorev2.position import PositionEstimator

# --- RASTGELELİĞİ SABİTLE (tekrarlanabilirlik için) ---
random.seed(42)
np.random.seed(42)

VIDEO = "/home/rana/Downloads/THYZ_2026_Ornek_Veri_1.MP4"
HEALTHY_FRAMES = 450
TEST_FRAMES = 900   # toplam 450+900=1350 kare (yaklaşık 3 dk)

# Gerçek pozisyon verisini oku (29.97 fps -> 7.5 fps için her 4. kare)
gt_raw = []
with open("data/ground_truth.csv") as f:
    for row in csv.DictReader(f):
        gt_raw.append((float(row["translation_x"]),
                       float(row["translation_y"]),
                       float(row["translation_z"])))
gt = gt_raw[::4]
print(f"GT: {len(gt)} kare @7.5fps")

loader = VideoLoader(VIDEO, target_fps=7.5)
vo = VisualOdometry(calib_path="data/calibration.json", calib_key="rgb_1080p")
est = PositionEstimator(vo)

errors = []
z_errors = []   # Z ekseni hatasını ayrıca takip et

for i in range(TEST_FRAMES):
    result = loader.next_frame()
    if result is None or i >= len(gt):
        break
    _, frame = result
    g = gt[i]

    # Health simülasyonu: ilk HEALTHY_FRAMES sağlıklı, sonrası kesik
    gps = {
        "translation_x": g[0],
        "translation_y": g[1],
        "translation_z": g[2],
        "gps_health_status": 1 if i < HEALTHY_FRAMES else 0
    }

    x, y, z = est.update(frame, gps)

    if i >= HEALTHY_FRAMES:
        err = float(np.linalg.norm(np.array([x, y, z]) - np.array(g)))
        errors.append(err)
        z_err = abs(z - g[2])
        z_errors.append(z_err)
        if (i - HEALTHY_FRAMES) % 100 == 0:
            print(f"kesinti+{i-HEALTHY_FRAMES}: tahmin=({x:.1f},{y:.1f},{z:.1f}) "
                  f"gercek=({g[0]:.1f},{g[1]:.1f},{g[2]:.1f}) hata={err:.2f}m | z-hatası={z_err:.2f}m")

loader.release()

errors = np.array(errors)
z_errors = np.array(z_errors)

print(f"\n--- SONUC (Denklem 2) ---")
print(f"Ortalama toplam hata: {errors.mean():.3f} m")
print(f"Max toplam hata:      {errors.max():.3f} m")
print(f"Std toplam hata:      {errors.std():.3f} m")
print(f"\n--- Z EKSENİ ÖZEL ---")
print(f"Ortalama z-hatası:    {z_errors.mean():.3f} m")
print(f"Max z-hatası:         {z_errors.max():.3f} m")

# Trend analizi
print(f"\n--- ZAMANSAL TREND ---")
print(f"Ilk 100 kare (kesinti sonrası): {errors[:100].mean():.3f} m")
print(f"Son 100 kare:                    {errors[-100:].mean():.3f} m")