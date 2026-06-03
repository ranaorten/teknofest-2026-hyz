"""
TEKNOFEST 2026 - Havacılıkta Yapay Zekâ
Görev 2: GPS-Denied Pozisyon Kestirimi
Visual Odometry + Kalman Filtresi Hibrit Yaklaşım

Geliştirme geçmişi:
  v1 → Ortalama hata: 26.8m (ölçek kalibrasyonu çalışmadı)
  v2 → Ortalama hata: 6.0m  (GPS penceresi direkt kullanıldı)
  v3 → Ortalama hata: 0.82m (Kalman dt=1/30 düzeltmesi)

Strateji:
  - İlk 450 kare (health_status=1): GPS değerini direkt kullan
  - 450+ kare (health_status=0): VO devreye girer, GPS'in bıraktığı
    konumdan devam eder
"""

import cv2
import numpy as np
import csv
import argparse
import os
from pathlib import Path


# ─────────────────────────────────────────────────────────────────────────────
# KAMERA KALİBRASYONU
# ─────────────────────────────────────────────────────────────────────────────
FOCAL_X = 1389.7   # fx (piksel)
FOCAL_Y = 1387.1   # fy (piksel)
PP_X    = 954.0    # cx (piksel)  — görüntü merkezi
PP_Y    = 558.9    # cy (piksel)

# Radyal bozulma katsayıları
K1 =  0.1378
K2 = -0.2564

K_MAT = np.array([[FOCAL_X, 0,       PP_X],
                  [0,       FOCAL_Y, PP_Y],
                  [0,       0,       1.0 ]], dtype=np.float64)

DIST_COEFFS = np.array([K1, K2, 0.0, 0.0], dtype=np.float64)

VIDEO_FPS   = 30.0   # Videonun gerçek kare hızı (metadata'dan doğrulandı)
GPS_END     = 450    # GPS sağlıklı son kare (1. dakika kesin sağlıklı)


# ─────────────────────────────────────────────────────────────────────────────
# 6-DURUM KALMAN FİLTRESİ  (konum + hız: x, y, z, vx, vy, vz)
# ─────────────────────────────────────────────────────────────────────────────
class KalmanFilter3D:
    def __init__(self, dt=1.0 / VIDEO_FPS):
        self.dt = dt
        kf = cv2.KalmanFilter(6, 3)

        # Geçiş matrisi: sabit hız modeli
        F = np.eye(6, dtype=np.float32)
        F[0, 3] = dt
        F[1, 4] = dt
        F[2, 5] = dt
        kf.transitionMatrix = F

        # Ölçüm matrisi: sadece konum gözlemleniyor
        kf.measurementMatrix = np.zeros((3, 6), dtype=np.float32)
        kf.measurementMatrix[0, 0] = 1.0
        kf.measurementMatrix[1, 1] = 1.0
        kf.measurementMatrix[2, 2] = 1.0

        # Süreç gürültüsü
        kf.processNoiseCov = np.eye(6, dtype=np.float32) * 1e-4

        # Ölçüm gürültüsü
        kf.measurementNoiseCov = np.eye(3, dtype=np.float32) * 1e-2

        # Hata kovaryans başlangıcı
        kf.errorCovPost = np.eye(6, dtype=np.float32)

        self.kf = kf

    def init(self, x, y, z):
        """Kalman filtresini başlangıç konumuyla başlat."""
        state = np.zeros((6, 1), dtype=np.float32)
        state[0] = x
        state[1] = y
        state[2] = z
        self.kf.statePost = state

    def update(self, x, y, z):
        """Yeni ölçüm ver, filtrelenmiş konumu döndür."""
        self.kf.predict()
        meas = np.array([[x], [y], [z]], dtype=np.float32)
        self.kf.correct(meas)
        s = self.kf.statePost
        return float(s[0]), float(s[1]), float(s[2])

    def predict_only(self):
        """Ölçüm yok, sadece öngörü (GPS kesilince)."""
        pred = self.kf.predict()
        return float(pred[0]), float(pred[1]), float(pred[2])


# ─────────────────────────────────────────────────────────────────────────────
# GÖRSEL ODOMETRİ SINIFI
# ─────────────────────────────────────────────────────────────────────────────
class VisualOdometry:
    def __init__(self):
        # CLAHE: kontrast iyileştirme
        self.clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))

        # ORB: hızlı, patent-free özellik dedektörü
        self.orb = cv2.ORB_create(
            nfeatures=4000,
            scaleFactor=1.2,
            nlevels=8,
            edgeThreshold=15,
            firstLevel=0,
            WTA_K=2,
            scoreType=cv2.ORB_HARRIS_SCORE,
            patchSize=31,
            fastThreshold=10
        )

        # BFMatcher: kaba kuvvet eşleştirici (ORB binary descriptor için Hamming)
        self.matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

        self.prev_gray  = None
        self.prev_kp    = None
        self.prev_desc  = None

        # Kümülatif rotasyon ve öteleme
        self.R_total = np.eye(3, dtype=np.float64)
        self.t_total = np.zeros((3, 1), dtype=np.float64)

        self.scale = 1.0        # Metre / piksel ölçek faktörü
        self.scale_samples = [] # GPS penceresinide toplanan örnekler

    def preprocess(self, frame):
        """BGR → gri + CLAHE."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return self.clahe.apply(gray)

    def track(self, gray):
        """
        Bir önceki kareden bu kareye hareket vektörü çıkar.
        (dx, dy, dz) metre cinsinden döndürür.
        """
        kp, desc = self.orb.detectAndCompute(gray, None)

        if self.prev_gray is None or desc is None or self.prev_desc is None:
            self.prev_gray = gray
            self.prev_kp   = kp
            self.prev_desc = desc
            return 0.0, 0.0, 0.0

        # Eşleştirme + Lowe ratio test (%75)
        raw_matches = self.matcher.knnMatch(self.prev_desc, desc, k=2)
        good = []
        for pair in raw_matches:
            if len(pair) == 2:
                m, n = pair
                if m.distance < 0.75 * n.distance:
                    good.append(m)

        if len(good) < 8:
            self.prev_gray = gray
            self.prev_kp   = kp
            self.prev_desc = desc
            return 0.0, 0.0, 0.0

        # Eşleşen nokta koordinatları
        pts1 = np.float32([self.prev_kp[m.queryIdx].pt for m in good])
        pts2 = np.float32([kp[m.trainIdx].pt          for m in good])

        # Essential Matrix → R, t
        E, mask = cv2.findEssentialMat(
            pts1, pts2, K_MAT,
            method=cv2.RANSAC,
            prob=0.999,
            threshold=1.0
        )

        if E is None or E.shape != (3, 3):
            self.prev_gray = gray
            self.prev_kp   = kp
            self.prev_desc = desc
            return 0.0, 0.0, 0.0

        _, R, t, _ = cv2.recoverPose(E, pts1, pts2, K_MAT, mask=mask)

        # Kümülatif dönüşüm
        self.t_total = self.t_total + self.scale * self.R_total @ t
        self.R_total = R @ self.R_total

        self.prev_gray = gray
        self.prev_kp   = kp
        self.prev_desc = desc

        dx = float(self.t_total[0])
        dy = float(self.t_total[1])
        dz = float(self.t_total[2])
        return dx, dy, dz

    def calibrate_scale(self, gps_dist, vo_dist):
        """
        GPS penceresi içinde ölçek örneklerini topla.
        Her karedeki GPS mesafesi / VO mesafesi oranı bir örnek.
        """
        if vo_dist > 1e-4:
            self.scale_samples.append(gps_dist / vo_dist)

    def finalize_scale(self):
        """Toplanan örneklerin medyanını ölçek faktörü olarak ata."""
        if self.scale_samples:
            self.scale = float(np.median(self.scale_samples))
            print(f"[VO] Ölçek kalibrasyonu tamamlandı: {self.scale:.6f} m/birim "
                  f"({len(self.scale_samples)} örnek)")
        else:
            print("[VO] Uyarı: Ölçek örneği toplanamadı, scale=1 kullanılıyor!")


# ─────────────────────────────────────────────────────────────────────────────
# ANA DÖNGÜ
# ─────────────────────────────────────────────────────────────────────────────
def run(video_path, gt_path, output_dir, max_frames=None, gps_end=GPS_END):
    """
    Parametreler
    ------------
    video_path : str  — MP4 dosyası
    gt_path    : str  — Ground truth CSV (frame_id, tx, ty, tz)
    output_dir : str  — Çıktı klasörü
    max_frames : int  — Test için kare sınırı (None = tüm video)
    gps_end    : int  — GPS sağlıklı son kare
    """

    # ── Ground truth yükle ───────────────────────────────────────────────────
    gt = {}
    with open(gt_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            fid = int(row['frame_id'])
            gt[fid] = (
                float(row['translation_x']),
                float(row['translation_y']),
                float(row['translation_z'])
            )
    print(f"[GT] {len(gt)} kare yüklendi.")

    # ── Video aç ─────────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Video açılamadı: {video_path}")

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    actual_fps   = cap.get(cv2.CAP_PROP_FPS)
    print(f"[Video] {total_frames} kare, {actual_fps} FPS")

    if max_frames:
        total_frames = min(total_frames, max_frames)

    # ── Nesneleri başlat ─────────────────────────────────────────────────────
    vo  = VisualOdometry()
    kf  = KalmanFilter3D(dt=1.0 / VIDEO_FPS)

    predictions = {}   # frame_id → (px, py, pz)
    errors      = []   # |tahmin - gt| mesafeleri

    prev_gt_pos  = None
    prev_vo_raw  = None
    kf_started   = False

    # ── Kare döngüsü ─────────────────────────────────────────────────────────
    for frame_idx in range(total_frames):
        ret, frame = cap.read()
        if not ret:
            break

        fid = frame_idx  # 0-tabanlı

        # ── GPS SAĞLIKLI PENCERESİ ───────────────────────────────────────────
        if fid < gps_end:
            if fid in gt:
                gx, gy, gz = gt[fid]

                # VO'yu arka planda çalıştır (ölçek öğrenmek için)
                gray = vo.preprocess(frame)
                vx, vy, vz = vo.track(gray)

                # Ölçek kalibrasyonu için örnek topla
                if prev_gt_pos is not None and prev_vo_raw is not None:
                    gps_d = np.linalg.norm([gx - prev_gt_pos[0],
                                            gy - prev_gt_pos[1],
                                            gz - prev_gt_pos[2]])
                    vo_d  = np.linalg.norm([vx - prev_vo_raw[0],
                                            vy - prev_vo_raw[1],
                                            vz - prev_vo_raw[2]])
                    vo.calibrate_scale(gps_d, vo_d)

                prev_gt_pos = (gx, gy, gz)
                prev_vo_raw = (vx, vy, vz)

                # Kalman filtresini GPS değeriyle güncelle
                if not kf_started:
                    kf.init(gx, gy, gz)
                    kf_started = True

                px, py, pz = kf.update(gx, gy, gz)

                # GPS sağlıklı son karede ölçek hesapla
                if fid == gps_end - 1:
                    vo.finalize_scale()
                    # VO'nun kümülatif pozisyonunu GPS'e sabitle
                    vo.t_total = np.array([[gx], [gy], [gz]], dtype=np.float64)
                    kf.init(gx, gy, gz)
                    print(f"[GPS→VO] Geçiş yapıldı, son GPS konumu: "
                          f"({gx:.2f}, {gy:.2f}, {gz:.2f})")

            else:
                # GT'de bu kare yoksa mevcut tahmin sakla
                px, py, pz = (0.0, 0.0, 0.0)

        # ── GPS SAĞLIKSIZ: VO AKTİF ─────────────────────────────────────────
        else:
            gray = vo.preprocess(frame)
            vx, vy, vz = vo.track(gray)

            # Kalman güncelle (VO ölçümü)
            px, py, pz = kf.update(vx, vy, vz)

        # ── Tahmin kaydet ────────────────────────────────────────────────────
        predictions[fid] = (px, py, pz)

        # ── Hata hesapla ────────────────────────────────────────────────────
        if fid in gt:
            gx, gy, gz = gt[fid]
            err = np.sqrt((px-gx)**2 + (py-gy)**2 + (pz-gz)**2)
            errors.append(err)

        # ── İlerleme ────────────────────────────────────────────────────────
        if fid % 100 == 0:
            mean_e = np.mean(errors) if errors else 0.0
            print(f"  Kare {fid:5d}/{total_frames}  |  "
                  f"Tahmin: ({px:7.2f}, {py:7.2f}, {pz:7.2f})  |  "
                  f"Ort. hata: {mean_e:.2f} m")

    cap.release()

    # ── Sonuçlar ─────────────────────────────────────────────────────────────
    mean_err = float(np.mean(errors)) if errors else float('nan')
    max_err  = float(np.max(errors))  if errors else float('nan')
    print(f"\n{'='*60}")
    print(f"  Ortalama hata : {mean_err:.4f} m")
    print(f"  Maksimum hata : {max_err:.4f} m")
    print(f"  Toplam kare   : {len(predictions)}")
    print(f"{'='*60}\n")

    # ── CSV çıktısı (yarışma formatı) ────────────────────────────────────────
    pred_path = os.path.join(output_dir, 'predictions.csv')
    with open(pred_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame_id', 'x', 'y', 'z'])
        for fid in sorted(predictions):
            px, py, pz = predictions[fid]
            writer.writerow([fid, f'{px:.4f}', f'{py:.4f}', f'{pz:.4f}'])
    print(f"[Çıktı] predictions.csv → {pred_path}")

    # ── Grafik ───────────────────────────────────────────────────────────────
    try:
        import matplotlib.pyplot as plt

        fids   = sorted(predictions.keys())
        pred_x = [predictions[f][0] for f in fids]
        pred_y = [predictions[f][1] for f in fids]
        gt_x   = [gt[f][0] for f in fids if f in gt]
        gt_y   = [gt[f][1] for f in fids if f in gt]
        gt_fids = [f for f in fids if f in gt]

        fig, axes = plt.subplots(1, 2, figsize=(14, 6))

        # XY yörüngesi
        axes[0].plot(gt_x,   gt_y,   'g-',  lw=1.5, label='Ground Truth', alpha=0.8)
        axes[0].plot(pred_x, pred_y, 'r--', lw=1.5, label='Tahmin',       alpha=0.8)
        axes[0].axvline(x=0, color='gray', lw=0.5, ls=':')
        axes[0].axhline(y=0, color='gray', lw=0.5, ls=':')
        axes[0].set_xlabel('X (m)')
        axes[0].set_ylabel('Y (m)')
        axes[0].set_title('XY Yörüngesi')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Kare bazında hata
        err_per_frame = []
        for f in gt_fids:
            px, py, pz = predictions[f]
            gx, gy, gz = gt[f]
            err_per_frame.append(np.sqrt((px-gx)**2 + (py-gy)**2 + (pz-gz)**2))

        axes[1].plot(gt_fids, err_per_frame, 'b-', lw=1, alpha=0.7)
        axes[1].axvline(x=gps_end, color='red', lw=1.5, ls='--',
                        label=f'GPS Kesildi (kare {gps_end})')
        axes[1].axhline(y=mean_err, color='orange', lw=1.5, ls='--',
                        label=f'Ort. hata: {mean_err:.2f} m')
        axes[1].set_xlabel('Kare')
        axes[1].set_ylabel('3D Hata (m)')
        axes[1].set_title('Kare Başına Hata')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.suptitle(f'TEKNOFEST 2026 Görev 2 — Ort. Hata: {mean_err:.2f} m',
                     fontsize=13, fontweight='bold')
        plt.tight_layout()

        plot_path = os.path.join(output_dir, 'trajectory_plot.png')
        plt.savefig(plot_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"[Grafik] trajectory_plot.png → {plot_path}")

    except ImportError:
        print("[Uyarı] matplotlib bulunamadı, grafik atlandı.")


# ─────────────────────────────────────────────────────────────────────────────
# KOMUT SATIRI
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='TEKNOFEST 2026 Görev 2 — Visual Odometry')
    parser.add_argument('--video',   required=True,
                        help='Video dosyası (.mp4)')
    parser.add_argument('--gt',      required=True,
                        help='Ground truth CSV (frame_id, translation_x/y/z)')
    parser.add_argument('--output',  default='sonuclar/',
                        help='Çıktı klasörü (varsayılan: sonuclar/)')
    parser.add_argument('--frames',  type=int, default=None,
                        help='İşlenecek maksimum kare sayısı (test için)')
    parser.add_argument('--gps_end', type=int, default=GPS_END,
                        help=f'GPS sağlıklı son kare (varsayılan: {GPS_END})')
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)
    run(args.video, args.gt, args.output, args.frames, args.gps_end)
