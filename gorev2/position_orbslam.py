"""
ORB-SLAM3 tabanli PositionEstimator (production) - v2, derinlik tabanli Z.

Mimarisi:
  - X/Y: ORB-SLAM3'un ardisik kare pozisyon delta'lari, GPS'li donemde
    online olarak kalibre edilir (Procrustes: scale + rotasyon, ceviri yok).
    GPS'siz donemde bu kalibre edilmis delta'lar kumulatif olarak toplanir.
  - Z: Track edilen harita noktalarinin medyan derinligi (mono_stream'in
    4. ciktisi) ile GT Z arasinda GPS'li donemde dogrusal regresyon
    (z = a*depth + b) ogrenilir. GPS'siz donemde bu regresyon dogrudan
    uygulanir (kumulatif degil, her karede bagimsiz tahmin).

    NEDEN BU YONTEM SECILDI: Yarisma puanlamasi (Sartname 9.2, Denklem 2)
    SADECE ORTALAMA hatayi kullaniyor. Derinlik tabanli Z, donmus Z'ye
    gore ORTALAMA hatada daha iyi (10.71m vs 11.21-11.30m) - max hata
    biraz daha yuksek olsa da (puanlamaya girmiyor) resmi kritere gore
    daha iyi secim budur. Detayli deney kayitlari DEVAM_NOTU.md'de.
"""

import numpy as np
from collections import deque

from gorev2.orb_slam3_bridge import OrbSlam3Bridge


def procrustes_scale_rotation(src_vecs, dst_vecs):
    n, dim = src_vecs.shape
    cov = (dst_vecs.T @ src_vecs) / n

    U, D, Vt = np.linalg.svd(cov)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt

    var_src = (src_vecs ** 2).sum() / n
    if var_src < 1e-12:
        return 1.0, np.eye(dim)
    scale = np.trace(np.diag(D) @ S) / var_src

    return scale, R


class PositionEstimatorORB:
    def __init__(self, bridge: OrbSlam3Bridge, calib_buffer_size: int = 450,
                 min_calib_samples: int = 30, refit_every: int = 10):
        self.bridge = bridge

        self.pos_global = np.zeros(3)
        self.scale = 1.0
        self.R = np.eye(3)

        self._src_deltas = deque(maxlen=calib_buffer_size)
        self._dst_deltas = deque(maxlen=calib_buffer_size)

        # Derinlik -> Z regresyonu icin (a, b): z = a*depth + b
        self._depth_samples = deque(maxlen=calib_buffer_size)
        self._z_samples = deque(maxlen=calib_buffer_size)
        self.depth_a = 0.0
        self.depth_b = 0.0
        self.depth_calibrated = False

        self.min_calib_samples = min_calib_samples
        self.refit_every = refit_every
        self._frames_since_refit = 0

        self.prev_slam_pos = None
        self.prev_gps = None
        self.was_healthy = False
        self.is_calibrated = False

    def _refit_calibration(self):
        if len(self._src_deltas) >= self.min_calib_samples:
            src = np.array(self._src_deltas)
            dst = np.array(self._dst_deltas)
            try:
                self.scale, self.R = procrustes_scale_rotation(src, dst)
                self.is_calibrated = True
            except np.linalg.LinAlgError:
                pass

        if len(self._depth_samples) >= self.min_calib_samples:
            X = np.array(self._depth_samples)
            y = np.array(self._z_samples)
            A_fit = np.vstack([X, np.ones_like(X)]).T
            try:
                coef, *_ = np.linalg.lstsq(A_fit, y, rcond=None)
                self.depth_a, self.depth_b = coef
                self.depth_calibrated = True
            except np.linalg.LinAlgError:
                pass

    def update(self, frame, gps):
        """
        frame: BGR numpy array.
        gps: {"gps_health_status": 0|1, "translation_x/y/z": float} ya da None.
        Donus: (x, y, z) - guncel pozisyon tahmini (metre).
        """
        healthy_gps = bool(gps and int(gps.get("gps_health_status", 0)) == 1)

        result = self.bridge.push_frame(frame)   # (x,y,z,depth) veya None
        if result is not None:
            slam_pos = result[:3]
            depth = result[3]
        else:
            slam_pos = None
            depth = None

        if healthy_gps:
            cur_gps = np.array([
                float(gps["translation_x"]),
                float(gps["translation_y"]),
                float(gps["translation_z"]),
            ])

            if (slam_pos is not None and self.prev_slam_pos is not None
                    and self.was_healthy and self.prev_gps is not None):
                d_slam = np.array(slam_pos) - np.array(self.prev_slam_pos)
                d_gt = cur_gps - self.prev_gps
                if 0.05 < np.linalg.norm(d_gt) < 10.0:
                    self._src_deltas.append(d_slam)
                    self._dst_deltas.append(d_gt)

            if depth is not None and depth > 0:
                self._depth_samples.append(depth)
                self._z_samples.append(cur_gps[2])

            self._frames_since_refit += 1
            if self._frames_since_refit >= self.refit_every:
                self._refit_calibration()
                self._frames_since_refit = 0

            self.pos_global = cur_gps.copy()
            self.prev_gps = cur_gps.copy()
            self.was_healthy = True

        else:
            if (slam_pos is not None and self.prev_slam_pos is not None
                    and self.is_calibrated):
                d_slam = np.array(slam_pos) - np.array(self.prev_slam_pos)
                d_metric = self.scale * (self.R @ d_slam)
                self.pos_global[0] += d_metric[0]
                self.pos_global[1] += d_metric[1]

            if depth is not None and depth > 0 and self.depth_calibrated:
                self.pos_global[2] = self.depth_a * depth + self.depth_b
            # depth yoksa/kalibre degilse Z son degerinde kalir (fallback)

            self.was_healthy = False

        if slam_pos is not None:
            self.prev_slam_pos = slam_pos

        return float(self.pos_global[0]), float(self.pos_global[1]), float(self.pos_global[2])
