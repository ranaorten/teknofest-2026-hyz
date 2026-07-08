import cv2
import json
import numpy as np
from collections import deque
from dataclasses import dataclass

@dataclass
class Motion:
    dx_px: float        # X eksenindeki medyan piksel kayması
    dy_px: float        # Y eksenindeki medyan piksel kayması
    dtheta: float       # Tahmini Yaw değişimi (radyan)
    is_valid: bool      # Güvenilirlik durumu
    method: str         # "klt_essential" | "static" | "failed"

class VisualOdometry:
    def __init__(self, calib_path: str = None, calib_key: str = "rgb_1080p"):
        self.K, self.dist = self._load_calibration(calib_path, calib_key)
        self._undistort_maps = None
        
        self.prev_gray = None
        self.prev_pts = None

    def _load_calibration(self, path, key):
        if not path:
            return np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float64), np.zeros(5)
        try:
            with open(path) as f:
                c = json.load(f)[key]
            K = np.array([[c["fx"], 0, c["cx"]], [0, c["fy"], c["cy"]], [0, 0, 1]], dtype=np.float64)
            dist = np.array(c.get("radial_distortion", [0,0,0]) + c.get("tangential_distortion", [0,0]), dtype=np.float64)
            return K, dist
        except Exception:
            return np.array([[1000, 0, 960], [0, 1000, 540], [0, 0, 1]], dtype=np.float64), np.zeros(5)

    def _prepare_frame(self, frame):
        if self._undistort_maps is None:
            h, w = frame.shape[:2]
            self._undistort_maps = cv2.initUndistortRectifyMap(self.K, self.dist, None, self.K, (w, h), cv2.CV_16SC2)
        
        frame = cv2.remap(frame, self._undistort_maps[0], self._undistort_maps[1], cv2.INTER_LINEAR)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame.ndim == 3 else frame
        
        # Sadece karanlıksa CLAHE
        if np.mean(gray) < 80:
            gray = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8)).apply(gray)
        return gray

    def process(self, frame) -> Motion:
        gray = self._prepare_frame(frame)

        if self.prev_gray is None:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            return Motion(0.0, 0.0, 0.0, True, "first_frame")

        if self.prev_pts is None or len(self.prev_pts) < 10:
            self.prev_pts = cv2.goodFeaturesToTrack(self.prev_gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            if self.prev_pts is None:
                return Motion(0.0, 0.0, 0.0, False, "failed")

        # KLT Optik Akış ile noktaları takip et
        curr_pts, status, err = cv2.calcOpticalFlowPyrLK(self.prev_gray, gray, self.prev_pts, None)
        
        good_prev = self.prev_pts[status == 1]
        good_curr = curr_pts[status == 1]

        if len(good_curr) < 20:
            self.prev_gray = gray
            self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
            return Motion(0.0, 0.0, 0.0, False, "failed")

        # Statik kontrol: Gürültüyü hareket zannetmemek için
        pixel_diffs = np.linalg.norm(good_curr - good_prev, axis=1)
        if np.median(pixel_diffs) < 0.5:
            self.prev_gray = gray
            return Motion(0.0, 0.0, 0.0, True, "static")

        # Essential Matrix ile outlier'ları temizle ve Yaw bul
        E, mask = cv2.findEssentialMat(good_prev, good_curr, self.K, cv2.RANSAC, 0.999, 1.0)
        
        if E is not None and mask is not None:
            _, R, _, _ = cv2.recoverPose(E, good_prev, good_curr, self.K, mask=mask)
            yaw = np.arctan2(R[1, 0], R[0, 0])
            
            # SADECE inlier noktalardan medyan piksel kaymasını hesapla (Scale problemini çözer)
            inlier_prev = good_prev[mask.ravel() == 1]
            inlier_curr = good_curr[mask.ravel() == 1]
            
            if len(inlier_curr) > 0:
                dx_px = np.median(inlier_curr[:, 0] - inlier_prev[:, 0])
                dy_px = np.median(inlier_curr[:, 1] - inlier_prev[:, 1])
                
                self.prev_gray = gray
                self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
                return Motion(dx_px, dy_px, yaw, True, "klt_essential")

        # Fallback
        self.prev_gray = gray
        self.prev_pts = cv2.goodFeaturesToTrack(gray, maxCorners=1000, qualityLevel=0.01, minDistance=10)
        return Motion(0.0, 0.0, 0.0, False, "failed")

class PositionEstimator:
    def __init__(self, vo: VisualOdometry, buffer_size=300):
        self.vo = vo
        self.pos_global = np.zeros(3)
        self.yaw_global = 0.0
        
        self._X_buf = deque(maxlen=buffer_size)
        self._Y_buf = deque(maxlen=buffer_size)
        
        # Kamera Piksel -> GPS Metre Dönüşüm Matrisi
        self.A = np.eye(2) * -0.001 
        
        self.prev_gps = None
        self.was_healthy = False
        
        # Z ekseni için basit fallback
        self.last_healthy_z = 10.0

    def _update_online_calibration(self):
        if len(self._X_buf) < 30:
            return
            
        X = np.hstack(self._X_buf)  # (2, N)
        Y = np.hstack(self._Y_buf)  # (2, N)
        
        try:
            # A = Y * X^T * (X * X^T)^-1
            X_XT_inv = np.linalg.pinv(X @ X.T)
            self.A = Y @ X.T @ X_XT_inv
        except np.linalg.LinAlgError:
            pass

    def update(self, frame, gps: dict | None):
        motion = self.vo.process(frame)
        healthy_gps = bool(gps and int(gps.get("gps_health_status", 0)) == 1)

        # Yaw Entegrasyonu
        if motion.is_valid:
            self.yaw_global += motion.dtheta
            
        current_z = float(gps["translation_z"]) if healthy_gps else self.last_healthy_z

        if motion.is_valid and motion.method != "static":
            # 1. Pikselleri Yükseklik (Z) ile ölçekle
            # Z yüksekliğindeki bir cismin piksel hareketi Z ile orantılı metrik hareket yaratır
            scaled_shift = np.array([[motion.dx_px * current_z], 
                                     [motion.dy_px * current_z]])
            
            # 2. Global eksene döndür (Sadece 2D Yaw rotasyonu)
            c, s = np.cos(self.yaw_global), np.sin(self.yaw_global)
            R_yaw = np.array([[c, -s], 
                              [s,  c]])
            
            rotated_shift = R_yaw @ scaled_shift

            if healthy_gps and self.prev_gps is not None and self.was_healthy:
                cur_gps = np.array([float(gps["translation_x"]), float(gps["translation_y"]), float(gps["translation_z"])])
                d_gps_xy = np.array([[cur_gps[0] - self.prev_gps[0]], 
                                     [cur_gps[1] - self.prev_gps[1]]])
                
                # Sıçrama ve gürültü filtresi
                if 0.05 < np.linalg.norm(d_gps_xy) < 10.0:
                    self._X_buf.append(rotated_shift)
                    self._Y_buf.append(d_gps_xy)
                    self._update_online_calibration()

            elif not healthy_gps:
                # DEAD RECKONING
                # Öğrendiğimiz A matrisini dönmüş ve ölçeklenmiş piksel kaymasına uygula
                metric_dxy = self.A @ rotated_shift
                self.pos_global[0] += float(metric_dxy[0][0])
                self.pos_global[1] += float(metric_dxy[1][0])
                # GPS koptuğunda Z'yi sabit kabul ediyoruz (şimdilik)

        if healthy_gps:
            self.pos_global = np.array([float(gps["translation_x"]), float(gps["translation_y"]), float(gps["translation_z"])])
            self.prev_gps = self.pos_global.copy()
            self.last_healthy_z = current_z
            self.was_healthy = True
        else:
            self.was_healthy = False

        return float(self.pos_global[0]), float(self.pos_global[1]), float(self.pos_global[2])