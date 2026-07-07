import numpy as np
from collections import deque

class PositionEstimator:
    def __init__(self, vo, buffer_size=300, min_calib_samples=30):
        self.vo = vo
        self.min_calib_samples = min_calib_samples
        
        # Global Pozisyon ve Oryantasyon
        self.pos_global = np.zeros(3)      # [x, y, z] (Metre)
        self.R_global = np.eye(3)          # 3x3 Global Rotation Matrix
        
        # Çevrimiçi Kalibrasyon Havuzu (Sliding Window)
        self.calib_buffer_size = buffer_size
        self._vo_xy = deque(maxlen=buffer_size)   # VO'dan gelen [tx, ty]
        self._gps_dxy = deque(maxlen=buffer_size) # GPS'ten gelen [dx, dy]
        self._vo_z = deque(maxlen=buffer_size)    # VO'dan gelen tz
        self._gps_dz = deque(maxlen=buffer_size)  # GPS'ten gelen dz
        
        # Öğrenilecek Kalibrasyon Parametreleri
        self.A = np.eye(2) * 0.05  # 2x2 (px -> metre) X, Y ekseni için
        self.alpha_z = 0.05        # Z ekseni ölçek katsayısı
        self.beta_z = 0.0          # Z ekseni sabit sapması
        
        # Durum Değişkenleri
        self.prev_gps = None
        self.was_healthy = False
        self.frame_count = 0

    def _update_online_calibration(self):
        """ 
        GPS sağlıklıyken her karede piksel->metre dönüşüm matrisini (A) 
        ve Z ekseni katsayılarını günceller.
        """
        if len(self._vo_xy) < self.min_calib_samples:
            return

        # --- X ve Y ekseni için A Matrisinin Öğrenilmesi ---
        # GPS_dxy = A @ VO_xy denklemini çözer
        X_xy = np.array(self._vo_xy).T       # (2, N) matrisi
        Y_xy = np.array(self._gps_dxy).T     # (2, N) matrisi
        
        try:
            # Pseudo-inverse kullanarak güvenli çözüm: A = Y * X^T * (X * X^T)^-1
            X_XT_inv = np.linalg.pinv(X_xy @ X_xy.T)
            self.A = Y_xy @ X_xy.T @ X_XT_inv
        except np.linalg.LinAlgError:
            pass

        # --- Z ekseni için Doğrusal Regresyon ---
        # GPS_dz = alpha_z * VO_tz + beta_z
        X_z = np.vstack([self._vo_z, np.ones(len(self._vo_z))]).T  # (N, 2)
        Y_z = np.array(self._gps_dz).reshape(-1, 1)                # (N, 1)
        
        try:
            coeff, *_ = np.linalg.lstsq(X_z, Y_z, rcond=None)
            self.alpha_z = float(coeff[0][0])
            self.beta_z = float(coeff[1][0])
        except np.linalg.LinAlgError:
            pass

    def update(self, frame, gps: dict | None):
        # 1. Visual Odometry'den göreli (relative) hareketi al
        motion = self.vo.process(frame)
        self.frame_count += 1

        healthy_gps = bool(gps and int(gps.get("gps_health_status", 0)) == 1)
        
        # 2. Global Oryantasyonu Güncelle
        # R_global = R_global @ R_delta
        if motion.is_valid and motion.method != "static":
            self.R_global = self.R_global @ motion.R
            
        # Göreli ötelemeyi global eksene hizala (Henüz ölçeksiz, piksel/keyfi birimde)
        # t_global = R_global @ t_relative
        t_global = self.R_global @ motion.t
        vo_tx, vo_ty, vo_tz = float(t_global[0][0]), float(t_global[1][0]), float(t_global[2][0])

        if healthy_gps:
            # GPS Verisini al
            cur_gps = np.array([
                float(gps["translation_x"]),
                float(gps["translation_y"]),
                float(gps["translation_z"])
            ])
            
            # Kalibrasyon havuzunu doldur
            if self.prev_gps is not None and self.was_healthy and motion.is_valid and motion.method != "static":
                d_gps = cur_gps - self.prev_gps
                
                # Sadece makul hareketleri (gürültü olmayan) havuza ekle
                if 0.1 < np.linalg.norm(d_gps) < 50.0:
                    self._vo_xy.append([vo_tx, vo_ty])
                    self._gps_dxy.append([d_gps[0], d_gps[1]])
                    self._vo_z.append(vo_tz)
                    self._gps_dz.append(d_gps[2])
                    
                    # Havuz güncellendikçe modeli kalibre et
                    self._update_online_calibration()

            # Pozisyonu GPS ile düzelt
            self.pos_global = cur_gps.copy()
            self.prev_gps = cur_gps.copy()
            self.was_healthy = True

        else:
            self.was_healthy = False
            
            # --- DEAD RECKONING (GPS Kesintisi) ---
            if motion.is_valid and motion.method != "static":
                vo_xy_arr = np.array([[vo_tx], [vo_ty]])
                
                # Öğrenilen A matrisi ile pikselden metreye X-Y dönüşümü
                est_dxy = self.A @ vo_xy_arr
                
                # Öğrenilen katsayılar ile Z dönüşümü
                est_dz = self.alpha_z * vo_tz + self.beta_z
                
                # Tahmini hareketi global pozisyona ekle
                self.pos_global[0] += float(est_dxy[0][0])
                self.pos_global[1] += float(est_dxy[1][0])
                self.pos_global[2] += est_dz

        return float(self.pos_global[0]), float(self.pos_global[1]), float(self.pos_global[2])