"""
Piksel -> metre online kalibrasyon (irtifa-bagimli dinamik olcek).

Fizik: metre/piksel orani = Z / f  (Z: yerden yukseklik, f: odak uzakligi)
Yani drone yukseldikce ayni piksel hareketi daha fazla metreye karsilik gelir.

Kalibrasyon doneminde:
  - A matrisi ogrenilir (kalibrasyon irtifasindaki px->m donusumu)
  - k_z ogrenilir: dz = k_z * log(scale). Fiziksel olarak k_z ~ mutlak
    irtifa Z0'i kodlar (dz = Z * dlog(scale) iliskisinden).

Kesinti doneminde:
  - Anlik irtifa takip edilir: Z = Z0 + (kumulatif dz)
  - A matrisi anlik irtifayla olceklenir: A_etkin = A * (Z / Z0)
"""
import numpy as np


class PixelToMeterCalibration:
    def __init__(self):
        self.A = None          # 2x2 donusum matrisi (kalibrasyon irtifasinda)
        self.k_z = None        # log(scale) -> dz katsayisi (~ mutlak irtifa Z0)
        self.Z0 = None         # kalibrasyon donemindeki etkin mutlak irtifa
        self.z_rel = 0.0       # kalibrasyon sonundan itibaren goreli z değişimi
        self._px = []
        self._scale = []
        self._m = []

    def add_sample(self, dx_px, dy_px, scale, dx_m, dy_m, dz_m):
        """Saglikli donemde her kare icin cagrilir."""
        self._px.append((dx_px, dy_px))
        self._scale.append(scale)
        self._m.append((dx_m, dy_m, dz_m))

    def fit(self, min_samples: int = 30) -> bool:
        if len(self._px) < min_samples:
            return False

        P = np.array(self._px)
        M = np.array(self._m)
        S = np.log(np.clip(self._scale, 1e-6, None))

        At, *_ = np.linalg.lstsq(P, M[:, :2], rcond=None)
        self.A = At.T

        denom = float(S @ S)
        self.k_z = float(S @ M[:, 2]) / denom if denom > 1e-12 else 0.0

        # Mutlak irtifa kestirimi: |k_z| ~ Z0.
        # (dz = Z*dlog(scale); z ekseni asagi-pozitifse k_z negatif cikar,
        #  isaretten bagimsiz buyuklugu irtifadir.)
        self.Z0 = max(abs(self.k_z), 1.0)   # 0'a bolunmeyi onle
        self.z_rel = 0.0
        return True

    def is_ready(self) -> bool:
        return self.A is not None

    def predict(self, dx_px, dy_px, scale):
        """Kare-arasi piksel hareketini metreye cevirir (irtifa-duyarli)."""
        if not self.is_ready():
            return 0.0, 0.0, 0.0

        Z_now = max(self.Z0 + self.z_rel, 1.0)
        ratio = Z_now / self.Z0

        xy = (self.A * ratio) @ np.array([dx_px, dy_px])
        dz = self.k_z * np.log(max(scale, 1e-6))

        # Irtifa takibi: scale < 1 => zemin kuculuyor => yukseliyoruz.
        # log(scale) negatifken irtifa artmali:
        self.z_rel += -np.log(max(scale, 1e-6)) * Z_now

        return float(xy[0]), float(xy[1]), float(dz)