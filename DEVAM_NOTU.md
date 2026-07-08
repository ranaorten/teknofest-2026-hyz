# Devam Notu - ORB-SLAM3 Hibrit Denemesi

## Şu anki durum
- `orbslam3-migration` branch'indeyiz (classical pipeline `classical-vo-baseline` tag'inde korunuyor)
- ORB-SLAM3 kuruldu ve çalışıyor: `~/ORB_SLAM3/` (loop closure System.cc:214'te kapatıldı)
- Kalibre edilmiş kamera dosyası: `~/ORB_SLAM3/my_camera.yaml`
- Test sekansı: `orbslam3_full_seq/` (2256 kare, gitignore'da - yeniden üretilebilir)
- Son trajectory: `~/ORB_SLAM3/KeyFrameTrajectory.txt` (754 keyframe, loop closure kapalı)

## Sonuç
Hibrit yaklaşım (ORB-SLAM3 delta dead-reckoning X/Y + donmuş Z):
**11.08m ortalama, 23.3m max** (678 GPS'siz kare üzerinde)
Classical pipeline: ~11m ortalama, ~27m max
→ Şu an başa baş, net kazanç yok ama worst-case'de ORB-SLAM3 hibrit daha iyi.

## Sıradaki adım (kaldığımız yer)
Z modelini iyileştirmek. Denenenler:
- Donmuş Z (son bilinen değer): 11.08m — ŞU ANKİ EN İYİ
- Sabit hız extrapolasyonu (v_z=0.86 m/s ile): 103m — BAŞARISIZ, kullanma

Denenmemiş fikirler:
- Kısa pencereli/damped hız modeli (v_z zamanla sönümlenerek 0'a gitsin)
- Kalman filtresi (Z için sabit-ivme modeli, GPS'siz dönemde sadece predict)
- ORB-SLAM3'ün kendi Z deltalarını (24.8m hataya sebep olan) düzeltmek yerine
  farklı bir sinyalle (örn. barometrik olmasa da optik akış büyüklüğü) harmanlamak

## Komutları tekrar çalıştırmak için
```bash
source ~/Desktop/teknofest-2026-hyz/orbslam3_env/bin/activate
cd ~/Desktop/teknofest-2026-hyz
# Kalibrasyon + hibrit test scriptleri: compute_calibration.py,
# relative_delta_dead_reckoning.py, evaluate_full_trajectory.py
```

## Güncelleme: Z modeli deneyleri tamamlandı (doğru metodoloji ile)

Denenen tüm Z stratejileri (GPS'siz bölge, gerçek test):
- Donmuş Z: **11.08m** (EN İYİ)
- Sabit hız extrapolasyonu: 103m (başarısız)
- Sönümlenmiş hız (τ=1..20, doğru train/val ayrımıyla seçildi): 12.21m (donmuş Zden kötü)

**Kesin bulgu:** GPS'li 60s'lik kalibrasyon penceresinde Z varyansı çok düşük (std=0.57m),
bu yüzden hiçbir hız/ivme tabanlı Z modeli güvenilir öğrenilemiyor. Doğru train/val
ayrımıyla bile en iyi τ seçimi gerçek GPS'siz bölgede donmuş Z'den kötü performans verdi.
Donmuş Z, tembellik değil, veri kısıtı altında istatistiksel olarak en savunulabilir seçim.

**Bu oturumun nihai sonucu:** ORB-SLAM3 hibrit (delta dead-reckoning X/Y + donmuş Z)
= 11.08m ortalama / 23.3m max, classical pipeline (~11m / ~27m) ile pratik eşitlik,
worst-case'de üstünlük. Buradan sonraki olası yönler: (1) barometrik/IMU gibi
ek sensör sinyali olmadan Z'de daha fazla ilerleme zor görünüyor, (2) X/Y kazancını
(1.4m/2.2m) production koduna taşımak tek başına değerli olabilir.
