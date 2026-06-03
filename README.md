# TEKNOFEST 2026 — Havacılıkta Yapay Zekâ

GPS-denied navigasyon, hava aracı nesne tespiti ve görüntü eşleştirme görevleri için geliştirilmiş yapay zekâ sistemi.

---

## Görevler

### Görev 1 — Nesne Tespiti (%25)

Hava aracı kamerası görüntüsünden 4 nesne sınıfının tespiti:

| Sınıf | Ek Bildirim |
|-------|-------------|
| Taşıt | `motion_status`: hareketli / hareketsiz |
| İnsan | — |
| UAP (Uçan Araba Parkı) | `landing_status`: iniş uygun / değil |
| UAİ (Uçan Ambulans İniş) | `landing_status`: iniş uygun / değil |

**Puanlama:** mAP@IoU=0.5 — her kare için yalnızca 1 tahmin gönderilmeli, fazlası false positive sayılır.

**Pipeline:**
- YOLOv8 / YOLOv9 nesne tespiti
- SAHI (Slicing Aided Hyper Inference) — küçük nesne tespiti için
- Optik akış — taşıt hareket durumu tespiti
- IoU ≥ 0.5 eşik kontrolü

---

### Görev 2 — GPS-Denied Pozisyon Kestirimi (%40)

GPS kesintiye girdiğinde yalnızca kamera görüntüsünden x/y/z pozisyon tahmini.

**Strateji: Hibrit GPS + Visual Odometry**

```
İlk 450 kare (health_status=1)  →  GPS değeri direkt kullanılır
450+ kare    (health_status=0)  →  Visual Odometry devreye girer
```

**Algoritma bileşenleri:**

| Bileşen | Yöntem |
|---------|--------|
| Ön işleme | CLAHE kontrast iyileştirme |
| Özellik tespiti | ORB (4000 nokta) |
| Özellik eşleştirme | BFMatcher + Lowe ratio test (%75) |
| Hareket kestirimi | Essential Matrix + RANSAC → recoverPose |
| Ölçek kalibrasyonu | GPS penceresi medyanı (metre/piksel) |
| Konum filtresi | 6-durum Kalman filtresi (konum + hız) |

**Kamera parametreleri:**
```
fx = 1389.7 px    fy = 1387.1 px
cx = 954.0  px    cy = 558.9  px
k1 = 0.1378       k2 = -0.2564
FPS = 30
```

**Geliştirme sonuçları:**

| Versiyon | Değişiklik | Ortalama Hata |
|----------|-----------|---------------|
| v1 | Temel VO | 26.8 m |
| v2 | GPS penceresi direkt kullanımı | 6.0 m |
| v3 | Kalman dt = 1/30 düzeltmesi | **0.82 m** |

**Puanlama formülü:**

```
E = (1/N) × Σ √((x̂ᵢ-xᵢ)² + (ŷᵢ-yᵢ)² + (ẑᵢ-zᵢ)²)
```

**Kullanım:**
```bash
python3 visual_odometry.py \
  --video VIDEO.MP4 \
  --gt ground_truth.csv \
  --output sonuclar/ \
  --gps_end 450
```

---

### Görev 3 — Görüntü Eşleştirme (%25)

Oturum başında verilen referans nesne görsellerinin, video akışında tespiti.

**Zorluklar:**
- Farklı kamera modalitesi (termal ↔ RGB)
- Farklı açı ve irtifa
- Uydu görüntüsü ↔ hava görüntüsü eşleştirme
- Görüntü işleme filtrelerinden geçmiş referanslar

**Planlanan yöntem:** ORB / SIFT descriptor matching + homografi tahmini

---

## Proje Yapısı

```
teknofest-2026-hyz/
├── visual_odometry.py   # Görev 2 — GPS-denied pozisyon kestirimi
├── README.md
└── sonuclar/            # CSV çıktıları ve grafikler (git'e eklenmez)
```

---

## Gereksinimler

```bash
pip install opencv-python numpy matplotlib
```

- Python 3.10+
- OpenCV 4.13+

---

## Puanlama Özeti

| Görev | Ağırlık | Metrik |
|-------|---------|--------|
| Görev 1 — Nesne Tespiti | %25 | mAP@0.5 |
| Görev 2 — Pozisyon Kestirimi | %40 | Ortalama 3D Öklid Hatası (m) |
| Görev 3 — Görüntü Eşleştirme | %25 | mAP@0.5 |
| Final Raporu | %5 | — |
| Sunum | %5 | — |
