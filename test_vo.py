"""VO modulunu gercek video uzerinde test eder (ilk 200 kare @7.5fps)."""
import sys
sys.path.insert(0, ".")
from common.video_loader import VideoLoader
from gorev2.visual_odometry import VisualOdometry
import time

loader = VideoLoader("/home/rana/Downloads/THYZ_2026_Ornek_Veri_1.MP4", target_fps=7.5)
vo = VisualOdometry()

cum_x, cum_y, cum_scale = 0.0, 0.0, 1.0
methods = {}
t0 = time.time()

for i in range(200):
    result = loader.next_frame()
    if result is None:
        break
    idx, frame = result
    m = vo.process(frame)
    cum_x += m.dx
    cum_y += m.dy
    cum_scale *= m.scale
    methods[m.method] = methods.get(m.method, 0) + 1
    if idx % 50 == 0:
        print(f"Kare {idx}: dx={m.dx:+.1f} dy={m.dy:+.1f} scale={m.scale:.4f} "
              f"yontem={m.method} inlier={m.n_inliers}")

elapsed = time.time() - t0
loader.release()

print(f"\n--- OZET ---")
print(f"Islenen kare: {sum(methods.values())}, sure: {elapsed:.1f}s "
      f"({sum(methods.values())/elapsed:.1f} kare/sn)")
print(f"Yontem dagilimi: {methods}")
print(f"Kumulatif kayma (px): x={cum_x:.1f}, y={cum_y:.1f}")
print(f"Kumulatif olcek: {cum_scale:.3f} (1'den kucuk=yukseldi, buyuk=alcaldi)")