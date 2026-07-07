"""
Video yükleyici: Videoyu açar, kareleri sıralı olarak tek tek verir.

Yarışmada kareler sunucudan tek tek alınacağı için (toplu indirme
yasak), aynı akış mantığını burada taklit ediyoruz. Böylece yarışma
günü sadece bu sınıfı server_io ile değiştirmek yeterli olacak.
"""
import cv2


class VideoLoader:
    def __init__(self, video_path: str, target_fps: float = None):
        self.cap = cv2.VideoCapture(video_path)
        if not self.cap.isOpened():
            raise FileNotFoundError(f"Video acilamadi: {video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.current_index = -1

        # Yarışma 7.5 FPS verecek; ham video daha yüksek FPS ise kare atla
        self.skip = 1
        if target_fps is not None and self.fps > target_fps:
            self.skip = round(self.fps / target_fps)


    def info(self) -> dict:
        return {
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "frame_count": self.frame_count,
        }

    def next_frame(self):
        """Sıradaki kareyi döner: (frame_index, frame) ya da video bittiyse None."""
        frame = None
        for _ in range(self.skip):
            ret, f = self.cap.read()
            if not ret:
                return None
            frame = f
        self.current_index += 1
        return self.current_index, frame

    def release(self):
        self.cap.release()


if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Kullanim: python3 common/video_loader.py <video_yolu>")
        sys.exit(1)

    loader = VideoLoader(sys.argv[1], target_fps=7.5)
    print("Video bilgileri:", loader.info())
    print(f"Kare atlama (skip): {loader.skip} -> efektif FPS: {loader.fps / loader.skip:.2f}")
    # İlk 5 kareyi okuyup boyutlarını yazdıralım (test)
    for _ in range(5):
        result = loader.next_frame()
        if result is None:
            print("Video bitti.")
            break
        idx, frame = result
        print(f"Kare {idx}: shape={frame.shape}")

    loader.release()