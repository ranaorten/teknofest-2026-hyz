"""
ORB-SLAM3'un gercek-zamanli C++ wrapper'i (mono_stream v3, ham byte
protokolu + medyan derinlik) ile subprocess uzerinden haberlesen bridge.
"""

import subprocess
import struct
import numpy as np
import cv2


class OrbSlam3Bridge:
    def __init__(self, vocab_path, settings_path, executable_path, fps=7.4925):
        self.vocab_path = vocab_path
        self.settings_path = settings_path
        self.executable_path = executable_path
        self.fps = fps
        self.proc = None

    def start(self):
        self.proc = subprocess.Popen(
            [self.executable_path, self.vocab_path, self.settings_path, str(self.fps)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=None,
            bufsize=0,
        )

    def push_frame(self, frame_bgr):
        """
        Donus: (x, y, z, median_depth) tuple veya None (tracking kaybi).
        """
        if self.proc is None or self.proc.poll() is not None:
            raise RuntimeError("mono_stream subprocess calismiyor - once start() cagirin")

        if not frame_bgr.flags["C_CONTIGUOUS"]:
            frame_bgr = np.ascontiguousarray(frame_bgr)

        height, width = frame_bgr.shape[:2]
        header = struct.pack("<ii", width, height)

        self.proc.stdin.write(header)
        self.proc.stdin.write(frame_bgr.tobytes())
        self.proc.stdin.flush()

        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("mono_stream beklenmedik sekilde kapandi (stdout bos)")

        line = line.decode("utf-8").strip()
        if line == "LOST":
            return None

        parts = line.split()
        if len(parts) != 4:
            raise RuntimeError(f"Beklenmeyen mono_stream ciktisi: {line!r}")

        x, y, z, depth = (float(p) for p in parts)
        return (x, y, z, depth)

    def stop(self):
        if self.proc is None:
            return
        try:
            self.proc.stdin.write(struct.pack("<i", 0))
            self.proc.stdin.flush()
            self.proc.stdin.close()
            self.proc.wait(timeout=10)
        except Exception:
            self.proc.kill()
        finally:
            self.proc = None


if __name__ == "__main__":
    import sys
    import time

    if len(sys.argv) < 4:
        print("Kullanim: python3 orb_slam3_bridge.py <vocab> <settings> <video_yolu> [n_frames]")
        sys.exit(1)

    vocab, settings, video_path = sys.argv[1], sys.argv[2], sys.argv[3]
    n_frames = int(sys.argv[4]) if len(sys.argv) > 4 else 20

    bridge = OrbSlam3Bridge(
        vocab_path=vocab,
        settings_path=settings,
        executable_path="/home/rana/ORB_SLAM3/Examples/Monocular/mono_stream",
        fps=7.4925,
    )
    bridge.start()
    time.sleep(2)

    cap = cv2.VideoCapture(video_path)
    skip = 4
    frame_idx = 0
    sent = 0
    t0 = time.time()

    while cap.isOpened() and sent < n_frames:
        ret, frame = cap.read()
        if not ret:
            break
        if frame_idx % skip == 0:
            result = bridge.push_frame(frame)
            print(f"kare {sent}: {result}")
            sent += 1
        frame_idx += 1

    cap.release()
    bridge.stop()

    elapsed = time.time() - t0
    print(f"\n{sent} kare, {elapsed:.2f}s ({sent/elapsed:.2f} kare/s)")
