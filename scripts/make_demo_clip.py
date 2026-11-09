"""Cut a short 720p demo clip from a long or high-resolution video."""
import argparse

import cv2

ap = argparse.ArgumentParser()
ap.add_argument("--src", required=True)
ap.add_argument("--dst", required=True)
ap.add_argument("--seconds", type=float, default=20)
ap.add_argument("--width", type=int, default=1280)
a = ap.parse_args()
cap = cv2.VideoCapture(a.src)
assert cap.isOpened(), "cannot open " + a.src
fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
w0, h0 = int(cap.get(3)), int(cap.get(4))
w = a.width - a.width % 2
h = int(round(h0 * w / w0 / 2) * 2)
vw = cv2.VideoWriter(a.dst, cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
n, limit = 0, int(fps * a.seconds)
while n < limit:
    ok, f = cap.read()
    if not ok:
        break
    vw.write(cv2.resize(f, (w, h), interpolation=cv2.INTER_AREA))
    n += 1
cap.release()
vw.release()
print("wrote", a.dst, ":", n, "frames at", round(fps, 1), "fps,", w, "x", h)
