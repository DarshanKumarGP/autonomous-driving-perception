"""YOLO11 ONNX detector (CPU, onnxruntime). No torch needed, so it also runs in the ROS2 container."""
import argparse
import time

import cv2
import numpy as np
import onnxruntime as ort

NAMES = ["person", "rider", "car", "truck", "bus", "train", "motor", "bike", "traffic light", "traffic sign"]


class OnnxYolo:
    def __init__(self, path, imgsz=640, conf=0.35, iou=0.5, threads=0, classes=None):
        so = ort.SessionOptions()
        if threads:
            so.intra_op_num_threads = int(threads)
        self.sess = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        self.inp = self.sess.get_inputs()[0].name
        self.imgsz, self.conf, self.iou = imgsz, conf, iou
        self.cls_arr = None if classes is None else np.array(list(classes))
        self.last_infer_ms = 0.0

    def _prep(self, bgr):
        h, w = bgr.shape[:2]
        r = min(self.imgsz / h, self.imgsz / w)
        nh, nw = int(round(h * r)), int(round(w * r))
        img = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_LINEAR)
        top, left = (self.imgsz - nh) // 2, (self.imgsz - nw) // 2
        canvas = np.full((self.imgsz, self.imgsz, 3), 114, np.uint8)
        canvas[top:top + nh, left:left + nw] = img
        x = np.ascontiguousarray(canvas[:, :, ::-1].transpose(2, 0, 1)[None]).astype(np.float32) / 255.0
        return x, r, left, top

    def detect(self, bgr):
        """Returns a list of (x1, y1, x2, y2, score, class_id) in the pixel coordinates of bgr."""
        h, w = bgr.shape[:2]
        x, r, left, top = self._prep(bgr)
        t0 = time.perf_counter()
        out = self.sess.run(None, {self.inp: x})[0]
        self.last_infer_ms = (time.perf_counter() - t0) * 1000
        p = out[0].T                                   # (8400, 4 + number of classes)
        scores_all = p[:, 4:]
        cls = scores_all.argmax(1)
        score = scores_all[np.arange(len(p)), cls]
        keep = score >= self.conf
        if self.cls_arr is not None:
            keep &= np.isin(cls, self.cls_arr)
        p, cls, score = p[keep], cls[keep], score[keep]
        if len(p) == 0:
            return []
        cx, cy, bw, bh = p[:, 0], p[:, 1], p[:, 2], p[:, 3]
        x1 = np.clip((cx - bw / 2 - left) / r, 0, w - 1)
        y1 = np.clip((cy - bh / 2 - top) / r, 0, h - 1)
        x2 = np.clip((cx + bw / 2 - left) / r, 0, w - 1)
        y2 = np.clip((cy + bh / 2 - top) / r, 0, h - 1)
        dets = []
        for c in np.unique(cls):
            m = np.where(cls == c)[0]
            boxes = [[float(x1[i]), float(y1[i]), float(x2[i] - x1[i]), float(y2[i] - y1[i])] for i in m]
            idx = cv2.dnn.NMSBoxes(boxes, [float(score[i]) for i in m], self.conf, self.iou)
            for k in np.array(idx).reshape(-1):
                i = m[int(k)]
                dets.append((float(x1[i]), float(y1[i]), float(x2[i]), float(y2[i]), float(score[i]), int(c)))
        return dets


def draw(frame, dets):
    out = frame.copy()
    for x1, y1, x2, y2, s, c in dets:
        cv2.rectangle(out, (int(x1), int(y1)), (int(x2), int(y2)), (0, 220, 0), 2)
        cv2.putText(out, "%s %.2f" % (NAMES[c], s), (int(x1), max(12, int(y1) - 4)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 220, 0), 1)
    return out


def main():
    ap = argparse.ArgumentParser(description="ONNX YOLO CPU check")
    ap.add_argument("--model", required=True)
    ap.add_argument("--video")
    ap.add_argument("--image")
    ap.add_argument("--runs", type=int, default=30)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.35)
    ap.add_argument("--save")
    a = ap.parse_args()
    det = OnnxYolo(a.model, conf=a.conf, threads=a.threads)
    if a.image:
        frame = cv2.imread(a.image)
    else:
        cap = cv2.VideoCapture(a.video)
        cap.set(cv2.CAP_PROP_POS_FRAMES, 60)
        ok, frame = cap.read()
        cap.release()
    assert frame is not None, "could not read an image or frame"
    for _ in range(3):
        det.detect(frame)
    tot, inf = [], []
    for _ in range(a.runs):
        t0 = time.perf_counter()
        dets = det.detect(frame)
        tot.append((time.perf_counter() - t0) * 1000)
        inf.append(det.last_infer_ms)
    counts = {}
    for d in dets:
        counts[NAMES[d[5]]] = counts.get(NAMES[d[5]], 0) + 1
    print("onnxruntime", ort.__version__, "| providers in use:", det.sess.get_providers())
    print("frame size:", frame.shape[1], "x", frame.shape[0], "| detections:", len(dets), counts)
    print("model-only ms: median %.1f, p95 %.1f" % (np.median(inf), np.percentile(inf, 95)))
    print("total ms (letterbox + model + NMS): median %.1f, p95 %.1f -> %.1f FPS" % (
        np.median(tot), np.percentile(tot, 95), 1000 / np.median(tot)))
    if a.save:
        cv2.imwrite(a.save, draw(frame, dets))
        print("saved", a.save)


if __name__ == "__main__":
    main()
