"""ROS2 node: ONNX detector + IoU tracker + warning rule on the camera stream."""
import json
import os
import re
import sys
import time

sys.path.insert(0, "/app")
import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage, Image
from std_msgs.msg import String

from common import msg_to_frame
from iou_tracker import IouTracker
from onnx_yolo import NAMES, OnnxYolo
from warning_logic import DEFAULT_CORRIDOR, assess

ROAD_USERS = [0, 1, 2, 3, 4, 6, 7]   # person, rider, car, truck, bus, motor, bike
LEVELS = {0: ("CLEAR", (0, 140, 0)), 1: ("CAUTION: vehicle ahead", (0, 140, 255)), 2: ("BRAKE: obstacle close", (0, 0, 220))}


class PerceptionNode(Node):
    def __init__(self):
        super().__init__("perception_node")
        defaults = dict(model="/models/yolo11s_bdd.onnx", conf=0.35, threads=0, out_dir="/out",
                        name="demo", caution_y=0.55, brake_y=0.68)
        for k, v in defaults.items():
            self.declare_parameter(k, v)
        g = lambda k: self.get_parameter(k).value
        self.name, self.out_dir = str(g("name")), str(g("out_dir"))
        os.makedirs(self.out_dir, exist_ok=True)
        self.caution_y, self.brake_y = float(g("caution_y")), float(g("brake_y"))
        self.det = OnnxYolo(str(g("model")), conf=float(g("conf")), threads=int(g("threads")), classes=ROAD_USERS)
        self.tracker = IouTracker()
        self.log = open("%s/%s_frames.jsonl" % (self.out_dir, self.name), "w")
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.RELIABLE)
        qos10 = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(Image, "/camera/image_raw", self.on_image, qos)
        self.pub_img = self.create_publisher(CompressedImage, "/perception/annotated/compressed", qos10)
        self.pub_det = self.create_publisher(String, "/perception/detections", qos10)
        self.pub_warn = self.create_publisher(String, "/perception/warning", qos10)
        self.pub_status = self.create_publisher(String, "/perception/status", qos10)
        self.n, self.t_pipe, self.t_model, self.t_cb = 0, [], [], []
        self.t_rx, self.t_render, self.t_enc, self.t_pub = [], [], [], []
        self.level_frames, self.first_level, self.id_cls = {0: 0, 1: 0, 2: 0}, {1: None, 2: None}, {}
        self.finish_time, self.done = None, False
        self.beat()
        self.create_timer(2.0, self.beat)
        self.create_timer(0.5, self.announce)
        self.get_logger().info("perception ready: model %s, conf %.2f" % (g("model"), float(g("conf"))))

    def beat(self):
        try:
            open("/tmp/heartbeat", "w").close()
        except OSError:
            pass

    def announce(self):
        if self.finish_time is None:
            return
        age = time.time() - self.finish_time
        if age > 1.5:
            self.pub_status.publish(String(data="DONE"))
        if age > 15:
            self.done = True

    def render(self, frame, tracks, level, culprit, ms):
        out = frame.copy()
        h, w = out.shape[:2]
        poly = np.array([(int(x * w), int(y * h)) for x, y in DEFAULT_CORRIDOR], np.int32)
        cv2.polylines(out, [poly], True, (255, 255, 0), 1)
        for t in tracks:
            x1, y1, x2, y2 = [int(v) for v in t["box"]]
            hot = culprit is not None and t["id"] == culprit[0]
            col = (0, 0, 255) if (hot and level == 2) else ((0, 140, 255) if hot else (0, 220, 0))
            cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
            if (x2 - x1) >= 60 or hot: cv2.putText(out, "%s #%d" % (NAMES[t["cls"]], t["id"]), (x1, max(16, y1 - 6)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
        txt, col = LEVELS[level]
        cv2.rectangle(out, (0, 0), (w, 46), col, -1)
        cv2.putText(out, txt, (12, 32), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.putText(out, "%.0f ms/frame (CPU, ONNX Runtime)" % ms, (w - 360, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        cv2.putText(out, "recorded footage - warning only, no vehicle control (CARLA not used)", (10, h - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1)
        return out

    def publish_jpeg(self, out, stamp, fid):
        ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, 90])
        t_enc = time.perf_counter()
        if not ok:
            return t_enc
        cm = CompressedImage()
        cm.header.stamp = stamp
        cm.header.frame_id = fid
        cm.format = "jpeg"
        flat = buf.reshape(-1)
        try:
            cm.data = flat
        except (AssertionError, TypeError):
            cm.data = flat.tobytes()
        self.pub_img.publish(cm)
        return t_enc

    def on_image(self, msg):
        t0 = time.perf_counter()
        frame = msg_to_frame(msg)
        rx_ms = (time.perf_counter() - t0) * 1000
        fid = msg.header.frame_id
        idx = int(re.sub(r"\D", "", fid) or 0)
        last = fid.startswith("last")
        h, w = frame.shape[:2]
        dets = self.det.detect(frame)
        tracks = self.tracker.update(dets)
        level, culprit = assess(tracks, w, h, caution_y=self.caution_y, brake_y=self.brake_y)
        ms_pipe = (time.perf_counter() - t0) * 1000
        self.n += 1
        self.t_pipe.append(ms_pipe)
        self.t_model.append(self.det.last_infer_ms)
        self.t_rx.append(rx_ms)
        self.level_frames[level] += 1
        if level and self.first_level[level] is None:
            self.first_level[level] = idx
        for t in tracks:
            self.id_cls[t["id"]] = NAMES[t["cls"]]
        t1 = time.perf_counter()
        out = self.render(frame, tracks, level, culprit, ms_pipe)
        t2 = time.perf_counter()
        t_enc = self.publish_jpeg(out, msg.header.stamp, fid)
        t4 = time.perf_counter()
        self.t_render.append((t2 - t1) * 1000)
        self.t_enc.append((t_enc - t2) * 1000)
        self.t_pub.append((t4 - t_enc) * 1000)
        payload = dict(frame=idx, level=level, culprit=culprit, ms=round(ms_pipe, 1),
                       tracks=[dict(id=t["id"], cls=NAMES[t["cls"]], score=round(t["score"], 2),
                                    box=[int(round(v)) for v in t["box"]]) for t in tracks])
        text = json.dumps(payload)
        self.pub_det.publish(String(data=text))
        self.pub_warn.publish(String(data=LEVELS[level][0]))
        self.log.write(text + "\n")
        self.log.flush()
        self.t_cb.append((time.perf_counter() - t0) * 1000)
        if self.n % 10 == 0:
            self.get_logger().info("progress: %d frames processed, last callback %.0f ms" % (self.n, self.t_cb[-1]))
        if last:
            self.finish(idx)

    def finish(self, last_idx):
        published = last_idx + 1
        per_class = {}
        for c in self.id_cls.values():
            per_class[c] = per_class.get(c, 0) + 1
        med = lambda v: round(float(np.median(v)), 1)
        stats = dict(run=self.name, frames_published=published, frames_processed=self.n,
                     frames_dropped=max(0, published - self.n),
                     receive_ms_median=med(self.t_rx), model_ms_median=med(self.t_model),
                     pipeline_ms_median=med(self.t_pipe), pipeline_ms_p95=round(float(np.percentile(self.t_pipe, 95)), 1),
                     render_ms_median=med(self.t_render), jpeg_encode_ms_median=med(self.t_enc),
                     publish_annotated_ms_median=med(self.t_pub),
                     callback_ms_median=med(self.t_cb), callback_p95_ms=round(float(np.percentile(self.t_cb, 95)), 1),
                     callback_fps=round(1000.0 / float(np.median(self.t_cb)), 1),
                     frames_by_warning_level={"clear": self.level_frames[0], "caution": self.level_frames[1],
                                              "brake": self.level_frames[2]},
                     first_caution_frame=self.first_level[1], first_brake_frame=self.first_level[2],
                     confirmed_tracks_by_class=per_class,
                     note="pipeline = receive + letterbox + ONNX model + NMS + tracking + warning rule; callback also includes drawing, "
                          "JPEG encoding, publishing and logging; IoU tracker (not ByteTrack); warning thresholds are uncalibrated heuristics")
        with open("%s/%s_stats.json" % (self.out_dir, self.name), "w") as f:
            json.dump(stats, f, indent=1)
        self.get_logger().info("perception finished: " + json.dumps(stats))
        self.finish_time = time.time()


def main():
    rclpy.init()
    node = PerceptionNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()

