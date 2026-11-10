"""ROS2 node: plays a recorded driving video as a camera topic (/camera/image_raw)."""
import time

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from common import frame_to_msg


class CameraNode(Node):
    def __init__(self):
        super().__init__("camera_node")
        self.declare_parameter("video", "/data/demo720.mp4")
        self.declare_parameter("publish_hz", 4.0)
        self.declare_parameter("max_seconds", 20.0)
        self.declare_parameter("publish_width", 960)
        video = self.get_parameter("video").value
        self.cap = cv2.VideoCapture(video)
        if not self.cap.isOpened():
            raise RuntimeError("cannot open video " + str(video))
        self.src_fps = self.cap.get(cv2.CAP_PROP_FPS) or 30.0
        hz = float(self.get_parameter("publish_hz").value)
        self.step = max(1, int(round(self.src_fps / hz)))
        self.max_frames = int(self.src_fps * float(self.get_parameter("max_seconds").value))
        self.width = int(self.get_parameter("publish_width").value)
        qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=2, reliability=ReliabilityPolicy.RELIABLE)
        qos10 = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.pub = self.create_publisher(Image, "/camera/image_raw", qos)
        self.create_subscription(String, "/perception/status", self.on_status, qos10)
        self.read_frames, self.sent, self.last_frame = 0, 0, None
        self.started, self.finished_at, self.done = False, None, False
        self.t_wait = time.time()
        self.timer = self.create_timer(1.0 / hz, self.tick)
        self.create_timer(1.0, self.watchdog)
        self.get_logger().info("camera: %s | source %.1f fps | publishing every %d source frames (%.1f Hz), width %d, up to %d frames"
                               % (video, self.src_fps, self.step, hz, self.width, self.max_frames))

    def on_status(self, msg):
        if msg.data.startswith("DONE"):
            self.done = True

    def watchdog(self):
        if self.finished_at is not None and time.time() - self.finished_at > 120:
            self.get_logger().warn("no DONE message within 120 s, stopping")
            self.done = True

    def tick(self):
        if not self.started:
            if self.pub.get_subscription_count() < 1:
                if time.time() - self.t_wait > 60:
                    self.get_logger().error("no subscriber after 60 s, stopping")
                    self.done = True
                return
            self.started = True
        got, ended = None, False
        for _ in range(self.step):
            if self.read_frames >= self.max_frames:
                ended = True
                break
            ok, f = self.cap.read()
            if not ok:
                ended = True
                break
            got = f
            self.read_frames += 1
        if self.read_frames >= self.max_frames:
            ended = True
        if got is None:
            got = self.last_frame          # video ended on a step boundary: repeat the last frame as the end marker
        if got is None:
            self.done = True
            return
        self.last_frame = got
        h, w = got.shape[:2]
        if w != self.width:
            got = cv2.resize(got, (self.width, int(round(h * self.width / w))), interpolation=cv2.INTER_AREA)
        fid = ("last%d" if ended else "cam%d") % self.sent
        self.pub.publish(frame_to_msg(got, self.get_clock().now().to_msg(), fid))
        self.sent += 1
        if ended:
            self.timer.cancel()
            self.finished_at = time.time()
            self.get_logger().info("camera: published %d frames (last one marked), waiting for DONE" % self.sent)


def main():
    rclpy.init()
    node = CameraNode()
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
