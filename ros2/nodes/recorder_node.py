"""ROS2 node: decodes /perception/annotated/compressed (JPEG) and writes it to an mp4 file."""
import os

import cv2
import numpy as np
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import CompressedImage
from std_msgs.msg import String


class RecorderNode(Node):
    def __init__(self):
        super().__init__("recorder_node")
        self.declare_parameter("out_dir", "/out")
        self.declare_parameter("name", "demo")
        self.declare_parameter("fps", 4.0)
        self.out_dir = str(self.get_parameter("out_dir").value)
        self.name = str(self.get_parameter("name").value)
        self.fps = float(self.get_parameter("fps").value)
        os.makedirs(self.out_dir, exist_ok=True)
        self.writer, self.n, self.done = None, 0, False
        self.path = "%s/%s_annotated.mp4" % (self.out_dir, self.name)
        qos10 = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.RELIABLE)
        self.create_subscription(CompressedImage, "/perception/annotated/compressed", self.on_img, qos10)
        self.create_subscription(String, "/perception/status", self.on_status, qos10)
        self.get_logger().info("recorder: writing %s at %.1f fps" % (self.path, self.fps))

    def on_img(self, m):
        f = cv2.imdecode(np.frombuffer(m.data, dtype=np.uint8), cv2.IMREAD_COLOR)
        if f is None:
            self.get_logger().warn("could not decode a frame")
            return
        if self.writer is None:
            h, w = f.shape[:2]
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
        self.writer.write(f)
        self.n += 1

    def on_status(self, m):
        if m.data.startswith("DONE") and not self.done:
            if self.writer is not None:
                self.writer.release()
            self.get_logger().info("recorder: saved %d frames to %s" % (self.n, self.path))
            self.done = True


def main():
    rclpy.init()
    node = RecorderNode()
    try:
        while rclpy.ok() and not node.done:
            rclpy.spin_once(node, timeout_sec=0.2)
    except KeyboardInterrupt:
        pass
    finally:
        if node.writer is not None and not node.done:
            node.writer.release()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
