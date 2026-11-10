"""ROS2 node: writes /perception/annotated to an mp4 file."""
import os

import cv2
import rclpy
from rclpy.node import Node
from rclpy.qos import HistoryPolicy, QoSProfile, ReliabilityPolicy
from sensor_msgs.msg import Image
from std_msgs.msg import String

from common import msg_to_frame


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
        self.create_subscription(Image, "/perception/annotated", self.on_img, qos10)
        self.create_subscription(String, "/perception/status", self.on_status, qos10)
        self.get_logger().info("recorder: writing %s at %.1f fps" % (self.path, self.fps))

    def on_img(self, m):
        f = msg_to_frame(m)
        if self.writer is None:
            self.writer = cv2.VideoWriter(self.path, cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (m.width, m.height))
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
