import sys

import numpy
import cv2
import onnxruntime

print("numpy", numpy.__version__, "| opencv", cv2.__version__, "| onnxruntime", onnxruntime.__version__)
import rclpy
from sensor_msgs.msg import Image
from std_msgs.msg import String

print("rclpy + message types import OK")
sys.path.insert(0, "/app")
import iou_tracker, onnx_yolo, warning_logic

print("project modules import OK")
