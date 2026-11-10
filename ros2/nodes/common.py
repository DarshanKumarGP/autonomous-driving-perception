"""Helpers shared by the nodes. No cv_bridge, so there are no numpy ABI surprises."""
import numpy as np
from sensor_msgs.msg import Image


def frame_to_msg(frame_bgr, stamp, frame_id):
    m = Image()
    m.header.stamp = stamp
    m.header.frame_id = frame_id
    m.height, m.width = int(frame_bgr.shape[0]), int(frame_bgr.shape[1])
    m.encoding = "bgr8"
    m.is_bigendian = 0
    m.step = m.width * 3
    flat = np.ascontiguousarray(frame_bgr, dtype=np.uint8).reshape(-1)
    try:
        m.data = flat                      # fast path: numpy uint8 array
    except (AssertionError, TypeError):
        m.data = flat.tobytes()            # fallback
    return m


def msg_to_frame(m):
    assert m.encoding == "bgr8", "expected bgr8, got " + m.encoding
    return np.frombuffer(m.data, dtype=np.uint8).reshape(m.height, m.width, 3)
