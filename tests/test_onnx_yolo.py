import os

import numpy as np
import pytest

from onnx_yolo import OnnxYolo

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
MODEL = os.path.join(ROOT, "models", "yolo11s_bdd.onnx")
FRAME = np.zeros((720, 1280, 3), np.uint8)


class FakeSession:
    def __init__(self, out):
        self.out = out

    def run(self, _names, _feed):
        return [self.out]

    def get_providers(self):
        return ["FakeProvider"]


def make_detector(out, conf=0.35, classes=None):
    d = OnnxYolo.__new__(OnnxYolo)
    d.sess = FakeSession(out)
    d.inp = "images"
    d.imgsz, d.conf, d.iou = 640, conf, 0.5
    d.cls_arr = None if classes is None else np.array(list(classes))
    d.last_infer_ms = 0.0
    return d


def blank_output(candidates):
    out = np.zeros((1, 14, 8400), np.float32)
    for k, (cx, cy, w, h, cls, score) in enumerate(candidates):
        out[0, 0:4, k] = (cx, cy, w, h)
        out[0, 4 + cls, k] = score
    return out


def test_preprocess_shape_and_range():
    d = make_detector(blank_output([]))
    x, r, left, top = d._prep(FRAME)
    assert x.shape == (1, 3, 640, 640) and x.dtype == np.float32
    assert 0.0 <= float(x.min()) and float(x.max()) <= 1.0
    assert r == 0.5 and left == 0 and top == 140


def test_box_is_mapped_back_to_original_pixels():
    d = make_detector(blank_output([(320, 320, 100, 100, 2, 0.9)]))
    dets = d.detect(FRAME)
    assert len(dets) == 1
    x1, y1, x2, y2, score, cls = dets[0]
    assert cls == 2 and abs(score - 0.9) < 1e-3
    assert abs(x1 - 540) < 1 and abs(y1 - 260) < 1 and abs(x2 - 740) < 1 and abs(y2 - 460) < 1


def test_low_confidence_is_filtered():
    d = make_detector(blank_output([(320, 320, 100, 100, 2, 0.2)]))
    assert d.detect(FRAME) == []


def test_class_filter():
    out = blank_output([(320, 320, 100, 100, 2, 0.9), (100, 300, 60, 60, 8, 0.9)])
    d = make_detector(out, classes=[2])
    assert [x[5] for x in d.detect(FRAME)] == [2]


def test_nms_keeps_best_of_overlapping_same_class_boxes():
    out = blank_output([(320, 320, 100, 100, 2, 0.9), (324, 320, 100, 100, 2, 0.8)])
    dets = make_detector(out).detect(FRAME)
    assert len(dets) == 1 and abs(dets[0][4] - 0.9) < 1e-3


@pytest.mark.skipif(not os.path.exists(MODEL), reason="model file not present (it is not stored in git)")
def test_real_model_runs_on_a_blank_image():
    d = OnnxYolo(MODEL)
    assert isinstance(d.detect(FRAME), list)
