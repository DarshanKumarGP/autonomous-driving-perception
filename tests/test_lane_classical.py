import types

import cv2
import numpy as np

import lane_classical as lc

ARGS = types.SimpleNamespace(top=0.5, bottom=0.97, top_half_width=0.12, bottom_half_width=0.5,
                             hough_threshold=30, alpha=0.25, hold=0, no_color=False)


def road(with_lines=True):
    img = np.zeros((540, 960, 3), np.uint8)
    if with_lines:
        cv2.line(img, (200, 520), (420, 300), (255, 255, 255), 8)
        cv2.line(img, (760, 520), (540, 300), (255, 255, 255), 8)
    return img


def test_both_lane_lines_are_found_on_a_synthetic_road():
    raw = lc.detect(road(), ARGS, {})
    assert raw["left"] is not None and raw["right"] is not None
    la, lb, _ = raw["left"]
    ra, rb, _ = raw["right"]
    assert la < 0 < ra
    assert abs((la * 500 + lb) - 220) < 30
    assert abs((ra * 500 + rb) - 740) < 30


def test_blank_road_gives_no_lines():
    raw = lc.detect(road(False), ARGS, {})
    assert raw["left"] is None and raw["right"] is None


def test_draw_overlays_red_lines_when_both_found():
    state = {}
    lc.detect(road(), ARGS, state)
    out = lc.draw(road(), ARGS, state)
    assert out.shape == road().shape
    red = (out[..., 2] > 200) & (out[..., 1] < 60) & (out[..., 0] < 60)
    assert red.any()


def test_draw_does_not_crash_without_lines():
    out = lc.draw(road(False), ARGS, {})
    assert out.shape == (540, 960, 3)
