"""Classical lane detection with OpenCV (no learning).

Per frame: resize -> edges (Canny) -> keep edges on white/yellow pixels -> region-of-interest
mask -> probabilistic Hough lines -> split left/right by slope -> least-squares line per
side -> smooth over time. There is no ground truth here, so the stats it writes count how
often each lane was found, not how accurate the lines are.
"""
import argparse
import json
import os
import time

import cv2
import numpy as np


def ensure_dir(path):
    d = os.path.dirname(path)
    if d:
        os.makedirs(d, exist_ok=True)


def roi_polygon(w, h, a):
    cx = w / 2.0
    return np.array([[
        (int(cx - a.bottom_half_width * w), int(a.bottom * h)),
        (int(cx - a.top_half_width * w), int(a.top * h)),
        (int(cx + a.top_half_width * w), int(a.top * h)),
        (int(cx + a.bottom_half_width * w), int(a.bottom * h)),
    ]], dtype=np.int32)


def feature_map(frame, use_color):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 50, 150)
    if not use_color:
        return edges
    hls = cv2.cvtColor(frame, cv2.COLOR_BGR2HLS)
    white = cv2.inRange(hls, np.array([0, 170, 0]), np.array([255, 255, 255]))
    yellow = cv2.inRange(hls, np.array([12, 60, 90]), np.array([38, 255, 255]))
    color = cv2.dilate(cv2.bitwise_or(white, yellow), np.ones((7, 7), np.uint8))
    return cv2.bitwise_and(edges, color)


def fit_side(segs, cx, side):
    """Return (a, b, y_far): x = a*y + b, and y_far = smallest y (farthest point) seen."""
    xs, ys, ws = [], [], []
    for x1, y1, x2, y2 in segs:
        if x2 == x1:
            continue
        slope = (y2 - y1) / float(x2 - x1)
        if not (0.4 < abs(slope) < 4.0):
            continue
        if side == "left" and slope < 0 and x1 < cx and x2 < cx:
            pass
        elif side == "right" and slope > 0 and x1 > cx and x2 > cx:
            pass
        else:
            continue
        length = float(np.hypot(x2 - x1, y2 - y1))
        xs += [x1, x2]
        ys += [y1, y2]
        ws += [length, length]
    if len(xs) < 2 or np.ptp(ys) < 10:
        return None
    a, b = np.polyfit(ys, xs, 1, w=ws)
    return (float(a), float(b), float(min(ys)))


def smooth(state, side, fit, alpha, hold):
    key = "miss_" + side
    if fit is not None:
        old = state.get(side)
        if old is None:
            state[side] = fit
        else:
            state[side] = tuple(alpha * n + (1 - alpha) * o for n, o in zip(fit, old))
        state[key] = 0
    else:
        state[key] = state.get(key, 0) + 1
        if state[key] > hold:
            state[side] = None


def detect(frame, a, state):
    h, w = frame.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    cv2.fillPoly(mask, roi_polygon(w, h, a), 255)
    feat = cv2.bitwise_and(feature_map(frame, not a.no_color), mask)
    segs = cv2.HoughLinesP(feat, 1, np.pi / 180, a.hough_threshold,
                           minLineLength=int(0.05 * h), maxLineGap=int(0.12 * h))
    segs = [] if segs is None else np.asarray(segs).reshape(-1, 4).tolist()
    raw = {side: fit_side(segs, w / 2.0, side) for side in ("left", "right")}
    for side in ("left", "right"):
        smooth(state, side, raw[side], a.alpha, a.hold)
    return raw


def draw(frame, a, state):
    h, w = frame.shape[:2]
    y_bot = int(a.bottom * h)
    y_roi_top = int(a.top * h)
    out = frame.copy()
    fl, fr = state.get("left"), state.get("right")
    tops = {}
    for side, f in (("left", fl), ("right", fr)):
        if f is not None:
            tops[side] = max(y_roi_top, int(f[2]))      # only draw where segments were seen
    if fl is not None and fr is not None:
        da = fl[0] - fr[0]
        y_vp = (fr[1] - fl[1]) / da if abs(da) > 1e-6 else None
        if y_vp is None or y_vp >= y_bot - 5:
            fl = fr = None                              # lines never meet above the bottom: reject both
            tops = {}
        else:
            common = max(max(tops.values()), int(y_vp + 0.04 * h))
            tops = {"left": common, "right": common}    # stop where the lines meet
    pts = {}
    for side, f in (("left", fl), ("right", fr)):
        if f is None or side not in tops:
            continue
        yt = tops[side]
        if yt >= y_bot - 10:
            continue
        pts[side] = ((int(f[0] * y_bot + f[1]), y_bot), (int(f[0] * yt + f[1]), yt))
    if len(pts) == 2:
        overlay = out.copy()
        poly = np.array([pts["left"][0], pts["left"][1], pts["right"][1], pts["right"][0]], np.int32)
        cv2.fillPoly(overlay, [poly], (0, 200, 0))
        out = cv2.addWeighted(overlay, 0.3, out, 0.7, 0)
    for p in pts.values():
        cv2.line(out, p[0], p[1], (0, 0, 255), 4)
    label = "lanes: " + ("L " if "left" in pts else "- ") + ("R" if "right" in pts else "-")
    cv2.putText(out, label, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return out


def resize_to(frame, width):
    s = width / float(frame.shape[1])
    return cv2.resize(frame, (width, int(round(frame.shape[0] * s))))


def main():
    p = argparse.ArgumentParser(description="Classical lane detection (OpenCV)")
    p.add_argument("--video")
    p.add_argument("--image")
    p.add_argument("--out", required=True)
    p.add_argument("--still", help="save one mid-clip frame (jpg)")
    p.add_argument("--stats", help="save stats json")
    p.add_argument("--max-seconds", type=float, default=20)
    p.add_argument("--width", type=int, default=960)
    p.add_argument("--top", type=float, default=0.60)
    p.add_argument("--bottom", type=float, default=0.95)
    p.add_argument("--top-half-width", type=float, default=0.08)
    p.add_argument("--bottom-half-width", type=float, default=0.45)
    p.add_argument("--hough-threshold", type=int, default=30)
    p.add_argument("--alpha", type=float, default=0.25)
    p.add_argument("--hold", type=int, default=8)
    p.add_argument("--no-color", action="store_true")
    a = p.parse_args()
    assert a.video or a.image, "give --video or --image"
    ensure_dir(a.out)

    if a.image:
        img = cv2.imread(a.image)
        assert img is not None, "cannot read " + a.image
        img = resize_to(img, a.width)
        state = {}
        detect(img, a, state)
        cv2.imwrite(a.out, draw(img, a, state))
        print("saved", a.out)
        return

    cap = cv2.VideoCapture(a.video)
    assert cap.isOpened(), "cannot open " + a.video
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    limit = int(fps * a.max_seconds)
    state, n, vw, mid = {}, 0, None, None
    c = dict(raw_left=0, raw_right=0, raw_both=0, shown_both=0)
    t0 = time.perf_counter()
    while n < limit:
        ok, frame = cap.read()
        if not ok:
            break
        frame = resize_to(frame, a.width)
        raw = detect(frame, a, state)
        c["raw_left"] += raw["left"] is not None
        c["raw_right"] += raw["right"] is not None
        c["raw_both"] += (raw["left"] is not None and raw["right"] is not None)
        c["shown_both"] += (state.get("left") is not None and state.get("right") is not None)
        vis = draw(frame, a, state)
        if vw is None:
            vw = cv2.VideoWriter(a.out, cv2.VideoWriter_fourcc(*"mp4v"), fps,
                                 (vis.shape[1], vis.shape[0]))
        vw.write(vis)
        n += 1
        if n == max(limit // 2, 1):
            mid = vis.copy()
    cap.release()
    if vw is not None:
        vw.release()
    secs = time.perf_counter() - t0
    stats = dict(video=os.path.basename(a.video), frames=n, seconds_of_video=round(n / fps, 1),
                 left_found_pct=round(100.0 * c["raw_left"] / max(n, 1), 1),
                 right_found_pct=round(100.0 * c["raw_right"] / max(n, 1), 1),
                 both_found_pct_raw=round(100.0 * c["raw_both"] / max(n, 1), 1),
                 both_shown_pct_after_smoothing=round(100.0 * c["shown_both"] / max(n, 1), 1),
                 processing_fps_incl_decode=round(n / max(secs, 1e-6), 1),
                 params={k: v for k, v in vars(a).items() if k not in ("video", "image", "out", "still", "stats")},
                 note="No ground truth: percentages count how often a lane line was found, not accuracy.")
    print(json.dumps(stats, indent=2))
    if a.still and mid is not None:
        ensure_dir(a.still)
        cv2.imwrite(a.still, mid, [cv2.IMWRITE_JPEG_QUALITY, 88])
    if a.stats:
        ensure_dir(a.stats)
        json.dump(stats, open(a.stats, "w"), indent=2)


if __name__ == "__main__":
    main()
