"""Rule-based obstacle warning. A heuristic on image position, NOT a calibrated distance."""

DEFAULT_CORRIDOR = [(0.30, 1.00), (0.44, 0.52), (0.56, 0.52), (0.70, 1.00)]   # fractions of width/height


def point_in_poly(x, y, poly):
    inside = False
    j = len(poly) - 1
    for i in range(len(poly)):
        xi, yi = poly[i]
        xj, yj = poly[j]
        if (yi > y) != (yj > y):
            if x < (xj - xi) * (y - yi) / (yj - yi) + xi:
                inside = not inside
        j = i
    return inside


def assess(tracks, w, h, corridor=DEFAULT_CORRIDOR, caution_y=0.55, brake_y=0.68):
    """Returns (level, culprit): level 0 clear, 1 caution, 2 brake; culprit = (track id, bottom edge as fraction of height)."""
    poly = [(x * w, y * h) for x, y in corridor]
    level, culprit = 0, None
    for t in tracks:
        x1, y1, x2, y2 = t["box"]
        if not point_in_poly((x1 + x2) / 2.0, y2, poly):
            continue
        frac = y2 / float(h)
        lv = 2 if frac >= brake_y else (1 if frac >= caution_y else 0)
        if lv == 0:
            continue
        if lv > level or (lv == level and frac > culprit[1]):
            level, culprit = lv, (t["id"], round(frac, 3))
    return level, culprit
