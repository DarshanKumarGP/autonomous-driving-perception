"""Very small IoU tracker (greedy matching). Pure Python, so it is easy to test."""


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


class IouTracker:
    def __init__(self, iou_thr=0.25, max_age=3, min_hits=2):
        self.iou_thr, self.max_age, self.min_hits = iou_thr, max_age, min_hits
        self.tracks = {}
        self.next_id = 1

    def update(self, dets):
        """dets: list of (x1, y1, x2, y2, score, cls). Returns confirmed tracks seen in this frame."""
        ids = list(self.tracks)
        pairs = []
        for ti in ids:
            t = self.tracks[ti]
            for di, d in enumerate(dets):
                if d[5] != t["cls"]:
                    continue
                v = iou(t["box"], d[:4])
                if v >= self.iou_thr:
                    pairs.append((v, ti, di))
        pairs.sort(reverse=True)
        used_t, used_d = set(), set()
        for v, ti, di in pairs:
            if ti in used_t or di in used_d:
                continue
            used_t.add(ti)
            used_d.add(di)
            t = self.tracks[ti]
            t["box"], t["score"] = tuple(dets[di][:4]), dets[di][4]
            t["hits"] += 1
            t["age"] = 0
            t["seen"] = True
        for ti in ids:
            if ti not in used_t:
                self.tracks[ti]["age"] += 1
                self.tracks[ti]["seen"] = False
        for di, d in enumerate(dets):
            if di not in used_d:
                self.tracks[self.next_id] = dict(box=tuple(d[:4]), cls=d[5], score=d[4], hits=1, age=0, seen=True)
                self.next_id += 1
        for ti in [k for k, t in self.tracks.items() if t["age"] > self.max_age]:
            del self.tracks[ti]
        return [dict(id=ti, **t) for ti, t in self.tracks.items() if t["seen"] and t["hits"] >= self.min_hits]
