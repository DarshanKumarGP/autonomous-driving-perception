from iou_tracker import IouTracker, iou


def box(x, y=100, w=100, h=100):
    return (x, y, x + w, y + h)


def test_iou_identical_and_disjoint():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1.0
    assert iou((0, 0, 10, 10), (20, 20, 30, 30)) == 0.0


def test_iou_half_overlap():
    assert abs(iou((0, 0, 10, 10), (5, 0, 15, 10)) - 1 / 3) < 1e-9


def test_track_confirmed_on_second_sighting():
    t = IouTracker()
    assert t.update([box(100) + (0.9, 2)]) == []
    out = t.update([box(105) + (0.9, 2)])
    assert len(out) == 1 and out[0]["id"] == 1


def test_id_stays_the_same_while_object_moves():
    t = IouTracker()
    ids = set()
    for k in range(6):
        for tr in t.update([box(100 + 6 * k) + (0.9, 2)]):
            ids.add(tr["id"])
    assert ids == {1}


def test_two_objects_get_two_ids():
    t = IouTracker()
    dets = [box(50) + (0.9, 2), box(600) + (0.8, 2)]
    t.update(dets)
    out = t.update(dets)
    assert sorted(tr["id"] for tr in out) == [1, 2]


def test_different_class_is_not_matched():
    t = IouTracker()
    t.update([box(100) + (0.9, 2)])
    t.update([box(100) + (0.9, 3)])
    assert len(t.tracks) == 2


def test_track_is_dropped_after_max_age_and_new_id_is_issued():
    t = IouTracker(max_age=1)
    t.update([box(100) + (0.9, 2)])
    t.update([box(100) + (0.9, 2)])
    t.update([])
    t.update([])
    assert t.tracks == {}
    t.update([box(100) + (0.9, 2)])
    assert list(t.tracks) == [2]
