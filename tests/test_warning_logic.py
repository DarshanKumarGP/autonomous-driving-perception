from warning_logic import assess, point_in_poly

W, H = 1280, 720


def trk(i, x1, y1, x2, y2):
    return dict(id=i, box=(x1, y1, x2, y2))


def test_point_in_poly():
    sq = [(0, 0), (10, 0), (10, 10), (0, 10)]
    assert point_in_poly(5, 5, sq)
    assert not point_in_poly(15, 5, sq)


def test_clear_when_no_tracks():
    assert assess([], W, H) == (0, None)


def test_brake_when_close_vehicle_is_in_corridor():
    level, culprit = assess([trk(1, 560, 380, 700, 520)], W, H)
    assert level == 2 and culprit[0] == 1


def test_caution_for_mid_distance_vehicle():
    level, culprit = assess([trk(7, 600, 300, 680, 432)], W, H)
    assert level == 1 and culprit[0] == 7


def test_vehicle_outside_corridor_is_ignored():
    assert assess([trk(1, 100, 300, 200, 500)], W, H)[0] == 0


def test_distant_vehicle_stays_clear():
    assert assess([trk(1, 620, 330, 660, 360)], W, H)[0] == 0


def test_highest_level_wins():
    level, culprit = assess([trk(1, 600, 300, 680, 432), trk(2, 560, 380, 700, 520)], W, H)
    assert level == 2 and culprit[0] == 2
