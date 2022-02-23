from __future__ import annotations

from lapixdl.evaluation.model import BBox


def test_center_point_box():
    b0 = BBox(0, 0, 3, 3, 0)
    assert b0.center_point == (1, 1)
    b1 = BBox(100, 100, 100, 100, 0)
    assert b1.center_point == (149, 149)
    b2 = BBox(500, 500, 100, 100, 0)
    assert b2.center_point == (549, 549)
    b3 = BBox(100, 200, 76, 76, 0)
    assert b3.center_point == (137, 237)
