import pytest
import numpy as np

from lapixdl.evaluation.evaluate import calculate_pairwise_bbox_ious
from lapixdl.evaluation.model import BBox


def test_calculate_pairwise_bbox_ious():
    gt_bboxes = [
        BBox(39, 63, 203-39+1, 112-63+1, 0),
        BBox(49, 75, 203-49+1, 125-75+1, 0),
        BBox(31, 69, 201-31+1, 125-69+1, 0),
        BBox(50, 72, 197-50+1, 121-72+1, 0),
        BBox(35, 51, 196-35+1, 110-51+1, 0),
        BBox(35, 51, 196, 110, 0),
        BBox(0, 0, 5, 5, 0),
    ]
    pred_bboxes = [
        BBox(54, 66, 198-54+1, 114-66+1, 0),
        BBox(42, 78, 186-42+1, 126-78+1, 0),
        BBox(18, 63, 235-18+1, 135-63+1, 0),
        BBox(54, 72, 198-54+1, 120-72+1, 0),
        BBox(36, 60, 180-36+1, 108-60+1, 0),
        BBox(35, 51, 196, 110, 0),
        BBox(6, 6, 5, 5, 0),
    ]

    ious = calculate_pairwise_bbox_ious(gt_bboxes, pred_bboxes)

    assert round(ious[0, 0], 3) == .798
    assert round(ious[1, 1], 3) == .79
    assert round(ious[2, 2], 3) == .612
    assert round(ious[3, 3], 3) == .947
    assert round(ious[4, 4], 3) == .731
    assert ious[5, 5] == 1
    assert ious[6, 6] == 0
