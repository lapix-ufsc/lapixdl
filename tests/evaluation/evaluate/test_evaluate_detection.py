import pytest
import numpy as np

from lapixdl.evaluation.evaluate import evaluate_detection
from lapixdl.evaluation.model import BBox
from ... import utils


def test_evaluation_detection_iou_metric():
    classes = ['kite']

    gt_bbox = BBox(110, 110, 320, 280, 0)

    pred_bbox = BBox(70, 50, 240, 220, 0)

    metrics = evaluate_detection([[gt_bbox]], [[pred_bbox]], classes)

    assert round(metrics.avg_iou, 3) == .290
    assert round(metrics.by_class[0].iou, 3) == .290
