from __future__ import annotations

from lapixdl.evaluation.evaluate import evaluate_detection
from lapixdl.formats.annotation import BBox


def test_evaluation_detection_iou_metric():
    classes = ['kite']

    gt_bbox = BBox(110, 110, 320, 280, 0)

    pred_bbox = BBox(70, 50, 240, 220, 0, .5)

    metrics = evaluate_detection([[gt_bbox]], [[pred_bbox]], classes)

    assert round(metrics.avg_iou, 3) == .290
    assert round(metrics.by_class[0].iou, 3) == .290


def test_evaluation_detection_iou_metric_w_more_classes():
    classes = ['kite', 'person']

    gt_bbox = BBox(110, 110, 320, 280, 0)

    pred_bbox = BBox(70, 50, 240, 220, 0, .5)

    metrics = evaluate_detection([[gt_bbox]], [[pred_bbox]], classes)

    assert round(metrics.avg_iou, 3) == .290
    assert round(metrics.by_class[0].iou, 3) == .290


def test_evaluation_detection_no_gt():
    classes = ['kite', 'person']

    pred_bbox = BBox(70, 50, 240, 220, 0)

    metrics = evaluate_detection([[]], [[pred_bbox]], classes)

    assert round(metrics.avg_iou, 3) == 0
    assert round(metrics.by_class[0].iou, 3) == 0
    assert metrics._by_class[0].FP == 1


def test_evaluation_detection_no_pred():
    classes = ['kite', 'person']

    gt_bbox = BBox(110, 110, 320, 280, 0)

    metrics = evaluate_detection([[gt_bbox]], [[]], classes)

    assert round(metrics.avg_iou, 3) == 0
    assert round(metrics.by_class[0].iou, 3) == 0
    assert metrics._by_class[0].FN == 1
