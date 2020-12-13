import pytest
import numpy as np

from lapixdl.evaluation.evaluate import evaluate_segmentation
from lapixdl.evaluation.model import BBox
from ... import utils


def test_evaluation_classification_metrics():
    classes = ['a', 'b', 'c']

    gt_masks = [
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2],
            [2, 2, 2, 2, 2]
        ]
    ]
    pred_masks = [
        [
            [0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0],
            [0, 0, 2, 2, 1],
            [1, 0, 0, 0, 2],
            [2, 2, 2, 2, 2]
        ]
    ]

    metrics = evaluate_segmentation(gt_masks, pred_masks, classes)

    assert metrics.count == 25
    assert round(metrics.accuracy, 3) == .48
    assert round(metrics.avg_recall, 3) == .511
    assert round(metrics.avg_precision, 3) == .547
    assert round(metrics.avg_f_score, 3) == .465
    assert round(metrics.avg_specificity, 3) == .757


def test_evalution_iou_with_image():
    classes = ['bkg', 'kite']

    mask_shape = (480, 640)

    gt_bbox = BBox(110, 110, 320, 280, 1)
    mask_bin_GT = utils.bin_mask_from_bb(mask_shape, gt_bbox)

    pred_bbox = BBox(70, 50, 240, 220, 1)
    mask_bin_pred = utils.bin_mask_from_bb(mask_shape, pred_bbox)

    metrics = evaluate_segmentation([mask_bin_GT], [mask_bin_pred], classes)

    assert metrics.count == 480 * 640
    assert round(metrics.by_class[1].iou, 3) == .290
    assert round(metrics.avg_iou_no_bkg, 3) == .290
    assert round(metrics.avg_iou, 3) == .502


def test_should_not_accept_single_class():
    classes_empty = []
    classes_bkg_only = ['bkg']

    with pytest.raises(AssertionError):
        evaluate_segmentation([], [], classes_empty)

    with pytest.raises(AssertionError):
        evaluate_segmentation([], [], classes_bkg_only)


def test_evalution_iou_multiclass_with_images():
    classes = ['bkg', 'kite', 'person']

    mask_shape = (480, 640)

    gt_bbox_1 = BBox(10, 10, 10, 10, 1)
    mask_bin_GT_1 = utils.bin_mask_from_bb(mask_shape, gt_bbox_1)

    pred_bbox_1 = BBox(10, 10, 10, 10, 1)
    mask_bin_pred_1 = utils.bin_mask_from_bb(mask_shape, pred_bbox_1)

    gt_bbox_2 = BBox(110, 110, 320, 280, 2)
    mask_bin_GT_2 = utils.bin_mask_from_bb(mask_shape, gt_bbox_2)

    pred_bbox_2 = BBox(70, 50, 240, 220, 2)
    mask_bin_pred_2 = utils.bin_mask_from_bb(mask_shape, pred_bbox_2)

    mask_multi_GT = np.maximum(mask_bin_GT_1, mask_bin_GT_2 * 2)
    mask_multi_pred = np.maximum(mask_bin_pred_1, mask_bin_pred_2 * 2)

    metrics = evaluate_segmentation([mask_multi_GT], [mask_multi_pred], classes)

    assert metrics.count == 480 * 640

    assert metrics.by_class[1].iou == 1.0
    assert round(metrics.by_class[2].iou, 3) == .290
    assert round(metrics.avg_iou_no_bkg, 3) == .645
