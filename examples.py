"""
Module examples

Requires:
    numpy
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from matplotlib.colors import ListedColormap

from lapixdl.evaluation.evaluate import evaluate_classification
from lapixdl.evaluation.evaluate import evaluate_detection
from lapixdl.evaluation.evaluate import evaluate_segmentation
from lapixdl.evaluation.model import BBox
from lapixdl.evaluation.model import Classification
from lapixdl.evaluation.model import Result
from lapixdl.evaluation.visualize import show_classifications
from lapixdl.evaluation.visualize import show_detections
from lapixdl.evaluation.visualize import show_segmentations


def main():
    # Model evaluation examples
    evaluate_segmentation_example()
    evaluate_detection_example()
    evaluate_classification_example()

    # Results visualization examples
    show_classification_example()
    show_segmentation_example()
    show_detection_example()


def evaluate_segmentation_example():
    # Class names - Background must be at 0 index
    classes = ['bkg', 'kite', 'person']

    # Image shape
    mask_shape = (480, 640)

    # Creating fake data
    # Creates a rectangle of 1s in a 0s array
    gt_bbox_1 = BBox(10, 10, 10, 10, 1)
    mask_bin_GT_1 = draw_rectangle(np.zeros(mask_shape, np.int),
                                   gt_bbox_1.upper_left_point,
                                   gt_bbox_1.bottom_right_point,
                                   1)

    pred_bbox_1 = BBox(10, 10, 10, 10, 1)
    mask_bin_pred_1 = draw_rectangle(np.zeros(mask_shape, np.int),
                                     pred_bbox_1.upper_left_point,
                                     pred_bbox_1.bottom_right_point,
                                     1)

    # Creates a rectangle of 2s in a 0s array
    gt_bbox_2 = BBox(110, 110, 320, 280, 2)
    mask_bin_GT_2 = draw_rectangle(np.zeros(mask_shape, np.int),
                                   gt_bbox_2.upper_left_point,
                                   gt_bbox_2.bottom_right_point,
                                   2)

    pred_bbox_2 = BBox(70, 50, 240, 220, 2)
    mask_bin_pred_2 = draw_rectangle(np.zeros(mask_shape, np.int),
                                     pred_bbox_2.upper_left_point,
                                     pred_bbox_2.bottom_right_point,
                                     2)

    # Merging masks
    mask_GT = np.maximum(mask_bin_GT_1, mask_bin_GT_2)
    mask_pred = np.maximum(mask_bin_pred_1, mask_bin_pred_2)

    # Creating data suplier iterator
    # It is not necessary here, but it's useful if you want to yield data
    # from the disk i.e. from a Pytorch DataLoader
    it_gt_masks = identity_iterator(mask_GT)
    it_pred_masks = identity_iterator(mask_pred)

    # Calculates and shows metrics
    metrics = evaluate_segmentation(it_gt_masks, it_pred_masks, classes)

    # Shows confusion matrix and returns its Figure and Axes
    fig, axes = metrics.show_confusion_matrix()

    # Shows confusion matrix for class `a`
    metrics.by_class[0].show_confusion_matrix()


def evaluate_detection_example():
    # Class names
    classes = ['kite', 'person']

    # Image shape
    mask_shape = (480, 640)

    # Creating fake data
    gt_bbox_1 = BBox(10, 10, 10, 10, 0, 1)
    pred_bbox_1 = BBox(10, 10, 10, 10, 0, 1)

    gt_bbox_2 = BBox(110, 110, 320, 280, 1, 1)
    pred_bbox_2 = BBox(70, 50, 240, 220, 1, 1)

    # Creating data suplier iterator
    # It is not necessary here, but it's useful if you want to yield data
    # from the disk i.e. from a Pytorch DataLoader
    it_gt_masks = identity_iterator([gt_bbox_1, gt_bbox_2])
    it_pred_masks = identity_iterator([pred_bbox_1, pred_bbox_2])

    # Calculates and shows metrics
    metrics = evaluate_detection(it_gt_masks, it_pred_masks, classes)

    # Shows confusion matrix and returns its Figure and Axes
    fig, axes = metrics.show_confusion_matrix()

    # Shows confusion matrix for class `a`
    metrics.by_class[0].show_confusion_matrix()


def evaluate_classification_example():
    # Class names
    classes = ['a', 'b', 'c']

    # Classifications based in array
    gt_class = [Classification(x) for x in [
        0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2]]
    # All predictions with .8 score
    pred_class = [Classification(x, .8) for x in [
        0, 0, 0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2, 2, 2, 2, 2]]

    # Calculates and shows metrics
    metrics = evaluate_classification(gt_class, pred_class, classes)

    # Shows confusion matrix and returns its Figure and Axes
    fig, axes = metrics.show_confusion_matrix()

    # Shows confusion matrix for class `a`
    metrics.by_class[0].show_confusion_matrix()


def show_classification_example():
    # Class names
    classes = ['a', 'b', 'c']

    # Classifications based in array
    gt_class = [Classification(x) for x in [
        0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2]]
    # All predictions with .8 score
    pred_class = [Classification(x, .8) for x in [
        0, 0, 2, 1, 0, 0, 0, 0, 0, 0, 2, 2, 1, 1, 0, 0, 0, 2, 2]]

    # Convert to results
    results = [Result(random_image(), gt, pred)
               for gt, pred in zip(gt_class, pred_class)]

    # GT only result
    results = [Result(random_image(), Classification(2))] + results

    # Shows results and returns its Figure and Axes
    fig, axes = show_classifications(results, classes, 5)


def show_segmentation_example():
    # Class names
    classes = ['bkg', 'kite', 'person', 'car']

    # Image/mask shape
    mask_shape = (480, 640)

    # Multiclass mask creation
    gt_bbox_1 = BBox(10, 10, 100, 100, 1)
    mask_bin_GT_1 = draw_bboxes(mask_shape, [gt_bbox_1])

    pred_bbox_1 = BBox(10, 10, 100, 100, 1)
    mask_bin_pred_1 = draw_bboxes(mask_shape, [pred_bbox_1])

    gt_bbox_2 = BBox(110, 110, 320, 280, 2)
    mask_bin_GT_2 = draw_bboxes(mask_shape, [gt_bbox_2])

    pred_bbox_2 = BBox(70, 50, 240, 220, 2)
    mask_bin_pred_2 = draw_bboxes(mask_shape, [pred_bbox_2])

    gt_bbox_3 = BBox(300, 300, 100, 100, 3)
    mask_bin_GT_3 = draw_bboxes(mask_shape, [gt_bbox_3])

    mask_multi_GT = np.maximum(np.maximum(
        mask_bin_GT_1, mask_bin_GT_2 * 2), mask_bin_GT_3 * 3)
    mask_multi_pred = np.maximum(mask_bin_pred_1, mask_bin_pred_2 * 2)

    # Convert to results
    results = [Result(random_image(mask_shape[0], mask_shape[1]),
                      mask_multi_GT, mask_multi_pred)]

    # GT only result
    results = [
        Result(random_image(mask_shape[0], mask_shape[1]), mask_multi_GT)] + results

    # Custom color map
    cmap = ListedColormap(['#ff0000', '#00ff00', '#0000ff', '#ffffff'])

    # Shows results and returns its Figure and Axes
    fig, axes = show_segmentations(results, classes, cmap=cmap)


def show_detection_example():
    # Class names
    classes = ['kite', 'person', 'car']

    # Image shape
    img_shape = (480, 640)

    # Bboxes creation
    gt_bbox_1 = BBox(10, 10, 100, 100, 0)
    pred_bbox_1 = BBox(10, 10, 100, 100, 0, .8212158464)

    gt_bbox_2 = BBox(110, 110, 320, 280, 1)
    pred_bbox_2 = BBox(70, 50, 240, 220, 1, .34844545)

    gt_bbox_3 = BBox(300, 300, 100, 100, 2)

    # Convert to results
    results = [Result(random_image(img_shape[0], img_shape[1]), [
                      gt_bbox_1, gt_bbox_2, gt_bbox_3], [pred_bbox_1, pred_bbox_2])]

    # GT only result
    results = [Result(random_image(img_shape[0], img_shape[1]), [
                      gt_bbox_1, gt_bbox_2, gt_bbox_3])] + results

    # Shows results and returns its Figure and Axes
    fig, axes = show_detections(results, classes, show_bbox_label=False)


def identity_iterator(value):
    yield value


def random_image(h=None, w=None):
    return (np.random.rand(h or 200, w or 400, 3) * 125).astype(np.int8)


def draw_bboxes(mask_shape, bboxes):
    mask = np.zeros(mask_shape, np.int)

    for bbox in bboxes:
        mask[
            bbox.upper_left_point[0]:bbox.bottom_right_point[0] + 1,
            bbox.upper_left_point[1]:bbox.bottom_right_point[1] + 1
        ] = 1

    return mask


def draw_rectangle(img: np.ndarray, pt1: Tuple[int, int], pt2: Tuple[int, int], fill: int):
    cp = img.copy()
    cp[slice(pt1[0], pt2[0] + 1), slice(pt1[1], pt2[1] + 1)] = fill
    return cp


main()
