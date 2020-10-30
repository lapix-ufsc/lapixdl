from typing import List, Iterable
import numpy as np
from progress.counter import Counter

from .model import *


def __merge_detection_metrics(acc: DetectionMetrics,
                              curr: DetectionMetrics) -> DetectionMetrics:
    pass


def evaluate_detection(gt_bboxes: List[List[BBox]],
                       pred_bboxes: List[List[BBox]],
                       iou_threshold=.5) -> DetectionMetrics:
    pass


def __flat_mask(mask: Mask) -> List[int]:
    return [item for sublist in mask for item in sublist]


def evaluate_segmentation(gt_masks: Iterable[Mask],
                          pred_masks: Iterable[Mask],
                          classes: List[str]) -> SegmentationMetrics:
    """Evaluates segmentation predictions

    The iterables should return one mask per iterations and the itens of
    `gt_masks` and `pred_masks` with same index should correspond to the
    same sample.

    Masks should be 2D arrays where each value corresponds to the class 
    index of the pixel the sample image.

    Args:
        gt_masks (Iterable[Mask]): Iterable of ground truth masks.
        pred_masks (Iterable[Mask]): Iterable of predicted masks.
        classes (List[str]): Class names.

    Returns:
        SegmentationMetrics: Pixel-based classification and segmentation metrics.
    """

    confusion_matrix = np.zeros((len(classes), len(classes)), np.int)

    with Counter('Evaluating ', suffix=' pixels') as counter:
        for (curr_gt_mask, curr_pred_mask) in zip(gt_masks, pred_masks):
            flat_gt = __flat_mask(curr_gt_mask)
            flat_pred = __flat_mask(curr_pred_mask)
            for (curr_pred, curr_gt) in zip(flat_pred, flat_gt):
                confusion_matrix[curr_pred, curr_gt] += 1
            counter.next()

    metrics = SegmentationMetrics(classes, confusion_matrix)

    print(metrics)

    return metrics


def evaluate_classification(gt_classifications: Iterable[Classification],
                            pred_classifications: Iterable[Classification],
                            classes: List[str]) -> ClassificationMetrics:
    """Evaluates classification predictions

    The iterables should return one classification per iterations and 
    the itens of `gt_classifications` and `pred_classifications` with 
    same index should correspond to the same sample.

    Args:
        gt_classifications (Iterable[Classification]): Ground truth classifications.
        pred_classifications (Iterable[Classification]): Predicted classifications.
        classes (List[str]): Class names.

    Returns:
        ClassificationMetrics: Classification metrics.
    """

    confusion_matrix = np.zeros((len(classes), len(classes)), np.int)

    with Counter('Evaluating ') as counter:
        for (curr_gt_classification, curr_pred_classification) in zip(gt_classifications, pred_classifications):
            confusion_matrix[curr_pred_classification.cls,
                             curr_gt_classification.cls] += 1
            counter.next()

    metrics = ClassificationMetrics(classes, confusion_matrix)

    print(metrics)

    return metrics


def evaluate_detection_single_image(gt_bboxes: List[BBox],
                                    pred_bboxes: List[BBox],
                                    iou_threshold=.5) -> DetectionMetrics:
    pass
