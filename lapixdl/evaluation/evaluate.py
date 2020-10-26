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
    confusion_matrix = np.zeros((len(classes), len(classes)), np.int)

    with Counter('Evaluating ') as counter:
        for (curr_gt_mask, curr_pred_mask) in zip(gt_masks, pred_masks):
            flat_gt = __flat_mask(curr_gt_mask)
            flat_pred = __flat_mask(curr_pred_mask)
            for (curr_pred, curr_gt) in zip(flat_pred, flat_gt):
                confusion_matrix[curr_pred, curr_gt] += 1
                counter.next()

    metrics = SegmentationMetrics(classes, confusion_matrix)

    print(metrics)

    return metrics


def evaluate_classification(gt_classifications: List[Classification],
                            pred_classifications: List[Classification],
                            classes: List[str]) -> ClassificationMetrics:
    confusion_matrix = np.zeros((len(classes), len(classes)), np.int)

    for (curr_gt_classification, curr_pred_classification) in zip(gt_classifications, pred_classifications):
        confusion_matrix[curr_pred_classification.cls,
                         curr_gt_classification.cls] += 1

    return ClassificationMetrics(classes, confusion_matrix)


def evaluate_detection_single_image(gt_bboxes: List[BBox],
                                    pred_bboxes: List[BBox],
                                    iou_threshold=.5) -> DetectionMetrics:
    pass
