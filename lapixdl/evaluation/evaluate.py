from typing import List
import numpy as np

from .model import *


def __merge_detection_metrics(acc: DetectionMetrics,
                              curr: DetectionMetrics) -> DetectionMetrics:
    pass


def evaluate_detection(gt_bboxes: List[List[BBox]],
                       pred_bboxes: List[List[BBox]],
                       iou_threshold=.5) -> DetectionMetrics:
    acc_metrics = DetectionMetrics()

    for (curr_gt_bboxes, curr_pred_bboxes) in zip(gt_bboxes, pred_bboxes):
        curr_metrics = evaluate_detection_single_image(
            curr_gt_bboxes, curr_pred_bboxes, iou_threshold)
        acc_metrics = __merge_detection_metrics(acc_metrics, curr_metrics)

    return acc_metrics


def __flat_mask(mask: Mask) -> List[int]:
    return [item for sublist in mask for item in sublist]


def evaluate_segmentation(gt_masks: List[Mask],
                          pred_masks: List[Mask],
                          classes: List[str]) -> SegmentationMetrics:
    confusion_matrix = np.zeros((len(classes), len(classes)), np.int)

    flat_gt = [cls for x in gt_masks for cls in __flat_mask(x)]
    flat_pred = [cls for x in pred_masks for cls in __flat_mask(x)]

    for (curr_gt, curr_pred) in zip(flat_gt, flat_pred):
        confusion_matrix[curr_pred, curr_gt] += 1

    return SegmentationMetrics(classes, confusion_matrix)


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
