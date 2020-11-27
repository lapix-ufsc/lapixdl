from typing import List, Iterable
import numpy as np
from tqdm import tqdm

from .model import *


def __flat_mask(mask: Mask) -> List[int]:
    return [item for sublist in mask for item in sublist]


def evaluate_segmentation(gt_masks: Iterable[Mask],
                          pred_masks: Iterable[Mask],
                          classes: List[str]) -> SegmentationMetrics:
    """Evaluates segmentation predictions

    The iterables should return one mask per iteration and the itens of
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

    for (curr_gt_mask, curr_pred_mask) in tqdm(zip(gt_masks, pred_masks), unit=' masks'):
        flat_gt = __flat_mask(curr_gt_mask)
        flat_pred = __flat_mask(curr_pred_mask)
        for (curr_pred, curr_gt) in zip(flat_pred, flat_gt):
            confusion_matrix[curr_pred, curr_gt] += 1

    metrics = SegmentationMetrics(classes, confusion_matrix)

    print(metrics)

    return metrics


def evaluate_classification(gt_classifications: Iterable[Classification],
                            pred_classifications: Iterable[Classification],
                            classes: List[str]) -> ClassificationMetrics:
    """Evaluates classification predictions

    The iterables should return one classification per iteration and 
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

    for (curr_gt_classification, curr_pred_classification) in tqdm(zip(gt_classifications, pred_classifications), unit=' samples'):
        confusion_matrix[curr_pred_classification.cls,
                         curr_gt_classification.cls] += 1

    metrics = ClassificationMetrics(classes, confusion_matrix)

    print(metrics)

    return metrics


def evaluate_detection_single_image(gt_bboxes: List[BBox],
                                    pred_bboxes: List[BBox],
                                    iou_threshold=.5) -> DetectionMetrics:
    pass


def __merge_detection_metrics(acc: DetectionMetrics,
                              curr: DetectionMetrics) -> DetectionMetrics:
    pass


def evaluate_detection(gt_bboxes: Iterable[List[BBox]],
                       pred_bboxes: Iterable[List[BBox]],
                       classes: List[str],
                       iou_threshold: float = .5,
                       undetected_cls_name: str = "_undetected_") -> DetectionMetrics:
    """Evaluates detection predictions

    The iterables should return the bboxes of one image per iteration 
    and the itens of `gt_bboxes` and `pred_bboxes` with same index 
    should correspond to the same image.

    Args:
        gt_bboxes (Iterable[List[BBox]]): Ground truth classifications.
        pred_bboxes (Iterable[List[BBox]]): Predicted classifications.
        classes (List[str]): Class names.
        iou_threshold (float): Minimum IoU threshold to consider a detection as a True Positive.
        undetected_cls_name (str): Name to be used for undetected instances class. Defaults to "_undetected_".

    Returns:
        DetectionMetrics: Detection metrics.
    """

    confusion_matrix = np.zeros((len(classes) + 1, len(classes) + 1), np.int)
    undetected_idx = len(classes)

    for (curr_gt_bboxes, curr_pred_bboxes) in tqdm(zip(gt_bboxes, pred_bboxes), unit=' samples'):
        pairwise_bbox_ious = calculate_pairwise_bbox_ious(
            curr_gt_bboxes, curr_pred_bboxes)

        no_hit_idxs = set(range(0, len(curr_pred_bboxes)))

        for i, gt_ious in enumerate(pairwise_bbox_ious):
            gt_cls_idx = curr_gt_bboxes[i].cls

            hits_idxs = [i for i, iou in enumerate(gt_ious) if iou >= iou_threshold]
            no_hit_idxs.difference_update(hits_idxs)

            if len(hits_idxs) == 0:
                confusion_matrix[undetected_idx, gt_cls_idx] += 1
            else: 
                for hit_idx in hits_idxs:
                    confusion_matrix[curr_pred_bboxes[hit_idx].cls, gt_cls_idx] += 1

        for no_hit_idx in no_hit_idxs:
            confusion_matrix[curr_pred_bboxes[no_hit_idx].cls, undetected_idx] += 1

    metrics = DetectionMetrics(classes + [undetected_cls_name], confusion_matrix)

    print(metrics)

    return metrics


def calculate_pairwise_bbox_ious(gt_bboxes: List[BBox],
                                 pred_bboxes: List[BBox]) -> List[List[float]]:
    """Calculates the [gt x pred] matrix of pairwise IoUs of GT and predicted bboxes of an image.

    Args:
        gt_bboxes (List[BBox]): GT bboxes
        pred_bboxes (List[BBox]): Predicted bboxes

    Returns:
        List[List[float]]: [gt x pred] matrix of pairwise IoUs of GT and predicted bboxes
    """

    ious = np.zeros((len(gt_bboxes), len(pred_bboxes)), np.float)

    for i, gt_bbox in enumerate(gt_bboxes):
        for j, pred_bbox in enumerate(pred_bboxes):
            ious[i, j] = calculate_bbox_iou(gt_bbox, pred_bbox)

    return ious


def calculate_bbox_iou(bbox_a: BBox, bbox_b: BBox) -> float:
    """Calculates the IoU between 2 bboxes.

    Args:
        bbox_a (BBox): BBox.
        bbox_b (BBox): BBox.

    Returns:
        float: IoU between the two bboxes.
    """

    # Gets each box upper left and bottom right coordinates
    (upr_lft_x_a, upr_lft_y_a) = bbox_a.upper_left_point
    (btm_rgt_x_a, btm_rgt_y_a) = bbox_a.bottom_right_point

    (upr_lft_x_b, upr_lft_y_b) = bbox_b.upper_left_point
    (btm_rgt_x_b, btm_rgt_y_b) = bbox_b.bottom_right_point

    # Calculates the intersection box upper left and bottom right coordinates
    (upr_lft_x_intersect, upr_lft_y_intersect) = (
        max(upr_lft_x_a, upr_lft_x_b), max(upr_lft_y_a, upr_lft_y_b))
    (btm_rgt_x_intersect, btm_rgt_y_intersect) = (
        min(btm_rgt_x_a, btm_rgt_x_b), min(btm_rgt_y_a, btm_rgt_y_b))

    # Calculates the height and width of the intersection box
    (w_intersect, h_intersect) = (btm_rgt_x_intersect -
                                  upr_lft_x_intersect + 1, btm_rgt_y_intersect - upr_lft_y_intersect + 1)

    # IoU = 0 if there is no intersection
    if (w_intersect <= 0) or (h_intersect <= 0):
        return 0.0

    intersect_area = float(w_intersect * h_intersect)
    union_area = float(bbox_a.area + bbox_b.area - intersect_area)

    return intersect_area / union_area
