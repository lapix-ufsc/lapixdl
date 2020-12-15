from typing import List, Iterable

import numpy as np
from tqdm import tqdm

from .model import *


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
    tot_images = 0
    classes_count = len(classes)

    cls_ious_sum = np.zeros(classes_count)
    confusion_matrix = np.zeros((classes_count + 1, classes_count + 1), np.int)
    undetected_idx = classes_count
    # Tracks detection scores to calculate the Precision x Recall curve and the Average Precision metric
    predictions_by_class: List[List[PredictionResult]] = [[] for i in range(len(classes))] 

    # For each image GT and predicted bbox set
    for (curr_gt_bboxes, curr_pred_bboxes) in tqdm(zip(gt_bboxes, pred_bboxes), unit=' samples'):
        pairwise_bbox_ious = calculate_pairwise_bbox_ious(
            curr_gt_bboxes, curr_pred_bboxes)

        # Sums classes' IoUs by image to calculate the average at the end
        curr_img_cls_ious = calculate_iou_by_class(
            curr_gt_bboxes, curr_pred_bboxes, classes_count)
        cls_ious_sum = np.add(cls_ious_sum, curr_img_cls_ious)
        tot_images += 1

        # Tracks prediction bboxes that does not correspond to any GT bbox to identify FPs
        no_hit_idxs = set(range(0, len(curr_pred_bboxes)))

        # Iterate through lines of predictions bbox IoUs for each GT bbox
        for i, gt_ious in enumerate(pairwise_bbox_ious):
            gt_cls_idx = curr_gt_bboxes[i].cls
            max_iou_idx = np.argmax(gt_ious)
            # Only the max IoU is considered TP, the others are FPs
            max_iou = gt_ious[max_iou_idx]

            if max_iou < iou_threshold: # FN - GT bbox not detected
                confusion_matrix[undetected_idx, gt_cls_idx] += 1
            else: # TP - GT bbox detected
                no_hit_idxs.remove(max_iou_idx) # Remove from FPs
                
                pred_cls_idx = curr_pred_bboxes[max_iou_idx].cls
                confusion_matrix[pred_cls_idx, gt_cls_idx] += 1
                predictions_by_class[pred_cls_idx]\
                    .append(PredictionResult(curr_pred_bboxes[max_iou_idx].score, PredictionResultType.TP))

        for no_hit_idx in no_hit_idxs: # FPs - Predictions that do not match any GT
            pred_cls_idx = curr_pred_bboxes[no_hit_idx].cls
            confusion_matrix[pred_cls_idx, undetected_idx] += 1
            predictions_by_class[pred_cls_idx]\
                .append(PredictionResult(curr_pred_bboxes[no_hit_idx].score, PredictionResultType.FP))

    metrics = DetectionMetrics(
        classes + [undetected_cls_name],
        confusion_matrix,
        cls_ious_sum / tot_images,
        predictions_by_class)

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

    intersect_area = bbox_a.intersection_area_with(bbox_b)
    union_area = bbox_a.union_area_with(bbox_b, intersect_area)

    return float(intersect_area) / float(union_area)


def calculate_iou_by_class(gt_bboxes: List[BBox],
                           pred_bboxes: List[BBox],
                           classes_count: int) -> List[float]:
    """Calculates the array of IoUs between GT and predicted bboxes of an image by class.
    This method considers bbox as segmentation masks to calculate the IoUs.

    Args:
        gt_bboxes (List[BBox]): GT bboxes
        pred_bboxes (List[BBox]): Predicted bboxes
        classes_count (int): Classes count

    Returns:
        List[float]: IoUs of an image indexed by class
    """

    ious = np.zeros(classes_count, np.float)

    for i in range(classes_count):
        ious[i] = __calculate_binary_iou([gt_bbox for gt_bbox in gt_bboxes if gt_bbox.cls == i],
                                         [pred_bbox for pred_bbox in pred_bboxes if pred_bbox.cls == i])

    return ious


def __calculate_binary_iou(gt_bboxes: List[BBox],
                           pred_bboxes: List[BBox]) -> float:

    bboxes_btm_right_points = [bbox.bottom_right_point for bbox in gt_bboxes]\
        + [bbox.bottom_right_point for bbox in pred_bboxes]
    max_x = max([point[0] for point in bboxes_btm_right_points])
    max_y = max([point[1] for point in bboxes_btm_right_points])

    gt_mask = __draw_bboxes((max_x + 1, max_y + 1), gt_bboxes)
    pred_mask = __draw_bboxes((max_x + 1, max_y + 1), pred_bboxes)

    tp, f = (0, 0)
    flat_gt = __flat_mask(gt_mask)
    flat_pred = __flat_mask(pred_mask)
    for (curr_pred, curr_gt) in zip(flat_pred, flat_gt):
        if curr_gt == 1 and curr_pred == 1:
            tp += 1
        elif curr_gt != curr_pred:
            f += 1

    return tp / (f + tp)


def __draw_bboxes(mask_shape: Tuple[int, int], bboxes: List[BBox]) -> List[List[int]]:
    mask = np.zeros(mask_shape, np.int)

    for bbox in bboxes:
        mask[
            bbox.upper_left_point[0]:bbox.bottom_right_point[0] + 1,
            bbox.upper_left_point[1]:bbox.bottom_right_point[1] + 1
        ] = 1

    return mask


def __flat_mask(mask: Mask) -> List[int]:
    return [item for sublist in mask for item in sublist]
