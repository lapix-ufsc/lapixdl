from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum
from functools import reduce
from typing import Generic
from typing import List
from typing import TypeVar
from typing import Union

import numpy as np
import pandas as pd

from . import plot

# A Mask is a 2D array representing the class index of each pixel of an image as values
Mask = List[List[int]]

# A Image is a 3D array representing RGB values or a 2D array representing grayscale values
Image = Union[List[List[List[int]]], List[List[int]]]

TResult = TypeVar('TResult')


def recall_string(recall: float) -> str:
    return 'No positive cases in GT' if math.isnan(recall) else str(recall)


def specificity_string(specificity: float) -> str:
    return 'No negative cases in GT' if math.isnan(specificity) else str(specificity)


@dataclass
class Result(Generic[TResult]):
    """Result of a GT versus Predictions

    Args:
        Generic (Result type): Type of the results.

    Attributes:
        image (Image): Image that the GT and predictions are based. Format: (Rows, Cols, Chanels) or (Rows, Cols) for grayscale.
        gt (TResult): Ground truth results.
        prediction (Optional[TResult]): Prediction results. None if only GT must be shown.
    """
    image: Image
    gt: TResult
    prediction: TResult | None = None


class PredictionResultType(Enum):
    TP = 0
    FP = 1
    TN = 2
    FN = 3


@dataclass
class PredictionResult:
    score: float
    type: PredictionResultType


@dataclass
class Classification:
    """Represents a classification

    Attributes:
        cls (int): Class index.
        score (Optional[float], optional): Prediction score. Defaults to None.
    """
    cls: int
    score: float | None = None


@dataclass
class BinaryClassificationMetrics:
    """Binary classification metrics

    Attributes:
        cls (str): Class name.
        TP (int): True Positive count. Defaults to 0.
        FP (int): False Positive count. Defaults to 0.
        TN (int): True Negative count. Defaults to 0.
        FN (int): False Negative count. Defaults to 0.
    """

    cls: str
    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0

    @property
    def has_instances(self) -> bool:
        """int: Indicates if the class has any ground truth or predicted instances."""
        return self.count > 0

    @property
    def count(self) -> int:
        """int: Total count of classified instances."""
        return self.TP + self.TN + self.FP + self.FN

    @property
    def accuracy(self) -> float:
        """int: Total count of classified instances."""
        if self.count == 0:
            return math.nan
        return (self.TP + self.TN) / self.count

    @property
    def recall(self) -> float:
        """float: Recall metric - TP / (TP + FN)."""
        if self.TP == 0 and self.FN == 0:
            return math.nan
        return self.TP / (self.TP + self.FN)

    @property
    def false_positive_rate(self) -> float:
        """float: False Positive Rate (FPR) metric - FP / (FP + TN)."""
        if self.FP == 0 and self.TN == 0:
            return math.nan
        return self.FP / (self.FP + self.TN)

    @property
    def specificity(self) -> float:
        """float: Specificity metric - TN / (FP + TN)."""
        if self.FP == 0 and self.TN == 0:
            return math.nan

        return self.TN / (self.FP + self.TN)

    @property
    def precision(self) -> float:
        """float: Precision metric - TP / (FP + TP)."""
        if self.FP == 0 and self.FN == 0:  # No GT instances
            return 1
        elif self.FP == 0 and self.TP == 0:
            return math.nan
        return self.TP / (self.FP + self.TP)

    @property
    def f_score(self) -> float:
        """float: F-Score/Dice metric - 2*TP / (FP + FN + 2*TP)."""
        quotient = (self.FP + self.FN + 2 * self.TP)
        if self.TP == 0 and self.FP == 0 and self.FN == 0:  # No GT instances
            return 1
        elif quotient == 0:
            return math.nan
        return 2 * self.TP / quotient

    @property
    def confusion_matrix(self) -> list[list[int]]:
        """List[List[int]]: Confusion matrix of all the classes"""
        return [[self.TP, self.FP], [self.FN, self.TN]]

    def show_confusion_matrix(self):
        """Plots de confusion matrix

        Return:
            Tuple[Figure, Axes]: Figure and Axes of the ploted confusion matrix
        """
        return plot.confusion_matrix(self.confusion_matrix, ['P', 'N'], f'Confusion Matrix for \"{self.cls}\" Class')

    def __str__(self):
        no_instances_string = ' [NO INSTANCES]' if not self.has_instances else ''
        return (
            f'{self.cls}{no_instances_string}:\n'
            f'\tTP: {self.TP}\n'
            f'\tTN: {self.TN}\n'
            f'\tFP: {self.FP}\n'
            f'\tFN: {self.FN}\n'
            f'\tFPR: {self.false_positive_rate}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tRecall: {recall_string(self.recall)}\n'
            f'\tPrecision: {self.precision}\n'
            f'\tSpecificity: {specificity_string(self.specificity)}\n'
            f'\tF-Score: {self.f_score}'
        )

    def to_dict(self) -> dict:
        if not self.has_instances:
            return {}
        else:
            return {
                'TP': self.TP,
                'TN': self.TN,
                'FP': self.FP,
                'FN': self.FN,
                'FPR': self.false_positive_rate,
                'Accuracy': self.accuracy,
                'Recall': recall_string(self.recall),
                'Precision': self.precision,
                'Specificity': specificity_string(self.specificity),
                'F-Score': self.f_score,
            }


class ClassificationMetrics:
    """Multiclass classification metrics

    Attributes:
        classes (List[str]): Class names.
        confusion_matrix (List[List[int]]): Confusion matrix of all the classes.
    """

    def __init__(self, classes: list[str], confusion_matrix: list[list[int]] = []):
        """
        Args:
            classes (List[str]): Class names.
            confusion_matrix (List[List[int]], optional): Confusion matrix of all the classes. Defaults to [].

        Raises:
            AssertException: If the confusion matrix is not a square matrix of order `len(classes)`
        """
        self._confusion_matrix = np.array(confusion_matrix)

        assert self._confusion_matrix.shape == (len(classes), len(
            classes)), 'The confusion matrix must be a square matrix of order len(classes)'

        self._classes = classes

        self._by_class = self.__get_by_class_metrics()
        self._by_class_w_instances = list(filter(
            lambda c: c.has_instances, self._by_class))
        self._count = self._confusion_matrix.sum()

    @property
    def by_class(self) -> list[BinaryClassificationMetrics]:
        """List[BinaryClassificationMetrics]: Binary metrics calculated for each class index."""
        return self._by_class

    @property
    def by_class_w_instances(self) -> list[BinaryClassificationMetrics]:
        """List[BinaryClassificationMetrics]: Binary metrics calculated for each class index with instances."""
        return self._by_class_w_instances

    @property
    def count(self) -> int:
        """int: Total count of classified instances."""
        return self._count

    @property
    def accuracy(self) -> float:
        """float: Accuracy metric - correct classifications / count."""
        return np.diagonal(self._confusion_matrix).sum() / self.count

    @property
    def avg_recall(self) -> float:
        """float: Macro average recall metric."""
        by_class_w_recall = [
            c for c in self.by_class_w_instances if not math.isnan(c.recall)]

        if (len(by_class_w_recall) == 0):
            return 1
        return reduce(lambda acc, curr: curr.recall + acc, by_class_w_recall, .0) / len(by_class_w_recall)

    @property
    def avg_precision(self) -> float:
        """float: Macro average precision metric."""
        return reduce(lambda acc, curr: (0 if math.isnan(curr.precision)
                                         else curr.precision) + acc,
                      self.by_class_w_instances, .0) / len(self.by_class_w_instances)

    @property
    def avg_specificity(self) -> float:
        """float: Macro average specificity metric."""
        by_class_w_specificity = [c for c in self.by_class_w_instances
                                  if not math.isnan(c.specificity)]
        return reduce(lambda acc, curr: curr.specificity + acc, by_class_w_specificity, .0) / len(by_class_w_specificity)

    @property
    def avg_f_score(self) -> float:
        """float: Macro average F-Score/Dice metric."""
        return reduce(lambda acc, curr: curr.f_score + acc, self.by_class_w_instances, .0) / len(self.by_class_w_instances)

    @property
    def avg_false_positive_rate(self) -> float:
        """float: Macro average False Positive Rate metric."""
        return reduce(lambda acc, curr: curr.false_positive_rate + acc,
                      self.by_class_w_instances, .0) / len(self.by_class_w_instances)

    @property
    def confusion_matrix(self) -> list[list[int]]:
        """List[List[int]]: Confusion matrix of all the classes"""
        return self._confusion_matrix

    def show_confusion_matrix(self):
        """Plots de confusion matrix

        Return:
            Tuple[Figure, Axes]: Figure and Axes of the plotted confusion
            matrix
        """
        return plot.confusion_matrix(self._confusion_matrix, self._classes)

    def __get_by_class_metrics(self):
        by_class = []
        for i in range(0, len(self._classes)):
            tp = self._confusion_matrix[i, i]
            tn = np.delete(np.delete(self._confusion_matrix, i, 0), i, 1).sum()
            fp = self._confusion_matrix[i, :].sum() - tp
            fn = self._confusion_matrix[:, i].sum() - tp

            by_class.append(BinaryClassificationMetrics(
                cls=self._classes[i],
                TP=tp,
                TN=tn,
                FP=fp,
                FN=fn
            ))
        return by_class

    def __str__(self):
        return (
            f'Classification Metrics:\n'
            f'\tSample Count: {self.count}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tAvg Recall: {self.avg_recall}\n'
            f'\tAvg Precision: {self.avg_precision}\n'
            f'\tAvg Specificity: {self.avg_specificity}\n'
            f'\tAvg FPR: {self.avg_false_positive_rate}\n'
            f'\tAvg F-Score: {self.avg_f_score}\n\n'
            f'By Class:\n\n'
        ) + '\n'.join([cls_metrics.__str__()
                       for cls_metrics in self.by_class])

    def to_dict(self) -> dict:
        return {
            'Sample Count': self.count,
            'Average': {
                'Accuracy': self.accuracy,
                'Recall': self.avg_recall,
                'Precision': self.avg_precision,
                'Specificity': self.avg_specificity,
                'FPR': self.avg_false_positive_rate,
                'F-Score': self.avg_f_score
            },
            'By Class': dict_by_class(self.by_class)
        }

    def to_dataframe(self) -> pd.DataFrame:
        as_dict = self.to_dict()
        dict_to_df = {
            'Average': as_dict['Average'],
            **as_dict['By Class']
        }
        return pd.DataFrame(dict_to_df)


class BinarySegmentationMetrics(BinaryClassificationMetrics):
    """Binary pixel-based classification metrics

    Attributes:
        TP (int): True Positive pixels count.
        FP (int): False Positive pixels count.
        TN (int): True Negative pixels count.
        FN (int): False Negative pixels count.
    """

    def __init__(self, classification_metrics: BinaryClassificationMetrics):
        """
        Args:
            classification_metrics (BinaryClassificationMetrics): Pixel-based binary classification metrics
        """
        super().__init__(
            cls=classification_metrics.cls,
            TP=classification_metrics.TP,
            TN=classification_metrics.TN,
            FP=classification_metrics.FP,
            FN=classification_metrics.FN
        )

    @property
    def iou(self) -> float:
        """float: IoU/Jaccard Index metric - TP / (FP + FN + TP)."""
        quotient = (self.FP + self.FN + self.TP)
        if quotient == 0:
            return math.nan
        return self.TP / quotient

    def __str__(self):
        return (
            f'{self.cls}:\n'
            f'\tTP: {self.TP}\n'
            f'\tTN: {self.TN}\n'
            f'\tFP: {self.FP}\n'
            f'\tFN: {self.FN}\n'
            f'\tIoU: {self.iou}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tRecall: {recall_string(self.recall)}\n'
            f'\tPrecision: {self.precision}\n'
            f'\tSpecificity: {specificity_string(self.specificity)}\n'
            f'\tFPR: {self.false_positive_rate}\n'
            f'\tF-Score: {self.f_score}\n'
        )

    def to_dict(self) -> dict:
        return {
            'TP': self.TP,
            'TN': self.TN,
            'FP': self.FP,
            'FN': self.FN,
            'IoU': self.iou,
            'Accuracy': self.accuracy,
            'Recall': recall_string(self.recall),
            'Precision': self.precision,
            'Specificity': specificity_string(self.specificity),
            'FPR': self.false_positive_rate,
            'F-Score': self.f_score
        }


class SegmentationMetrics(ClassificationMetrics):
    """Multiclass pixel-based classification metrics

    Attributes:
        classes (List[str]): Class names with `classes[0]` as the background class.
        confusion_matrix (List[List[int]]): Confusion matrix of all the classes.
    """

    def __init__(self, classes: list[str], confusion_matrix: list[list[int]] = []):
        """
        Args:
            classes (List[str]): Class names. It is expected the first class to be the background class.
            confusion_matrix (List[List[int]], optional): Confusion matrix of all the classes. Defaults to [].

        Raises:
            AssertException: If the confusion matrix is not a square matrix of order `len(classes)`.
            AssertException: If the number of classes is less than 2.
        """
        assert len(classes) > 1, \
            'There should be at least two classes (with background as the first)'
        super().__init__(classes, confusion_matrix)
        self._by_class = [BinarySegmentationMetrics(x) for x in self.by_class]
        self._by_class_w_instances = [
            x for x in self.by_class if x.has_instances]

    @property
    def avg_iou(self) -> float:
        """float: Macro average IoU/Jaccard Index metric."""
        return reduce(lambda acc, curr: curr.iou + acc, self._by_class_w_instances, .0) / len(self._by_class_w_instances)

    @property
    def avg_iou_no_bkg(self) -> float:
        """float: Macro average IoU/Jaccard Index metric without `background` class (index 0)."""
        return reduce(lambda acc, curr: curr.iou + acc,
                      self._by_class_w_instances[1:], .0) / (len(self._by_class_w_instances) - 1)

    def __str__(self):
        return (
            f'Segmentation Metrics:\n'
            f'\tPixel Count: {self.count}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tAvg Recall: {self.avg_recall}\n'
            f'\tAvg Precision: {self.avg_precision}\n'
            f'\tAvg Specificity: {self.avg_specificity}\n'
            f'\tAvg F-Score: {self.avg_f_score}\n'
            f'\tAvg FPR: {self.avg_false_positive_rate}\n'
            f'\tAvg IoU: {self.avg_iou}\n'
            f'\tAvg IoU w/o Background: {self.avg_iou_no_bkg}\n'
            f'By Class:\n\n'
        ) + '\n'.join(list(cls_metrics.__str__()
                           for cls_metrics in self.by_class))

    def to_dict(self) -> dict:
        return {
            'Pixel Count': self.count,
            'Average': {
                'Accuracy': self.accuracy,
                'Recall': self.avg_recall,
                'Precision': self.avg_precision,
                'Specificity': self.avg_specificity,
                'F-Score': self.avg_f_score,
                'FPR': self.avg_false_positive_rate,
                'IoU': self.avg_iou,
                'IoU w/o Background': self.avg_iou_no_bkg
            },
            'By Class': dict_by_class(self.by_class)
        }

    def to_dataframe(self) -> pd.DataFrame:
        as_dict = self.to_dict()
        dict_to_df = {
            'Average': as_dict['Average'],
            **as_dict['By Class']
        }
        return pd.DataFrame(dict_to_df)


class BinaryDetectionMetrics(BinaryClassificationMetrics):
    """Binary instance-based classification metrics

    Attributes:
        TP (int): True Positive instances count.
        FP (int): False Positive instances count.
        TN (int): Not used in object detection. Always 0.
        FN (int): False Negative instances count.
    """

    def __init__(self, classification_metrics: BinaryClassificationMetrics,
                 iou: float,
                 predictions: list[PredictionResult]):
        super().__init__(
            cls=classification_metrics.cls,
            TP=classification_metrics.TP,
            FP=classification_metrics.FP,
            FN=classification_metrics.FN
        )
        self._iou = iou
        self._precision_recall_curve = self.__calculate_precision_recall_curve(
            predictions) if self.gt_count > 0 else []

    @property
    def gt_count(self) -> int:
        """int: Total count of GT bboxes."""
        return self.TP + self.FN

    @property
    def predicted_count(self) -> int:
        """int: Total count of predicted bboxes."""
        return self.TP + self.FP

    @property
    def iou(self) -> float:
        """float: IoU/Jaccard Index metric.

        This metric is calculated as for segmentation, considering bboxes as pixel masks.
        """
        return self._iou

    @property
    def precision_recall_curve(self) -> list[tuple[float, float]]:
        """List[Tuple[float, float]]: Precision x Recall curve as a list of (Recall, Precision) tuples."""
        assert self.gt_count > 0, 'This class does not have instances.'
        return self._precision_recall_curve

    def average_precision(self, interpolation_points: int | None = None) -> float:
        """Calculates the Average Precision metric.

        Args:
            interpolation_points (Optional[int], optional): Number of points to use for interpolation.
            Uses all points if None. Defaults to None.

        Returns:
            float: Average Precision metric.
        """
        if interpolation_points is not None:
            return self._interpolated_average_precision(interpolation_points)
        return self._average_precision()

    def show_precision_recall_curve(self):
        """Plots de Precision x Recall curve.

        Return:
            Tuple[Figure, Axes]: Figure and Axes of the ploted Precision x Recall curve.
        """
        return plot.precision_recall_curve([self._precision_recall_curve], [self.cls])

    def __calculate_precision_recall_curve(self, predictions: list[PredictionResult]) -> list[tuple[float, float]]:
        assert self.gt_count > 0, 'The GT count must be greater than 0.'

        sorted_predictions = sorted(
            predictions, key=lambda p: float(p.score), reverse=True)

        curve: list[tuple[float, float]] = []
        tp = 0
        fp = 0
        gt_count = self.gt_count

        def calc_precision(tp, fp):
            return tp / (tp + fp)

        def calc_recall(tp):
            return tp / gt_count

        for prediction in sorted_predictions:
            if prediction.type == PredictionResultType.TP:
                tp += 1
            elif prediction.type == PredictionResultType.FP:
                fp += 1
            curve.append((calc_recall(tp), calc_precision(tp, fp)))

        return curve

    def _interpolated_average_precision(self, interpolation_points: int) -> float:
        precision_sum = .0
        recall_interpolation = np.linspace(0, 1, interpolation_points)

        for point in recall_interpolation:
            precision_sum += max([rp[1]
                                  for rp in self._precision_recall_curve if rp[0] >= point] + [0])

        return precision_sum / interpolation_points

    def _average_precision(self) -> float:
        ap_sum = .0
        prev_max_recall = .0

        greater_recall_points = self._precision_recall_curve
        while len(greater_recall_points) > 0:
            max_point = max(greater_recall_points, key=lambda rp: rp[1])
            ap_sum += (max_point[0] - prev_max_recall) * max_point[1]
            prev_max_recall = max_point[0]
            greater_recall_points = [
                rp for rp in greater_recall_points if rp[0] > prev_max_recall]

        return ap_sum

    def __str__(self):
        return (
            f'{self.cls}:\n'
            f'\tTP: {self.TP}\n'
            f'\tFP: {self.FP}\n'
            f'\tFN: {self.FN}\n'
            f'\tTN: [NA]\n'
            f'\tIoU: {self.iou}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tRecall: {recall_string(self.recall)}\n'
            f'\tPrecision: {self.precision}\n'
            f'\tF-Score: {self.f_score}\n'
            f'\tAverage Precision: {self.average_precision()}\n'
            f'\t11-point Average Precision: {self.average_precision(11)}\n'
        )

    def to_dict(self) -> dict:
        return {
            'TP': self.TP,
            'FP': self.FP,
            'FN': self.FN,
            'TN': math.nan,
            'IoU': self.iou,
            'Accuracy': self.accuracy,
            'Recall': recall_string(self.recall),
            'Precision': self.precision,
            'F-Score': self.f_score,
            'Average Precision': self.average_precision(),
            '11-point Average Precision': self.average_precision(11)
        }


class DetectionMetrics(ClassificationMetrics):
    """Multiclass detection metrics

    Attributes:
        classes (List[str]): Class names. The last must be the "undetected" class.
        confusion_matrix (List[List[int]]): Confusion matrix of all the classes.
        iou_by_class (List[float]): IoU values indexed by class.

        The last column and line must correspond to the "undetected" class.
    """

    def __init__(self, classes: list[str],
                 confusion_matrix: list[list[int]],
                 iou_by_class: list[float],
                 predictions_by_class: list[list[PredictionResult]]):
        super().__init__(classes, confusion_matrix)
        by_class_wo_undetected = self.by_class[slice(0, -1)]
        self._by_class = [BinaryDetectionMetrics(by_class, iou, predictions)
                          for by_class, iou, predictions in zip(by_class_wo_undetected, iou_by_class, predictions_by_class)]
        self._by_class_w_instances = [
            x for x in self.by_class if x.has_instances]

    @property
    def avg_iou(self):
        """float: Macro average IoU/Jaccard Index metric.

        This metric is calculated as for segmentation, considering bboxes as pixel masks.
        """
        return reduce(lambda acc, curr: curr.iou + acc, self._by_class_w_instances, .0) / len(self._by_class_w_instances)

    def mean_average_precision(self, interpolation_points: int | None = None) -> float:
        """Calculates the mean of Average Precision metrics of all classes.

        Args:
            interpolation_points (Optional[int], optional): Number of points to use for interpolation.
            Uses all points if None. Defaults to None.

        Returns:
            float: Mean Average Precision metric.
        """
        return reduce(lambda acc,
                      curr: curr.average_precision(interpolation_points) + acc, self._by_class_w_instances,
                      .0) / len(self._by_class_w_instances)

    def __str__(self):
        return (
            f'Detection Metrics:\n'
            f'\tBboxes Count: {self.count}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tAvg Recall: {self.avg_recall}\n'
            f'\tAvg Precision: {self.avg_precision}\n'
            f'\tAvg F-Score: {self.avg_f_score}\n'
            f'\tAvg IoU: {self.avg_iou}\n'
            f'By Class:\n\n'
        ) + '\n'.join(list(cls_metrics.__str__()
                           for cls_metrics in self.by_class))

    def to_dict(self) -> dict:
        return {
            'Bboxes Count': self.count,
            'Average': {
                'Accuracy': self.accuracy,
                'Recall': self.avg_recall,
                'Precision': self.avg_precision,
                'F-Score': self.avg_f_score,
                'IoU': self.avg_iou
            },
            'By Class': dict_by_class(self.by_class)
        }

    def to_dataframe(self) -> pd.DataFrame:
        as_dict = self.to_dict()
        dict_to_df = {
            'Average': as_dict['Average'],
            **as_dict['By Class']
        }
        return pd.DataFrame(dict_to_df)


def dict_by_class(by_class) -> dict:
    return {
        cls_metrics.cls: cls_metrics.to_dict() for cls_metrics in by_class
    }
