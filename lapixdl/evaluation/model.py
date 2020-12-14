from __future__ import annotations
from typing import Optional, Union, List, Tuple, Generic, TypeVar
from functools import reduce
import numpy as np

from . import plot

# A Mask is a 2D array representing the class index of each pixel of an image as values
Mask = List[List[int]]

# A Image is a 3D array representing RGB values or a 2D array representing grayscale values
Image = Union[List[List[List[int]]], List[List[int]]]

TResult = TypeVar('TResult')


class Result(Generic[TResult]):
    """Result of a GT versus Predictions

    Args:
        Generic (Result type): Type of the results.

    Attributes:
        image (Image): Image that the GT and predictions are based. Format: (Rows, Cols, Chanels) or (Rows, Cols) for grayscale.
        gt (TResult): Ground truth results.
        prediction (Optional[TResult]): Prediction results. None if only GT must be shown.
    """

    def __init__(self, image: Image, gt: TResult, prediction: Optional[TResult] = None):
        """

        Args:
            image (Image): Image that the GT and predictions are based. Format: (Rows, Cols, Chanels) or (Rows, Cols) for grayscale.
            gt (TResult): Ground truth results.
            prediction (Optional[TResult], optional): Prediction results. Pass None if only GT must be shown. Defaults to None.
        """
        self.image = image
        self.gt = gt
        self.prediction = prediction


class BBox:
    """Bounding Box data structure

    Attributes:
        upper_left_x (int): Uper left X position of the Bounding Box.
        upper_left_y (int): Upper left Y position of the Bounding Box.
        width (int): Width of the Bounding Box.
        height (int): Height of the Bounding Box.
        cls (int): Bounding Box class index.
        score (Optional[float]): Bounding Box prediction score.
    """

    def __init__(self,
                 upper_left_x: int, upper_left_y: int,
                 width: int, height: int,
                 cls: int, score: Optional[float] = None):
        """
        Args:
            upper_left_x (int): Uper left X position of the Bounding Box.
            upper_left_y (int): Upper left Y position of the Bounding Box.
            width (int): Width of the Bounding Box.
            height (int): Height of the Bounding Box.
            cls (int): Bounding Box class index.
            score (Optional[float], optional): Prediction score. Defaults to None.
        """
        self.height = height
        self.width = width
        self.upper_left_x = upper_left_x
        self.upper_left_y = upper_left_y
        self.cls = cls
        self.score = score

    @property
    def upper_left_point(self) -> Tuple[int, int]:
        """Tuple[int, int]: (X,Y) of the upper left point of the Bounding Box."""
        return (self.upper_left_x, self.upper_left_y)

    @property
    def bottom_right_point(self) -> Tuple[int, int]:
        """Tuple[int, int]: (X,Y) of the bottom right point of the Bounding Box."""
        return (self.upper_left_x + self.width - 1, self.upper_left_y + self.height - 1)
    
    @property
    def center_point(self) -> Tuple[int, int]:
        """Tuple[int, int]: (X,Y) of the center point of the Bounding Box."""
        return ((self.upper_left_x + self.width - 1) // 2, (self.upper_left_y + self.height - 1) // 2)

    @property
    def area(self) -> int:
        """int: Area of the Bounding Box."""
        return self.width * self.height

    def intersection_area_with(self: BBox, bbox: BBox) -> int:
        """Calculates the intersection area with another bbox

        Args:
            self (BBox): This bbox
            bbox (BBox): Bbox to intersect with

        Returns:
            int: The intersection area with the bbox
        """

        # Gets each box upper left and bottom right coordinates
        (upr_lft_x_a, upr_lft_y_a) = self.upper_left_point
        (btm_rgt_x_a, btm_rgt_y_a) = self.bottom_right_point

        (upr_lft_x_b, upr_lft_y_b) = bbox.upper_left_point
        (btm_rgt_x_b, btm_rgt_y_b) = bbox.bottom_right_point

        # Calculates the intersection box upper left and bottom right coordinates
        (upr_lft_x_intersect, upr_lft_y_intersect) = (
            max(upr_lft_x_a, upr_lft_x_b), max(upr_lft_y_a, upr_lft_y_b))
        (btm_rgt_x_intersect, btm_rgt_y_intersect) = (
            min(btm_rgt_x_a, btm_rgt_x_b), min(btm_rgt_y_a, btm_rgt_y_b))

        # Calculates the height and width of the intersection box
        (w_intersect, h_intersect) = (btm_rgt_x_intersect -
                                      upr_lft_x_intersect + 1, btm_rgt_y_intersect - upr_lft_y_intersect + 1)

        # If H or W <= 0, there is no intersection
        if (w_intersect <= 0) or (h_intersect <= 0):
            return 0

        return w_intersect * h_intersect

    def union_area_with(self: BBox, bbox: BBox, intersection_area: Optional[int] = None) -> int:
        """Calculates the union area with another bbox

        Args:
            self (BBox): This bbox
            bbox (BBox): Bbox to union with
            intersection_area (Optional[int], optional): The intersection area between this and bbox. Defaults to None.

        Returns:
            int: The union area with the bbox
        """

        return self.area + bbox.area - (intersection_area or self.intersection_area_with(bbox))


class Classification:
    """Represents a classification

    Attributes:
        cls (int): Class index.
        score (Optional[float], optional): Prediction score.
    """

    def __init__(self, cls: int, score: Optional[float] = None):
        """
        Args:
            cls (int): Class index.
            score (Optional[float], optional): Prediction score. Defaults to None.
        """
        self.cls = cls
        self.score = score


class BinaryClassificationMetrics:
    """Binary classification metrics

    Attributes:
        TP (int): True Positive count.
        FP (int): False Positive count.
        TN (int): True Negative count.
        FN (int): False Negative count.
    """

    def __init__(self, cls: str, tp: int = 0, fp: int = 0, tn: int = 0, fn: int = 0):
        """
        Args:
            cls (str): Class name.
            tp (int, optional): True Positive count. Defaults to 0.
            fp (int, optional): False Positive count. Defaults to 0.
            tn (int, optional): True Negative count. Defaults to 0.
            fn (int, optional): False Negative count. Defaults to 0.
        """
        self._cls = cls
        self.TP = tp
        self.FP = fp
        self.TN = tn
        self.FN = fn

    @property
    def cls(self) -> str:
        """str: Class name."""
        return self._cls

    @property
    def count(self) -> int:
        """int: Total count of classified instances."""
        return self.TP + self.TN + self.FP + self.FN

    @property
    def accuracy(self) -> float:
        """int: Total count of classified instances."""
        return (self.TP + self.TN)/self.count

    @property
    def recall(self) -> float:
        """float: Recall metric - TP / (TP + FN)."""
        return self.TP/(self.TP + self.FN)

    @property
    def false_positive_rate(self) -> float:
        """float: False Positive Rate (FPR) metric - FP / (FP + TN)."""
        return self.FP/(self.FP + self.TN)

    @property
    def specificity(self) -> float:
        """float: Specificity metric - TN / (FP + TN)."""
        return self.TN/(self.FP + self.TN)

    @property
    def precision(self) -> float:
        """float: Precision metric - TP / (FP + TP)."""
        return self.TP/(self.FP + self.TP)

    @property
    def f_score(self) -> float:
        """float: F-Score/Dice metric - 2*TP / (FP + FN + 2*TP)."""
        return 2*self.TP/(self.FP + self.FN + 2*self.TP)

    @property
    def confusion_matrix(self) -> List[List[int]]:
        """List[List[int]]: Confusion matrix of all the classes"""
        return [[self.TP, self.FP], [self.FN, self.TN]]

    def show_confusion_matrix(self):
        """Plots de confusion matrix

        Return:
            Tuple[Figure, Axes]: Figure and Axes of the ploted confusion matrix
        """
        return plot.confusion_matrix(self.confusion_matrix, ['P', 'N'], f'Confusion Matrix for \"{self.cls}\" Class')

    def __str__(self):
        return (
            f'{self.cls}:\n'
            f'\tTP: {self.TP}\n'
            f'\tTN: {self.TN}\n'
            f'\tFP: {self.FP}\n'
            f'\tFN: {self.FN}\n'
            f'\tFPR: {self.false_positive_rate}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tRecall: {self.recall}\n'
            f'\tPrecision: {self.precision}\n'
            f'\tSpecificity: {self.specificity}\n'
            f'\tF-Score: {self.f_score}'
        )


class ClassificationMetrics:
    """Multiclass classification metrics

    Attributes:
        classes (List[str]): Class names.
        confusion_matrix (List[List[int]]): Confusion matrix of all the classes.
    """

    def __init__(self, classes: List[str], confusion_matrix: List[List[int]] = []):
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
        self._count = self._confusion_matrix.sum()

    @property
    def by_class(self) -> List[BinaryClassificationMetrics]:
        """List[BinaryClassificationMetrics]: Binary metrics calculated for each class index."""
        return self._by_class

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
        return reduce(lambda acc, curr: curr.recall + acc, self.by_class, .0) / len(self.by_class)

    @property
    def avg_precision(self) -> float:
        """float: Macro average precision metric."""
        return reduce(lambda acc, curr: curr.precision + acc, self.by_class, .0) / len(self.by_class)

    @property
    def avg_specificity(self) -> float:
        """float: Macro average specificity metric."""
        return reduce(lambda acc, curr: curr.specificity + acc, self.by_class, .0) / len(self.by_class)

    @property
    def avg_f_score(self) -> float:
        """float: Macro average F-Score/Dice metric."""
        return reduce(lambda acc, curr: curr.f_score + acc, self.by_class, .0) / len(self.by_class)

    @property
    def avg_false_positive_rate(self) -> float:
        """float: Macro average False Positive Rate metric."""
        return reduce(lambda acc, curr: curr.false_positive_rate + acc, self.by_class, .0) / len(self.by_class)

    @property
    def confusion_matrix(self) -> List[List[int]]:
        """List[List[int]]: Confusion matrix of all the classes"""
        return self._confusion_matrix

    def show_confusion_matrix(self):
        """Plots de confusion matrix

        Return:
            Tuple[Figure, Axes]: Figure and Axes of the ploted confusion matrix
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
                tp=tp,
                tn=tn,
                fp=fp,
                fn=fn
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
            classification_metrics.cls,
            tp=classification_metrics.TP,
            tn=classification_metrics.TN,
            fp=classification_metrics.FP,
            fn=classification_metrics.FN
        )

    @property
    def iou(self) -> float:
        """float: IoU/Jaccard Index metric - TP / (FP + FN + TP)."""
        return self.TP / (self.FP + self.FN + self.TP)

    def __str__(self):
        return (
            f'{self.cls}:\n'
            f'\tTP: {self.TP}\n'
            f'\tTN: {self.TN}\n'
            f'\tFP: {self.FP}\n'
            f'\tFN: {self.FN}\n'
            f'\tIoU: {self.iou}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tRecall: {self.recall}\n'
            f'\tPrecision: {self.precision}\n'
            f'\tSpecificity: {self.specificity}\n'
            f'\tFPR: {self.false_positive_rate}\n'
            f'\tF-Score: {self.f_score}\n'
        )


class SegmentationMetrics(ClassificationMetrics):
    """Multiclass pixel-based classification metrics

    Attributes:
        classes (List[str]): Class names with `classes[0]` as the background class.
        confusion_matrix (List[List[int]]): Confusion matrix of all the classes.
    """

    def __init__(self, classes: List[str], confusion_matrix: List[List[int]] = []):
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

    @property
    def avg_iou(self) -> float:
        """float: Macro average IoU/Jaccard Index metric."""
        return reduce(lambda acc, curr: curr.iou + acc, self._by_class, .0) / len(self.by_class)

    @property
    def avg_iou_no_bkg(self) -> float:
        """float: Macro average IoU/Jaccard Index metric without `background` class (index 0)."""
        return reduce(lambda acc, curr: curr.iou + acc, self._by_class[1:], .0) / (len(self.by_class) - 1)

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
        ) + '\n'.join(list([cls_metrics.__str__()
                            for cls_metrics in self.by_class]))


class BinaryDetectionMetrics(BinaryClassificationMetrics):
    """Binary instance-based classification metrics

    Attributes:
        TP (int): True Positive instances count.
        FP (int): False Positive instances count.
        TN (int): Not used in object detection. Always 0.
        FN (int): False Negative instances count.
    """

    def __init__(self, classification_metrics: BinaryClassificationMetrics, iou: float):
        super().__init__(
            classification_metrics.cls,
            tp=classification_metrics.TP,
            fp=classification_metrics.FP,
            fn=classification_metrics.FN
        )
        self._iou = iou

    @property
    def iou(self):
        """float: IoU/Jaccard Index metric.

        This metric is calculated as for segmentation, considering bboxes as pixel masks.
        """
        return self._iou

    def __str__(self):
        return (
            f'{self.cls}:\n'
            f'\tTP: {self.TP}\n'
            f'\tFP: {self.FP}\n'
            f'\tFN: {self.FN}\n'
            f'\tTN: [NA]\n'
            f'\tIoU: {self.iou}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tRecall: {self.recall}\n'
            f'\tPrecision: {self.precision}\n'
            f'\tF-Score: {self.f_score}\n'
        )


class DetectionMetrics(ClassificationMetrics):
    """Multiclass detection metrics

    Attributes:
        classes (List[str]): Class names. The last must be the "undetected" class.
        confusion_matrix (List[List[int]]): Confusion matrix of all the classes.
        iou_by_class (List[float]): IoU values indexed by class.

        The last column and line must correspond to the "undetected" class.
    """

    def __init__(self, classes: List[str],
                 confusion_matrix: List[List[int]] = [],
                 iou_by_class: List[float] = []):
        super().__init__(classes, confusion_matrix)
        by_class_wo_undetected = self.by_class[slice(0, -1)]
        self._by_class = [BinaryDetectionMetrics(by_class, iou)
                          for by_class, iou in zip(by_class_wo_undetected, iou_by_class)]

    @property
    def avg_iou(self):
        """float: Macro average IoU/Jaccard Index metric.

        This metric is calculated as for segmentation, considering bboxes as pixel masks.
        """
        return reduce(lambda acc, curr: curr.iou + acc, self.by_class, .0) / len(self.by_class)

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
        ) + '\n'.join(list([cls_metrics.__str__()
                            for cls_metrics in self.by_class]))
