from typing import Optional, Union, List, Tuple
from functools import reduce
import numpy as np


class BBox:
    def __init__(self,
                 upper_left_x: int, upper_left_y: int,
                 width: int, height: int,
                 cls: int, score: float = .0):
        self.height = height
        self.width = width
        self.upper_left_x = upper_left_x
        self.upper_left_y = upper_left_y
        self.cls = cls
        self.score = score

    @property
    def upper_left_point(self) -> Tuple[int, int]:
        return (self.upper_left_x, self.upper_left_y)

    @property
    def bottom_right_point(self) -> Tuple[int, int]:
        return (self.upper_left_x + self.width, self.upper_left_y + self.height)


Mask = List[List[int]]


class Classification:
    def __init__(self, cls: int, score: Optional[float] = None):
        self.cls = cls
        self.score = score


class BinaryClassificationMetrics:
    TP: int = 0
    FP: int = 0
    TN: int = 0
    FN: int = 0

    def __init__(self, cls: str, tp: int = 0, fp: int = 0, tn: int = 0, fn: int = 0):
        self._cls = cls
        self.TP = tp
        self.FP = fp
        self.TN = tn
        self.FN = fn

    @property
    def cls(self) -> str:
        return self._cls

    @property
    def count(self) -> float:
        return self.TP + self.TN + self.FP + self.FN

    @property
    def accuracy(self) -> float:
        return (self.TP + self.TN)/self.count

    @property
    def recall(self) -> float:
        return self.TP/(self.TP + self.FN)

    @property
    def specificity(self) -> float:
        return self.TN/(self.FP + self.TN)

    @property
    def precision(self) -> float:
        return self.TP/(self.FP + self.TP)

    @property
    def f_score(self) -> float:
        return 2*self.TP/(self.FP + self.FN + 2*self.TP)

    def __str__(self):
        return (
            f'{self.cls}:\n'
            f'\tTP: {self.TP}\n'
            f'\tTN: {self.TN}\n'
            f'\tFP: {self.FP}\n'
            f'\tFN: {self.FN}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tRecall: {self.recall}\n'
            f'\tPrecision: {self.precision}\n'
            f'\tSpecificity: {self.specificity}\n'
            f'\tF-Score: {self.f_score}'
        )


class ClassificationMetrics:
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

    def __init__(self, classes: List[str], confusion_matrix: List[List[int]] = []):
        self._confusion_matrix = np.array(confusion_matrix)

        assert self._confusion_matrix.shape == (len(classes), len(
            classes)), 'The confusion matrix must be a square matrix of order len(classes)'

        self._classes = classes

        self._by_class = self.__get_by_class_metrics()
        self._count = self._confusion_matrix.sum()

    @property
    def by_class(self) -> List[BinaryClassificationMetrics]:
        return self._by_class

    @property
    def count(self) -> int:
        return self._count

    @property
    def accuracy(self) -> float:
        return np.diagonal(self._confusion_matrix).sum() / self.count

    @property
    def avg_recall(self) -> float:
        return reduce(lambda acc, curr: curr.recall + acc, self.by_class, .0) / len(self.by_class)

    @property
    def avg_precision(self) -> float:
        return reduce(lambda acc, curr: curr.precision + acc, self.by_class, .0) / len(self.by_class)

    @property
    def avg_specificity(self) -> float:
        return reduce(lambda acc, curr: curr.specificity + acc, self.by_class, .0) / len(self.by_class)

    @property
    def avg_f_score(self) -> float:
        return reduce(lambda acc, curr: curr.f_score + acc, self.by_class, .0) / len(self.by_class)

    def __str__(self):
        return (
            f'Classification Metrics:\n'
            f'\tSample Count: {self.count}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tAvg Recall: {self.avg_recall}\n'
            f'\tAvg Precision: {self.avg_precision}\n'
            f'\tAvg Specificity: {self.avg_specificity}\n'
            f'\tAvg F-Score: {self.avg_f_score}\n\n'
            f'By Class:\n\n'
        ) + '\n'.join([cls_metrics.__str__()
                       for cls_metrics in self.by_class])


class BinarySegmentationMetrics(BinaryClassificationMetrics):
    def __init__(self, classification_metrics: BinaryClassificationMetrics):
        super().__init__(
            classification_metrics.cls,
            tp=classification_metrics.TP,
            tn=classification_metrics.TN,
            fp=classification_metrics.FP,
            fn=classification_metrics.FN
        )

    @property
    def iou(self):
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
            f'\tF-Score: {self.f_score}\n'
        )


class SegmentationMetrics(ClassificationMetrics):
    def __init__(self, classes: List[str], confusion_matrix: List[List[int]] = []):
        assert len(classes) > 1, \
            'There should be at least two classes (with background as the first)'
        super().__init__(classes, confusion_matrix)
        self._by_class = [BinarySegmentationMetrics(x) for x in self.by_class]

    @property
    def avg_iou(self):
        return reduce(lambda acc, curr: curr.iou + acc, self.by_class, .0) / len(self.by_class)

    @property
    def avg_iou_no_bkg(self):
        return reduce(lambda acc, curr: curr.iou + acc, self.by_class[1:], .0) / (len(self.by_class) - 1)

    def __str__(self):
        return (
            f'Segmentation Metrics:\n'
            f'\tPixel Count: {self.count}\n'
            f'\tAccuracy: {self.accuracy}\n'
            f'\tAvg Recall: {self.avg_recall}\n'
            f'\tAvg Precision: {self.avg_precision}\n'
            f'\tAvg Specificity: {self.avg_specificity}\n'
            f'\tAvg F-Score: {self.avg_f_score}\n'
            f'\tAvg IoU: {self.avg_iou}\n'
            f'\tAvg IoU w/o Background: {self.avg_iou_no_bkg}\n'
            f'By Class:\n\n'
        ) + '\n'.join(list([cls_metrics.__str__()
                            for cls_metrics in self.by_class]))


class BinaryDetectionMetrics(BinaryClassificationMetrics):
    def __init__(self, classification_metrics: BinaryClassificationMetrics):
        super().__init__(
            classification_metrics.cls,
            tp=classification_metrics.TP,
            tn=classification_metrics.TN,
            fp=classification_metrics.FP,
            fn=classification_metrics.FN
        )

    @property
    def iou(self):
        pass

    @property
    def avg_precision(self):
        pass


class DetectionMetrics(ClassificationMetrics):
    def __init__(self, classes: List[str], confusion_matrix: List[List[int]] = []):
        super().__init__(classes, confusion_matrix)
        self._by_class = [BinarySegmentationMetrics(x) for x in self.by_class]

    @property
    def avg_iou(self):
        return reduce(lambda acc, curr: curr.iou + acc, self.by_class, .0) / len(self.by_class)

    @property
    def mean_avg_precision(self):
        return reduce(lambda acc, curr: curr.avg_precision + acc, self.by_class, .0) / len(self.by_class)
