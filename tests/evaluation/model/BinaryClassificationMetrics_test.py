import math
from lapixdl.evaluation.model import BinaryClassificationMetrics


def test_count():
    bin_class_A = BinaryClassificationMetrics(cls=['a', 'b'], FN=1, TP=2)

    assert bin_class_A.count == 3

    bin_class_B = BinaryClassificationMetrics(cls=['a', 'b'], FN=1, TP=2, FP=2)


    assert bin_class_B.count == 5


def test_accuracy():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], FN=2, TN=2, TP=2, FP=2)

    assert bin_class.accuracy == 0.5


def test_recall():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], TP=2, FN=6)

    assert bin_class.recall == 0.25

def test_recall_zero():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], TP=0, FN=0)

    assert math.isnan(bin_class.recall)


def test_fpr():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], FP=2, TN=6)

    assert bin_class.false_positive_rate == 0.25

def test_specificity():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], FP=2, TN=6)

    assert bin_class.specificity == 0.75

def test_specificity_zero():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], FP=0, TN=0)

    assert math.isnan(bin_class.specificity)


def test_precision():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], TP=8, FP=2)

    assert bin_class.precision == 0.8

def test_precision_zero():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], TP=0, FP=0)

    assert bin_class.precision == 1

def test_f_score():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], FN=4, TN=2, TP=5, FP=6)

    assert bin_class.f_score == 0.5

def test_f_score_zero():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], FN=0, TN=0, TP=0, FP=0)

    assert bin_class.f_score == 1
