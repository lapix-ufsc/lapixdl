from lapixdl.evaluation.model import ClassificationMetrics, BinaryClassificationMetrics
import pytest


def test_constructor_invalid():
    with pytest.raises(AssertionError):
        ClassificationMetrics(['a', 'b'], [[1],[2]])

def test_constructor_valid():
    ClassificationMetrics(['a', 'b'], [[1,2],[2,1]])

def test_count():
    metrics = ClassificationMetrics(['a', 'b', 'c'], 
    [
        [4,6,3],
        [1,2,0],
        [1,2,6]
    ])

    assert metrics.count == 25

    class_a = metrics.by_class[0]
    class_b = metrics.by_class[1]
    class_c = metrics.by_class[2]

    assert class_a.count == 25
    assert class_b.count == 25
    assert class_c.count == 25


def test_accuracy():
    metrics = ClassificationMetrics(['a', 'b', 'c'], 
    [
        [4,6,3],
        [1,2,0],
        [1,2,6]
    ])

    assert round(metrics.accuracy, 3) == .480

    class_a = metrics.by_class[0]
    class_b = metrics.by_class[1]
    class_c = metrics.by_class[2]

    assert round(class_a.accuracy, 3) == .560
    assert round(class_b.accuracy, 3) == .640
    assert round(class_c.accuracy, 3) == .760


def test_recall():
    metrics = ClassificationMetrics(['a', 'b', 'c'], 
    [
        [4,6,3],
        [1,2,0],
        [1,2,6]
    ])

    assert round(metrics.avg_recall, 3) == .511

    class_a = metrics.by_class[0]
    class_b = metrics.by_class[1]
    class_c = metrics.by_class[2]

    assert round(class_a.recall, 3) == .667
    assert round(class_b.recall, 3) == .200
    assert round(class_c.recall, 3) == .667



def test_specificity():
    metrics = ClassificationMetrics(['a', 'b', 'c'], 
    [
        [4,6,3],
        [1,2,0],
        [1,2,6]
    ])

    assert round(metrics.avg_specificity, 3) == .757

    class_a = metrics.by_class[0]
    class_b = metrics.by_class[1]
    class_c = metrics.by_class[2]

    assert round(class_a.specificity, 3) == .526
    assert round(class_b.specificity, 3) == .933
    assert round(class_c.specificity, 3) == .812


def test_precision():
    metrics = ClassificationMetrics(['a', 'b', 'c'], 
    [
        [4,6,3],
        [1,2,0],
        [1,2,6]
    ])

    assert round(metrics.avg_precision, 3) == .547

    class_a = metrics.by_class[0]
    class_b = metrics.by_class[1]
    class_c = metrics.by_class[2]

    assert round(class_a.precision, 3) == .308
    assert round(class_b.precision, 3) == .667
    assert round(class_c.precision, 3) == .667


def test_f_score():
    metrics = ClassificationMetrics(['a', 'b', 'c'], 
    [
        [4,6,3],
        [1,2,0],
        [1,2,6]
    ])

    assert round(metrics.avg_f_score, 3) == .465

    class_a = metrics.by_class[0]
    class_b = metrics.by_class[1]
    class_c = metrics.by_class[2]

    assert round(class_a.f_score, 3) == .421
    assert round(class_b.f_score, 3) == .308
    assert round(class_c.f_score, 3) == .667
