from lapixdl.evaluation.model import BinaryClassificationMetrics


def test_count():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'])
    bin_class.FN = 1
    bin_class.TP = 2

    assert bin_class.count == 3

    bin_class.FP = 2

    assert bin_class.count == 5


def test_accuracy():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'], FN=2, TN=2, TP=2, FP=2)

    assert bin_class.accuracy == 0.5


def test_recall():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'])
    bin_class.TP = 2
    bin_class.FN = 6

    assert bin_class.recall == 0.25


def test_fpr():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'])
    bin_class.FP = 2
    bin_class.TN = 6

    assert bin_class.false_positive_rate == 0.25

def test_specificity():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'])
    bin_class.TN = 6
    bin_class.FP = 2

    assert bin_class.specificity == 0.75


def test_precision():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'])
    bin_class.TP = 8
    bin_class.FP = 2

    assert bin_class.precision == 0.8


def test_f_score():
    bin_class = BinaryClassificationMetrics(cls=['a', 'b'])
    bin_class.FN = 4
    bin_class.TN = 2
    bin_class.TP = 5
    bin_class.FP = 6

    assert bin_class.f_score == 0.5
