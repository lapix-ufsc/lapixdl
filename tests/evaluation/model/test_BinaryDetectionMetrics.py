from lapixdl.evaluation.model import BinaryDetectionMetrics, BinaryClassificationMetrics, PredictionResult, PredictionResultType
import pytest

# From https://github.com/rafaelpadilla/Object-Detection-Metrics
predictions = [
    PredictionResult(.88, PredictionResultType.FP),
    PredictionResult(.70, PredictionResultType.TP),
    PredictionResult(.80, PredictionResultType.FP),
    PredictionResult(.71, PredictionResultType.FP),
    PredictionResult(.54, PredictionResultType.TP),
    PredictionResult(.74, PredictionResultType.FP),
    PredictionResult(.18, PredictionResultType.TP),
    PredictionResult(.67, PredictionResultType.FP),
    PredictionResult(.38, PredictionResultType.FP),
    PredictionResult(.91, PredictionResultType.TP),
    PredictionResult(.44, PredictionResultType.FP),
    PredictionResult(.35, PredictionResultType.FP),
    PredictionResult(.78, PredictionResultType.FP),
    PredictionResult(.45, PredictionResultType.FP),
    PredictionResult(.14, PredictionResultType.FP),
    PredictionResult(.62, PredictionResultType.TP),
    PredictionResult(.44, PredictionResultType.FP),
    PredictionResult(.95, PredictionResultType.TP),
    PredictionResult(.23, PredictionResultType.FP),
    PredictionResult(.45, PredictionResultType.FP),
    PredictionResult(.84, PredictionResultType.FP),
    PredictionResult(.43, PredictionResultType.FP),
    PredictionResult(.48, PredictionResultType.TP),
    PredictionResult(.95, PredictionResultType.FP)
]

TP = len([prediction for prediction in predictions if prediction.type ==
          PredictionResultType.TP])
FP = len([prediction for prediction in predictions if prediction.type ==
          PredictionResultType.FP])
FN = 15 - TP


def test_gt_count():
    bin_class = BinaryClassificationMetrics(cls='a', FN=FN, TP=TP, FP=FP)
    metrics = BinaryDetectionMetrics(bin_class, 0, predictions)

    assert metrics.gt_count == 15


def test_pred_count():
    bin_class = BinaryClassificationMetrics(cls='a', FN=FN, TP=TP, FP=FP)
    metrics = BinaryDetectionMetrics(bin_class, 0, predictions)

    assert metrics.predicted_count == 24


def test_iou():
    bin_class = BinaryClassificationMetrics(cls='a', FN=FN, TP=TP, FP=FP)
    metrics = BinaryDetectionMetrics(bin_class, 10, predictions)

    assert metrics.iou == 10


def test_precision_recall_curve():
    bin_class = BinaryClassificationMetrics(cls='a', FN=FN, TP=TP, FP=FP)
    metrics = BinaryDetectionMetrics(bin_class, 10, predictions)

    rounded_pr_curve = [(round(rp[0], 4), round(rp[1], 4))
                        for rp in metrics.precision_recall_curve]
    expected_curve = [
        (0.0667, 1.0000),
        (0.0667, 0.5000),
        (0.1333, 0.6667),
        (0.1333, 0.5000),
        (0.1333, 0.4000),
        (0.1333, 0.3333),
        (0.1333, 0.2857),
        (0.1333, 0.2500),
        (0.1333, 0.2222),
        (0.2000, 0.3000),
        (0.2000, 0.2727),
        (0.2667, 0.3333),
        (0.3333, 0.3846),
        (0.4000, 0.4286),
        (0.4000, 0.4000),
        (0.4000, 0.3750),
        (0.4000, 0.3529),
        (0.4000, 0.3333),
        (0.4000, 0.3158),
        (0.4000, 0.3000),
        (0.4000, 0.2857),
        (0.4000, 0.2727),
        (0.4667, 0.3043),
        (0.4667, 0.2917),
    ]

    assert rounded_pr_curve == expected_curve

def test_average_precision():
    bin_class = BinaryClassificationMetrics(cls='a', FN=FN, TP=TP, FP=FP)
    metrics = BinaryDetectionMetrics(bin_class, 10, predictions)

    assert round(metrics.average_precision(11), 4) == .2684
    assert round(metrics.average_precision(), 4) == .2457
