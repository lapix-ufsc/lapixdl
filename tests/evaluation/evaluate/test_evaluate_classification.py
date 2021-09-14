from lapixdl.evaluation.evaluate import evaluate_classification
from lapixdl.evaluation.model import Classification

def test_evaluation():
    classes = ['a', 'b', 'c']

    gt_class = [Classification(x) for x in [0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1, 2,2,2,2,2,2,2,2,2] ]
    pred_class = [Classification(x) for x in [0,0,0,0,2,1, 0,0,0,0,0,0,2,2,1,1, 0,0,0,2,2,2,2,2,2] ]


    metrics = evaluate_classification(gt_class, pred_class, classes)

    assert metrics.count == 25
    assert round(metrics.accuracy, 3) == .48
    assert round(metrics.avg_recall, 3) == .511
    assert round(metrics.avg_precision, 3) == .547
    assert round(metrics.avg_f_score, 3) == .465
    assert round(metrics.avg_specificity, 3) == .757

def test_evaluation_w_zeros():
    classes = ['a', 'b']

    gt_class = [Classification(x) for x in [0,0,0,0,0,0, 1,1,1,1,1,1,1,1,1,1] ]
    pred_class = [Classification(x) for x in [1,1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0] ]


    metrics = evaluate_classification(gt_class, pred_class, classes)

    assert metrics.count == 16
    assert metrics.accuracy == 0
    assert metrics.avg_recall == 0
    assert metrics.avg_precision == 0
    assert metrics.avg_f_score == 0
    assert metrics.avg_specificity == 0

def test_evaluation_w_only_gt_negatives_falses():
    classes = ['a', 'b']

    gt_class = [Classification(x) for x in [0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0] ]
    pred_class = [Classification(x) for x in [1,1,1,1,1,1, 1,1,1,1,1,1,1,1,1,1] ]


    metrics = evaluate_classification(gt_class, pred_class, classes)

    assert metrics.count == 16
    assert metrics.accuracy == 0
    assert metrics.avg_recall == 0
    assert metrics.avg_precision == 0
    assert metrics.avg_f_score == 0
    assert metrics.avg_specificity == 0

def test_evaluation_w_only_gt_negatives_trues():
    classes = ['a', 'b']

    gt_class = [Classification(x) for x in [0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0] ]
    pred_class = [Classification(x) for x in [0,0,0,0,0,0, 0,0,0,0,0,0,0,0,0,0] ]


    metrics = evaluate_classification(gt_class, pred_class, classes)

    assert metrics.count == 16
    assert metrics.accuracy == 1
    assert metrics.avg_recall == 1
    assert metrics.avg_precision == 1
    assert metrics.avg_f_score == 1
    assert metrics.avg_specificity == 1