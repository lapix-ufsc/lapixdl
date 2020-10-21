from lapixdl.evaluation.evaluate import evaluate_segmentation


def test_evaluation():
    classes = ['a', 'b', 'c']

    gt_masks = [
        [
            [0, 0, 0, 0, 0],
            [0, 1, 1, 1, 1],
            [1, 1, 1, 1, 1],
            [1, 2, 2, 2, 2],
            [2, 2, 2, 2, 2]
        ]
    ]
    pred_masks = [
        [
            [0, 0, 0, 0, 2],
            [1, 0, 0, 0, 0],
            [0, 0, 2, 2, 1],
            [1, 0, 0, 0, 2],
            [2, 2, 2, 2, 2]
        ]
    ]

    metrics = evaluate_segmentation(gt_masks, pred_masks, classes)

    assert metrics.count == 25
    assert round(metrics.accuracy, 3) == .48
    assert round(metrics.avg_recall, 3) == .511
    assert round(metrics.avg_precision, 3) == .547
    assert round(metrics.avg_f_score, 3) == .465
    assert round(metrics.avg_specificity, 3) == .757
