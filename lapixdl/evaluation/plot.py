from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sn
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def confusion_matrix(
    confusion_matrix: list[list[int]],
    classes: list[str],
    title: str | None = 'Confusion Matrix',
) -> tuple[Figure, Axes]:
    """Plots a confusion matrix

    Args:
        confusion_matrix (List[List[int]]): Confusion matrix.
        classes (List[str]): Class names.
        title (Optional[str], optional): Chart title. Defaults to
        "Confusion Matrix".

    Return:
        Tuple[Figure, Axes]: Figure and Axes of the plotted confusion
        matrix.
    """

    df_cm = pd.DataFrame(confusion_matrix, classes, classes)

    fig, ax = plt.subplots()
    fig.suptitle(title)

    ax = sn.heatmap(
        df_cm,
        annot=True,
        ax=ax,
        cmap='YlGnBu',
    )

    ax.set_xlabel('Actual')
    ax.set_ylabel('Predicted')
    plt.show()

    return fig, ax


def precision_recall_curve(
    recalls_precisions_by_class: list[list[tuple[float, float]]],
    classes: list[str],
    title: str | None = 'Precision x Recall Curve',
) -> tuple[Figure, Axes]:
    """Plots a Precision x Recall curve.

    Args:
        recalls_precisions_by_class (List[List[Tuple[float, float]]]): (Recall, Precision) tuples indexed by class.
        classes (List[str]): Class names.
        title (Optional[str], optional): Chart title. Defaults to "Precision x Recall Curve".

    Return:
        Tuple[Figure, Axes]: Figure and Axes of the ploted Precision x Recall curve.
    """

    table = []
    for i, recalls_precisions_cls in enumerate(recalls_precisions_by_class):
        for recall_precision in recalls_precisions_cls:
            table.append(
                (classes[i], recall_precision[0], recall_precision[1]),
            )

    df = pd.DataFrame(table, columns=['class', 'recall', 'precision'])

    fig, ax = plt.subplots()
    fig.suptitle(title)

    ax = sn.lineplot(
        data=df,
        y='precision',
        x='recall',
        hue='class',
        ax=ax,
    )

    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    plt.show()

    return fig, ax
