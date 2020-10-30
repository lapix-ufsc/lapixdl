from typing import List, Tuple, Optional

import pandas as pd
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes


def confusion_matrix(confusion_matrix: List[List[int]], classes: List[str], title: Optional[str] = None) -> Tuple[Figure, Axes]:
    """Plots a confusion matrix

    Args:
        confusion_matrix (List[List[int]]): Confusion matrix
        classes (List[str]): Class names

    Return:
        Tuple[Figure, Axes]: Figure and Axes of the ploted confusion matrix
    """

    df_cm = pd.DataFrame(confusion_matrix, classes, classes)

    fig, ax = plt.subplots()
    fig.suptitle('Confusion Matrix' if title is None else title)

    ax = sn.heatmap(df_cm,
                    annot=True,
                    ax=ax,
                    cmap="YlGnBu")

    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")

    plt.show()

    return fig, ax
