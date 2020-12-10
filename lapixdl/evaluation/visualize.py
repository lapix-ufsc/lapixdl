from typing import Optional, Union, List, Tuple, TypeVar, Generic
import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
import seaborn as sn

from .model import BBox, Mask, Classification, Result


def show_classifications(
        results: List[Result[Classification]],
        class_names: List[str],
        cols: int = 3) -> Tuple[Figure, Axes]:

    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols)
    fig.suptitle('Classifications')

    for i, result in enumerate(results):
        axis = axes[i // cols][i % cols]
        axis.set_title(
            f'GT: {class_names[result.gt.cls]}'
            f'\nPred: {class_names[result.prediction.cls]} ({result.prediction.score})' if not result.prediction is None else '',
            fontsize='small'
        )
        axis.axis('off')
        axis.imshow(result.image)

    for i in range(len(results), (cols * rows)):
        axis = axes[i // cols][i % cols]
        axis.axis('off')

    plt.tight_layout(pad=.2)
    plt.show()
    return fig, axes


def show_segmentation(
        results: List[Result[Mask]],
        class_names: List[str],
        cols: int = 1,
        palette: Optional[str] = None) -> Tuple[Figure, Axes]:

    rows = math.ceil(len(results) / cols)
    fig, ax = plt.subplots()
    fig.suptitle('Segmentations')

    cmap = sn.color_palette(palette)

    plt.show()
    return fig, ax


def show_detections(results: List[Result[List[BBox]]]) -> Tuple[Figure, Axes]:
    fig, ax = plt.subplots()
    fig.suptitle('Detections')

    plt.show()
    return fig, ax
