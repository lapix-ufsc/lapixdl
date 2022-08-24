from __future__ import annotations

import math
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import seaborn as sn
from matplotlib import cm
from matplotlib.axes import Axes
from matplotlib.colors import Colormap
from matplotlib.figure import Figure

from .model import Classification
from .model import Mask
from .model import Result
from lapixdl.formats.annotation import BBox

correct_color = sn.color_palette('Paired')[3]
incorrect_color = sn.color_palette('Paired')[5]

ColorMap = Union[str, Colormap]


def show_classifications(
        results: list[Result[Classification]],
        class_names: list[str],
        cols: int = 3,
        diff_correct_incorect: bool = True) -> tuple[Figure, Axes]:
    """Shows multiple classification results.

    Args:
        results (List[Result[Classification]]): List of classification results.
        class_names (List[str]): Class names.
        cols (int, optional): Number of colunms to show. Defaults to 3.
        diff_correct_incorect (bool, optional): Indicates if correct and incorrect
        results should be differentiated by color. Defaults to True.

    Returns:
        Tuple[Figure, Axes]: Figure and Axes of the ploted results.
    """

    rows = math.ceil(len(results) / cols)
    fig, axes = plt.subplots(rows, cols)
    if rows == 1:
        axes = [axes]
    fig.suptitle('Classifications')

    for i, result in enumerate(results):
        axe = axes[i // cols][i % cols]
        pred_s = (f'\nPred: {class_names[result.prediction.cls]} ({result.prediction.score})' if result.prediction is not None
                  else '')
        axe.set_title(
            f'GT: {class_names[result.gt.cls]}' + pred_s,
            fontsize='small',
            color='#333' if result.prediction is None or not diff_correct_incorect
            else (correct_color if result.prediction.cls == result.gt.cls else incorrect_color)
        )
        axe.axis('off')
        axe.imshow(result.image)

    for i in range(len(results), (cols * rows)):
        axe = axes[i // cols][i % cols]
        axe.axis('off')

    fig.set_size_inches((20, 10), forward=True)
    plt.tight_layout(w_pad=.2, h_pad=1.5)
    plt.show()
    return fig, axes


def show_segmentations(
        results: list[Result[Mask]],
        class_names: list[str],
        cmap: ColorMap = 'tab10',
        mask_alpha: float = .3) -> tuple[Figure, Axes]:
    """Shows segmentation results

    Args:
        results (List[Result[Mask]]): Segmentation results.
        class_names (List[str]): Class names.
        cmap (ColorMap, optional): Matplotlib color map for the mask or list of `class_index -> hex_color`. Defaults to 'tab10'.
        mask_alpha (float, optional): Alpha value for the mask colors. Defaults to .3.

    Returns:
        Tuple[Figure, Axes]: Figure and Axes of the ploted results.
    """

    cmap_colors = cm.get_cmap(cmap).colors

    assert len(cmap_colors) >= len(
        class_names), 'The color map length must be greater or equal the length of the class names.'

    rows = len(results)
    fig, axes = plt.subplots(rows, 3)
    if rows == 1:
        axes = [axes]

    fig.suptitle(' ', fontsize=40)  # To keep space for the legend
    lengend_handles = [mpatches.Patch(color=cmap_colors[i], label=code)
                       for i, code in enumerate(class_names)]
    fig.legend(handles=lengend_handles, fontsize='small',
               ncol=min(8, len(class_names)), loc='upper center')

    for i, result in enumerate(results):
        axe_img = axes[i][0]
        axe_GT = axes[i][1]
        axe_pred = axes[i][2]

        axe_img.set_title('Image', fontsize='small')
        axe_img.imshow(result.image)
        axe_img.axis('off')

        axe_GT.set_title('GT', fontsize='small')
        axe_GT.axis('off')
        axe_GT.imshow(result.image)
        axe_GT.imshow(result.gt, cmap=cmap, alpha=mask_alpha,
                      interpolation='none', vmin=0, vmax=len(cmap_colors) - 1)

        axe_pred.axis('off')
        if result.prediction is not None:
            axe_pred.set_title('Prediction', fontsize='small')
            axe_pred.imshow(result.image)
            axe_pred.imshow(result.prediction, cmap=cmap, alpha=mask_alpha,
                            interpolation='none', vmin=0, vmax=len(cmap_colors) - 1)

    fig.set_size_inches((20, 10), forward=True)
    plt.tight_layout(w_pad=.2, h_pad=3)
    plt.show()

    return fig, axes


def show_detections(results: list[Result[list[BBox]]],
                    class_names: list[str],
                    cmap: ColorMap = 'tab10',
                    show_bbox_label: bool = True) -> tuple[Figure, Axes]:
    """Shows detection results.

    Args:
        results (List[Result[List[BBox]]]): Detection results.
        class_names (List[str]): Class names.
        cmap (ColorMap, optional): Matplotlib color map for the mask or list of `class_index -> hex_color`. Defaults to 'tab10'.
        show_bbox_label (bool, optional): Indicates if the class label should be shown for each bbox. Defaults to True.

    Returns:
        Tuple[Figure, Axes]: Figure and Axes of the ploted results.
    """

    cmap_colors = cm.get_cmap(cmap).colors

    assert len(cmap_colors) >= len(
        class_names), 'The color map length must be greater or equal the length of the class names.'

    rows = len(results)
    fig, axes = plt.subplots(rows, 3)
    if rows == 1:
        axes = [axes]

    fig.suptitle(' ', fontsize=40)  # To keep space for the legend
    lengend_handles = [mpatches.Patch(color=cmap_colors[i], label=code)
                       for i, code in enumerate(class_names)]
    fig.legend(handles=lengend_handles, fontsize='small',
               ncol=min(8, len(class_names)), loc='upper center')

    for i, result in enumerate(results):
        axe_img = axes[i][0]
        axe_GT = axes[i][1]
        axe_pred = axes[i][2]

        axe_img.set_title('Image', fontsize='small')
        axe_img.imshow(result.image)
        axe_img.axis('off')

        axe_GT.set_title('GT', fontsize='small')
        axe_GT.axis('off')
        axe_GT.imshow(result.image)
        __draw_bboxes(axe_GT, result.gt, cmap_colors,
                      class_names, show_bbox_label)

        axe_pred.axis('off')
        if result.prediction is not None:
            axe_pred.set_title('Prediction', fontsize='small')
            axe_pred.imshow(result.image)
            __draw_bboxes(axe_pred, result.prediction,
                          cmap_colors, class_names, show_bbox_label)

    fig.set_size_inches((20, 10), forward=True)
    plt.tight_layout(w_pad=.2, h_pad=3)
    plt.show()

    return fig, axes


def __draw_bboxes(axe: plt.Axes, bboxes: list[BBox], cmap_colors: list[str], class_names: list[str], show_bbox_label: bool):
    for bbox in bboxes:
        color = cmap_colors[bbox.cls]
        label = (f'{class_names[bbox.cls]}' if show_bbox_label else '') + \
            (f' ({round(bbox.score, 3)})' if bbox.score else '')
        rect = mpatches.Rectangle(bbox.upper_left_point,
                                  bbox.width, bbox.height,
                                  linewidth=2,
                                  edgecolor=color,
                                  facecolor='none')
        axe.add_patch(rect)
        axe.annotate(label.strip(),
                     (bbox.upper_left_point[0] + 5,
                      bbox.upper_left_point[1] - 5),
                     color='w',
                     weight='bold',
                     fontsize=10,
                     ha='left', va='bottom',
                     bbox=dict(facecolor=color, edgecolor='none', pad=1.5))
