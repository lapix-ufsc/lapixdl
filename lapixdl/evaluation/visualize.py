from typing import Optional, Union, List, Tuple, TypeVar, Generic
import math

import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from matplotlib import cm
import matplotlib.patches as mpatches
import seaborn as sn
import numpy as np

from .model import BBox, Mask, Classification, Result, Image

correct_color = sn.color_palette("Paired")[3]
incorrect_color = sn.color_palette("Paired")[5]


def show_classifications(
        results: List[Result[Classification]],
        class_names: List[str],
        cols: int = 3,
        diff_correct_incorect: bool = True) -> Tuple[Figure, Axes]:
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
    fig.suptitle('Classifications')

    for i, result in enumerate(results):
        axe = axes[i // cols][i % cols]
        axe.set_title(
            f'GT: {class_names[result.gt.cls]}' +
            (f'\nPred: {class_names[result.prediction.cls]} ({result.prediction.score})' if not result.prediction is None else ''),
            fontsize='small',
            color='#333' if result.prediction is None or not diff_correct_incorect
            else (correct_color if result.prediction.cls ==
                  result.gt.cls else incorrect_color)
        )
        axe.axis('off')
        axe.imshow(result.image)

    for i in range(len(results), (cols * rows)):
        axe = axes[i // cols][i % cols]
        axe.axis('off')

    plt.tight_layout(w_pad=.2, h_pad=1.5)
    plt.show()
    return fig, axes


def show_segmentations(
        results: List[Result[Mask]],
        class_names: List[str],
        cmap: Optional[str] = 'tab10',
        mask_alpha: float = .3) -> Tuple[Figure, Axes]:
    """Shows segmentation results

    Args:
        results (List[Result[Mask]]): Segmentation results.
        class_names (List[str]): Class names.
        cmap (Optional[str], optional): Matplotlib color map for the mask. Defaults to 'tab10'.
        mask_alpha (float, optional): Alpha value for the mask colors. Defaults to .3.

    Returns:
        Tuple[Figure, Axes]: Figure and Axes of the ploted results.
    """

    cmap_colors = cm.get_cmap(cmap).colors

    rows = len(results)
    fig, axes = plt.subplots(rows, 3)

    fig.suptitle(' ', fontsize=40)  # To keep space for the legend
    lengend_handles = [mpatches.Patch(color=cmap_colors[i], label=code)
                       for i, code in enumerate(class_names)]
    fig.legend(handles=lengend_handles, fontsize='small',
               ncol=len(class_names), loc='upper center')

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
        if not result.prediction is None:
            axe_pred.set_title('Prediction', fontsize='small')
            axe_pred.imshow(result.image)
            axe_pred.imshow(result.prediction, cmap=cmap, alpha=mask_alpha,
                            interpolation='none', vmin=0, vmax=len(cmap_colors) - 1)

    plt.tight_layout(w_pad=.2, h_pad=3)
    plt.show()

    return fig, axes


def show_detections(results: List[Result[List[BBox]]],
                    class_names: List[str],
                    cmap: Optional[str] = 'tab10',
                    show_bbox_label: bool = True) -> Tuple[Figure, Axes]:
    """Shows detection results.

    Args:
        results (List[Result[List[BBox]]]): Detection results.
        class_names (List[str]): Class names.
        cmap (Optional[str], optional): Matplotlib color map for the bboxes. Defaults to 'tab10'.
        show_bbox_label (bool, optional): Indicates if the class label should be shown for each bbox. Defaults to True.

    Returns:
        Tuple[Figure, Axes]: Figure and Axes of the ploted results.
    """

    cmap_colors = cm.get_cmap(cmap).colors

    rows = len(results)
    fig, axes = plt.subplots(rows, 3)

    fig.suptitle(' ', fontsize=40)  # To keep space for the legend
    lengend_handles = [mpatches.Patch(color=cmap_colors[i], label=code)
                       for i, code in enumerate(class_names)]
    fig.legend(handles=lengend_handles, fontsize='small',
               ncol=len(class_names), loc='upper center')

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
        if not result.prediction is None:
            axe_pred.set_title('Prediction', fontsize='small')
            axe_pred.imshow(result.image)
            __draw_bboxes(axe_pred, result.prediction,
                          cmap_colors, class_names, show_bbox_label)

    plt.tight_layout(w_pad=.2, h_pad=3)
    plt.show()

    return fig, axes


def __draw_bboxes(axe: plt.Axes, bboxes: List[BBox], cmap_colors: List, class_names: List[str], show_bbox_label: bool):
    for bbox in bboxes:
        color = cmap_colors[bbox.cls]
        label = (f'{class_names[bbox.cls]}' if show_bbox_label else '') + \
            (f' ({bbox.score})' if bbox.score else '')
        rect = mpatches.Rectangle(bbox.upper_left_point,
                                  bbox.width, bbox.height,
                                  linewidth=2,
                                  edgecolor=color,
                                  facecolor='none',
                                  label='teste')
        axe.add_patch(rect)
        axe.annotate(label.strip(),
                     (bbox.upper_left_point[0] + 5,
                      bbox.upper_left_point[1] + 5),
                     color='w',
                     weight='bold',
                     fontsize=8,
                     ha='left', va='top',
                     bbox=dict(facecolor=color, edgecolor='none', pad=1.5))
