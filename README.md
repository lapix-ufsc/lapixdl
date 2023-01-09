[![DOI](https://zenodo.org/badge/306032350.svg)](https://zenodo.org/badge/latestdoi/306032350) [![CodeFactor](https://www.codefactor.io/repository/github/lapix-ufsc/lapixdl/badge)](https://www.codefactor.io/repository/github/lapix-ufsc/lapixdl) [![PyPI](https://img.shields.io/pypi/v/lapixdl?color=blue&label=pypi%20version)](https://pypi.org/project/lapixdl/) [![tests](https://github.com/lapix-ufsc/lapixdl/actions/workflows/tests.yml/badge.svg?branch=main&event=schedule)](https://github.com/lapix-ufsc/lapixdl/actions/workflows/tests.yml)


# LAPiX DL - Utils for Computer Vision Deep Learning research

This package contains utilitary functions to support train and evaluation of Deep Learning models applied to images.

Three computer vision approaches are covered: Segmentation, Detection and Classification.

## How to use

### For Model Evaluation

This module exports the following functions for model evaluation:
```python
from lapixdl.evaluation.evaluate import evaluate_segmentation
from lapixdl.evaluation.evaluate import evaluate_detection
from lapixdl.evaluation.evaluate import evaluate_classification
```

All model evaluation methods need two iterators: **one for the ground truth itens and one for the predictions**.

These iterators must be sorted equaly, assuring that the ground truth and the prediction of the same sample are at the same position.

#### Example of segmentation model evaluation using **PyTorch**:

```python
from lapixdl.evaluation.evaluate import evaluate_segmentation

classes = ['background', 'object']

# Iterator for GT masks
# `dl` is a PyTorch DataLoader
def gt_mask_iterator_from_dl(dl):
  for imgs, masks in iter(dl):
    for mask in masks:
      yield mask

# Iterator for prediction masks
# `predict` a function that, given an image, predicts the mask.
def pred_mask_iterator_from_dl(dl, predict):
  for imgs, masks in iter(dl):
    for img in imgs:
      yield predict(img)

gt_masks = gt_mask_iterator_from_dl(validation_dl)
pred_masks = pred_mask_iterator_from_dl(validation_dl, prediction_function)

# Calculates and shows metrics
eval = evaluate_segmentation(gt_masks, pred_masks, classes)

# Shows confusion matrix and returns its Figure and Axes
fig, axes = eval.show_confusion_matrix()
```

#### Examples with third libraries

##### How to log the results of LAPiX DL evaluations in the Weights & Biases platform
About [Weights & Biases](https://docs.wandb.ai/).

```python
from lapixdl.evaluation.evaluate import evaluate_segmentation
import wandb

# init wandb ...
...

eval_test = evaluate_segmentation(gt_masks, pred_masks, categories)

...

# If you want to log everything
wandb.log({'test_evaluation':  eval_test.to_dict()['By Class']})

# If you want to choose specific categories to log
selected_cats = ['A', 'B', 'C']
metrics_by_cat = {k: v for k, v in eval_test.to_dict()['By Class'].items() if k in selected_cats}
wandb.log({'test_evaluation': metrics_by_cat})
```

##### Computing using GPU with `torchmetrics`
About [torchmetrics](https://torchmetrics.readthedocs.io/en/stable/).

The lapixdl package calculates the confusion matrix first (on the CPU), which
this will be slower than calculating using `torchmetrics` which uses `pytorch`
**tensors**. So a trick here, to not calculate each metric separately in
`torchmetrics`, is to calculate a **confusion matrix** using `torchmetrics`
and then calculate all the metrics at once using `lapixdl`.

A simple example for a Segmentation case:

```python
import torchmetrics
from lapixdl.evaluation.model import SegmentationMetrics

classes = ['background', 'object']

confMat = torchmetrics.ConfusionMatrix(
    reduce="macro", mdmc_reduce="global", num_classes=len(classes)
)

confusion_matrix = confMat(pred, target)
confusion_matrix = confusion_matrix.numpy()

metrics = SegmentationMetrics(
    classes=classes, confusion_matrix=confusion_matrix
)
```

### For Results Visualization

This module exports the following functions for results visualization:
```python
from lapixdl.evaluation.visualize import show_segmentations
from lapixdl.evaluation.visualize import show_classifications
from lapixdl.evaluation.visualize import show_detections
```

The available color maps are the [ones from matplotlib](https://matplotlib.org/3.1.1/gallery/color/colormap_reference.html).

### For Data Conversion
This module exports the functions for data conversion.
```python
from lapixdl.convert import labelbox_to_lapix
from lapixdl.convert import labelbox_to_coco
```

#### Example of conversion from **Labelbox** to **COCO** labels format:

```python
import json

from lapixdl.formats import labelbox_to_coco

# A map categories between labelbox schematic id and category ID
map_categories = {
  '<schematic id from labelbox>': 1 # category id
}

# The categories section in the COCO format
categories_coco = [{
  'supercategory': None,
  'name': 'example_category',
  'id': 1
}]

# Convert it and create the COCO OD data
coco_dict = labelbox_to_coco(
  'labelbox_export_file.json',
  map_categories,
  categories_coco,
  target = 'object detection',
  image_shape = (1200, 1600)
)

# Saves converted json
with open('./coco.json', 'w') as out_file:
    json.dump(coco_dict, out_file)
```
