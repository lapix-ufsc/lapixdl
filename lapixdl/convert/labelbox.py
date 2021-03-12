from typing import List, Optional, Tuple
import numpy as np
import itertools


def __bounding_rect_from_polygon(polygon: List[Tuple[int, int]]) -> List[int]:
    arr = np.array(polygon)

    max = arr.max(axis=0)
    min = arr.min(axis=0)
    h = max[1] - min[1]
    w = max[0] - min[0]

    return [min[0], min[1], w, h]


def __generate_coco_annotations(img_labels: List[dict], img_id: int, annotation_first_id: int, classes: List[str]) -> List[dict]:
    annotations = []

    for img_label in img_labels:
        class_name = img_label['value']
        polygon = img_label['polygon']
        poly_convert = [(p['x'], p['y']) for p in polygon]
        bbox = __bounding_rect_from_polygon(poly_convert)
        segmentation = list(itertools.chain.from_iterable(poly_convert))
        if class_name not in classes:
            classes.append(class_name)

        annotations.append({
            'segmentation': [segmentation],
            'iscrowd': 0,
            'image_id': img_id,
            'category_id': classes.index(class_name),
            'id': annotation_first_id,
            'bbox': bbox,
            'area': __calculate_area(segmentation)
        })

        annotation_first_id += 1

    return annotations


def __calculate_area(segmentation: List[float]) -> float:
    x = segmentation[0:-1:2]
    y = segmentation[1::2]
    return 0.5*np.abs(np.dot(x, np.roll(y, 1))-np.dot(y, np.roll(x, 1))) #Shoelace


def __generate_coco_file(lblbox_annotations: dict, img_names_to_include: Optional[List[str]] = None) -> dict:
    annotation_id = 1
    image_id = 1

    annotations: List[dict] = []
    images: List[dict] = []
    classes: List[str] = []

    ftr_lblbox_annotations = (filter(lambda ant: ant['External ID'] in (img_names_to_include or []),
                                     lblbox_annotations)) if img_names_to_include is not None else lblbox_annotations

    for images_labels in ftr_lblbox_annotations:
        img_filename = images_labels['External ID']
        img_labels = images_labels['Label']['objects'] if 'objects' in images_labels['Label'] else [
        ]

        new_annotations = __generate_coco_annotations(
            img_labels, image_id, annotation_id, classes)
        annotations += new_annotations

        images.append({
            "file_name": img_filename,
            "height": 1200,
            "width": 1600,
            "id": image_id
        })

        image_id += 1
        annotation_id += len(new_annotations)

    categories = [{
        "supercategory": "none",
        "name": cls,
        "id": i
    } for i, cls in enumerate(classes)]

    final_json = {
        "type": "instances",
        "images": images,
        "categories": categories,
        "annotations": annotations
    }

    return final_json


def labelbox_to_coco(lblbox_annotations: dict, img_names_to_include: Optional[List[str]] = None) -> dict:
    """Converts from Labelbox format to the COCO format

    Args:
        lblbox_annotations (dict): Dict of labelbox annotations parsed from the exported json.
        img_names_to_include (Optional[List[str]], optional): Image names (External IDs) to include in the converted file. Defaults to None.

    Returns:
        dict: Dict in the COCO format.
    """
    return __generate_coco_file(lblbox_annotations, img_names_to_include)
