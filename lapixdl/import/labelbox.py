
import numpy as np
import itertools


def __generate_coco_annotations(img_labels, img_id, annotation_first_id, classes):
    annotations = []

    for img_label in img_labels:
        class_name = img_label['value']
        polygon = img_label['polygon']
        poly_convert = [(p['x'], p['y']) for p in polygon]
        bbox = cv2.boundingRect(np.array(poly_convert, np.int32))
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
        })

        annotation_first_id += 1

    return annotations


def __generate_coco_file(lblbox_annotations, img_names_to_include):
    annotation_id = 1
    image_id = 1
    annotations = []
    images = []
    classes = []

    for images_labels in filter(lambda ant: ant['External ID'] in img_names_to_include, lblbox_annotations):
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


def labelbox_to_coco(lblbox_annotations, img_names_to_include):
    return __generate_coco_file(lblbox_annotations, img_names_to_include)
