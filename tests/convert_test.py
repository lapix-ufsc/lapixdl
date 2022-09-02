from __future__ import annotations

from lapixdl.convert import create_coco_od
from lapixdl.convert import labelbox_to_lapix
from lapixdl.convert import lapix_to_od_coco_annotations
from lapixdl.formats import labelbox
from lapixdl.formats import lapix
from lapixdl.formats.lapix import LapixDataFrame


def test_labelbox_to_lapix(labelbox_filename):
    labelbox_df = labelbox.load(labelbox_filename)

    categories_map = {
        '<Unique ID for category square>': 1
    }
    lapix_df = labelbox_to_lapix(labelbox_df, categories_map)

    assert lapix_df.shape == (1, 3)
    assert type(lapix_df) is LapixDataFrame


def test_lapix_to_od_coco_annotations(lapix_filename):
    lapix_df = lapix.load(lapix_filename)

    annotations_coco = lapix_to_od_coco_annotations(lapix_df)

    assert len(annotations_coco) == 2

    assert annotations_coco[0]['id'] == 1
    assert annotations_coco[1]['image_id'] == 2
    assert annotations_coco[1]['category_id'] == 1
    assert annotations_coco[0]['bbox'] == [0., 0., 15., 28.]
    coco_od_labels = ['id', 'image_id', 'category_id', 'bbox', 'segmentation', 'area', 'iscrowd']
    assert all(k in coco_od_labels for k in annotations_coco[0].keys())


def test_create_coco_od(lapix_filename):
    lapix_df = lapix.load(lapix_filename)

    categories_coco = [{
        'supercategory': None,
        'name': 'example_category',
        'id': 1
    }]

    coco_od = create_coco_od(lapix_df, categories_coco)

    coco_od_labels = ['categories', 'images', 'annotations']
    assert all(k in coco_od.keys() for k in coco_od_labels)

    info_coco = {
        'year': '2022',
        'version': '1.0',
        'description': 'Any description here',
        'contributor': 'Names here',
        'url': 'URL here',
        'date_created': '2022-01-01',
    }

    coco_od = create_coco_od(lapix_df, categories_coco, info_coco=info_coco)

    assert coco_od['info'] == info_coco
