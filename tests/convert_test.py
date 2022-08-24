from lapixdl.convert import from_labelbox
from lapixdl.convert import to_OD_COCO_annotations
from lapixdl.convert import create_COCO_OD
from lapixdl.formats import labelbox
from lapixdl.formats import lapix
from lapixdl.formats.lapix import Lapix


def test_from_labelbox(labelbox_filename):
    labelbox_df = labelbox.load(labelbox_filename)

    categories_map = {
        '<Unique ID for category square>': 1
    }
    lapix_df = from_labelbox(labelbox_df, categories_map)

    assert lapix_df.shape == (1, 3)
    assert type(lapix_df) is Lapix


def test_to_OD_COCO_annotations(lapix_filename):
    lapix_df = lapix.load(lapix_filename)

    annotations_coco = to_OD_COCO_annotations(lapix_df)

    assert len(annotations_coco) == 2

    assert annotations_coco[0]['id'] == 1
    assert annotations_coco[1]['image_id'] == 2
    assert annotations_coco[1]['category_id'] == 1
    assert annotations_coco[0]['bbox'] == [0., 0., 15., 28.]
    coco_od_labels = ['id', 'image_id', 'category_id', 'bbox', 'segmentation', 'area', 'iscrowd']
    assert all(k in coco_od_labels for k in annotations_coco[0].keys())


def test_create_COCO_OD(lapix_filename):
    lapix_df = lapix.load(lapix_filename)

    categories_coco = [{
        'supercategory': None,
        'name': 'example_category',
        'id': 1
    }]

    coco_od = create_COCO_OD(lapix_df, categories_coco)

    coco_od_labels = ['categories', 'images', 'annotations']
    assert all(k in coco_od_labels for k in coco_od.keys())

    info_coco = {
        'year': '2022',
        'version': '1.0',
        'description': 'Any description here',
        'contributor': 'Names here',
        'url': 'URL here',
        'date_created': '2022-01-01',
    }

    coco_od = create_COCO_OD(lapix_df, categories_coco, info_coco=info_coco)

    assert coco_od['info'] == info_coco
