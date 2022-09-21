from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
from PIL import Image
from shapely.geometry import Polygon

from lapixdl.convert import __save_masks_as_files
from lapixdl.convert import annotations_to_mask
from lapixdl.convert import draw_annotation
from lapixdl.convert import labelbox_to_coco
from lapixdl.convert import labelbox_to_lapix
from lapixdl.convert import lapix_to_annotations
from lapixdl.convert import lapix_to_od_coco_annotations
from lapixdl.convert import save_lapixdf_as_masks
from lapixdl.convert import sort_annotations_to_draw
from lapixdl.formats import labelbox
from lapixdl.formats import lapix
from lapixdl.formats.annotation import Annotation
from lapixdl.formats.lapix import LapixDataFrame


def test_labelbox_to_lapix(labelbox_filename, labelbox_map_categories):
    labelbox_df = labelbox.load(labelbox_filename)

    lapix_df = labelbox_to_lapix(labelbox_df, labelbox_map_categories)

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


def test_lapix_to_annotations(lapix_raw, polygon_example):
    anns = lapix_to_annotations(lapix_raw)

    assert anns == [Annotation(polygon_example, 1)] * 2


def test_order_annotations_to_draw(polygon_example):
    ann_A = Annotation(polygon_example, 1)
    ann_B = Annotation(polygon_example, 2)
    anns = [ann_B, ann_A]

    sorted_anns = sort_annotations_to_draw(anns)

    assert sorted_anns == [ann_A, ann_B]

    anns = [ann_B, ann_A, ann_B, ann_B, ann_A]

    sorted_anns = sort_annotations_to_draw(anns, (2, 1))

    assert sorted_anns == [ann_B, ann_B, ann_B, ann_A, ann_A]


def test_order_annotations_to_draw_wrong_draw_order(polygon_example):
    with pytest.raises(ValueError):
        sort_annotations_to_draw([Annotation(polygon_example, 0)], [])


def test_draw_annotation():
    coords = [(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)]
    pol = Polygon(coords)

    img_example = Image.fromarray(np.zeros((100, 100), dtype=np.uint8))

    out_img = draw_annotation(img_example, Annotation(pol, 1), 1)

    out_arr = np.array(out_img)

    expected_arr = np.zeros((100, 100), dtype=np.uint8)

    expected_arr[:11, :11] = 1

    assert np.array_equal(out_arr, expected_arr)


def test_annotations_to_mask():
    pol_A = Polygon([(0, 0), (0, 10), (10, 10), (10, 0), (0, 0)])
    pol_B = Polygon([(0, 0), (0, 55), (55, 55), (55, 0), (0, 0)])

    out_msk = annotations_to_mask([Annotation(pol_A, 1), Annotation(pol_B, 2)], 100, 100)

    expected_arr = np.zeros((100, 100), dtype=np.uint8)

    expected_arr[:56, :56] = 2

    assert np.array_equal(out_msk.categorical, expected_arr)


def test_lapix_to_masks_single_process(lapix_raw, tmpdir):
    __save_masks_as_files(lapix_raw, str(tmpdir))
    assert len(tmpdir.listdir()) == 2


def test_save_lapixdf_as_masks(lapix_raw, tmpdir):
    save_lapixdf_as_masks(lapix_raw, str(tmpdir), '.png', processes=2)
    assert len(tmpdir.listdir()) == 2


def test_save_lapixdf_as_masks_without_images(capsys):
    save_lapixdf_as_masks(pd.DataFrame(columns=['image_id', 'image_name']), '')
    _, err = capsys.readouterr()
    assert 'There is no annotation to save as mask.\n' == err


def test_labelbox_to_coco_od(labelbox_filename, labelbox_map_categories):

    categories_coco = [{
        'supercategory': None,
        'name': 'example_category',
        'id': 1
    }]

    coco_od = labelbox_to_coco(
        labelbox_filename,
        labelbox_map_categories,
        categories_coco,
        image_shape=(1000, 1000)
    )

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

    coco_od = labelbox_to_coco(
        labelbox_filename,
        labelbox_map_categories,
        categories_coco,
        info_coco=info_coco,
        image_shape=(1000, 1000)
    )

    assert coco_od['info'] == info_coco


def test_labelbox_to_coco_wrong_target(labelbox_filename, labelbox_map_categories):
    with pytest.raises(NotImplementedError):
        labelbox_to_coco(labelbox_filename, labelbox_map_categories, [], target='Wrong target', image_shape=(1000, 1000))


def test_labelbox_to_coco_wrong_image_shape(labelbox_filename, labelbox_map_categories):
    with pytest.raises(NotImplementedError):
        labelbox_to_coco(labelbox_filename, labelbox_map_categories, [], target='Wrong target', image_shape=None)


@pytest.fixture
def labelbox_example(tmpdir):
    labelbox_raw = [
        {
            'ID': 'asasasasasdasd',
            'DataRow ID': 'asasasasasdasd',
            'Labeled Data': 'asasasasasdasd',
            'Label': {
                'objects': [
                    {
                        'featureId': 'cka2gfhtl036o0y7v66em102s',
                        'schemaId': 'ck9ct9vgzl61z074205gy328i',
                        'title': 'Normal',
                        'value': 'normal',
                        'color': '#FF8000',
                        'polygon': [
                            {
                                'x': 42,
                                'y': 894
                            },
                            {
                                'x': 41,
                                'y': 895
                            },
                            {
                                'x': 74,
                                'y': 919
                            },
                            {
                                'x': 67,
                                'y': 908
                            },
                            {
                                'x': 48,
                                'y': 896
                            },
                            {
                                'x': 44,
                                'y': 894
                            }
                        ],
                        'instanceURI': 'asasasasasdasd'
                    },
                ],
                'classifications': []
            },
            'Created By': 'xxx@xxx.xxx',
            'Project Name': 'Papanicolaou',
            'Created At': '2020-05-12T12:28:27.000Z',
            'Updated At': '2020-09-25T17:21:16.000Z',
            'Seconds to Label': 778.39,
            'External ID': 'filename_example_image.png',
            'Agreement': -1,
            'Benchmark Agreement': -1,
            'Benchmark ID': 'null',
            'Dataset Name': 'Papanicolaou dataset',
            'Reviews': [
                {
                    'score': 1,
                    'ID': 'asasasasasdasd',
                    'createdAt': '2020-09-25T17:21:16.000Z',
                    'instanceURI': 'aaaa@aaaa.aaa'
                }
            ],
            'View Label': 'xxx.xxx.xxx'
        }
    ]

    filename = str(tmpdir.join('labelbox_real_example.json'))
    with open(filename, 'w+') as outfile:
        json.dump(labelbox_raw, outfile)

    schematic_to_category_id = {'ck9ct9vgzl61z074205gy328i': 1}

    return filename, schematic_to_category_id


@pytest.fixture
def object_detection_coco_example():
    return {
        'categories': [{'supercategory': None, 'name': 'example_category', 'id': 1}],
        'images': [{'file_name': 'filename_example_image', 'height': 1000, 'width': 1000, 'id': 1}],
        'annotations': [
            {'id': 1,
             'image_id': 1,
             'category_id': 1,
             'bbox': [41.0, 894.0, 33.0, 25.0],
             'segmentation': [[42.0, 894.0, 41.0, 895.0, 74.0, 919.0, 67.0, 908.0, 48.0, 896.0, 44.0, 894.0, 42.0, 894.0]],
             'area': 136.0,
             'iscrowd': 0}
        ],
        'info': {}
    }


def test_labelbox_to_coco_od_complete(labelbox_example, object_detection_coco_example):
    labelbox_filename, labelbox_map_categories = labelbox_example

    categories_coco = [{
        'supercategory': None,
        'name': 'example_category',
        'id': 1
    }]

    coco_od = labelbox_to_coco(
        labelbox_filename,
        labelbox_map_categories,
        categories_coco,
        image_shape=(1000, 1000)
    )

    assert coco_od == object_detection_coco_example
