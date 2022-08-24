from __future__ import annotations
from shapely.geometry import Point
from shapely.geometry import MultiPolygon

from lapixdl.formats.annotation import BBox
from lapixdl.formats.annotation import Annotation
from lapixdl.formats.annotation import bounds_to_BBox

import pytest

from shapely.geometry import Polygon


def test_bbox():
    bbox = BBox(0, 0, 10, 15, 0)
    assert bbox.upper_left_point == (0, 0)
    assert bbox.upper_right_point == (9, 0)
    assert bbox.bottom_right_point == (9, 14)
    assert bbox.bottom_left_point == (0, 14)
    assert bbox.center_point == (4, 7)
    assert bbox.area == 150
    assert bbox.coords == ((0, 0),
                           (9, 0),
                           (9, 14),
                           (0, 14))
    assert bbox.xy == ([0, 9, 9, 0],
                       [0, 0, 14, 14])
    assert bbox.slice_x == slice(0, 9)
    assert bbox.slice_y == slice(0, 14)


def test_bbox_intersection_and_union_area_with():
    bbox_A = BBox(0, 0, 10, 15, 0)
    bbox_B = BBox(5, 5, 20, 25, 0)

    intersection_area = bbox_A.intersection_area_with(bbox_B)
    assert intersection_area == 50

    union_area = bbox_A.union_area_with(bbox_B)
    assert union_area == 600


def test_bbox_to_polygon():
    bbox_A = BBox(0, 0, 10, 15, 0)

    out = bbox_A.to_polygon()

    coords = [(0, 0),
              (9, 0),
              (9, 14),
              (0, 14)]
    assert out.equals_exact(Polygon(coords), 0)


def test_annotation(polygon_example):
    ann = Annotation(polygon_example, 0)

    assert ann.bbox == BBox(0, 0, 15, 28, 0)
    assert ann._geo_type == 'Polygon'
    assert ann.coco_bbox == [0, 0, 15, 28]


def test_annotation_iter(polygon_example):
    ann = Annotation(polygon_example, 1)

    assert len([geo for geo in ann]) == 1

    mult_p = MultiPolygon([polygon_example, polygon_example.buffer(10)])
    ann = Annotation(mult_p, 1)

    assert len([geo for geo in ann]) == 2


def test_annotation_iter_wrong_geo():
    with pytest.raises(TypeError):
        iter(Annotation(Point(0, 0), 0))


def test_annotation_copy(polygon_example):
    ann = Annotation(polygon_example, 1)

    assert ann == ann.copy()


def test_bounds_to_BBox(polygon_example):
    bbox = bounds_to_BBox(polygon_example.bounds, 0)
    assert bbox.upper_left_x == 0
    assert bbox.upper_left_y == 0
    assert bbox.width == 15
    assert bbox.height == 28
    assert bbox.cls == 0
