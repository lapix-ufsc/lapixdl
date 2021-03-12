from lapixdl.convert.labelbox import labelbox_to_coco, __calculate_area
import json

def test_labelbox_to_coco():
    with open('./tests/convert/labelbox_test.json') as file:
        labelbox_file = json.load(file)

    with open('./tests/convert/coco_expect.json') as file:
        coco_expect = json.load(file)

    conversion = labelbox_to_coco(labelbox_file)

    assert conversion == coco_expect

def test_labelbox_to_coco_w_filter():
    imgs_to_include = [
        "2019_07_10__16_23__0048_b0s0c0x135537-1600y50862-1200m6486.png",
        "2019_07_10__16_23__0048_b0s0c0x123913-1600y119932-1200m15182.png"
    ]

    with open('./tests/convert/labelbox_test.json') as file:
        labelbox_file = json.load(file)

    with open('./tests/convert/coco_expect_filtered.json') as file:
        coco_expect = json.load(file)

    conversion = labelbox_to_coco(labelbox_file, imgs_to_include)

    assert conversion == coco_expect

def test_calculate_area():
    segmentation = [0,0, 0,2, 2,2, 2,0]
    area = __calculate_area(segmentation)

    assert area == 4