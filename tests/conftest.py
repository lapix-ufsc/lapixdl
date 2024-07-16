from __future__ import annotations

import json

import pandas as pd
import pytest
from shapely.geometry import Polygon


@pytest.fixture
def polygon_example():
    return Polygon([
        (0, 0),
        (0, 10),
        (10, 10),
        (10, 25),
        (15, 28),
        (15, 15),
        (10, 5),
        (5, 5),
    ])


@pytest.fixture
def labelbox_raw(polygon_example):
    return [
        {
            'ID': 'a1', 'External ID': 'tmp/A_xxx.png', 'Skipped': False, 'Reviews': [{
                'score': 1,
                'labelId': 'a1',
            }],
            'Label': {
                'objects': [{
                    'featureId': '<ID for this annotation - 26>',
                    'schemaId': 'Unique_schematicID_for_category_square',
                    'color': '#1CE6FF',
                    'title': 'square',
                    'value': 'square',
                    'polygon': [{'x': x, 'y': y} for x, y in zip(*polygon_example.exterior.xy)],
                    'instanceURI': '<URL for this annotation>',
                }],
            },
        },
        {'ID': 'a2', 'External ID': 'tmp/B_xxx', 'Skipped': True},
    ]


@pytest.fixture
def labelbox_filename(labelbox_raw, tmpdir):
    filename = str(tmpdir.join('labelbox_example.json'))
    with open(filename, 'w+') as outfile:
        json.dump(labelbox_raw, outfile)

    return filename


@pytest.fixture
def labelbox_map_categories():
    return {
        'Unique_schematicID_for_category_square': 1,
    }


@pytest.fixture
def lapix_raw(polygon_example):
    return pd.DataFrame(
        {
            'image_name': ['A.png', 'B.png'],
            'geometry': [polygon_example, polygon_example],
            'category_id': [1, 1],
            'area': [polygon_example.area, polygon_example.area],
            'image_id': [1, 2],
            'iscrowd': [0, 0],
            'image_width': [1000, 1000],
            'image_height': [2000, 2000],
        },
    )


@pytest.fixture
def lapix_filename(lapix_raw, tmpdir):
    filename = str(tmpdir.join('lapix_example.parquet.gzip'))

    lapix_raw['geometry'] = lapix_raw['geometry'].apply(lambda x: x.wkt)
    lapix_raw.to_parquet(filename, index=False, compression='gzip')

    return filename


def pytest_report_header(config):
    import numpy as np
    import matplotlib
    import pandas as pd
    import PIL
    import pyarrow
    import seaborn as sbn
    import shapely

    return f"""\
    main deps:
        - numpy-{np.__version__}
        - matplotlib-{matplotlib.__version__}
        - pandas-{pd.__version__}
        - pillow-{PIL.__version__}
        - pyarrow-{pyarrow.__version__}
        - seaborn-{sbn.__version__}
        - shapely-{shapely.__version__}
"""
