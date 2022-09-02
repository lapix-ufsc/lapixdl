from __future__ import annotations

import multiprocessing
from itertools import chain
from typing import Any

import numpy as np
import pandas as pd
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from lapixdl.base import basename
from lapixdl.formats.lapix import LapixDataFrame


# -----------------------------------------------------------------------
# Functions to work with data from LabelBox (lbox)
def lbox_geo_to_shapely(object: dict[str, Any]) -> Polygon | Point | np.nan:
    keys = object.keys()

    if 'polygon' in keys:
        polygon = object['polygon']
        geometry = Polygon(np.array([(p['x'], p['y']) for p in polygon]))
    elif 'point' in keys:
        point = object['point']
        geometry = Point(np.array([point['x'], point['y']]))
    else:
        geometry = np.NaN
    return geometry


def __lbox_has_review(reviews: list[dict[str, Any]]) -> bool:
    for x in reviews:
        if x['score'] == 1:
            return True
    else:
        return False


def __lbox_drop_duplicate_labels(labelbox_df: pd.DataFrame) -> pd.DataFrame:
    duplicated_idx = labelbox_df['image_name'].duplicated(keep=False)
    df_duplicated = labelbox_df.loc[duplicated_idx, :].copy()

    if df_duplicated.empty:
        return labelbox_df

    # Check the labels that has review
    df_duplicated['have_review'] = df_duplicated.apply(lambda row: __lbox_has_review(row['Reviews']), axis=1)

    # Count the quantity of labels for each row
    df_duplicated['len'] = df_duplicated.apply(lambda row: len(row['Label']['objects']), axis=1)

    # Sort the DF by the quantity of labels
    df_duplicated = df_duplicated.sort_values(['image_name', 'len'], ascending=False)

    # Drop the duplicates labels and keep the first label will be that have more labels
    df_to_keep = df_duplicated.drop_duplicates(['image_name'], keep='first')

    id_to_remove = df_duplicated.loc[~df_duplicated['ID'].isin(df_to_keep['ID'].to_numpy()), 'ID']
    # the rows without review
    id_to_remove = pd.concat([id_to_remove, df_duplicated.loc[~df_duplicated['have_review'], 'ID']])

    df_without_duplicated = labelbox_df[~labelbox_df['ID'].isin(id_to_remove)].copy()

    return df_without_duplicated


def __lbox_explode_images_annotations(labelbox_df: pd.DataFrame) -> pd.DataFrame:
    labelbox_df['objects'] = labelbox_df.apply(lambda row: row['Label']['objects'], axis=1)
    labelbox_df = labelbox_df.explode('objects')
    labelbox_df = labelbox_df.reset_index()

    labelbox_df = labelbox_df.drop(['index', 'Label'], axis=1)
    return labelbox_df


def __lbox_cast_geometries(labelbox_df: pd.DataFrame) -> pd.DataFrame:
    labelbox_df['geometry'] = labelbox_df['objects'].apply(lambda obj: lbox_geo_to_shapely(obj))
    df_out = labelbox_df.dropna(axis=0, subset=['geometry'])

    if labelbox_df.shape != df_out.shape:
        print(f'Some NaN geometries have been deleted! Original shape = {labelbox_df.shape} | out shape = {df_out.shape}')

    if df_out.empty:
        raise ValueError('Data without valid geometries! After transform the geometries the dataframe stay empty.')

    return df_out


def labelbox_to_lapix(
    labelbox_df: pd.DataFrame,
    schematic_to_id: dict[str, int]
) -> LapixDataFrame:
    '''Transform the raw dataframe from LabelBox data to Lapix dataframe'''

    df_out = labelbox_df.copy()

    # Drop ignored images at labelling process
    if 'Skipped' in df_out.columns:
        df_out = df_out.drop(df_out[df_out['Skipped']].index)

    # Drop irrelevant columns
    df_out = df_out.drop(
        [
            'DataRow ID', 'Labeled Data', 'Created By', 'Project Name', 'Dataset Name', 'Created At', 'Updated At',
            'Seconds to Label', 'Agreement', 'Benchmark Agreement', 'Benchmark ID', 'View Label',
            'Has Open Issues', 'Skipped',
        ], axis=1, errors='ignore',
    )

    # Get image names
    df_out['image_name'] = df_out.apply(lambda row: basename(row['External ID']), axis=1)
    df_out = df_out.drop(['External ID'], axis=1)

    # Remove duplicated labels
    df_out = __lbox_drop_duplicate_labels(df_out)

    # Explode annotations to each row
    df_out = __lbox_explode_images_annotations(df_out)

    # Transform labelbox annotation to a geometry
    df_out = __lbox_cast_geometries(df_out)

    # Map category IDs
    df_out['category_id'] = df_out.apply(lambda row: schematic_to_id[row['objects']['schemaId']], axis=1)

    df_out = df_out.drop(['ID', 'objects', 'Reviews'], axis=1)

    lapix_df = LapixDataFrame(df_out)

    return lapix_df


# -----------------------------------------------------------------------
# Functions work with data to COCO
def bounds_to_coco_bb(
    bounds: tuple[float],
    decimals: int = 2,
) -> list[float]:
    # bounds is in  (minx, miny, maxx, maxy)
    # bb  of coco is in [min(x), min(y), max(x)-min(x), max(y)-min(y)]
    b = tuple(np.round(x, decimals) for x in bounds)
    min_x, min_y, max_x, max_y = b
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def pol_to_coco_segment(
    geo: Polygon | MultiPolygon,
    decimals: int = 2,
) -> list[list[float]]:
    # polygon of shapely is a class
    # polygon or segmentation at coco is a list of [[x0, y0, x1, y1 ...]]

    def coco_pol(geometry: Polygon, decimals: int = decimals) -> list[float]:
        list_of_points = list(zip(*geometry.exterior.coords.xy))
        geometry = list(chain(*list_of_points))
        geometry = list(np.around(np.array(geometry), decimals))
        return geometry

    if geo.geom_type == 'Polygon':
        return [coco_pol(geo)]
    elif geo.geom_type == 'MultiPolygon':
        return [coco_pol(g) for g in geo.geoms]
    else:
        raise TypeError(f'Geometry shape is not a polygon or MultiPolygon. This is a {geo.geom_type}.')


def __lapix_to_od_coco_annotations(
    lapix_df: LapixDataFrame,
    decimals: int = 2,
) -> list[dict[str, Any]]:
    cols = lapix_df.columns
    if not all(c in cols for c in ['area', 'image_id', 'iscrowd']):
        raise KeyError('The dataframe need to have the columns `area`, `image_id`, `iscrowd`!')

    return lapix_df.apply(
        lambda row: {
            'id': row.name,
            'image_id': row['image_id'],
            'category_id': row['category_id'],
            'bbox': bounds_to_coco_bb(row['geometry'].bounds, decimals),
            'segmentation': pol_to_coco_segment(row['geometry'], decimals),
            'area': np.round(row['area'], decimals),
            'iscrowd': row['iscrowd'],
        },
        axis=1,
    ).to_numpy().tolist()


def lapix_to_od_coco_annotations(
    lapix_df: LapixDataFrame,
    decimals: int = 2,
    processes: int = 1
) -> list[dict[str, Any]]:

    cols = lapix_df.columns
    if not all(c in cols for c in ['area', 'image_id', 'iscrowd']):
        raise KeyError('The dataframe need to have the columns `area`, `image_id`, `iscrowd`!')

    # To ensure that will have sequential index
    lapix_df.reset_index(drop=True, inplace=True)
    lapix_df.index = lapix_df.index + 1

    # Split equally the annotations for the processes quantity
    ann_ids_splitted = np.array_split(lapix_df.index.tolist(), processes)
    print(f'Number of processes: {processes}, annotations per process: {len(ann_ids_splitted[0])}')

    workers = multiprocessing.Pool(processes=processes)
    procs = []
    for ann_ids in ann_ids_splitted:
        df_to_process = lapix_df.loc[lapix_df.index.isin(ann_ids), :]
        p = workers.apply_async(__lapix_to_od_coco_annotations, (df_to_process, decimals))
        procs.append(p)

    annotations_coco = []
    for p in procs:
        annotations_coco.extend(p.get())

    return annotations_coco


def create_coco_od(
    lapix_df: LapixDataFrame,
    categories_coco: list[dict[str, str]],
    decimals: int = 2,
    processes: int = 1,
    info_coco: dict[str, Any] | None = None
) -> dict[str, Any]:

    lapix_df_clean = lapix_df.drop_duplicates('image_name')

    images_coco = [
        {
            'file_name': row['image_name'],
            'height': row['image_height'],
            'width': row['image_width'],
            'id': row['image_id'],
        } for _, row in lapix_df_clean.iterrows()
    ]

    annotations_coco = lapix_to_od_coco_annotations(lapix_df, decimals, processes)

    coco_od = {
        'categories': categories_coco,
        'images': images_coco,
        'annotations': annotations_coco,
        'info': info_coco if isinstance(info_coco, dict) else {}
    }

    return coco_od
