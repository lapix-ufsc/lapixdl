from __future__ import annotations

import multiprocessing
import os
import sys
from collections import defaultdict
from itertools import chain
from typing import Any
from typing import Iterator

import numpy as np
import pandas as pd
from PIL import Image
from PIL import ImageDraw
from shapely.geometry import MultiPolygon
from shapely.geometry import Point
from shapely.geometry import Polygon

from lapixdl.base import basename
from lapixdl.formats.annotation import Annotation
from lapixdl.formats.lapix import LapixDataFrame
from lapixdl.formats.mask import Mask


# -----------------------------------------------------------------------
# Functions to work with data from LabelBox (lbox)
def lbox_geo_to_shapely(object: dict[str, Any]) -> Polygon | Point | np.nan:
    """Convert the labelbox geometries into shapely geometries
    (For polygons or Points)
    """
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
    """Verify if the labelbox review field has at least one review"""
    for x in reviews:
        if x['score'] == 1:
            return True
    else:
        return False


def __lbox_drop_duplicate_labels(labelbox_df: pd.DataFrame) -> pd.DataFrame:
    """Remove duplicated annotations from labelbox dataframe"""

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
    """Explode image annotations of the Labelbox dataframe"""
    labelbox_df['objects'] = labelbox_df.apply(lambda row: row['Label']['objects'], axis=1)
    labelbox_df = labelbox_df.explode('objects')
    labelbox_df = labelbox_df.reset_index()

    labelbox_df = labelbox_df.drop(['index', 'Label'], axis=1)
    return labelbox_df


def __lbox_cast_geometries(labelbox_df: pd.DataFrame) -> pd.DataFrame:
    """Cast the labelbox geometries into shapely geometries"""
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
    """Generate a bbox in coco format from a tuple of bounds

    Args:
        bounds (tuple[float]): A tuple of floats in the following order
    (min_x, min_y, max_x, max_y)
        decimals: The quantity of decimals to round the data

    Returns:
        list[float]: An bbox into coco format based on the bounds
    """
    # bounds is in  (minx, miny, maxx, maxy)
    # bb  of coco is in [min(x), min(y), max(x)-min(x), max(y)-min(y)]
    b = tuple(np.round(x, decimals) for x in bounds)
    min_x, min_y, max_x, max_y = b
    return [min_x, min_y, max_x - min_x, max_y - min_y]


def pol_to_coco_segment(
    geo: Polygon | MultiPolygon,
    decimals: int = 2,
) -> list[list[float]]:
    """Generate a coco segment from shapely polygon"""
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
    """Convert lapix dataframe annotations into object detection
    annotations in a single process"""
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
    """Convert lapix dataframe annotations into object detection
    annotations in a multiprocesses

    Args:
        lapix_df (LapixDataFrame): The data into Lapix format, need to
    have the columns: `area`, `image_id`, `iscrowd` and `geometry`
        decimals (int): The quantity of decimals desired on the coco
    annotations
        processes (int): The number of processes to be triggered to
    perform the conversion

    Results:
        list[dict[str, Any]]: A list with the annotations in the coco
    for object detection format.
    """

    cols = lapix_df.columns
    if not all(c in cols for c in {'area', 'image_id', 'iscrowd', 'geometry'}):
        raise KeyError('The dataframe need to have the columns `area`, `image_id`, `iscrowd` and `geometry`!')

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


def __generate_coco_od_file(
    lapix_df: LapixDataFrame,
    categories_coco: list[dict[str, str]],
    decimals: int = 2,
    processes: int = 1,
    info_coco: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Create the object detection COCO data

    Args:
        lapix_df (LapixDataFrame): The data into Lapix format, need to
    have the columns: `area`, `image_id`, `iscrowd` and `geometry`.
        categories_coco (list[dict[str, str]]): A list with dicts within
    the categories into coco format.
        decimals (int): The quantity of decimals desired on the coco
    annotations
        processes (int): The number of processes to be triggered to
    perform the conversion
        info_coco (dict[str, Any] | None ): The data to be the `info`
    field of the COCO data.
    """
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


# -----------------------------------------------------------------------
# Functions to work with lapix DataFrame
def lapix_to_annotations(lapix_df: LapixDataFrame) -> list[Annotation]:
    return [
        Annotation(
            row['geometry'],
            row['category_id'],
            row['iscrowd'] if 'iscrowd' in row else 0
        )
        for _, row in lapix_df.iterrows()
    ]


def lapix_to_masks(
    lapix_df: LapixDataFrame,
    draw_order: tuple[int, ...] | None = None,
) -> Iterator[tuple[str, Mask]]:
    df_groupped = lapix_df.groupby('image_id')
    for _, df_by_img in df_groupped:
        annotations = lapix_to_annotations(df_by_img)

        w = df_by_img['image_width'].unique()[0]
        h = df_by_img['image_height'].unique()[0]
        image_name = basename(df_by_img.iloc[0]['image_name'])

        yield (
            image_name,
            annotations_to_mask(annotations, int(w), int(h), draw_order)
        )


# -----------------------------------------------------------------------
# Functions to generate masks
def sort_annotations_to_draw(
    annotations: list[Annotation],
    draw_order: tuple[int, ...] | None = None
) -> list[Annotation]:
    '''Sorts the annotations based on the draw_order. If draw_order
    is None sorts in ascending order based on the category value'''
    items = defaultdict(list)

    for ann in annotations:
        items[ann.category_id].append(ann)

    if draw_order is None:
        draw_order = tuple(np.unique(list(items)))
    elif not isinstance(draw_order, tuple):
        raise ValueError('Unexpected value for `draw_order`. The value needs to be a tuple of int or None.')

    out = [ann for cat in draw_order for ann in items[cat]]

    return out


def draw_annotation(
    target: Image.Image,
    annotation: Annotation,
    value: int | tuple[int, int, int],
) -> Image.Image:
    '''Image: Draw an Annotation into the target (an Image)'''
    for geo in annotation:
        pol_x, pol_y = geo.exterior.coords.xy
        coords = list(zip(pol_x, pol_y))
        ImageDraw.Draw(target).polygon(coords, fill=value)

    return target


def annotations_to_mask(
    annotations: list[Annotation],
    width: int = 1600,
    height: int = 1200,
    draw_order: tuple[int, ...] | None = None,
) -> Mask:
    """Creates a Mask with the specified shape based on the annotations"""
    shape = (height, width)

    annotations_sorted = sort_annotations_to_draw(annotations, draw_order)

    out = Image.fromarray(np.zeros(shape, dtype=np.uint8))
    for ann in annotations_sorted:
        out = draw_annotation(out, ann, ann.category_id)

    return Mask(np.array(out))


def __save_masks_as_files(
    lapix_df: LapixDataFrame,
    output_directory: str,
    mask_extension: str = '.png',
    draw_order: tuple[int, ...] | None = None,
) -> None:

    for image_name, mask in lapix_to_masks(lapix_df, draw_order):
        out_path = os.path.join(
            output_directory,
            image_name + mask_extension
        )
        mask.save(out_path)


def save_lapixdf_as_masks(
    lapix_df: LapixDataFrame,
    output_directory: str,
    mask_extension: str = '.png',
    draw_order: tuple[int, ...] | None = None,
    processes: int = 1
) -> None:
    '''Saves masks as files based on the annotations from a Lapix DataFrame

    Args:
        lapix_df (LapixDataFrame): The data in the Lapix format. Needs to
    have the following columns: `image_id` and `geometry`.
        output_directory (str): The directory where the images should be
    saved.
        mask_extension (str): The image file extension/type. Defaults to
    '.png'.
        draw_order (tuple[int, ...] | None): If draw_order is None sorts
    in ascending order based on the category value. Otherwise this will
    draw the masks based on the tuple sequence.
        processes (int): The number of processes to be used to perform
    the conversion.

    '''
    img_ids = lapix_df['image_id'].unique()

    if len(img_ids) == 0:
        print('There is no annotation to save as mask.', file=sys.stderr)
        return

    images_ids_splitted = np.array_split(img_ids, processes)
    print(
        f'Starting to save {len(img_ids)} semantic segmentation masks using '
        f'{processes} processes with {len(images_ids_splitted[0])} masks per process...',
    )

    workers = multiprocessing.Pool(processes=processes)
    procs = []

    for images_ids in images_ids_splitted:
        df_to_process = lapix_df.loc[lapix_df['image_id'].isin(images_ids), :]
        p = workers.apply_async(__save_masks_as_files, (df_to_process, output_directory, mask_extension, draw_order))
        procs.append(p)

    workers.close()
    workers.join()


# -----------------------------------------------------------------------
# Functions to convert from x to y directly (ex: labelbox to coco)
def labelbox_to_coco(
    labelbox_filename: str,
    schematic_to_id: dict[str, int],
    categories_coco: list[dict[str, str]],
    *,
    target: str = 'OD',
    decimals: int = 2,
    processes: int = 1,
    info_coco: dict[str, Any] | None = None,
    image_shape: tuple[int, int] | None = None
) -> dict[str, Any]:
    """Generate dictionary into COCO format from labelbox

    Args:
        labelbox_filename (str): The full filename for the labelbox
    JSON file
        schematic_to_id (dict[str, int]): A map dictionary between
    schematic id at labelbox and the matching category
        categories_coco (list[dict[str, str]]): A list with dicts within
    the categories into coco format.
        target (str): The desired target format of the COCO file, this
    can be (`OD` or `OBJECT DETECTION`)
        decimals (int): The quantity of decimals desired on the coco
    annotations
        processes (int): The number of processes to be triggered to
    perform the conversion
        info_coco (dict[str, Any] | None ): The data to be the `info`
    field of the COCO data.
        image_shape (tuple[int, int] | None): If a tuple will set this
    shape (height, width) for all images, if None, will try open and
    load the shape of each image.

    Return:
        dict[str, Any]: A dictionary with the data into the COCO target
    format
    """
    from lapixdl.formats import labelbox
    from lapixdl.formats import lapix

    labelbox_df = labelbox.load(labelbox_filename)

    lapix_df = labelbox_to_lapix(labelbox_df, schematic_to_id)

    lapix_df['image_id'] = lapix.generate_ids(lapix_df['image_name'])
    lapix_df['area'] = lapix.geometries_area(lapix_df)
    lapix_df['iscrowd'] = 0  # TODO: Auto check this?

    if isinstance(image_shape, tuple):
        height, width = image_shape[:2]
        lapix_df['image_height'] = height
        lapix_df['image_width'] = width
    else:
        raise NotImplementedError

    if target.upper() in {'OD', 'OBJECT DETECTION'}:
        coco_dict = __generate_coco_od_file(
            lapix_df,
            categories_coco,
            decimals,
            processes,
            info_coco
        )
    else:
        raise NotImplementedError

    return coco_dict
