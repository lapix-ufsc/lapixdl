from __future__ import annotations

from typing import Any

import pandas as pd
import shapely.wkt

from lapixdl.base import FileTypeError


class LapixDataFrame(pd.DataFrame):
    """An abstraction of pandas DataFrame for Lapix builtin type"""
    _metadata = ['geometry', 'category']

    @property
    def _constructor(self):
        return LapixDataFrame


def load(
    filename: str,
    **kwargs: Any
) -> LapixDataFrame:
    """Loads the lapix parquet file into a Lapix dataframe

    Args:
        filename (str): The full filename for the parquet file.
        kwargs (Any): Kwargs for the read parquet method from pandas.

    Returns: A Lapix DataFrame from parquet file file.
    """
    if not filename.endswith(('.parquet.gzip', '.parquet')):
        raise FileTypeError('The file is not a parquet file.')

    df = pd.read_parquet(filename, **kwargs)

    lapix_df = LapixDataFrame(df)

    if 'geometry' in df.columns:
        # buffer(0) applied to fix invalid geomtries. From shapely issue #278
        df['geometry'] = df['geometry'].apply(lambda x: shapely.wkt.loads(x).buffer(0))

    return lapix_df


def save(
        lapix_df: LapixDataFrame,
        filename: str,
        compression: str = 'gzip',
        **kwargs: Any
) -> None:
    """Save a Lapix DataFrame into a parquet file

    Args:
        lapix_df (LapixDataFrame): The lapix dataframe to be saved.
        filename (str): The full filename for the parquet file.
        compression (str): The type of compression to be applied need to
    be: `snappy`, `gzip`, `brotli` or None
        kwargs (Any): Kwargs for the `to_parquet` method from pandas.
    """
    df_out = lapix_df.copy()
    df_out['geometry'] = df_out['geometry'].apply(lambda x: x.wkt)
    df_out.to_parquet(filename, compression=compression, **kwargs)


def generate_ids(
    column: pd.Series,
) -> pd.Series:
    column = column.astype('category')
    return column.cat.codes + 1


def geometries_area(
    lapix_df: LapixDataFrame,
) -> pd.Series:
    return pd.Series(lapix_df['geometry'].apply(lambda x: x.area))
