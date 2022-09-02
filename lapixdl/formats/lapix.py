from __future__ import annotations

from typing import Any

import pandas as pd
import shapely.wkt

from lapixdl.base import FileTypeError


class LapixDataFrame(pd.DataFrame):
    _metadata = ['geometry', 'category']

    @property
    def _constructor(self):
        return LapixDataFrame


def load(
    filename: str,
    **kwargs: Any
) -> LapixDataFrame:

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
    df_out = lapix_df.copy()
    df_out['geometry'] = df_out['geometry'].apply(lambda x: x.wkt)
    df_out.to_parquet(filename, compression=compression, **kwargs)
